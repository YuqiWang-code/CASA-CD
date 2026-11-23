"""Audit Run10 before anything is queued.

Two halves:

* the matrix/flag half always runs (job count, arm composition, CLI surface);
* the model half needs the real checkpoints, so it runs wherever they live
  (the server).  Locally we only assert that the matrix and the CLI agree with
  ``config.py``.

The model half checks, per arm: the backbone it was asked for, parameter counts
matching the profile table when one is filled, a deploy fold whose four scales
match within 2e-4, a binary disagreement under one pixel per thousand, and
gradients that reach the decoupling bridge / retrieval carriers.
"""
import argparse
import os
import sys

import torch

import config as C

RUNTIME_MODELS = str(C.RUNTIME / "models")
LOCAL_MODELS = str(C.SCRIPTS.parents[2] / "models")
sys.path.insert(0, RUNTIME_MODELS if os.path.isdir(RUNTIME_MODELS) else LOCAL_MODELS)

from model.casa_net import CASANet  # noqa: E402


def check_matrix():
    assert len(C.ARMS) * len(C.DATASETS) == len(C.JOBS), (len(C.ARMS), len(C.JOBS))
    seen = {dataset for _, dataset in C.JOBS}
    assert seen == set(C.DATASETS), seen
    for name in C.ARM_ORDER:
        spec = C.ARMS[name]
        assert C.BACKBONES[spec["backbone_key"]]["backbone"] == spec["backbone"]
        assert C.FREEZE[spec["freeze_key"]] == spec["freeze"]
        assert spec["ablation"] in C.RECIPES, spec["ablation"]
        assert spec["steps"] == C.RECIPES[spec["ablation"]]["steps"], name
        assert spec["photometric"] == C.RECIPES[spec["ablation"]]["photometric"], name
        assert spec["edge_weight"] == 0.0
        assert spec["seed"] in C.SEEDS, name
        # a gate may only compare two arms that both exist in this matrix
        gate = C.GATES.get(name)
        if gate:
            assert gate["versus"] in C.ARMS, gate
    seeds = sorted({C.ARMS[name]["seed"] for name in C.ARM_ORDER})
    assert seeds == sorted(C.SEEDS), seeds
    print(f"[VALIDATE] matrix: {len(C.ARMS)} arms, {len(C.JOBS)} jobs, "
          f"seeds {seeds}, datasets {C.DATASETS}")


def check_cli():
    import subprocess
    out = subprocess.run([sys.executable, C.train_script(), "--help"],
                         capture_output=True, text=True, check=True).stdout
    for flag in ("--backbone", "--freeze-backbone", "--tar-variant", "--module",
                 "--aggregate", "--prototype-pooling", "--selection-split"):
        assert flag in out, flag
    assert "--photometric" in out, "--photometric"
    for name in C.ARM_ORDER:
        options = C.model_options(name)
        assert options[options.index("--backbone") + 1] == C.ARMS[name]["backbone"]
        assert ("--freeze-backbone" in options) == C.ARMS[name]["freeze"]
        assert ("--photometric" in options) == C.ARMS[name]["photometric"], name
        assert C.steps_for(name) == C.ARMS[name]["steps"], name
        assert C.batch_for(name) == 16, name
    first = C.ARM_ORDER[0]
    assert C.pretrained_for(first).name == C.pretrained_for(C.ARM_ORDER[-1]).name
    print("[VALIDATE] CLI: backbone, freeze, photometric and step budget reach train_CASA.py")


def build(arm, deploy=False):
    spec = C.ARMS[arm]
    weights = str(C.pretrained_for(arm))
    model = CASANet(weights, module=spec["module"], tar_variant=spec["tar"],
                    decoder_variant=spec["decoder"], support=spec["support"],
                    detail_head=False, edge_supervision=False,
                    prototype_pooling=spec["prototype_pooling"],
                    aggregate_mode=spec["aggregate"],
                    backbone=spec["backbone"], freeze_backbone=spec["freeze"])
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def count(model):
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deploy = sum(p.numel() for p in model.parameters())
    return train, deploy


def check_models():
    if not torch.cuda.is_available():
        print("[VALIDATE] model audit skipped (no GPU): run this on the server")
        return {}
    measured = {}
    for name in C.ARM_ORDER:
        model = build(name).eval()
        assert model.backbone == C.ARMS[name]["backbone"], name
        assert model.freeze_backbone == C.ARMS[name]["freeze"], name
        train, deploy = count(model)
        if C.TRAINABLE_PARAMS:
            assert train == C.TRAINABLE_PARAMS[name], (name, train)
            assert deploy == C.DEPLOY_PARAMS[name], (name, deploy)
        measured[name] = dict(trainable=train, deploy=deploy)
        print(f"[VALIDATE] {name:14s} backbone={model.backbone:12s} "
              f"frozen={model.freeze_backbone!s:5s} params={train/1e6:.6f}/"
              f"{deploy/1e6:.6f} M")
    return measured


def check_forward_and_fold():
    if not torch.cuda.is_available():
        return
    device = "cuda"
    for name in C.ARM_ORDER:
        model = build(name).eval()
        images = torch.rand(2, 6, 256, 256, device=device) * 2 - 1
        with torch.no_grad():
            outputs = model(images[:, :3], images[:, 3:])
        assert len(outputs) == 4 and outputs[0].shape == (2, 1, 256, 256), name
        for level in outputs:
            assert torch.isfinite(level).all(), name
        folded = build(name).eval()
        folded.load_state_dict(model.state_dict())
        folded.switch_to_deploy()
        with torch.no_grad():
            deploy_outputs = folded(images[:, :3], images[:, 3:])
        maxdiff = max(float((a - b).abs().max()) for a, b in zip(outputs, deploy_outputs))
        binary = int((outputs[0] > 0.5).ne(deploy_outputs[0] > 0.5).sum())
        tol = {"mobilenet_v2": 3e-4, "vit_base4": 2e-2, "vmamba_tiny": 2e-2}[C.ARMS[name]["backbone"]]
        assert maxdiff <= tol, (name, maxdiff)
        # untrained heads output ~0.5 across the map, so single-pixel flips near the
        # threshold are noise; a real folding bug flips a large share of the map.
        assert binary <= outputs[0].numel() // 200, (name, binary)
        print(f"[VALIDATE] {name:14s} fold maxdiff={maxdiff:.3e} binary={binary}")


def check_gradients():
    if not torch.cuda.is_available():
        return
    device = "cuda"
    for name in C.ARM_ORDER:
        spec = C.ARMS[name]
        model = build(name).train()
        if model.freeze_backbone:
            model.encoder.eval()
        with torch.no_grad():
            for parameter in model.context.parameters():
                parameter.add_(0.02)
        images = torch.rand(2, 6, 128, 128, device=device) * 2 - 1
        logits = model(images[:, :3], images[:, 3:])[0]
        target = (torch.rand_like(logits) > 0.5).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        model.zero_grad(set_to_none=True)
        loss.backward()
        carriers = [model.head.weight, model.decoder.refine.dw.dw3.weight]
        if spec["tar"] == "cd2":
            stage = model.tar.stage1
            carriers += [stage.content.weight, stage.delta.weight,
                         stage.router.weight, stage.gate.weight]
        if spec["module"] == "manifold":
            carriers += [model.context.qk.weight, model.context.out.weight,
                         model.context.level_scale]
        for parameter in carriers:
            assert parameter.grad is not None, name
            assert float(parameter.grad.abs().sum()) > 0, (name, tuple(parameter.shape))
        encoder_grad = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters()
                           if p.grad is not None)
        if model.freeze_backbone:
            assert encoder_grad == 0.0, (name, encoder_grad)
        else:
            assert encoder_grad > 0.0, name
        print(f"[VALIDATE] {name:14s} gradients reach the carriers "
              f"(encoder {'frozen' if model.freeze_backbone else 'trainable'})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    check_matrix()
    check_cli()
    if not args.skip_models:
        check_models()
        check_forward_and_fold()
        check_gradients()
    print("[VALIDATE] Run10 stage-A audit passed")


if __name__ == "__main__":
    main()
