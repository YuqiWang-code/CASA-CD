"""Run10 preflight: runtime snapshot, CPU audit, one-step smoke per job, readiness."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

import config as C

SOURCE_MODELS = Path(os.environ.get(
    "MINOR_SOURCE_MODELS", "/home/yqwang/projects/CASA-CD/models"))
RUNTIME_MODELS = Path(C.RUNTIME / "models")
sys.path.insert(0, str(RUNTIME_MODELS if RUNTIME_MODELS.is_dir() else SOURCE_MODELS))

from model.casa_net import CASANet  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def count(model):
    return sum(p.numel() for p in model.parameters())


def build_runtime():
    """Fingerprint the staged runtime; only copy when bootstrap did not stage one.

    bootstrap.py is the staging path (it packs the local tree and normalises shell
    scripts). prepare.py must never delete the staged tree: on this layout
    ``C.SCRIPTS.parents[2] / "models"`` *is* ``RUNTIME / "models"``.
    """
    if RUNTIME_MODELS.is_dir():
        source = RUNTIME_MODELS
        mode = "staged"
    else:
        source = SOURCE_MODELS
        mode = "copied"
        if not source.is_dir():
            raise SystemExit(f"no staged runtime and no source tree at {source}")
        shutil.copytree(source, RUNTIME_MODELS,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {}
    for path in sorted(RUNTIME_MODELS.rglob("*.py")):
        files[str(path.relative_to(RUNTIME_MODELS))] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    write(C.ROOT / "runtime_manifest.json", dict(source=str(source), files=files))
    print(f"[PREPARE] runtime {mode}: {len(files)} python files -> {RUNTIME_MODELS}")
    return files


def smoke(arm):
    spec = C.ARMS[arm]
    torch.manual_seed(C.seed_for(arm))
    model = CASANet(None, **C.context_kwargs(arm)).train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    a, b = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
    target = (torch.rand(2, 1, 256, 256) > 0.9).float()
    outputs = model(a, b)
    # ``forward`` returns sigmoid probabilities, so the supervised loss has to be
    # taken on their logits (what train_CASA.py does on the head outputs).
    logits = [torch.logit(value.clamp(1e-6, 1 - 1e-6)) for value in outputs]
    loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(value, target)
               for value in logits) / 4
    if spec["edge_weight"] > 0:
        from model.edge_supervision import edge_loss, sobel_edge_map
        _, auxiliary = model(a, b, return_aux=True)
        edge_target = sobel_edge_map(target, dilate=C.EDGE_DILATE)
        edge_term = sum(edge_loss(map_, torch.nn.functional.interpolate(
            edge_target, size=map_.shape[-2:], mode="area"))
            for map_ in auxiliary["edge_logits"])
        loss = loss + spec["edge_weight"] * edge_term
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss), arm
    deploy = CASANet(None, **C.context_kwargs(arm)).eval().switch_to_deploy()
    payload = dict(arm=arm, module=spec["module"], tar_variant=spec["tar"],
                   decoder_variant="reparam", support=spec["support"],
                   detail_head=spec["detail"], seed=C.seed_for(arm),
                   optimizer_steps=1, loss=loss.detach().item(),
                   trainable_params=count(model), deploy_params=count(deploy))
    expected = C.TRAINABLE_PARAMS.get(arm, 0)
    if expected:
        assert payload["trainable_params"] == expected, payload
        assert payload["deploy_params"] == C.DEPLOY_PARAMS[arm], payload
    print(f"[PREPARE] smoke {arm}: loss={loss.item():.4f} params={count(model)}/{count(deploy)}")
    del model, deploy
    return payload


def fold_on_device(device_name):
    """Fold every arm; CPU folding is enough to prove the deploy graph is equal."""
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA folding but no device is visible")
    report = {}
    for arm in C.ARM_ORDER:
        spec = C.ARMS[arm]
        torch.manual_seed(C.seed_for(arm))
        model = CASANet(None, **C.context_kwargs(arm)).to(device).eval()
        image = torch.randn(2, 6, 256, 256, device=device)
        with torch.no_grad():
            reference = model(image[:, :3], image[:, 3:])
        folded = copy.deepcopy(model).switch_to_deploy().eval()
        with torch.no_grad():
            result = folded(image[:, :3], image[:, 3:])
        difference = float((reference[0] - result[0]).abs().max())
        assert difference < 2e-4, (arm, difference)
        report[arm] = dict(max_diff=difference, deploy_params=count(folded),
                           binary_diff=int(((reference[0] > 0.5) != (result[0] > 0.5)).sum()))
        print(f"[PREPARE] fold {arm} on {device_name}: maxdiff={difference:.3e} "
              f"binary={report[arm]['binary_diff']}")
        del model, folded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-device", default=os.environ.get("MINOR_FOLD_DEVICE", "cpu"))
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    C.ROOT.mkdir(parents=True, exist_ok=True)
    if not args.skip_runtime:
        build_runtime()
    subprocess.run([sys.executable, str(C.SCRIPTS / "validate.py")], check=True)
    results = [smoke(arm) for arm in C.ARM_ORDER]
    folds = fold_on_device(args.fold_device)
    schedule = dict(
        protocol=C.PROTOCOL, slots_per_gpu=C.SLOTS, free_headroom_mib=C.HEADROOM,
        gpu_budget_mib={str(gpu): 32607 - C.HEADROOM for gpu in C.GPUS},
        variant_reserved_mib={arm: C.reservation_mib(arm) for arm in C.ARM_ORDER})
    write(C.ROOT / "scheduling.json", schedule)
    params = dict(trainable={r["arm"]: r["trainable_params"] for r in results},
                  deploy={r["arm"]: r["deploy_params"] for r in results})
    write(C.ROOT / "params.json", params)
    readiness = dict(
        protocol=C.PROTOCOL, passed=True, calibration_split="val", selection_split=C.SELECTION,
        independent_test=True, arms=list(C.ARM_ORDER), gpus=list(C.GPUS), slots=C.SLOTS,
        jobs=[f"{arm}/{dataset}" for arm, dataset in C.experiments()],
        smoke=results, accelerated_folding=folds, fold_device=args.fold_device,
        per_image_metrics=True,
        trainable_params={arm: C.TRAINABLE_PARAMS[arm] for arm in C.ARM_ORDER},
        deploy_params={arm: C.DEPLOY_PARAMS[arm] for arm in C.ARM_ORDER})
    write(C.ROOT / "readiness.json", readiness)
    print(f"[READY] Run10 audit and smoke tests passed: {len(results)} arms, "
          f"{len(C.experiments())} jobs on GPUs {C.GPUS}")
    for arm in C.ARM_ORDER:
        print(f"[PARAMS] {arm} train={params['trainable'][arm]} deploy={params['deploy'][arm]}")


if __name__ == "__main__":
    main()
