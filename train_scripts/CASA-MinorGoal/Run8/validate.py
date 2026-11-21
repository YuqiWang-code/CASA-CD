"""CPU audit for Run8: the single pooling axis, its parameters, folding and gradients."""
import copy
import os
import subprocess
import sys

import torch

import config as C

RUNTIME_MODELS = str(C.RUNTIME / "models")
LOCAL_MODELS = str(C.SCRIPTS.parents[2] / "models")
sys.path.insert(0, RUNTIME_MODELS if os.path.isdir(RUNTIME_MODELS) else LOCAL_MODELS)

from model.casa_net import CASANet  # noqa: E402


def count(model):
    return sum(p.numel() for p in model.parameters())


def build(arm):
    spec = C.ARMS[arm]
    return CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                   decoder_variant=spec["decoder"], support=spec["support"],
                   detail_head=spec["detail"],
                   edge_supervision=spec["edge_weight"] > 0,
                   edge_detach=spec.get("edge_detach", "retrieval"),
                   aggregate_mode=spec.get("aggregate", "lorentz"),
                   prototype_pooling=spec.get("prototype_pooling", "euclidean")).eval()


def check_matrix():
    jobs = C.experiments()
    assert len(jobs) == 2, jobs
    assert {job for job in jobs} == {(arm, dataset) for dataset in C.DATASETS
                                     for arm in C.ARM_ORDER}, jobs
    assert C.DATASETS == ("SYSU-CD-256",), C.DATASETS
    assert all(C.seed_for(arm) == 2334 for arm in C.ARM_ORDER)
    assert C.GPUS == (1,), C.GPUS
    expected = {"POOL_EUCLID": ("manifold", "cg", "euclidean"),
                "POOL_LORENTZ": ("manifold", "cg", "lorentz")}
    for arm, (module, tar, pooling) in expected.items():
        spec = C.ARMS[arm]
        assert (spec["module"], spec["tar"], spec["prototype_pooling"]) == (module, tar, pooling), arm
        assert spec["edge_weight"] == 0.0, arm
    assert C.ARMS["POOL_EUCLID"]["prototype_pooling"] == "euclidean"
    assert C.ARMS["POOL_LORENTZ"]["prototype_pooling"] == "lorentz"
    assert all(C.ARMS[arm]["edge_scales"] == "all" for arm in C.ARM_ORDER)
    assert all(C.ARMS[arm]["aggregate"] == "lorentz" for arm in C.ARM_ORDER)
    # the pooling switch is a pure geometric operation: identical capacity by construction
    assert len({C.TRAINABLE_PARAMS[a] for a in C.ARM_ORDER}) == 1
    assert len({C.DEPLOY_PARAMS[a] for a in C.ARM_ORDER}) == 1
    print("[VALIDATE] matrix")


def check_cli():
    script = str(C.train_script())
    if not os.path.isfile(script):
        script = os.path.join(LOCAL_MODELS, "train_CASA.py")
    help_text = subprocess.run([sys.executable, script, "--help"],
                               capture_output=True, text=True, check=True).stdout
    for flag in ("--module {none,hyperbolic,qkv,matched,difference,manifold}",
                 "--tar-variant {reparam,plain,diff,spp,spp1,cg}",
                 "--decoder-variant {reparam,plain,none}",
                 "--selection-split {test,val}", "--support",
                 "--edge-weight", "--edge-dilate", "--edge-scales {all,fine}",
                 "--aggregate {lorentz,euclidean}", "--per-image-metrics"):
        assert flag in help_text, flag
    print("[VALIDATE] CLI")


def check_models():
    torch.manual_seed(0)
    a, b = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    torch.manual_seed(11)
    control = CASANet(None, module="none", tar_variant="reparam").eval()
    with torch.no_grad():
        control_output = control(a, b)[0]
    for arm in C.ARM_ORDER:
        model = build(arm)
        assert count(model) == C.TRAINABLE_PARAMS[arm], (arm, count(model))
        with torch.no_grad():
            values = model(a, b)
        assert len(values) == 4 and all(tuple(v.shape) == (1, 1, 128, 128) for v in values), arm
        assert all(torch.isfinite(v).all() for v in values), arm
        assert not torch.allclose(values[0], control_output, atol=1e-4), f"{arm} is a no-op"
        folded = copy.deepcopy(model).switch_to_deploy()
        assert count(folded) == C.DEPLOY_PARAMS[arm], (arm, count(folded))
        with torch.no_grad():
            deployed = folded(a, b)
        delta = max(float((x - y).abs().max()) for x, y in zip(values, deployed))
        assert delta <= 2e-4, (arm, delta)
        binary = sum(int((x > 0.5).sum()) for x in values) - sum(int((y > 0.5).sum()) for y in deployed)
        assert binary == 0, (arm, binary)
        print(f"[VALIDATE] {arm} trainable={count(model)} deploy={count(folded)} "
              f"fold_maxdiff={delta:.3e}")
        del model, folded


def check_pooling_modes():
    """The two arms must differ in the pooling location and in nothing else."""
    euclid = build("POOL_EUCLID")
    lorentz = build("POOL_LORENTZ")
    assert euclid.context.prototype_pooling == "euclidean"
    assert lorentz.context.prototype_pooling == "lorentz"
    both = [euclid, lorentz]
    assert all(model.context.aggregate_mode == "lorentz" for model in both)
    assert all(model.tar_variant == "cg" for model in both)
    # the bridge must not carry a learnable scale between the two streams
    names = [name for name, _ in lorentz.named_parameters()]
    assert not any("beta" in name for name in names), names
    # identical capacity: the switch is geometry, not parameters
    assert count(euclid) == count(lorentz)
    print("[VALIDATE] pooling modes and the bridge scale")


def check_manifold_aggregation():
    model = build("POOL_LORENTZ")
    context = model.context
    with torch.no_grad():
        values = torch.randn(2, 1, 7, 8)
        weights = torch.softmax(torch.randn(2, 1, 5, 7), dim=-1)
        manifold = context.aggregate(weights, values)
        euclidean = weights @ values
    difference = float((manifold - euclidean).abs().max())
    assert difference > 1e-3, difference
    print(f"[VALIDATE] manifold vs euclidean aggregation maxdiff={difference:.3e}")


def check_gradients():
    torch.manual_seed(0)
    a, b = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
    target = (torch.rand(2, 1, 128, 128) > 0.9).float()
    # carriers that every arm must train: the bridge and the segmentation head
    common = ("head.weight",)
    bridge = ("tar.stage1.content.weight", "tar.stage1.difference.weight",
              "tar.stage1.gate.weight")
    # carriers that only the retrieval side owns
    retrieval = ("context.qk.weight", "context.out.weight", "context.gate.weight")
    for arm in C.ARM_ORDER:
        model = build(arm).train()
        if C.ARMS[arm]["module"] == "manifold":
            with torch.no_grad():
                model.context.gamma_context.fill_(0.5)
                model.context.level_scale.fill_(0.5)
        values = model(a, b)
        logits = [torch.logit(value.clamp(1e-6, 1 - 1e-6)) for value in values]
        sum(torch.nn.functional.binary_cross_entropy_with_logits(value, target)
            for value in logits).backward()
        names = list(common)
        if C.ARMS[arm]["tar"] == "cg":
            names += list(bridge)
        if C.ARMS[arm]["module"] == "manifold":
            names += list(retrieval) + ["context.level_scale", "context.support_head.weight"]
        for name in names:
            grad = model.get_parameter(name).grad
            assert grad is not None and float(grad.norm()) > 0, (arm, name)
        print(f"[VALIDATE] {arm} carrier gradients non-zero")


def main():
    check_matrix()
    check_cli()
    check_models()
    check_pooling_modes()
    check_manifold_aggregation()
    check_gradients()
    print("[VALIDATE] Run8 audit passed")


if __name__ == "__main__":
    main()
