"""CPU audit for Run6: job matrix, CLI, parameters, folding, edge heads, gradients."""
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
                   aggregate_mode=spec.get("aggregate", "lorentz")).eval()


def check_matrix():
    jobs = C.experiments()
    assert len(jobs) == 3, jobs
    assert {job for job in jobs} == {(arm, dataset) for dataset in C.DATASETS
                                     for arm in C.ARM_ORDER}, jobs
    assert C.DATASETS == ("SYSU-CD-256",), C.DATASETS
    assert all(C.seed_for(arm) == 2334 for arm in C.ARM_ORDER)
    assert C.GPUS == (1,), C.GPUS
    for arm in C.ARM_ORDER:
        spec = C.ARMS[arm]
        assert spec["module"] == "manifold" and spec["tar"] == "cg", arm
    for arm in C.ARM_ORDER:
        assert C.ARMS[arm]["edge_weight"] == 0.1, arm
        assert C.ARMS[arm]["edge_detach"] == "retrieval", arm
    assert C.ARMS["EDGE_ALL"]["edge_scales"] == "all"
    assert C.ARMS["EDGE_FINE"]["edge_scales"] == "fine"
    assert C.ARMS["EDGE_ALL"]["aggregate"] == "lorentz"
    assert C.ARMS["EUCLID_ALL"]["aggregate"] == "euclidean"
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


def check_edge_heads():
    """Only the edge arm carries the two 65-parameter taps, and they disappear on deploy."""
    full, cut = build("EDGE_ALL"), build("EDGE_FINE")
    for model in (full, cut):
        assert model.tar.edge_head is not None and model.context.edge_head is not None
        taps = sum(p.numel() for p in model.tar.edge_head.parameters())
        taps += sum(p.numel() for p in model.context.edge_head.parameters())
        assert taps == 130, taps
        assert count(model) == C.TRAINABLE_PARAMS["EDGE_ALL"], count(model)
    euclid = build("EUCLID_ALL")
    assert full.context.edge_detach == "retrieval"
    assert cut.context.edge_detach == "retrieval"
    assert full.context.aggregate_mode == "lorentz"
    assert euclid.context.aggregate_mode == "euclidean"
    armed = full

    # the edge target must not fire on the crop border
    from model.edge_supervision import sobel_edge_map
    full = torch.ones(1, 1, 64, 64)
    assert int(sobel_edge_map(full, dilate=1).gt(0).sum()) == 0

    # training mode exposes the maps, inference does not pay for them
    a, b = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    armed.train()
    _, auxiliary = armed(a, b, return_aux=True)
    maps = auxiliary["edge_logits"]
    assert maps is not None and len(maps) == 8, None if maps is None else len(maps)
    with torch.no_grad():
        armed.eval()
        armed.switch_to_deploy()
        assert armed.tar.edge_head is None and armed.context.edge_head is None
        assert count(armed) == C.DEPLOY_PARAMS["EDGE_ALL"]
    print(f"[VALIDATE] edge heads: 130 trainable parameters, 8 maps, dropped on deploy")


def check_manifold_aggregation():
    model = build("EDGE_ALL")
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
    target_mask = torch.zeros(2, 1, 128, 128)
    target_mask[:, :, 32:96, 24:104] = 1.0
    for arm in C.ARM_ORDER:
        model = build(arm).train()
        # open the residual gates so the context branch can receive gradients
        with torch.no_grad():
            model.context.gamma_context.fill_(0.5)
            model.context.level_scale.fill_(0.5)
        values = model(a, b)
        logits = [torch.logit(value.clamp(1e-6, 1 - 1e-6)) for value in values]
        sum(torch.nn.functional.binary_cross_entropy_with_logits(value, target)
            for value in logits).backward()
        for name in ("tar.stage1.content.weight", "tar.stage1.difference.weight",
                     "tar.stage1.gate.weight", "context.qk.weight", "context.out.weight",
                     "context.gate.weight", "context.level.0.weight",
                     "context.support_head.weight", "context.level_scale"):
            grad = model.get_parameter(name).grad
            assert grad is not None and float(grad.norm()) > 0, (arm, name)
        if C.ARMS[arm]["edge_weight"] > 0:
            # the edge taps only see the edge loss, so audit them with that loss
            from model.edge_supervision import edge_loss, sobel_edge_map
            model.zero_grad(set_to_none=True)
            _, auxiliary = model(a, b, return_aux=True)
            maps = auxiliary["edge_logits"]
            assert maps is not None and len(maps) == 8, None if maps is None else len(maps)
            target = sobel_edge_map(target_mask, dilate=C.EDGE_DILATE)
            loss = sum(edge_loss(m, torch.nn.functional.interpolate(
                target, size=m.shape[-2:], mode="area")) for m in maps)
            loss.backward()
            for name in ("tar.edge_head.proj.weight", "context.edge_head.proj.weight"):
                grad = model.get_parameter(name).grad
                assert grad is not None and float(grad.norm()) > 0, (arm, name)
        else:
            assert getattr(model.tar, "edge_head", None) is None
            assert getattr(model.context, "edge_head", None) is None
        print(f"[VALIDATE] {arm} carrier gradients non-zero")
        del model


def main():
    check_matrix()
    check_cli()
    check_models()
    check_edge_heads()
    check_manifold_aggregation()
    check_gradients()
    print("[VALIDATE] Run6 audit passed")


if __name__ == "__main__":
    main()
