"""CPU audit for Run11: job matrix, CLI, parameters, folding, HRA axes, gradients."""
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
    return CASANet(None, **C.context_kwargs(arm)).eval()


def check_matrix():
    jobs = C.experiments()
    assert len(jobs) == len(C.ARM_ORDER) * len(C.DATASETS), jobs
    assert {job for job in jobs} == {(arm, dataset) for dataset in C.DATASETS
                                     for arm in C.ARM_ORDER}, jobs
    assert C.DATASETS == ("SYSU-CD-256", "LEVIR-CD-256"), C.DATASETS
    assert C.SEEDS == (2334, 2335), C.SEEDS
    for base in C.BASE_ORDER:
        seeds = sorted(C.seed_of(arm) for arm in C.ARM_ORDER if C.base_of(arm) == base)
        assert seeds == list(C.SEEDS), (base, seeds)
    # HRA and E-HRA differ only on the three geometry axes
    h32, e32 = C.BASE_ARMS["H32"], C.BASE_ARMS["E32"]
    assert C.BASE_ARMS["H224"] == dict(h32, hra_grids=C.FLAT), C.BASE_ARMS["H224"]
    assert (e32["hra_pool"], e32["hra_distance"], e32["hra_value"]) == ("euclidean",) * 3, e32
    assert (h32["hra_pool"], h32["hra_distance"], h32["hra_value"]) == ("lorentz",) * 3, h32
    hv = C.BASE_ARMS["E32_HV"]
    assert hv == dict(e32, hra_value="lorentz"), hv
    assert C.BASE_ARMS["CTRL"]["module"] == "none", C.BASE_ARMS["CTRL"]
    assert C.BASE_ARMS["LOCAL"]["module"] == "local", C.BASE_ARMS["LOCAL"]
    assert set(C.GATES) == set(C.BASE_ORDER) - {"CTRL"}, C.GATES
    for name, gate in C.GATES.items():
        assert gate["versus"] in C.BASE_ORDER, (name, gate)
    print("[VALIDATE] matrix")


def check_cli():
    script = str(C.train_script())
    if not os.path.isfile(script):
        script = os.path.join(LOCAL_MODELS, "train_CASA.py")
    help_text = subprocess.run([sys.executable, script, "--help"],
                               capture_output=True, text=True, check=True).stdout
    for flag in ("--module {none,hyperbolic,qkv,matched,difference,manifold,hra,local}",
                 "--tar-variant {reparam,plain,diff,spp,spp1,cg}",
                 "--decoder-variant {reparam,plain,none}",
                 "--selection-split {test,val}", "--support",
                 "--hra-pool {euclidean,lorentz}", "--hra-distance {euclidean,lorentz}",
                 "--hra-value {euclidean,lorentz}", "--hra-grids",
                 "--guide-source {tar,diff}",
                 "--aggregate {lorentz,euclidean}", "--prototype-pooling",
                 "--retrieval-scales", "--compact-prototypes",
                 "--ds-weights", "--per-image-metrics"):
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
        expected = C.TRAINABLE_PARAMS.get(arm, 0)
        if expected:
            assert count(model) == expected, (arm, count(model))
        with torch.no_grad():
            values = model(a, b)
        assert len(values) == 4 and all(tuple(v.shape) == (1, 1, 128, 128) for v in values), arm
        assert all(torch.isfinite(v).all() for v in values), arm
        assert not torch.allclose(values[0], control_output, atol=1e-4), f"{arm} is a no-op"
        folded = copy.deepcopy(model).switch_to_deploy()
        if expected:
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


def check_hra_axes():
    """With the write-back open, the three axes must each be a real switch."""
    torch.manual_seed(0)
    a, b = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    outputs, specs = {}, {}
    for base in C.BASE_ORDER:
        arm = C.arm_name(base, C.SEEDS[0])
        spec = C.ARMS[arm]
        if spec["module"] != "hra":
            continue
        model = build(arm)
        with torch.no_grad():
            model.context.gamma_context.fill_(0.5)
            model.context.level_scale.fill_(0.5)
            outputs[base] = model(a, b)[0]
        context = model.context
        assert tuple(context.grid_sizes) == tuple(spec["hra_grids"]), (base, context.grid_sizes)
        assert sum(context.tokens_per_scale) == (224 if is_flat(spec["hra_grids"]) else 32), base
        assert context.support_prior == "mass", base
        assert float(context.mean_gate.detach()) == 0.0, base
        report = context.axis_report()
        assert report["member_pool"] == spec["hra_pool"], (base, report)
        assert report["distance"] == spec["hra_distance"], (base, report)
        assert report["value_aggregate"] == spec["hra_value"], (base, report)
        specs[base] = report
    reference = outputs["H32"]
    for base in ("E32", "E32_HV", "H224"):
        delta = float((outputs[base] - reference).abs().max())
        assert delta > 1e-5, (base, "geometry switch is a no-op", delta)
        print(f"[VALIDATE] {base} differs from H32 by {delta:.3e}")
    for base in ("LOCAL", "CTRL"):
        arm = C.arm_name(base, C.SEEDS[0])
        model = build(arm)
        assert sum(model.context.tokens_per_scale) == 0, base
        if base == "LOCAL":
            assert not hasattr(model.context, "qk") and not hasattr(model.context, "v"), base
        else:
            assert not hasattr(model.context, "qk"), base
    print(f"[VALIDATE] HRA axis reports {specs}")


def is_flat(grids):
    return tuple(grids) == tuple(C.FLAT)


def check_bridge_is_a_competition():
    """The change gate must mix, not boost: weights sum to one at init."""
    from model.str_tar import ChangeGateTARStage
    stage = ChangeGateTARStage(64, 64)
    p = torch.randn(2, 64, 8, 8)
    q = torch.randn(2, 64, 8, 8)
    fused, delta = stage(p, q)
    assert fused.shape == delta.shape == (2, 64, 8, 8)
    with torch.no_grad():
        gate = stage.gate(torch.zeros(2, 64, 8, 8) + 1e-3).sigmoid()
    assert torch.allclose(gate, torch.full_like(gate, 0.5), atol=1e-4), gate.mean()
    # delta is still a signed stream, so the retrieval sees both directions
    assert float(delta.mean()) < 0.5 and float(delta.min()) < 0.0
    print("[VALIDATE] bridge gate is a 50/50 mix at init and delta stays signed")


def check_gradients():
    torch.manual_seed(0)
    a, b = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
    target = (torch.rand(2, 1, 128, 128) > 0.9).float()
    for arm in C.ARM_ORDER:
        model = build(arm).train()
        if C.ARMS[arm]["module"] == "hra":
            with torch.no_grad():
                model.context.gamma_context.fill_(0.5)
                model.context.level_scale.fill_(0.5)
        elif C.ARMS[arm]["module"] == "local":
            # gamma_detail is the zero-initialised write-back of the local branch
            with torch.no_grad():
                model.context.gamma_detail.fill_(0.5)
        values = model(a, b)
        logits = [torch.logit(value.clamp(1e-6, 1 - 1e-6)) for value in values]
        sum(torch.nn.functional.binary_cross_entropy_with_logits(value, target)
            for value in logits).backward()
        names = ["head.weight", "fine.out.weight"]
        if C.ARMS[arm]["tar"] == "cg":
            names += ["tar.stage1.content.weight", "tar.stage1.difference.weight",
                      "tar.stage1.gate.weight"]
        if C.ARMS[arm]["module"] == "hra":
            names += ["context.qk.weight", "context.v.weight", "context.out.weight",
                      "context.gate.weight", "context.mean_gate", "context.level.0.weight",
                      "context.support_head.weight"]
        if C.ARMS[arm]["module"] == "local":
            names += ["context.local_dw.weight", "context.local_out.weight"]
        for name in names:
            grad = model.get_parameter(name).grad
            assert grad is not None and float(grad.norm()) > 0, (arm, name)
        print(f"[VALIDATE] {arm} carrier gradients non-zero")


def main():
    check_matrix()
    check_cli()
    check_models()
    check_hra_axes()
    check_bridge_is_a_competition()
    check_gradients()
    print("[VALIDATE] Run11 audit passed")


if __name__ == "__main__":
    main()
