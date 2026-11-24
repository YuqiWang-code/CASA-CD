"""CPU audit for Run10: job matrix, CLI, parameters, folding, retrieval options."""
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
    # one factor per arm, relative to R_FULL
    full = C.BASE_ARMS["R_FULL"]
    assert C.BASE_ARMS["R_RES"]["retrieval_scales"] == (2, 3)
    assert C.BASE_ARMS["R_RES"]["compact"] is False
    assert C.BASE_ARMS["R_COMPACT"]["compact"] is True
    assert C.BASE_ARMS["R_COMPACT"]["support"] == "confidence"
    assert C.BASE_ARMS["R_LEAN"] == dict(full, retrieval_scales=(2, 3), compact=True,
                                         support="confidence")
    assert C.BASE_ARMS["D_CTRL"]["module"] == "none"
    assert C.BASE_ARMS["D_CG"]["module"] == "none" and C.BASE_ARMS["D_CG"]["tar"] == "cg"
    assert C.BASE_ARMS["R_FULL"]["module"] == "manifold"
    assert set(C.GATES) == set(C.BASE_ORDER) - {"D_CTRL"}, C.GATES
    for name, gate in C.GATES.items():
        assert gate["versus"] in C.BASE_ORDER, (name, gate)
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


def check_retrieval_options():
    """The lean arms must really shrink the retrieval, and the full arm must not."""
    for arm in C.ARM_ORDER:
        spec = C.ARMS[arm]
        model = build(arm)
        if spec["module"] != "manifold":
            assert model.module == "none", arm
            continue
        context = model.context
        assert context.retrieval_scales == tuple(spec["retrieval_scales"]), (arm, context.retrieval_scales)
        expected_tokens = 32 if spec["compact"] else 224
        assert sum(context.tokens_per_scale) == expected_tokens, (arm, sum(context.tokens_per_scale))
        assert context.support_prior == spec["support"], (arm, context.support_prior)
        # the centring gate starts closed: keeping the mean is the default
        assert float(context.mean_gate.detach()) == 0.0, arm
    print("[VALIDATE] retrieval scales, prototype budget and priors")


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
        if C.ARMS[arm]["module"] == "manifold":
            with torch.no_grad():
                model.context.gamma_context.fill_(0.5)
                model.context.level_scale.fill_(0.5)
        values = model(a, b)
        logits = [torch.logit(value.clamp(1e-6, 1 - 1e-6)) for value in values]
        sum(torch.nn.functional.binary_cross_entropy_with_logits(value, target)
            for value in logits).backward()
        names = ["head.weight", "fine.out.weight"]
        if C.ARMS[arm]["tar"] == "cg":
            names += ["tar.stage1.content.weight", "tar.stage1.difference.weight",
                      "tar.stage1.gate.weight"]
        if C.ARMS[arm]["module"] == "manifold":
            names += ["context.qk.weight", "context.out.weight", "context.gate.weight",
                      "context.mean_gate"]
        for name in names:
            grad = model.get_parameter(name).grad
            assert grad is not None and float(grad.norm()) > 0, (arm, name)
        print(f"[VALIDATE] {arm} carrier gradients non-zero")


def main():
    check_matrix()
    check_cli()
    check_models()
    check_retrieval_options()
    check_bridge_is_a_competition()
    check_gradients()
    print("[VALIDATE] Run10 audit passed")


if __name__ == "__main__":
    main()
