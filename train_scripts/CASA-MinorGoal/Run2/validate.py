"""CPU audit for Run2: matrix, CLI, parameter counts, folding, gradients and no-op guards."""
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
from model.str_tar import RepSPPBlock  # noqa: E402


def count(model):
    return sum(p.numel() for p in model.parameters())


def check_matrix():
    jobs = C.experiments()
    assert len(jobs) == len(C.ARM_ORDER) * len(C.DATASETS), jobs
    assert {arm for arm, _ in jobs} == set(C.ARM_ORDER)
    assert {dataset for _, dataset in jobs} == set(C.DATASETS)
    assert all(arm in C.ARMS and dataset in C.DATASETS for arm, dataset in jobs)
    assert all(C.ARMS[arm]["module"] != "none" or C.ARMS[arm]["tar"] in ("reparam", "spp", "spp1")
               for arm in C.ARM_ORDER)
    assert {C.seed_for(arm) for arm in C.ARM_ORDER} == {2333, 2334, 2335}
    assert C.ARMS["DIFF_NOSPP"]["tar"] == "diff" and C.ARMS["DIFF_NOSPP"]["module"] == "difference"
    print("[VALIDATE] matrix")


def check_cli():
    script = str(C.train_script())
    if not os.path.isfile(script):
        script = os.path.join(LOCAL_MODELS, "train_CASA.py")
    help_text = subprocess.run([sys.executable, script, "--help"],
                               capture_output=True, text=True, check=True).stdout
    for flag in ("--module {none,hyperbolic,qkv,matched,difference}",
                 "--tar-variant {reparam,plain,diff,spp,spp1}",
                 "--decoder-variant {reparam,plain,none}",
                 "--selection-split {test,val}"):
        assert flag in help_text, flag
    print("[VALIDATE] CLI")


def build(arm):
    spec = C.ARMS[arm]
    return CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                   decoder_variant=spec["decoder"]).eval()


def check_models():
    torch.manual_seed(0)
    a, b = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    outputs = {}
    for arm in C.ARM_ORDER:
        model = build(arm)
        assert count(model) == C.TRAINABLE_PARAMS[arm], (arm, count(model))
        with torch.no_grad():
            values = model(a, b)
        assert len(values) == 4 and all(tuple(v.shape) == (1, 1, 128, 128) for v in values), arm
        assert all(torch.isfinite(v).all() for v in values), arm
        outputs[arm] = values[0].clone()
        folded = copy.deepcopy(model).switch_to_deploy()
        assert count(folded) == C.DEPLOY_PARAMS[arm], (arm, count(folded))
        with torch.no_grad():
            deployed = folded(a, b)
        assert max(float((x - y).abs().max()) for x, y in zip(values, deployed)) <= 2e-3, arm
        del model, folded
    for arm in ("SPP_S2334", "DIFF_S2334", "SPP1", "DIFF_SPP1", "DIFF_NOSPP"):
        assert not torch.allclose(outputs[arm], outputs["CTRL_S2334"], atol=1e-4), \
            f"{arm} is a silent no-op"
    print("[VALIDATE] parameter counts, output shapes, folding and no-op guards")


def check_gradients():
    torch.manual_seed(0)
    a, b = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
    target = (torch.rand(2, 1, 128, 128) > 0.9).float()

    spp = build("SPP_S2334").train()
    values = spp(a, b)
    sum(torch.nn.functional.binary_cross_entropy(v, target) for v in values).backward()
    assert float(spp.tar.stage1.spp.bn_branches[3].weight.grad.norm()) > 0, "7x7 branch never opens"
    print("[VALIDATE] SPP branch scales carry gradient")

    spp1 = build("SPP1").train()
    values = spp1(a, b)
    sum(torch.nn.functional.binary_cross_entropy(v, target) for v in values).backward()
    assert len(spp1.tar.stage1.spp.dw_branches) == 1, "spp1 must keep a single 1x1 branch"
    assert float(spp1.tar.stage1.spp.dw_branches[0].weight.grad.norm()) > 0
    assert float(spp1.tar.stage1.difference.proj.weight.grad.norm()) > 0
    print("[VALIDATE] 1x1-only pooling is a live branch")

    diff = build("DIFF_S2334").train()
    values = diff(a, b)
    sum(torch.nn.functional.binary_cross_entropy(v, target) for v in values).backward()
    assert float(diff.context.gamma_context.grad.norm()) > 0
    diff.zero_grad()
    with torch.no_grad():
        diff.context.gamma_context.fill_(0.5)
    values = diff(a, b)
    sum(torch.nn.functional.binary_cross_entropy(v, target) for v in values).backward()
    for name in ("qk.weight", "v.weight", "out.weight", "norm_diff.weight"):
        grad = dict(diff.context.named_parameters())[name].grad
        assert grad is not None and float(grad.norm()) > 0, name
    assert float(diff.tar.stage1.difference.proj.weight.grad.norm()) > 0
    print("[VALIDATE] difference retrieval learns on both sides")

    nospp = build("DIFF_NOSPP").train()
    values = nospp(a, b)
    sum(torch.nn.functional.binary_cross_entropy(v, target) for v in values).backward()
    assert float(nospp.context.gamma_context.grad.norm()) > 0
    nospp.zero_grad()
    with torch.no_grad():
        nospp.context.gamma_context.fill_(0.5)
    values = nospp(a, b)
    sum(torch.nn.functional.binary_cross_entropy(v, target) for v in values).backward()
    for name in ("qk.weight", "v.weight", "out.weight", "norm_diff.weight"):
        grad = dict(nospp.context.named_parameters())[name].grad
        assert grad is not None and float(grad.norm()) > 0, name
    assert float(nospp.tar.stage1.temporal.proj.weight.grad.norm()) > 0
    print("[VALIDATE] decoupled difference retrieval learns without the SPP stage")


def check_block():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 16, 16)
    for kernels in ((1, 3, 5, 7), (1,)):
        block = RepSPPBlock(16, kernels=kernels).eval()
        with torch.no_grad():
            reference = block(x)
        folded = copy.deepcopy(block).switch_to_deploy()
        with torch.no_grad():
            assert float((reference - folded(x)).abs().max()) <= 2e-4, kernels
        assert folded.dw.kernel_size == (max(kernels), max(kernels))
    print("[VALIDATE] multi-scale and single-window blocks both fold")


def main():
    check_matrix()
    check_cli()
    check_models()
    check_gradients()
    check_block()
    print("[VALIDATE] Run2 audit passed")


if __name__ == "__main__":
    main()
