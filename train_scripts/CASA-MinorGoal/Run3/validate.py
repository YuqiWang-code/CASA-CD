"""CPU audit for Run3: job matrix, CLI, parameter counts, folding, detail parity."""
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


def build(arm, seed=None):
    spec = C.ARMS[arm]
    return CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                   decoder_variant=spec["decoder"], support=spec["support"],
                   detail_head=spec["detail"]).eval()


def check_matrix():
    jobs = C.experiments()
    assert len(jobs) == 4, jobs
    assert {job for job in jobs} == {
        ("ARM_C", "WHU-CD-256"), ("ARM_C", "SYSU-CD-256"),
        ("SUPP_PIX", "SYSU-CD-256"), ("DETAIL", "LEVIR-CD-256")}, jobs
    assert C.seed_for("ARM_C") == C.seed_for("SUPP_PIX") == 2334
    assert C.seed_for("DETAIL") == 2333
    assert C.ARMS["SUPP_PIX"]["support"] == "pixel"
    assert C.ARMS["ARM_C"]["support"] == "mass"
    assert C.ARMS["DETAIL"]["detail"] and C.ARMS["DETAIL"]["module"] == "difference"
    assert C.GPUS == (1,), C.GPUS
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
                 "--selection-split {test,val}",
                 "--support {mass,cell,pixel,none}",
                 "--per-image-metrics", "--detail-head"):
        assert flag in help_text, flag
    print("[VALIDATE] CLI")


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
        delta = max(float((x - y).abs().max()) for x, y in zip(values, deployed))
        assert delta <= 2e-4, (arm, delta)
        binary = sum(int((x > 0.5).sum()) for x in values) - sum(int((y > 0.5).sum()) for y in deployed)
        assert binary == 0, (arm, binary)
        print(f"[VALIDATE] {arm} trainable={count(model)} deploy={count(folded)} "
              f"fold_maxdiff={delta:.3e}")
        del model, folded
    return outputs


def check_detail_parity():
    """An untrained detail head is exactly a no-op: the arm must equal DIFF bit for bit."""
    torch.manual_seed(1234)
    diff = CASANet(None, module="difference", tar_variant="spp", decoder_variant="reparam",
                   support="mass").eval()
    torch.manual_seed(1234)
    detail = build("DETAIL")
    left_state, right_state = diff.state_dict(), detail.state_dict()
    extra = set(right_state) - set(left_state)
    assert extra == {"detail.0.weight", "detail.1.weight", "detail.3.weight", "detail.3.bias"}, extra
    for key in left_state:
        assert torch.equal(left_state[key], right_state[key]), key
    with torch.no_grad():
        a, b = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
        left, right = diff(a, b)[0], detail(a, b)[0]
    difference = float((left - right).abs().max())
    print(f"[VALIDATE] DETAIL vs DIFF untrained maxdiff={difference:.3e} "
          f"binary_diff={int(((left > 0.5) != (right > 0.5)).sum())}")
    assert difference == 0.0, difference


def check_support_pixel():
    """`--support pixel` must change the logits, otherwise the arm is a silent no-op."""
    torch.manual_seed(7)
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    mass, pixel = build("ARM_C"), build("SUPP_PIX")
    assert mass.context.support_prior == "mass" and pixel.context.support_prior == "pixel"
    with torch.no_grad():
        left, right = mass(a, b)[0], pixel(a, b)[0]
    difference = float((left - right).abs().max())
    # ``forward`` returns probabilities, so the range check is on probabilities.
    print(f"[VALIDATE] support mass vs pixel maxdiff={difference:.3e} "
          f"probability_range=({float(left.min()):.3f},{float(left.max()):.3f})")
    assert 0.0 <= float(left.min()) and float(left.max()) <= 1.0
    assert difference > 1e-4, "support pixel is a no-op"


def check_gradients():
    torch.manual_seed(0)
    a, b = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
    target = (torch.rand(2, 1, 128, 128) > 0.9).float()
    for arm in ("ARM_C", "SUPP_PIX"):
        model = build(arm).train()
        values = model(a, b)
        logits = [torch.logit(value.clamp(1e-6, 1 - 1e-6)) for value in values]
        sum(torch.nn.functional.binary_cross_entropy_with_logits(value, target)
            for value in logits).backward()
        assert float(model.context.gamma_context.grad.norm()) > 0, arm
        assert float(model.tar.stage1.spp.bn_branches[3].weight.grad.norm()) > 0, arm
        print(f"[VALIDATE] {arm} context and 7x7 SPP branch carry gradient")
        del model

    from train_CASA import loss_function, training_losses  # noqa: E402
    model = build("DETAIL").train()
    outputs, aux = model(a, b, return_aux=True)
    main = loss_function(outputs, target)
    detail_logits = aux["detail_logits"]
    detail_target = (torch.nn.functional.adaptive_avg_pool2d(target, detail_logits.shape[-2:])
                     if aux["detail_target"] == "area" else
                     torch.nn.functional.interpolate(target, size=detail_logits.shape[-2:],
                                                     mode="nearest-exact"))
    detail = loss_function((detail_logits,), detail_target)
    total, total_main, auxiliary = training_losses(outputs, aux, target)
    print(f"[VALIDATE] detail loss: main={float(main.detach()):.5f} "
          f"detail={float(detail.detach()):.5f} ratio={float(detail.detach() / main.detach()):.3f} "
          f"main_with_detail={float(total_main.detach()):.5f} "
          f"auxiliary={float(auxiliary.detach()):.5f} total={float(total.detach()):.5f} "
          f"additive={abs(float(total_main.detach() - main.detach() - detail.detach())) < 1e-6}")
    assert float(detail.detach()) > 0


def main():
    check_matrix()
    check_cli()
    check_models()
    check_detail_parity()
    check_support_pixel()
    check_gradients()
    print("[VALIDATE] Run3 audit passed")


if __name__ == "__main__":
    main()
