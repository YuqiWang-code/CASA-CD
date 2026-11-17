"""Run1 preflight: CPU audit, one-step smoke per arm, scheduling and readiness."""
import copy
import json
import os
import subprocess
import sys

import torch

import config as C

RUNTIME_MODELS = str(C.RUNTIME / "models")
LOCAL_MODELS = str(C.SCRIPTS.parents[2] / "models")
sys.path.insert(0, RUNTIME_MODELS if os.path.isdir(RUNTIME_MODELS) else LOCAL_MODELS)

from model.casa_net import CASANet  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def count(model):
    return sum(p.numel() for p in model.parameters())


def smoke(arm):
    spec = C.ARMS[arm]
    torch.manual_seed(C.SEED)
    model = CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                    decoder_variant=spec["decoder"]).train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    a, b = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
    target = (torch.rand(2, 1, 256, 256) > 0.9).float()
    outputs = model(a, b)
    loss = sum(torch.nn.functional.binary_cross_entropy(value, target) for value in outputs) / 4
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss), arm
    deploy = CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                     decoder_variant=spec["decoder"]).eval().switch_to_deploy()
    payload = dict(arm=arm, module=spec["module"], tar_variant=spec["tar"],
                   decoder_variant=spec["decoder"], selection_split=C.SELECTION,
                   optimizer_steps=1, loss=loss.detach().item(),
                   trainable_params=count(model), deploy_params=count(deploy))
    assert payload["trainable_params"] == C.TRAINABLE_PARAMS[arm], payload
    assert payload["deploy_params"] == C.DEPLOY_PARAMS[arm], payload
    del model, deploy
    return payload


def fold_on_accelerator():
    """Fold every arm on the GPU: CPU folding cannot expose device-placement bugs."""
    if not torch.cuda.is_available():
        print("[PREPARE] no CUDA device, skipping the accelerated folding check")
        return None
    device = torch.device("cuda", 0)
    report = {}
    for arm in C.ARM_ORDER:
        spec = C.ARMS[arm]
        torch.manual_seed(C.SEED)
        model = CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                        decoder_variant=spec["decoder"]).to(device).eval()
        image = torch.randn(2, 6, 256, 256, device=device)
        with torch.no_grad():
            reference = model(image[:, :3], image[:, 3:])
        folded = copy.deepcopy(model).switch_to_deploy().eval()
        with torch.no_grad():
            result = folded(image[:, :3], image[:, 3:])
        difference = float((reference[0] - result[0]).abs().max())
        assert difference < 2e-4, (arm, difference)
        report[arm] = dict(max_diff=difference, deploy_params=count(folded))
        del model, folded
        torch.cuda.empty_cache()
    print("[PREPARE] accelerated folding", json.dumps(report))
    return report


def main():
    torch.set_num_threads(4)
    subprocess.run([sys.executable, str(C.SCRIPTS / "validate.py")], check=True)
    results = [smoke(arm) for arm in C.ARM_ORDER]
    folds = fold_on_accelerator()
    schedule = dict(
        protocol=C.PROTOCOL, slots_per_gpu=C.SLOTS, free_headroom_mib=C.HEADROOM,
        gpu_budget_mib={str(gpu): 32607 - C.HEADROOM for gpu in C.GPUS},
        variant_reserved_mib={arm: C.reservation_mib(arm) for arm in C.ARM_ORDER})
    write(C.ROOT / "scheduling.json", schedule)
    readiness = dict(
        protocol=C.PROTOCOL, passed=True, calibration_split="val", selection_split=C.SELECTION,
        independent_test=True, arms=list(C.ARM_ORDER), datasets=list(C.DATASETS),
        experiments=len(C.experiments()), smoke=results, accelerated_folding=folds,
        trainable_params={arm: C.TRAINABLE_PARAMS[arm] for arm in C.ARM_ORDER},
        deploy_params={arm: C.DEPLOY_PARAMS[arm] for arm in C.ARM_ORDER})
    write(C.ROOT / "readiness.json", readiness)
    print(f"[READY] Run2 audit and smoke tests passed: {len(results)} arms, "
          f"{len(C.experiments())} jobs on GPUs {C.GPUS}")


if __name__ == "__main__":
    main()
