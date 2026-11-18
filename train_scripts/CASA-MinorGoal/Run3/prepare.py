"""Run3 preflight: runtime snapshot, CPU audit, one-step smoke per job, readiness."""
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

LOCAL_MODELS = Path(C.SCRIPTS.parents[2] / "models")
RUNTIME_MODELS = Path(C.RUNTIME / "models")
sys.path.insert(0, str(RUNTIME_MODELS if RUNTIME_MODELS.is_dir() else LOCAL_MODELS))

from model.casa_net import CASANet  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def count(model):
    return sum(p.numel() for p in model.parameters())


def build_runtime():
    """Freeze the current ``models/`` tree so later edits cannot change this run."""
    if RUNTIME_MODELS.exists():
        shutil.rmtree(RUNTIME_MODELS)
    shutil.copytree(LOCAL_MODELS, RUNTIME_MODELS,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {}
    for path in sorted(RUNTIME_MODELS.rglob("*.py")):
        files[str(path.relative_to(RUNTIME_MODELS))] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    write(C.ROOT / "runtime_manifest.json", dict(source=str(LOCAL_MODELS), files=files))
    print(f"[PREPARE] runtime frozen: {len(files)} python files -> {RUNTIME_MODELS}")
    return files


def smoke(arm):
    spec = C.ARMS[arm]
    torch.manual_seed(C.seed_for(arm))
    model = CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                    decoder_variant=spec["decoder"], support=spec["support"],
                    detail_head=spec["detail"]).train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    a, b = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
    target = (torch.rand(2, 1, 256, 256) > 0.9).float()
    outputs = model(a, b)
    # ``forward`` returns sigmoid probabilities, so the supervised loss has to be
    # taken on their logits (what train_CASA.py does on the head outputs).
    logits = [torch.logit(value.clamp(1e-6, 1 - 1e-6)) for value in outputs]
    loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(value, target)
               for value in logits) / 4
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss), arm
    deploy = CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                     decoder_variant=spec["decoder"], support=spec["support"],
                     detail_head=spec["detail"]).eval().switch_to_deploy()
    payload = dict(arm=arm, module=spec["module"], tar_variant=spec["tar"],
                   decoder_variant=spec["decoder"], support=spec["support"],
                   detail_head=spec["detail"], seed=C.seed_for(arm),
                   optimizer_steps=1, loss=loss.detach().item(),
                   trainable_params=count(model), deploy_params=count(deploy))
    assert payload["trainable_params"] == C.TRAINABLE_PARAMS[arm], payload
    assert payload["deploy_params"] == C.DEPLOY_PARAMS[arm], payload
    print(f"[PREPARE] smoke {arm}: loss={loss.item():.4f} params={count(model)}/{count(deploy)}")
    del model, deploy
    return payload


def detail_loss_ratio():
    spec = C.ARMS["DETAIL"]
    torch.manual_seed(11)
    model = CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                    decoder_variant=spec["decoder"], support=spec["support"],
                    detail_head=True).train()
    a, b = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
    target = (torch.rand(2, 1, 256, 256) > 0.9).float()
    outputs, aux = model(a, b, return_aux=True)
    from train_CASA import loss_function, training_losses
    main = loss_function(outputs, target)
    detail_logits = aux["detail_logits"]
    detail_target = (torch.nn.functional.adaptive_avg_pool2d(target, detail_logits.shape[-2:])
                     if aux["detail_target"] == "area" else
                     torch.nn.functional.interpolate(target, size=detail_logits.shape[-2:],
                                                     mode="nearest-exact"))
    detail = loss_function((detail_logits,), detail_target)
    total, total_main, auxiliary = training_losses(outputs, aux, target)
    payload = dict(main=float(main.detach()), detail=float(detail.detach()),
                   ratio=float(detail.detach() / main.detach()),
                   main_with_detail=float(total_main.detach()),
                   auxiliary=float(auxiliary.detach()), total=float(total.detach()),
                   additive=abs(float(total_main.detach() - main.detach() - detail.detach())) < 1e-6)
    print(f"[PREPARE] detail loss {payload}")
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
        model = CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                        decoder_variant=spec["decoder"], support=spec["support"],
                        detail_head=spec["detail"]).to(device).eval()
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
    detail = detail_loss_ratio()
    folds = fold_on_device(args.fold_device)
    schedule = dict(
        protocol=C.PROTOCOL, slots_per_gpu=C.SLOTS, free_headroom_mib=C.HEADROOM,
        gpu_budget_mib={str(gpu): 32607 - C.HEADROOM for gpu in C.GPUS},
        variant_reserved_mib={arm: C.reservation_mib(arm) for arm in C.ARM_ORDER})
    write(C.ROOT / "scheduling.json", schedule)
    write(C.ROOT / "detail_loss.json", detail)
    readiness = dict(
        protocol=C.PROTOCOL, passed=True, calibration_split="val", selection_split=C.SELECTION,
        independent_test=True, arms=list(C.ARM_ORDER), gpus=list(C.GPUS), slots=C.SLOTS,
        jobs=[f"{arm}/{dataset}" for arm, dataset in C.experiments()],
        smoke=results, accelerated_folding=folds, fold_device=args.fold_device,
        detail_loss=detail, per_image_metrics=True,
        trainable_params={arm: C.TRAINABLE_PARAMS[arm] for arm in C.ARM_ORDER},
        deploy_params={arm: C.DEPLOY_PARAMS[arm] for arm in C.ARM_ORDER})
    write(C.ROOT / "readiness.json", readiness)
    print(f"[READY] Run3 audit and smoke tests passed: {len(results)} arms, "
          f"{len(C.experiments())} jobs on GPUs {C.GPUS}")


if __name__ == "__main__":
    main()
