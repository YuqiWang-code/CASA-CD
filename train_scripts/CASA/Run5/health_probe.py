"""Dataset-specific health training with the formal schedule and task gradients."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
from train_CASA import (ASSIGNMENT_WEIGHT, PROTOCOL, assignment_loss, learning_rate,
                        loss_function, make_loader, make_optimizer, source_files,
                        training_objective)
from model.casa_net import CASANet


def rms(tensor):
    return float(tensor.detach().double().square().mean().sqrt())


def inspect(model, image, target):
    model.eval()
    report = {}
    with torch.no_grad():
        a = model.encoder_fusion(*model.encoder(image[:, :3]))
        b = model.encoder_fusion(*model.encoder(image[:, 3:]))
        for i, (op, x, y) in enumerate(zip(model.stage_modules, a, b), 1):
            branches = []
            hook = op.out.register_forward_hook(lambda layer, inputs, output: branches.append(output))
            op.collect_diag = True
            _, _, logits = op(x, y)
            op.collect_diag = False
            hook.remove()
            branch = torch.cat(branches)
            report[f"stage{i}"] = dict(op.last_diag,
                branch_rms=rms(branch), residual_rms=rms(op.gamma * branch),
                weight_rms={n: rms(p) for n, p in op.named_parameters()})
            probability = logits.sigmoid()
            occupancy = torch.nn.functional.adaptive_avg_pool2d(target, probability.shape[-2:])
            report[f"stage{i}"]["assignment_positive_mean"] = (
                float((probability * occupancy).sum() / occupancy.sum())
                if occupancy.sum() > 0 else None)
            report[f"stage{i}"]["assignment_negative_mean"] = float(
                (probability * (1-occupancy)).sum() / (1-occupancy).sum().clamp_min(1e-6))
    model.zero_grad(set_to_none=True)
    outputs, assignments = model(image[:, :3], image[:, 3:], return_aux=True)
    loss_function(outputs, target).backward()
    for i, op in enumerate(model.stage_modules, 1):
        report[f"stage{i}"]["task_grad_rms"] = {
            n: rms(p.grad) if p.grad is not None else 0.0 for n, p in op.named_parameters()}
    model.zero_grad(set_to_none=True)
    auxiliary = assignment_loss(assignments, target)
    (ASSIGNMENT_WEIGHT * auxiliary).backward()
    for i, op in enumerate(model.stage_modules, 1):
        report[f"stage{i}"]["assignment_aux_grad_rms"] = {
            n: rms(p.grad) if p.grad is not None else 0.0 for n, p in op.assignment.named_parameters()}
        report[f"stage{i}"]["assignment_loss"] = float(auxiliary.detach())
    model.zero_grad(set_to_none=True)
    return report


def healthy(report, mode):
    for stage in report.values():
        if not stage["finite"] or stage["branch_rms"] <= 0 or stage["residual_rms"] <= 0:
            return False
        if not all(math.isfinite(x) and x > 0 for x in stage["task_grad_rms"].values()):
            return False
        if not all(math.isfinite(x) and x > 0 for x in stage["assignment_aux_grad_rms"].values()):
            return False
        if mode == "hyperbolic":
            if stage["radius_cv"] < 0.02 or stage["query_clip_fraction"] > 0.20:
                return False
            if stage["direction_resultant"] > 0.95:
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["hyperbolic", "qkv"], required=True)
    parser.add_argument("--dataset", choices=["LEVIR-CD-256", "SYSU-CD-256"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(2333)
    np.random.seed(2333)
    torch.manual_seed(2333)
    torch.cuda.manual_seed_all(2333)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    dataset = Path("/share_datasets/CD") / args.dataset
    train = make_loader(dataset, "train", 32, 1)
    validation = make_loader(dataset, "val", 2, 0)
    # Match the formal run's model and data-generator initialization order.
    model = CASANet(str(ROOT / "pretrained_weight/mobilenet_v2-b0353104.pth"),
                          module=args.module).to(device)
    optimizer = make_optimizer(model)
    with torch.random.fork_rng(devices=[]):
        image, target = next(iter(validation))
    image, target = image.float().to(device), target.float().to(device)
    total = math.ceil(40000 / len(train)) * len(train)
    manifest = dict(protocol=PROTOCOL, module=args.module, purpose="module health probe",
                    dataset=args.dataset, epochs=args.epochs, steps=args.epochs*len(train), scheduler_total=total, batch_size=32,
                    weight_decay=1e-4, module_weight_decay=0.0,
                    assignment_auxiliary_weight=ASSIGNMENT_WEIGHT,
                    assignment_auxiliary_gradient="heads only; main loss end-to-end",
                    assignment_target="area-averaged change occupancy", source_files=source_files())
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    step, history, started = 0, [], time.time()
    for epoch in range(args.epochs):
        model.train()
        for image_train, target_train in train:
            image_train = image_train.to(device).float()
            target_train = target_train.to(device).float()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(5e-4, epoch, step, total)
            optimizer.zero_grad(set_to_none=True)
            outputs, assignments = model(image_train[:, :3], image_train[:, 3:], return_aux=True)
            loss, main_loss, auxiliary_loss = training_objective(outputs, assignments, target_train)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"probe loss at {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                            error_if_nonfinite=True, foreach=True)
            optimizer.step()
            step += 1
        report = inspect(model, image, target)
        entry = dict(epoch=epoch+1, step=step, stages=report)
        history.append(entry)
        print(json.dumps(entry), flush=True)
        (args.output_dir / "health_history.json").write_text(json.dumps(history, indent=2))
        # Check geometry and task gradients throughout the health training.
        if epoch >= 1 and not healthy(report, args.module):
            raise RuntimeError(f"module health failed after epoch {epoch+1}")
    result = dict(passed=True, protocol=PROTOCOL, module=args.module, dataset=args.dataset, steps=step,
                  epochs=args.epochs, elapsed_seconds=time.time()-started, final=history[-1],
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    manifest=manifest, step=step), args.output_dir / "probe_checkpoint.pth")
    (args.output_dir / "health_result.json").write_text(json.dumps(result, indent=2))
    print(f"[PROBE-PASS] {args.module} {args.dataset} epochs={args.epochs} steps={step}", flush=True)


if __name__ == "__main__":
    main()
