"""Eight SYSU epochs with the full-budget schedule and per-stage task gradients."""
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
from train_wdmf import (PROTOCOL, learning_rate, loss_function, make_loader,
                        make_optimizer, source_identity)
from model.casa_wdmf_str_net import CASAWDMFSTRNet


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
            op(x, y)
            op.collect_diag = False
            hook.remove()
            branch = torch.cat(branches)
            report[f"stage{i}"] = dict(op.last_diag,
                branch_rms=rms(branch), residual_rms=rms(op.gamma * branch),
                weight_rms={n: rms(p) for n, p in op.named_parameters()})
    model.zero_grad(set_to_none=True)
    loss_function(model(image[:, :3], image[:, 3:]), target).backward()
    for i, op in enumerate(model.stage_modules, 1):
        report[f"stage{i}"]["task_grad_rms"] = {
            n: rms(p.grad) for n, p in op.named_parameters()}
    model.zero_grad(set_to_none=True)
    return report


def healthy(report, mode):
    for stage in report.values():
        if not stage["finite"] or stage["branch_rms"] <= 0 or stage["residual_rms"] <= 0:
            return False
        if not all(math.isfinite(x) and x > 0 for x in stage["task_grad_rms"].values()):
            return False
        if mode == "hyperbolic" and stage["radius_cv"] <= 0:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["hyperbolic", "qkv"], required=True)
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
    dataset = Path("/share_datasets/CD/SYSU-CD-256")
    train = make_loader(dataset, "train", 32, 1)
    validation = make_loader(dataset, "val", 2, 0)
    # Match the formal run's model and data-generator initialization order.
    model = CASAWDMFSTRNet(str(ROOT / "pretrained_weight/mobilenet_v2-b0353104.pth"),
                          module=args.module).to(device)
    optimizer = make_optimizer(model)
    with torch.random.fork_rng(devices=[]):
        image, target = next(iter(validation))
    image, target = image.float().to(device), target.float().to(device)
    total = math.ceil(40000 / len(train)) * len(train)
    manifest = dict(protocol=PROTOCOL, module=args.module, purpose="module health probe",
                    epochs=8, steps=8*len(train), scheduler_total=total, batch_size=32,
                    weight_decay=1e-4, module_weight_decay=0.0, source_files=source_identity())
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    step, history, started = 0, [], time.time()
    for epoch in range(8):
        model.train()
        for image_train, target_train in train:
            image_train = image_train.to(device).float()
            target_train = target_train.to(device).float()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(5e-4, epoch, step, total)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(image_train[:, :3], image_train[:, 3:]), target_train)
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
        # Every epoch after the original onset must retain live task gradients.
        if epoch >= 1 and not healthy(report, args.module):
            raise RuntimeError(f"module health failed after epoch {epoch+1}")
    result = dict(passed=True, protocol=PROTOCOL, module=args.module, steps=step,
                  epochs=8, elapsed_seconds=time.time()-started, final=history[-1])
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    manifest=manifest, step=step), args.output_dir / "probe_checkpoint.pth")
    (args.output_dir / "health_result.json").write_text(json.dumps(result, indent=2))
    print(f"[PROBE-PASS] {args.module} epochs=8 steps={step}", flush=True)


if __name__ == "__main__":
    main()
