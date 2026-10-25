"""Measure full-batch training memory and throughput for CASA variants."""
import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from model.casa_net import CASANet
from train_CASA import PROTOCOL, make_loader, training_losses, make_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    image, target = next(iter(make_loader(args.dataset_root, "train")))
    image, target = image.cuda().float(), target.cuda().float()
    results = {}
    for variant in ["none", "hyperbolic", "qkv"]:
        torch.manual_seed(2333)
        model = CASANet(args.pretrained, variant).cuda().train()
        optimizer = make_optimizer(model)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(6):
            torch.cuda.synchronize()
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            outputs, guide = model(image[:, :3], image[:, 3:], return_aux=True)
            loss, main_loss, auxiliary_loss = training_losses(outputs, guide, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.time() - start)
        model.context.collect_diag = True
        with torch.no_grad():
            model(image[:, :3], image[:, 3:])
        used = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True)
        process_mib = next(float(row.split(",")[1]) for row in used.splitlines()
                           if int(row.split(",")[0]) == os.getpid())
        context_mib = process_mib - torch.cuda.memory_reserved() / 1024**2
        result = dict(protocol=PROTOCOL, batch_size=image.shape[0],
                      grid_sizes=list(model.context.grid_sizes),
                      contexts_per_scale=list(model.context.tokens_per_scale),
                      joint_contexts=sum(model.context.tokens_per_scale),
                      qk_geometry=model.context.geometry, value_geometry=model.context.value_geometry,
                      context_projection_weight_decay=1e-4, context_other_weight_decay=0.0,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                      context_overhead_mib=context_mib, process_memory_mib=process_mib,
                      seconds_per_step=sum(times[2:]) / len(times[2:]), loss=float(loss.detach()),
                      params=sum(p.numel() for p in model.parameters()), diagnostics=model.diagnostics())
        results[variant] = result
        print(f"[PROFILE] {variant} {json.dumps(result)}", flush=True)
        del model, optimizer, outputs, guide, loss, main_loss, auxiliary_loss
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
