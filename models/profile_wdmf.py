"""Measure full-batch training memory and throughput before scheduling GPU0."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch

from model.casa_wdmf_str_net import CASAWDMFSTRNet
from train_wdmf import make_loader, loss_function


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
        model = CASAWDMFSTRNet(args.pretrained, variant).cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), 5e-4, (0.9, 0.99), weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(6):
            torch.cuda.synchronize()
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image[:, :3], image[:, 3:])
            loss = loss_function(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.time() - start)
        for op in model.stage_modules:
            op.collect_diag = True
        with torch.no_grad():
            model(image[:, :3], image[:, 3:])
        result = dict(peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                      seconds_per_step=sum(times[2:]) / len(times[2:]), loss=float(loss.detach()),
                      params=sum(p.numel() for p in model.parameters()), diagnostics=model.diagnostics())
        results[variant] = result
        print(f"[PROFILE] {variant} {json.dumps(result)}", flush=True)
        del model, optimizer, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
