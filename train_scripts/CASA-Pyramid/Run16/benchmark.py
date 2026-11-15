"""Batch-1 FP32 latency per version and arm (deployment form)."""
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import config as C

WARMUP, REPEATS = 20, 50


def load_net(version, arm):
    directory = str(C.RUNTIME / C.model_dir(version))
    for name in [n for n in list(sys.modules) if n == "model" or n.startswith("model.")]:
        del sys.modules[name]
    sys.path.insert(0, directory)
    module = importlib.import_module("model.casa_net")
    net = module.CASANet(module=arm)
    net.switch_to_deploy()
    return net.eval()


def measure(net, device):
    a = torch.rand(1, 3, 256, 256, device=device)
    b = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad():
        for _ in range(WARMUP):
            net(a, b)
        torch.cuda.synchronize()
        samples = []
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            net(a, b)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
    return dict(median_ms=round(statistics.median(samples), 3),
                p10_ms=round(sorted(samples)[len(samples) // 10], 3),
                p90_ms=round(sorted(samples)[9 * len(samples) // 10], 3),
                repeats=REPEATS)


def main():
    device = torch.device('cuda')
    payload = dict(device=torch.cuda.get_device_name(0), precision='fp32', input='1x3x256x256',
                   warmup=WARMUP, results={})
    for version in C.VERSIONS:
        for arm in C.arms(version):
            net = load_net(version, arm).to(device)
            key = f"{version}/{arm}"
            payload["results"][key] = dict(measure(net, device),
                                           deploy_params=sum(p.numel() for p in net.parameters()))
            print(f"[BENCH] {key} {payload['results'][key]}", flush=True)
            del net
            torch.cuda.empty_cache()
    (C.ROOT / 'inference_benchmark.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == "__main__":
    main()
