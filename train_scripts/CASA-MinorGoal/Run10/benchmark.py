"""Inference benchmark for the Run10 arms (deploy form, batch 1, 256x256, FP32)."""
import json
import os
import statistics
import sys
import time

import torch

import config as C

RUNTIME_MODELS = str(C.RUNTIME / "models")
LOCAL_MODELS = str(C.SCRIPTS.parents[2] / "models")
sys.path.insert(0, RUNTIME_MODELS if os.path.isdir(RUNTIME_MODELS) else LOCAL_MODELS)

from model.casa_net import CASANet  # noqa: E402

WARMUP, REPEATS = 20, 50


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    a = torch.randn(1, 3, 256, 256, device=device)
    b = torch.randn(1, 3, 256, 256, device=device)
    payload = {}
    for arm in C.ARM_ORDER:
        spec = C.ARMS[arm]
        model = CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                        decoder_variant=spec["decoder"], support=spec["support"],
                        detail_head=spec["detail"],
                        edge_supervision=spec["edge_weight"] > 0,
                    prototype_pooling=spec.get("prototype_pooling", "euclidean"),
                    aggregate_mode=spec.get("aggregate", "lorentz")).to(device).eval().switch_to_deploy()
        with torch.no_grad():
            for _ in range(WARMUP):
                model.predict(a, b)
            if device == "cuda":
                torch.cuda.synchronize()
            timings = []
            for _ in range(REPEATS):
                start = time.perf_counter()
                model.predict(a, b)
                if device == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1e3)
        payload[arm] = dict(median_ms=statistics.median(timings),
                            deploy_params=sum(p.numel() for p in model.parameters()))
        print(f"[BENCH] {arm}: {payload[arm]['median_ms']:.3f} ms, "
              f"{payload[arm]['deploy_params']} deploy params")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    (C.ROOT / "inference_benchmark.json").write_text(
        json.dumps(dict(device=device, batch=1, input=256, precision="float32", arms=payload), indent=2))


if __name__ == "__main__":
    main()
