"""Measure full-batch training and batch-one deployed MoE execution on GPU 1."""
import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from model.casa_net import CASANet
from model.layers.candidate_context import MODEL_VARIANTS
from train_CASA import (PROTOCOL, make_loader, make_optimizer, training_losses,
                        task_gradient_diagnostics, loss_function, SpatialMetrics)


def process_memory_mib():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True)
    for row in output.splitlines():
        pid, memory = row.split(",")
        if int(pid) == os.getpid():
            return float(memory)
    return None


def profile_variant(variant, image, target, pretrained, training_steps, inference_steps):
    torch.manual_seed(2333)
    torch.cuda.manual_seed_all(2333)
    image, target = image.cuda().float(), target.cuda().float()
    model = CASANet(pretrained, variant).cuda().train()
    optimizer = make_optimizer(model)
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for step in range(training_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary = model(image[:, :3], image[:, 3:], return_aux=True)
        loss, main, guide = training_losses(logits, auxiliary, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{variant}: non-finite profiling loss")
        if step == 2 and variant != "none":
            task_gradient_diagnostics(model, main)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                      error_if_nonfinite=True, foreach=True)
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    # Formal epoch evaluation collects full-batch context and spatial diagnostics.
    model.eval()
    model.context.collect_diag = True
    with torch.no_grad():
        validation_logits, _ = model(image[:, :3], image[:, 3:], return_aux=True)
        loss_function(validation_logits, target)
        spatial = SpatialMetrics(image.device)
        spatial.update(validation_logits[0] > 0, target)
        spatial.scores()
    model.context.collect_diag = False
    del validation_logits, spatial
    result = dict(
        protocol=PROTOCOL, physical_gpu=1, variant=variant, batch_size=image.shape[0],
        training_outputs=4, inference_outputs=1, training_steps=training_steps,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        process_memory_mib=process_memory_mib(),
        seconds_per_step=statistics.mean(timings[3:]),
        memory_scope="training including task-gradient and full-batch epoch diagnostics",
        training_step_seconds=timings,
        loss=float(loss.detach()), main_loss=float(main.detach()), guide_loss=float(guide.detach()),
        params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        context_params=sum(p.numel() for p in model.context.parameters()),
        configuration=model.context_metadata(),
        optimizer_groups=[dict(name=g["group_name"], weight_decay=g["weight_decay"],
                               decoupled_weight_decay=g["decoupled_weight_decay"])
                          for g in optimizer.param_groups],
        precision=dict(tensor="float32", cudnn_tf32=True, matmul_tf32=False),
        device_name=torch.cuda.get_device_name(),
    )
    if result["process_memory_mib"] is not None:
        result["context_overhead_mib"] = result["process_memory_mib"] - torch.cuda.memory_reserved() / 1024**2
    a, b = image[:1, :3].clone(), image[:1, 3:].clone()
    del optimizer, logits, auxiliary, loss, main, guide, image, target
    model.zero_grad(set_to_none=True)
    model = model.eval().switch_to_deploy()
    gc.collect()
    torch.cuda.empty_cache()
    # One image pair models the final inference API, excluding deep supervision heads.
    with torch.inference_mode():
        for _ in range(10):
            model.predict(a, b)
        torch.cuda.synchronize()
        persistent_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        events = []
        for _ in range(inference_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            prediction = model.predict(a, b)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        milliseconds = [start.elapsed_time(end) for start, end in events]
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"{variant}: non-finite deployed prediction")
    result["deploy"] = dict(
        params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        batch_size=1, measured_iterations=inference_steps,
        median_ms=statistics.median(milliseconds), mean_ms=statistics.mean(milliseconds),
        p90_ms=sorted(milliseconds)[max(0, int(.9 * len(milliseconds)) - 1)],
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        incremental_peak_mib=(torch.cuda.max_memory_allocated() - persistent_bytes) / 1024**2,
        memory_scope="deployed model, one image pair and inference temporaries",
    )
    del model, prediction, events, a, b
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modules", nargs="+", choices=MODEL_VARIANTS, default=MODEL_VARIANTS)
    parser.add_argument("--training-steps", type=int, default=6)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=1 for this GPU-1 exploration")
    if args.training_steps < 4 or args.inference_steps < 1:
        parser.error("training-steps must be >= 4 and inference-steps must be >= 1")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    image, target = next(iter(make_loader(args.dataset_root, "train", batch_size=args.batch_size)))
    results = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for variant in args.modules:
        results[variant] = profile_variant(variant, image, target, args.pretrained,
                                           args.training_steps, args.inference_steps)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[PROFILE] {variant} {json.dumps(results[variant])}", flush=True)


if __name__ == "__main__":
    main()
