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
from train_CASA import PROTOCOL, GUIDE_OBJECTIVE, guide_effective_coefficients, auxiliary_weights, make_loader, training_losses, make_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modules", nargs="+", choices=("none", "hyperbolic", "qkv", "matched"),
                        default=("none", "hyperbolic", "matched"))
    parser.add_argument("--guide-weight", type=float)
    parser.add_argument("--retrieval-weight", type=float)
    parser.add_argument("--interior-positive-weight", type=float, default=1.0)
    parser.add_argument("--context-heads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--detail-head", action="store_true")
    parser.add_argument("--detail-target", choices=("area", "nearest-exact"), default="area")
    parser.add_argument("--tar-variant", choices=("reparam", "plain"), default="reparam")
    parser.add_argument("--decoder-variant", choices=("reparam", "plain"), default="reparam")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    image, target = next(iter(make_loader(args.dataset_root, "train")))
    image, target = image.cuda().float(), target.cuda().float()
    results = {}
    for variant in args.modules:
        torch.manual_seed(2333)
        model = CASANet(args.pretrained, variant, context_heads=args.context_heads,
                        detail_head=args.detail_head, detail_target=args.detail_target,
                        tar_variant=args.tar_variant, decoder_variant=args.decoder_variant).cuda().train()
        optimizer = make_optimizer(model)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(6):
            torch.cuda.synchronize()
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(image[:, :3], image[:, 3:], return_aux=True)
            loss, main_loss, auxiliary_loss = training_losses(logits, auxiliary, target, args.guide_weight, args.retrieval_weight, args.interior_positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.time() - start)
        model.context.collect_diag = True
        with torch.no_grad():
            model.predict(image[:, :3], image[:, 3:])
        used = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True)
        process_mib = next(float(row.split(",")[1]) for row in used.splitlines()
                           if int(row.split(",")[0]) == os.getpid())
        context_mib = process_mib - torch.cuda.memory_reserved() / 1024**2
        result = dict(protocol=PROTOCOL, batch_size=image.shape[0],
                      training_outputs=4, training_prediction_format="logits",
                      guide_objective=GUIDE_OBJECTIVE if variant in ("hyperbolic", "matched") else "raw_area_bce",
                      guide_effective_coefficients=guide_effective_coefficients(variant, auxiliary_weights(variant, args.guide_weight, args.retrieval_weight)["guide"]),
                      auxiliary_weights=auxiliary_weights(variant, args.guide_weight, args.retrieval_weight),
                      compression="first_order_class_pooling" if variant in ("hyperbolic", "matched") else "none", encoder_lr_mult=1.0,
                      inference_method="predict", inference_outputs=1,
                      grid_sizes=list(model.context.grid_sizes),
                      contexts_per_scale=list(model.context.tokens_per_scale),
                      joint_contexts=sum(model.context.tokens_per_scale),
                      initial_compression_power=model.context.compression_power,
                      interior_positive_weight=args.interior_positive_weight,
                      detail_head=args.detail_head,
                      compression_power=model.context.compression_power,
                      support_power=model.context.support_power,
                      self_reference=model.context.self_reference,
                      qk_geometry=model.context.geometry, value_geometry=model.context.value_geometry,
                      heads=model.context.heads if variant != "none" else 0,
                      qkv_dim=model.context.qk.out_channels if variant != "none" else 0,
                      hyperbolic_radius=model.context.radius_mode if variant in ("hyperbolic", "matched") else None,
                      context_projection_weight_decay=1e-4, context_projection_weight_decay_mode="decoupled",
                      body_weight_decay_mode="coupled_l2", context_other_weight_decay=0.0,
                      optimizer_groups=[dict(name=g["group_name"], weight_decay=g["weight_decay"],
                                             decoupled_weight_decay=g["decoupled_weight_decay"])
                                        for g in optimizer.param_groups],
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                      context_overhead_mib=context_mib, process_memory_mib=process_mib,
                      seconds_per_step=sum(times[2:]) / len(times[2:]), loss=float(loss.detach()),
                      params=sum(p.numel() for p in model.parameters()), diagnostics=model.diagnostics())
        results[variant] = result
        print(f"[PROFILE] {variant} {json.dumps(result)}", flush=True)
        del model, optimizer, logits, auxiliary, loss, main_loss, auxiliary_loss
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
