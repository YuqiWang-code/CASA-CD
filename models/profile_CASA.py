"""Profile CASA variants: training memory (legacy), deployed GFLOPs, and latency.

Modes
-----
memory   (default) full-batch training memory/throughput, as before.
flops    parameter count + GFLOPs at 1x3x256x256, eval, after switch_to_deploy().
latency  batch-1 fp32 inference latency (needs a free GPU).
all      flops + latency in one process.

FLOPs are cross-checked with two independent counters:
  * torch.utils.flop_counter.FlopCounterMode
  * fvcore.nn.FlopCountAnalysis
A >2% disagreement is reported so the caller can inspect uncounted ops
(e.g. F.interpolate / upsample is not a FLOP-counted kernel in fvcore).
"""
import argparse
import gc
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from model.casa_net import CASANet

# Run1 / Run3 arm table.  Params are checked against the published deployment
# counts (CTRL 1,935,810 / SPP 1,995,714 / DIFF 2,007,126).
ARMS = {
    "CTRL": dict(module="none", tar="reparam", decoder="reparam", trained=True,
                 checkpoint="Run1/CTRL"),
    "SPP": dict(module="none", tar="spp", decoder="reparam", trained=True,
                checkpoint="Run1/SPP"),
    "DIFF": dict(module="difference", tar="spp", decoder="reparam", trained=True,
                 checkpoint="Run1/DIFF"),
    # ARM-C is the parameter-matched content control (hyperbolic + spp); no
    # trained checkpoint exists, architecture/FLOPs are what matter here.
    "ARM-C": dict(module="hyperbolic", tar="spp", decoder="reparam", trained=False,
                  checkpoint=None),
    # ARM-D (difference_matched) is not implemented in models/model, so it
    # cannot be instantiated; it is reported as unavailable in the JSON.
    "ARM-D": dict(module=None, tar="spp", decoder="reparam", trained=False,
                  checkpoint=None),
    # DETAIL is Run3's boundary arm: DIFF plus the zero-initialised detail head,
    # which stays in the deployment graph.
    "DETAIL": dict(module="difference", tar="spp", decoder="reparam", trained=False,
                   checkpoint=None, detail_head=True),
}


def build_model(spec, device, pretrained=None):
    torch.manual_seed(2333)
    model = CASANet(pretrained, spec["module"], tar_variant=spec["tar"],
                    decoder_variant=spec["decoder"], context_heads=1,
                    detail_head=spec.get("detail_head", False))
    return model.eval().to(device)


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def deploy_fold(model, device):
    """switch_to_deploy() and verify the folded graph matches the original.

    Runs under no_grad(), not inference_mode(): folding materialises new
    parameters, and inference tensors stored as parameters break fvcore's
    jit trace with "Inference tensors cannot be saved for backward".
    """
    a = torch.randn(1, 3, 256, 256, device=device)
    b = torch.randn_like(a)
    with torch.no_grad():
        before = model.predict(a, b).clone()
        model.switch_to_deploy()
        after = model.predict(a, b)
    return dict(maxdiff=(before - after).abs().max().item(),
                binary_diff=int(((before > 0.5) != (after > 0.5)).sum()),
                pixels=int(after.numel()))


def _sum_nested(node):
    """FlopCounterMode.get_flop_counts() nests module -> op -> shapes -> count."""
    if isinstance(node, dict):
        return sum(_sum_nested(value) for value in node.values())
    return float(node)


def measure_head_mac(model, device):
    """Head cost at its real input resolution (the decoder's finest scale)."""
    shapes = []

    def hook(module, inputs):
        shapes.append(tuple(inputs[0].shape))

    handle = model.head.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model.predict(torch.randn(1, 3, 256, 256, device=device),
                          torch.randn(1, 3, 256, 256, device=device))
    finally:
        handle.remove()
    shape = shapes[0] if shapes else (1, model.head.in_channels, 64, 64)
    mac = shape[2] * shape[3] * model.head.in_channels
    return float(mac), list(shape)


def count_flops(model, device):
    a = torch.randn(1, 3, 256, 256, device=device)
    b = torch.randn_like(a)
    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:
            model.predict(a, b)
    torch_total = float(counter.get_total_flops())
    torch_by = {k: _sum_nested(v) for k, v in counter.get_flop_counts().items()}
    head_mac, head_shape = measure_head_mac(model, device)

    class _Single(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x, y):
            return self.wrapped.predict(x, y)

    wrapper = _Single(model)
    try:
        from fvcore.nn import FlopCountAnalysis
        analysis = FlopCountAnalysis(wrapper, (a, b))
        analysis.unsupported_ops_warnings(False)
        if hasattr(analysis, "uncalled_modules_warnings"):
            analysis.uncalled_modules_warnings(False)
        fv_total = float(analysis.total())
        fv_by = {k: float(v) for k, v in analysis.by_module().items()}
        # fvcore counts one fused multiply-add as one FLOP; disable the
        # elementwise/BN/upsample handles so the remaining number is pure
        # conv/matmul MACs, directly comparable to torch_counter/2.
        for op in ("aten::batch_norm", "aten::group_norm", "aten::layer_norm",
                   "aten::instance_norm", "aten::upsample_nearest2d",
                   "aten::upsample_bilinear2d", "aten::adaptive_avg_pool2d",
                   "aten::grid_sampler"):
            analysis.set_op_handle(op, None)
        fv_mac = float(analysis.total())
        fv_error = None
    except Exception as error:  # fvcore can choke on dynamic ops; keep the torch number
        fv_total, fv_by, fv_mac, fv_error = None, {}, None, f"{type(error).__name__}: {error}"

    torch_half = torch_total / 2.0
    if fv_mac is None:
        diff_pct = None
    else:
        diff_pct = abs(torch_half - fv_mac) / max(torch_half, fv_mac) * 100
    return dict(
        # convention A: papers/thop style, one fused multiply-add = 1 FLOP
        mac_gflops=None if fv_mac is None else fv_mac / 1e9,
        # convention B: torch FlopCounterMode, multiply and add counted apart
        flop2_gflops=torch_total / 1e9,
        torch_counter_gflops=torch_total / 1e9,
        fvcore_gflops=None if fv_total is None else fv_total / 1e9,
        cross_check_pct=diff_pct,
        cross_check_rule="|torch_counter/2 - fvcore(conv+matmul only)| / max < 2%",
        fvcore_error=fv_error,
        head_mac_gflops=head_mac / 1e9,
        head_input_shape=head_shape,
        head_share_pct=None if fv_mac is None else head_mac / fv_mac * 100,
        no_head_mac_gflops=None if fv_mac is None else (fv_mac - head_mac) / 1e9,
        by_module_torch={k: v / 1e9 for k, v in torch_by.items() if v},
        by_module_fvcore={k: v / 1e9 for k, v in fv_by.items() if v},
    )


def _percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bench_latency(model, device, warmup=100, iters=1000):
    model.eval()
    a = torch.randn(1, 3, 256, 256, device=device)
    b = torch.randn_like(a)
    if device.type != "cuda":
        times = []
        with torch.inference_mode():
            for _ in range(warmup):
                model.predict(a, b)
            for _ in range(iters):
                start = time.perf_counter()
                model.predict(a, b)
                times.append((time.perf_counter() - start) * 1000.0)
        return dict(n=len(times), warmup=warmup, device=str(device),
                    median_ms=_percentile(times, 50), p90_ms=_percentile(times, 90),
                    mean_ms=sum(times) / len(times), min_ms=min(times), max_ms=max(times))
    torch.backends.cudnn.benchmark = True
    with torch.inference_mode():
        for _ in range(warmup):
            model.predict(a, b)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for index in range(iters):
            starts[index].record()
            model.predict(a, b)
            ends[index].record()
        torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return dict(n=len(times), warmup=warmup, device=str(device),
                median_ms=_percentile(times, 50), p90_ms=_percentile(times, 90),
                mean_ms=sum(times) / len(times), min_ms=min(times), max_ms=max(times),
                std_ms=(sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5)


def wait_for_idle_gpu(mem_mib, util_pct, timeout_s, poll_s=60, only=None):
    """Block until at least one GPU is idle enough not to perturb a co-tenant run.

    Used while Run2 owns both 5090s; the benchmark then runs on the first GPU
    that frees up (Run2 keeps two slots per card, so a card only counts as free
    once its memory drops to the idle footprint).
    """
    deadline = time.time() + timeout_s
    while True:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        free = []
        for row in rows:
            index, used, util = (part.strip() for part in row.split(","))
            if only is not None and int(index) not in only:
                continue
            if float(used) <= mem_mib and float(util) <= util_pct:
                free.append(int(index))
        if free:
            return free[0], rows
        if time.time() >= deadline:
            raise TimeoutError(f"no idle GPU within {timeout_s}s: {rows}")
        print(f"[WAIT] {time.strftime('%H:%M:%S')} busy: {rows}", flush=True)
        time.sleep(poll_s)


def gpu_identity(index):
    props = torch.cuda.get_device_properties(index)
    out = dict(index=index, name=props.name, total_memory_mib=props.total_memory / 1024**2,
               capability=f"{props.major}.{props.minor}",
               uuid=str(getattr(props, "uuid", "")))
    try:
        row = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,clocks.sm,clocks.mem,power.draw",
             "--format=csv,noheader,nounits", "-i", str(index)], text=True).strip()
        out["nvidia_smi"] = row
    except Exception as error:
        out["nvidia_smi_error"] = str(error)
    return out


def run_memory(args):
    from train_CASA import (PROTOCOL, GUIDE_OBJECTIVE, guide_effective_coefficients,
                            auxiliary_weights, make_loader, training_losses, make_optimizer)
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
        for _ in range(6):
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


def run_flops_latency(args):
    device = torch.device(args.device)
    if args.wait_idle and device.type == "cuda":
        print(f"[WAIT] looking for an idle GPU (mem<={args.idle_mem_mib} MiB, "
              f"util<={args.idle_util}%), timeout {args.wait_idle}s", flush=True)
        only = None if args.wait_gpu is None else {args.wait_gpu}
        index, rows = wait_for_idle_gpu(args.idle_mem_mib, args.idle_util, args.wait_idle,
                                        only=only)
        device = torch.device(f"cuda:{index}")
        print(f"[WAIT] using cuda:{index}; snapshot={rows}", flush=True)
    elif args.wait_idle and device.type == "cpu":
        print("[WAIT] --wait-idle ignored on cpu device", flush=True)
    report = dict(
        meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), script="models/profile_CASA.py",
                  torch=torch.__version__, python=platform.python_version(),
                  device=str(device), input="1x3x256x256", dtype="float32",
                  mode="eval+switch_to_deploy", batch_size=1,
                  flops_counters=["torch.utils.flop_counter.FlopCounterMode",
                                  "fvcore.nn.FlopCountAnalysis"],
                  cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                  notes=["GFLOPs exclude upsample/activation kernels unless the counter lists them",
                         "ARM-D is not implemented in models/model (no difference_matched context)"]),
        arms={})
    if device.type == "cuda":
        report["meta"]["gpu"] = gpu_identity(device.index or 0)
    last = None
    for name in args.arms:
        spec = ARMS[name]
        if spec["module"] is None:
            report["arms"][name] = dict(available=False,
                                        reason="module 'difference_matched' not implemented in models/model")
            print(f"[PROFILE] {name} unavailable: {report['arms'][name]['reason']}", flush=True)
            continue
        model = build_model(spec, device)
        params = param_count(model)
        fold = deploy_fold(model, device)
        params_deploy = param_count(model)
        entry = dict(available=True, module=spec["module"], tar_variant=spec["tar"],
                     decoder_variant=spec["decoder"], trained=spec["trained"],
                     checkpoint=spec["checkpoint"], params=params,
                     params_deploy=params_deploy, deploy_fold=fold)
        if args.mode in ("flops", "all"):
            entry["flops"] = count_flops(model, device)
            flops = entry["flops"]
            print(f"[FLOPS] {name:6s} params={params:,} deploy={params_deploy:,} "
                  f"MAC={flops['mac_gflops']} GFLOP(2x)={flops['flop2_gflops']:.4f} "
                  f"cross={flops['cross_check_pct']} head={flops['head_mac_gflops']:.5f} "
                  f"fold_maxdiff={fold['maxdiff']:.3g}", flush=True)
        if args.mode in ("latency", "all"):
            latency = bench_latency(model, device, args.warmup, args.iters)
            entry["latency"] = latency
            gflops = entry.get("flops", {}).get("mac_gflops")
            if gflops:
                entry["effective_tflops"] = gflops / (latency["median_ms"] / 1000.0) / 1000.0
            print(f"[LATENCY] {name:6s} median={latency['median_ms']:.4f}ms "
                  f"p90={latency['p90_ms']:.4f}ms n={latency['n']}", flush=True)
        report["arms"][name] = entry
        last = model
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del last
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[PROFILE] wrote {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("memory", "flops", "latency", "all"), default="memory")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS),
                        default=("CTRL", "SPP", "DIFF", "ARM-C", "ARM-D", "DETAIL"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--wait-gpu", type=int, default=None,
                        help="restrict --wait-idle to this physical GPU index "
                             "(e.g. 1 = GPU1 only, GPU0 reserved for other users)")
    parser.add_argument("--wait-idle", type=float, default=0.0,
                        help="seconds to wait for an idle GPU before latency (0 = fail fast)")
    parser.add_argument("--idle-mem-mib", type=float, default=2000.0)
    parser.add_argument("--idle-util", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    # legacy training-memory options
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--pretrained")
    parser.add_argument("--modules", nargs="+", choices=("none", "hyperbolic", "qkv", "matched", "difference"),
                        default=("none", "hyperbolic", "matched"))
    parser.add_argument("--guide-weight", type=float)
    parser.add_argument("--retrieval-weight", type=float)
    parser.add_argument("--interior-positive-weight", type=float, default=1.0)
    parser.add_argument("--context-heads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--detail-head", action="store_true")
    parser.add_argument("--detail-target", choices=("area", "nearest-exact"), default="area")
    parser.add_argument("--tar-variant", choices=("reparam", "plain", "diff", "spp", "spp1"), default="reparam")
    parser.add_argument("--decoder-variant", choices=("reparam", "plain", "none"), default="reparam")
    args = parser.parse_args()
    if args.mode == "memory":
        if args.dataset_root is None or args.pretrained is None:
            parser.error("--mode memory needs --dataset-root and --pretrained")
        run_memory(args)
    else:
        run_flops_latency(args)


if __name__ == "__main__":
    main()
