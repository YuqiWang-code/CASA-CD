#!/usr/bin/env python
"""Run4 并发显存 probe（方案 §8.4 / R8）。

每个 GPU 同时启动 **2 个**本脚本：各自构造 batch32 的 E6_FET1 训练图，跑 `--steps` 次
forward+backward+Adam step（随机张量，不读数据集），报告 `torch.cuda.max_memory_allocated`。
任一进程 OOM ⇒ run_all.sh 判定并发不可行，则保持 batch32 改为**同卡串行、跨 GPU 并行**，
**不得**改 batch size / 梯度累积 / 训练协议。

    CUDA_VISIBLE_DEVICES=0 python gpu_concurrency_probe.py --variant E6_FET1 \
        --device cuda:0 --batch-size 32 --steps 2
"""
import argparse
import os
import sys
import time

import torch

_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "..", "..", "models"))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.casa_tvim_str_net import CASATViMSTRNet   # noqa: E402
from model.utils import BCEDiceLoss                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, default="E6_FET1", choices=["E6_FET1", "M1_R4CTRL"])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--tinyvim-pretrained-weight-path", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[PROBE][SKIP] CUDA not available")
        return 0

    torch.manual_seed(16)
    model = CASATViMSTRNet(
        args.tinyvim_pretrained_weight_path, caacp=True, rep_mode="full", str_dim=96,
        caacp_score_mode="rank", frh=False, caacp_residual_mode="current", fs_tar=False,
        fine_tap=(args.variant == "E6_FET1")).float().to(args.device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), 2e-4, (0.9, 0.99), eps=1e-8, weight_decay=1e-4)

    torch.cuda.reset_peak_memory_stats()
    B = args.batch_size
    t0 = time.time()
    for step in range(args.steps):
        pre = torch.randn(B, 3, args.size, args.size, device=args.device)
        post = torch.randn(B, 3, args.size, args.size, device=args.device)
        tgt = (torch.rand(B, 1, args.size, args.size, device=args.device) > 0.9).float()
        out = model(pre, post, tgt)
        loss = BCEDiceLoss(out, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[PROBE] step={step + 1}/{args.steps} bs={B} loss={loss.item():.4f} "
              f"peak_allocated={peak:.0f} MB", flush=True)
        del pre, post, tgt, out, loss
        torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"[PROBE][OK] variant={args.variant} bs={B} peak={peak:.0f} MB / total={total:.0f} MB "
          f"({100.0 * peak / total:.1f}%) elapsed={time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
