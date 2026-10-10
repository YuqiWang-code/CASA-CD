#!/usr/bin/env python
"""Run4 训练期 checkpoint 备份看门狗（零代码改动的崩溃缓解）。

背景（Run4 T11 实测）：`train.py::_save_last` 直接 `torch.save(..., "last.pth")`，不是原子写。
进程若在写盘期间被 OOM-kill / 断电 / SIGKILL，`last.pth` 会变成半截文件，
`_load_resume` 抛 `EOFError` → 自动恢复失败 → 该 run 只能按设计文档用**独立新目录**从头重跑。

T11 同时证明本项目的训练**跨进程不可逐位复现**（同 seed 的两次连续 40 步对照 run
也有 1098/1302 个张量不同），因此「精确恢复」不可声称；能保证的是
**协议精确**（步数 / per-group LR / optimizer / RNG 状态逐项恢复）。

本看门狗因此只做一件零风险的事：周期性把**能成功 torch.load 的** `last.pth`
原子复制为 `last.pth.bak`。恢复失败时可人工 `cp last.pth.bak last.pth` 继续，
最多损失一个 epoch。绝不修改 `last.pth`、绝不介入训练进程。

    nohup python train_scripts/CASA-TViM/Run4/ckpt_backup_watchdog.py \
        --ckpt-root /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4 \
        --interval 30 > /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/ckpt_backup_watchdog.log 2>&1 &
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch

VARIANTS = ["M1_R4CTRL", "E6_FET1"]
DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]


def snapshot(ckpt_root, state, log):
    """mtime 触发的「验证后原子备份」：只有能成功反序列化的 last.pth 才会成为 .bak。"""
    n_ok = 0
    for v in VARIANTS:
        for d in DATASETS:
            p = os.path.join(ckpt_root, v, d, "last.pth")
            if not os.path.isfile(p):
                continue
            key = f"{v}/{d}"
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if state.get(key) == mt:
                continue
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                epoch, steps = ck.get("epoch"), ck.get("actual_steps")
            except Exception as e:                      # 半截文件：不动，等下一次完整写入
                log(f"[SKIP] {key} unreadable ({type(e).__name__}); waiting for the next complete save")
                continue
            bak = p + ".bak"
            tmp = bak + ".tmp"
            shutil.copy2(p, tmp)
            os.replace(tmp, bak)                        # 原子替换
            state[key] = mt
            n_ok += 1
            log(f"[BAK] {key} epoch={epoch} actual_steps={steps} -> {os.path.basename(bak)}")
    return n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--max-hours", type=float, default=48.0)
    ap.add_argument("--state-file", default=None)
    args = ap.parse_args()

    state_file = args.state_file or os.path.join(args.ckpt_root, "ckpt_backup_state.json")
    state = {}
    if os.path.isfile(state_file):
        try:
            with open(state_file, encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"watchdog start ckpt_root={args.ckpt_root} interval={args.interval}s")
    t0 = time.time()
    while time.time() - t0 < args.max_hours * 3600:
        try:
            n = snapshot(args.ckpt_root, state, log)
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=1)
            if n:
                log(f"backed up {n} checkpoint(s)")
        except Exception as e:
            log(f"[WARN] snapshot failed: {type(e).__name__}: {e}")
        time.sleep(args.interval)
    log("watchdog exit (max-hours reached)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
