#!/usr/bin/env python
"""Run4 进度/ETA 精确采样，并写一份可恢复的状态快照到服务器。

    python analyse/run4_progress_eta.py --log-root <Run4 log root> --variant M1_R4CTRL \
        --interval 180 --write-status
"""
import argparse
import json
import os
import time

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
MAX_STEPS = 80000


def read_steps(log_root, variant, dataset):
    p = os.path.join(log_root, variant, dataset, "train_log.txt")
    if not os.path.isfile(p):
        return None
    last = None
    with open(p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if "ACTUAL-OPT-STEPS" in ln and "cumulative=" in ln:
                last = ln
    if last is None:
        return None
    return int(last.split("cumulative=")[1].split("/")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--variant", default="M1_R4CTRL")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--write-status", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    a = {d: read_steps(args.log_root, args.variant, d) for d in DATASETS}
    time.sleep(args.interval)
    t1 = time.time()
    b = {d: read_steps(args.log_root, args.variant, d) for d in DATASETS}
    dt = (t1 - t0) / 60.0

    rows = {}
    print(f"variant={args.variant} sampled over {dt:.2f} min")
    print(f"{'dataset':13s} {'steps':>7s} {'rate/min':>9s} {'remain':>7s} {'ETA min':>8s} {'ETA clock':>10s}")
    clock_now = time.strftime("%H:%M")
    for d in DATASETS:
        if a[d] is None or b[d] is None:
            print(f"{d:13s} (not started)")
            continue
        rate = (b[d] - a[d]) / dt if dt > 0 else 0.0
        remain = MAX_STEPS - b[d]
        eta_min = remain / rate if rate > 0 else float("inf")
        eta_clock = time.strftime("%H:%M", time.localtime(t1 + eta_min * 60)) if rate > 0 else "-"
        rows[d] = {"steps": b[d], "rate_per_min": round(rate, 1), "remaining": remain,
                   "eta_min": round(eta_min, 1), "eta_clock": eta_clock}
        print(f"{d:13s} {b[d]:7d} {rate:9.1f} {remain:7d} {eta_min:8.1f} {eta_clock:>10s}")

    if rows:
        slowest = max(rows.values(), key=lambda r: r["eta_min"])
        print(f"wave-1 ETA (slowest run) ~ {slowest['eta_clock']}  ({slowest['eta_min']:.0f} min from now, "
              f"clock at sample {clock_now})")

    payload = {"variant": args.variant, "log_root": args.log_root, "sampled_at": clock_now,
               "interval_min": round(dt, 2), "runs": rows}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("wrote", args.out)
    if args.write_status:
        p = os.path.join(args.log_root, "STATUS.md")
        lines = [f"# Run4 运行状态快照（{time.strftime('%Y-%m-%d %H:%M:%S')}）", "",
                 f"变体：`{args.variant}`（第 1 波 CTRL；E6 波次在其后自动启动）", "",
                 "| run | steps | rate/min | 剩余 | ETA |", "|---|---:|---:|---:|---|"]
        for d, r in rows.items():
            lines.append(f"| {d} | {r['steps']}/80000 | {r['rate_per_min']} | {r['remaining']} | {r['eta_clock']} |")
        lines += ["", "收口命令（两波全部结束后）：",
                  "```bash",
                  "GPU=0 bash train_scripts/CASA-TViM/Run4/post_train.sh",
                  "```",
                  "五段：协议审计 → 日志硬门 ×8 → 部署图对象指标 ×8 → 三层裁决 → README markdown；",
                  "前两段失败即 fail-closed 中止（exit 3 / 4）。证据索引见 "
                  "`train_scripts/CASA-TViM/Run4/EVIDENCE_INDEX.md`。", ""]
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
