#!/usr/bin/env python
"""Run7 正式结果收集与对照（**只读** train_log.txt 的最后一个完整 TEST 区块）。

规则（沿用 Run4/Run5/Run6 纪律）：
  * 正式结果**只取** `train_log.txt` 里**最后一个完整** `=== TEST RESULTS === ... === END ===` 区块；
    README/表格不得替代原始日志；
  * 该区块内必须同时出现 `[ACTUAL-OPT-STEPS] 80000` 与
    `[PROTOCOL-VERSION] run7_hf_edge_exact80k_v1`，否则该 run 记 `INVALID`；
  * 六个指标 + Params/FLOPs 只从该区块解析。

    python train_scripts/CASA-TViM/Run7/collect_results.py \
      --log-root /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7 \
      --out /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/_RESULTS/summary.json
"""
import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJ, "analyse"))

from tvim_diag_common import parse_test_block  # noqa: E402

PROTOCOL = "run7_hf_edge_exact80k_v1"
VARIANTS = ["M1_R7CTRL", "E9_HFAE_A", "E10_HFAE_B"]
DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
METRICS = ["Recall", "Precision", "OA", "F1", "IoU", "Kappa"]


def _re_first(pat, text, cast=float):
    m = re.search(pat, text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (TypeError, ValueError):
        return m.group(1)


def collect_one(log_root, variant, dataset):
    log = os.path.join(log_root, variant, dataset, "train_log.txt")
    rec = {"variant": variant, "dataset": dataset, "log": log}
    if not os.path.isfile(log):
        rec.update({"status": "MISSING_LOG"})
        return rec
    block, info, span = parse_test_block(log)
    if block is None:
        rec.update({"status": "NO_COMPLETE_TEST_BLOCK"})
        return rec
    steps = _re_first(r"\[ACTUAL-OPT-STEPS\]\s+(\d+)", block, int)
    proto = _re_first(r"\[PROTOCOL-VERSION\]\s+(\S+)", block, str)
    rec.update({
        "test_block_lines": span,
        "opt_steps": steps,
        "protocol_version": proto,
        "metrics": {k: (info or {}).get(k) for k in METRICS},
        "params_M": (info or {}).get("Params(M)"),
        "trainable_M": (info or {}).get("Trainable(M)"),
        "flops_G": (info or {}).get("FLOPs(G)"),
        "best_f1_logged": (info or {}).get("BEST_F1"),
        "hfae_form": _re_first(r"\[HFAE-FORM\]\s+(\S+)", block, str),
        "hfae_deploy_params": _re_first(r"\[HFAE-DEPLOY\].*deploy_params=(\d+)", block, int),
        "fold_real_disagree": _re_first(r"\[REPARAM-REAL-ARGMAX-DISAGREE\]\s+([0-9.eE+-]+)",
                                       block, float),
        # 折叠门配套读数：真实 batch 的 max|Δp|，供刀锋豁免与证据文件对拍（见 pre-reg §7.6）
        "fold_real_max_abs_error": _re_first(r"\[REPARAM-REAL-MAX-ABS-ERROR\]\s+([0-9.eE+-]+)",
                                            block, float),
    })
    ok = (steps == 80000 and proto == PROTOCOL)
    rec["status"] = "VALID" if ok else "INVALID"
    if not ok:
        rec["invalid_reason"] = f"opt_steps={steps} protocol={proto} (expect 80000 / {PROTOCOL})"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rows = [collect_one(args.log_root, v, d) for v in variants for d in datasets]

    print(f"{'dataset':16s}{'variant':14s}{'status':10s}{'F1':>9s}{'IoU':>9s}"
          f"{'Recall':>9s}{'Prec':>9s}{'OA':>9s}{'Kappa':>9s}{'params':>10s}{'steps':>8s}")
    for r in rows:
        m = r.get("metrics") or {}
        def f(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
        print(f"{r['dataset']:16s}{r['variant']:14s}{r['status']:10s}{f('F1'):>9s}{f('IoU'):>9s}"
              f"{f('Recall'):>9s}{f('Precision'):>9s}{f('OA'):>9s}{f('Kappa'):>9s}"
              f"{str(r.get('params_M')):>10s}{str(r.get('opt_steps')):>8s}")

    print("\n=== Δ vs same-wave M1_R7CTRL (F1) ===")
    by = {(r["variant"], r["dataset"]): r for r in rows}
    for d in datasets:
        ctrl = (by.get(("M1_R7CTRL", d)) or {}).get("metrics") or {}
        c = ctrl.get("F1")
        line = f"  {d:16s} CTRL={c if c is None else f'{c:.4f}'}"
        for v in ("E9_HFAE_A", "E10_HFAE_B"):
            m = (by.get((v, d)) or {}).get("metrics") or {}
            x = m.get("F1")
            line += (f"  {v}={x:.4f} (Δ{x-c:+.4f})"
                     if isinstance(x, (int, float)) and isinstance(c, (int, float))
                     else f"  {v}=n/a")
        print(line)

    n_valid = sum(1 for r in rows if r["status"] == "VALID")
    print(f"\n[COLLECT] {n_valid}/{len(rows)} runs VALID (80000 steps + {PROTOCOL})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"protocol": PROTOCOL, "rows": rows,
                   "n_valid": n_valid, "n_total": len(rows)}, fh, indent=2,
                  ensure_ascii=False, default=str)
    print(f"[COLLECT] wrote {args.out}")
    return 0 if n_valid == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
