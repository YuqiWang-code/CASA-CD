#!/usr/bin/env python
"""Run3 结果对比报告：outputs/ 下的 Run1(M1_FULL) 与 Run3(E4/E5) TEST RESULTS 自动对比，
含复盘文档 §13 预注册裁决（E4/E5 各自的成功/失败条件）。

用法（本地，先跑 .claude/_download_tvim_logs_run3.py 下载日志）：
    python analyse/run3_report.py
输出：docs/temporary/run3_report.md
"""
import os
import re

from extract_metrics_to_excel import extract_test_block, parse_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
DOCS_TMP = os.path.join(ROOT, "docs", "temporary")

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]

# 复盘文档 §13 预注册条件
REG = {
    "E4_RA_CAACP": {
        "hard": {"CDD-CD-256": 97.18, "WHU-CD-256": 95.07},
        "mech": "SYSU F1>83.47、SYSU Recall≥84.41、SYSU small F1≥0.195（D2 复测）",
        "levir": 91.07,
    },
    "E5_FS_TAR": {
        "hard": {"CDD-CD-256": 97.18, "WHU-CD-256": 95.07},
        "mech": "SYSU F1≥83.80、SYSU Recall>84.41、SYSU small F1≥0.205（D2 复测）",
        "levir": 91.20,
    },
}


def read_log(tag, run, variant, ds):
    path = os.path.join(OUTPUTS, tag, run, variant, ds, "train_log.txt")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    block = extract_test_block(text)
    if block is None:
        return None
    info = parse_block(block)
    if "F1" not in info:
        return None
    extra = {}
    m = re.search(r"\[CAACP-BETA\]\s+([0-9.eE+-]+)", block)
    if m:
        extra["beta"] = float(m.group(1))
    m = re.search(r"\[CAACP-WEIGHT-ENTROPY\]\s+([0-9.eE+-]+)", block)
    if m:
        extra["entropy"] = float(m.group(1))
    m = re.search(r"\[REPARAM-REAL-ARGMAX-DISAGREE\]\s+([0-9.eE+-]+)", block)
    if m:
        extra["disagree_real"] = float(m.group(1))
    m = re.search(r"\[DEPLOY-PARAMS\]\s+total=([0-9.]+)", block)
    if m:
        extra["deploy_total"] = float(m.group(1))
    m = re.search(r"\[BACKBONE-ADAPT-FINAL\]\s+rel_L2_from_pretrain=([0-9.eE+-]+)", block)
    if m:
        extra["rel_l2"] = float(m.group(1))
    return info, extra


def fmt(f1, iou):
    return f"{f1 * 100:.2f} / {iou * 100:.2f}"


def main():
    lines = []
    lines.append("# CASA-TViM Run3 结果对比（自动生成）\n")
    lines.append("基线：Run1 M1_FULL（CAACP rank + STR，deploy 4.880M）。"
                 "E4 deploy 4.880M；E5 deploy 4.954M。\n")

    base = {}
    for ds in DATASETS:
        r = read_log("CASA-TViM", "Run1", "M1_FULL", ds)
        base[ds] = r[0] if r else None
        if r:
            lines.append(f"- M1 {ds}: F1={r[0]['F1'] * 100:.2f} IoU={r[0]['IoU'] * 100:.2f} "
                         f"R={r[0]['Recall'] * 100:.2f} P={r[0]['Precision'] * 100:.2f}")
    lines.append("")

    for var in ("E4_RA_CAACP", "E5_FS_TAR"):
        lines.append(f"## {var}\n")
        lines.append("| 数据集 | M1 F1/IoU | " + var + " F1/IoU | ΔF1 | R/P | 硬条件 | β | disagree | deploy(M) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for ds in DATASETS:
            r = read_log("CASA-TViM", "Run3", var, ds)
            if r is None:
                lines.append(f"| {ds} | {fmt(base[ds]['F1'], base[ds]['IoU']) if base[ds] else '-'} | 未完成 | — | — | — | — | — | — |")
                continue
            info, extra = r
            b = base[ds]
            d_f1 = (info["F1"] - b["F1"]) * 100 if b else float("nan")
            hard = ""
            if ds in REG[var]["hard"]:
                ok = info["F1"] * 100 >= REG[var]["hard"][ds]
                hard = f"{'✓' if ok else '✗'} {info['F1'] * 100:.2f}>={REG[var]['hard'][ds]}"
            lines.append(
                f"| {ds} | {fmt(b['F1'], b['IoU']) if b else '-'} | {fmt(info['F1'], info['IoU'])} "
                f"| {d_f1:+.2f} | {info['Recall'] * 100:.2f} / {info['Precision'] * 100:.2f} "
                f"| {hard} | {extra.get('beta', float('nan')):.2e} | {extra.get('disagree_real', float('nan')):.2e} | "
                f"{extra.get('deploy_total', float('nan')):.3f} |")
        lines.append(f"\n预注册机制条件（{REG[var]['mech']}）→ 需 D2 复测（"
                     "python analyse/run2_zero_cost_diag.py --ckpt_run Run3 --variants "
                     f"{var} --datasets SYSU-CD-256）裁决；LEVIR 不显著低于 {REG[var]['levir']}。\n")

    out = os.path.join(DOCS_TMP, "run3_report.md")
    os.makedirs(DOCS_TMP, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
