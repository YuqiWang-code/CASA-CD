#!/usr/bin/env python
"""Run2 结果对比报告：outputs/ 下的 Run1(M1_FULL) 与 Run2(E1/E2/E3) TEST RESULTS
自动对比，输出 markdown 报告 + 达标判定（调研文档 §15 通过标准）。

用法（本地，先跑 .claude/_download_tvim_logs_run2.py 下载日志）：
    python analyse/run2_report.py
输出：docs/temporary/run2_report.md
"""
import os
import glob
import re

from extract_metrics_to_excel import extract_test_block, parse_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
DOCS_TMP = os.path.join(ROOT, "docs", "temporary")

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
RUN2_VARIANTS = ["E1_CP_CAACP", "E2_FRH", "E3_CP_FRH"]

# §15 通过标准（相对 M1 基线 F1）
TARGETS = {
    "E1_CP_CAACP": {"CDD-CD-256": ("gte", 97.18), "WHU-CD-256": ("gte", 95.07),
                    "LEVIR-CD-256": ("expect", 91.4), "SYSU-CD-256": ("gte", 83.47)},
    "E2_FRH": {"CDD-CD-256": ("gte", 97.18), "WHU-CD-256": ("gte", 95.07),
               "LEVIR-CD-256": ("expect", 91.5), "SYSU-CD-256": ("gte", 83.7)},
    "E3_CP_FRH": {"CDD-CD-256": ("gte", 97.18), "WHU-CD-256": ("gte", 95.07),
                  "LEVIR-CD-256": ("expect", 92.0), "SYSU-CD-256": ("gte", 84.0)},
}


def read_log(tag, run, variant, ds):
    if variant is None:
        path = os.path.join(OUTPUTS, tag, run, ds, "train_log.txt")
    else:
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
    return info, extra


def fmt(f1, iou):
    return f"{f1 * 100:.2f} / {iou * 100:.2f}"


def main():
    lines = []
    lines.append("# CASA-TViM Run2 结果对比（自动生成）\n")
    lines.append(f"基线：Run1 M1_FULL（CAACP rank + STR，deploy 4.880M）。Run2 变体部署 4.880~4.881M。\n")

    base = {}
    for ds in DATASETS:
        r = read_log("CASA-TViM", "Run1", "M1_FULL", ds)
        if r is None:
            print(f"[missing baseline] M1_FULL/{ds}")
            base[ds] = None
        else:
            base[ds] = r[0]
            lines.append(f"- M1 {ds}: F1={r[0]['F1'] * 100:.2f} IoU={r[0]['IoU'] * 100:.2f}")
    lines.append("")

    for var in RUN2_VARIANTS:
        lines.append(f"## {var}\n")
        hdr = "| 数据集 | M1 F1/IoU | {var} F1/IoU | ΔF1 | 达标（§15） | β | cell熵 | disagree | deploy(M) |".format(var=var)
        lines.append(hdr)
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for ds in DATASETS:
            r = read_log("CASA-TViM", "Run2", var, ds)
            if r is None:
                lines.append(f"| {ds} | {fmt(base[ds]['F1'], base[ds]['IoU']) if base[ds] else '-'} | 未完成 | — | — | — | — | — | — |")
                continue
            info, extra = r
            b = base[ds]
            d_f1 = (info["F1"] - b["F1"]) * 100 if b else float("nan")
            kind, target = TARGETS[var][ds]
            if kind == "gte":
                ok = info["F1"] * 100 >= target
            else:
                ok = info["F1"] * 100 >= target  # expect 也按数值判定，标注期望
            mark = "✓" if ok else "✗"
            lines.append(
                f"| {ds} | {fmt(b['F1'], b['IoU']) if b else '-'} | {fmt(info['F1'], info['IoU'])} "
                f"| {d_f1:+.2f} | {mark} {info['F1'] * 100:.2f}{'>=' if kind == 'gte' else '~'}{target} | "
                f"{extra.get('beta', float('nan')):.2e} | {extra.get('entropy', float('nan')):.3f} | "
                f"{extra.get('disagree_real', float('nan')):.2e} | {extra.get('deploy_total', float('nan')):.3f} |")
        lines.append("")

    out = os.path.join(DOCS_TMP, "run2_report.md")
    os.makedirs(DOCS_TMP, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
