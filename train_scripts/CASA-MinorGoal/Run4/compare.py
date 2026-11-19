"""Turn the Run4 TEST blocks into the pre-registered verdicts.

Each arm has exactly one matched reference at the same seed (2334):
TAR_CG vs Run2 CTRL, HRA_SPP vs Run3 ARM_C (same bridge, different readout),
CG_HRA vs Run2 CTRL. The gates live in ``config.GATES`` and were fixed before
the first job started.
"""
import argparse
import json
import re
from pathlib import Path

import config as C

BLOCK = re.compile(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", re.S)
F1 = re.compile(r"F1=([0-9.]+)")
EPOCH = re.compile(r"\[BEST-EPOCH\]\s*(\d+)")


def read_f1(directory):
    path = Path(directory) / "train_log.txt"
    if not path.is_file():
        return None, None
    match = BLOCK.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None, None
    block = match[-1]
    f1 = F1.search(block)
    epoch = EPOCH.search(block)
    if f1 is None:
        return None, None
    return float(f1.group(1)) * 100, (int(epoch.group(1)) if epoch else None)


def verdict(delta, gate):
    if delta is None:
        return "pending"
    if delta >= gate["minimum"]:
        return f"pass (>= {gate['minimum']:.2f})"
    if abs(delta) <= gate["noise"]:
        return f"inside noise ({gate['noise']:.2f})"
    if delta <= -gate["minimum"]:
        return f"worse (<= -{gate['minimum']:.2f})"
    return "directional"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(C.ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for arm, dataset in C.JOBS:
        gate = C.GATES[arm]
        reference_arm = gate["versus"]
        seed = C.seed_for(arm)
        reference = C.REFERENCE.get((reference_arm, dataset, seed))
        f1, epoch = read_f1(root / arm / dataset)
        delta = None if f1 is None or reference is None else round(f1 - reference, 3)
        rows.append(dict(arm=arm, dataset=dataset.split("-")[0], f1=f1, best_epoch=epoch,
                         reference_arm=reference_arm, reference=reference, delta_pp=delta,
                         gate=gate["minimum"], noise=gate["noise"], verdict=verdict(delta, gate)))
    lines = ["| arm | dataset | F1 (%) | control | control F1 | delta (pp) | gate (pp) | verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append("| {arm} | {dataset} | {f1} | {reference_arm} | {reference} | {delta_pp} | "
                     "{gate} | {verdict} |".format(**{k: ("—" if row[k] is None else row[k])
                                                     for k in row}))
    summary = {}
    for arm in C.ARM_ORDER:
        values = [row["delta_pp"] for row in rows if row["arm"] == arm and row["delta_pp"] is not None]
        summary[arm] = dict(n=len(values), mean_pp=(sum(values) / len(values) if values else None),
                            deltas=values)
    lines.append("")
    lines.append("| arm | n | mean delta (pp) | per-dataset |")
    lines.append("| --- | --- | --- | --- |")
    for arm, item in summary.items():
        mean = "—" if item["mean_pp"] is None else f"{item['mean_pp']:+.3f}"
        lines.append(f"| {arm} | {item['n']} | {mean} | "
                     + ", ".join(f"{v:+.3f}" for v in item["deltas"]) + " |")
    text = "\n".join(lines) + "\n"
    (root / "comparison.md").write_text(text, encoding="utf-8")
    (root / "comparison.json").write_text(json.dumps(dict(rows=rows, summary=summary), indent=2),
                                          encoding="utf-8")
    (root / "attribution.md").write_text(text, encoding="utf-8")
    (root / "attribution.json").write_text(json.dumps(dict(rows=rows, gates=C.GATES), indent=2),
                                           encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
