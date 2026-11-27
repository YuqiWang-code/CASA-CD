"""Turn the Run11 HRA TEST blocks into the pre-registered verdict.

Both arms live in this run, so the reference for the edge arm is its own
same-seed, same-schedule control rather than an earlier run.  The gate
lives in ``config.GATES`` and was fixed before the first job started.
"""
import argparse
import json
import re
from pathlib import Path

import config as C

BLOCK = re.compile(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", re.S)
F1 = re.compile(r"F1=([0-9.]+)")
EPOCH = re.compile(r"\[BEST-EPOCH\]\s*(\d+)")
RECALL = re.compile(r"Recall=([0-9.]+)")
PRECISION = re.compile(r"Precision=([0-9.]+)")


def read_block(directory):
    path = Path(directory) / "train_log.txt"
    if not path.is_file():
        return None
    match = BLOCK.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    block = match[-1]
    f1 = F1.search(block)
    if f1 is None:
        return None
    epoch = EPOCH.search(block)
    recall = RECALL.search(block)
    precision = PRECISION.search(block)
    return dict(f1=float(f1.group(1)) * 100,
                best_epoch=int(epoch.group(1)) if epoch else None,
                recall=round(float(recall.group(1)) * 100, 2) if recall else None,
                precision=round(float(precision.group(1)) * 100, 2) if precision else None)


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


def summary(root):
    """Mean F1 of the two seeds for every (base, dataset)."""
    table = {}
    for arm, dataset in C.JOBS:
        block = read_block(Path(root) / arm / dataset)
        if block is None:
            continue
        key = (C.base_of(arm), dataset)
        table.setdefault(key, []).append((arm, block))
    out = {}
    for key, rows in table.items():
        mean = sum(item[1]["f1"] for item in rows) / len(rows)
        out[key] = dict(f1=round(mean, 4), seeds=len(rows),
                        detail=[dict(arm=arm, **block) for arm, block in rows])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(C.ROOT))
    parser.add_argument("--json", default=str(C.ROOT / "comparison.json"))
    parser.add_argument("--markdown", default=str(C.ROOT / "comparison.md"))
    args = parser.parse_args()
    root = Path(args.root)
    table = summary(root)

    lines = ["# Run11 HRA axes: pre-registered verdict", "",
             "Two seeds per arm; each delta is vs the same-run control named in",
             "`config.GATES`. Gate = max(0.36, 1.5 sigma) = 0.59 pp for a module",
             "that must add something, 0.36 pp for the single value axis, and",
             "-0.36 pp for the 32-vs-224 budget check.", "",
             "The attribution question H32 vs E32 is reported as-is: a tie means the",
             "three axes are not one geometry, not that the measurement failed.", ""]
    payload = dict(protocol=C.PROTOCOL, reference={}, rows=[])
    for dataset in C.DATASETS:
        lines += [f"## {dataset}", "",
                  "| arm | F1 (%) | vs | delta (pp) | verdict |", "|---|---|---|---|---|"]
        for base in C.BASE_ORDER:
            entry = table.get((base, dataset))
            if entry is None:
                lines.append(f"| {base} | pending | - | - | - |")
                continue
            gate = C.GATES.get(base)
            if gate is None:
                lines.append(f"| {base} | {entry['f1']:.4f} | - | - | reference |")
                payload["rows"].append(dict(dataset=dataset, arm=base, f1=entry["f1"],
                                            versus=None, delta=None, verdict="reference"))
                continue
            ref = table.get((gate["versus"], dataset))
            if ref is None:
                lines.append(f"| {base} | {entry['f1']:.4f} | {gate['versus']} | pending | - |")
                continue
            delta = round(entry["f1"] - ref["f1"], 4)
            note = verdict(delta, gate)
            lines.append(f"| {base} | {entry['f1']:.4f} | {gate['versus']} | "
                         f"{delta:+.4f} | {note} |")
            payload["rows"].append(dict(dataset=dataset, arm=base, f1=entry["f1"],
                                        versus=gate["versus"], delta=delta, verdict=note))
        lines.append("")
    deltas = [row["delta"] for row in payload["rows"] if row["delta"] is not None]
    if deltas:
        lines += [f"mean |delta| over {len(deltas)} cells: "
                  f"{sum(deltas) / len(deltas):+.4f} pp", ""]
    text = "\n".join(lines)
    Path(args.markdown).write_text(text, encoding="utf-8")
    Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
