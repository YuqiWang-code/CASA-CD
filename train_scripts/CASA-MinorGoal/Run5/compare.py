"""Turn the Run5 TEST blocks into the pre-registered verdict.

Both arms live in this run, so the reference for the edge arm is its own
same-seed, same-schedule control (CG_HRA) rather than an earlier run.  The gate
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(C.ROOT))
    args = parser.parse_args()
    root = Path(args.root)

    rows = []
    for arm, dataset in C.JOBS:
        block = read_block(root / arm / dataset)
        gate = C.GATES.get(arm)
        delta, reference, reference_arm = None, None, None
        if gate is not None:
            reference_arm = gate["versus"]
            partner = read_block(root / reference_arm / dataset)
            reference = None if partner is None else partner["f1"]
            if block is not None and reference is not None:
                delta = round(block["f1"] - reference, 3)
        rows.append(dict(arm=arm,
                         dataset=dataset,
                         dataset_key=dataset.split("-")[0],
                         f1=None if block is None else round(block["f1"], 3),
                         best_epoch=None if block is None else block["best_epoch"],
                         recall=None if block is None else block["recall"],
                         precision=None if block is None else block["precision"],
                         versus=reference_arm,
                         reference=None if reference is None else round(reference, 3),
                         delta_pp=delta,
                         verdict="baseline" if gate is None else verdict(delta, gate),
                         gate=None if gate is None else dict(minimum=gate["minimum"],
                                                             noise=gate["noise"],
                                                             note=gate["note"])))

    # per-dataset pairing summary: one delta per library
    summary = {}
    for row in rows:
        if row["delta_pp"] is None:
            continue
        summary.setdefault(row["dataset_key"], []).append(row["delta_pp"])
    summary = {key: dict(n=len(values), mean_pp=round(sum(values) / len(values), 3))
               for key, values in summary.items()}

    payload = dict(protocol=C.PROTOCOL, selection=C.SELECTION, seed=C.SEED,
                   edge_weight=C.EDGE_WEIGHT, edge_dilate=C.EDGE_DILATE,
                   rows=rows, summary=summary)
    (root / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Run5 edge supervision: paired verdict", "",
             f"protocol `{C.PROTOCOL}`, val-selected, threshold 0.5, seed {C.SEED}, "
             f"edge weight {C.EDGE_WEIGHT:g} (dilate {C.EDGE_DILATE})", "",
             "| arm | dataset | F1 (%) | R / P | best ep | control | delta (pp) | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| {arm} | {dataset_key} | {f1} | {rp} | {ep} | {ref} | {delta} | {verdict} |".format(
            arm=row["arm"], dataset_key=row["dataset_key"],
            f1="—" if row["f1"] is None else f"{row['f1']:.2f}",
            rp="—" if row["recall"] is None else f"{row['recall']:.2f} / {row['precision']:.2f}",
            ep="—" if row["best_epoch"] is None else row["best_epoch"],
            ref="—" if row["reference"] is None else f"{row['reference']:.2f}",
            delta="—" if row["delta_pp"] is None else f"{row['delta_pp']:+.3f}",
            verdict=row["verdict"]))
    lines += ["", "## Pairing summary", ""]
    for key in ("SYSU", "LEVIR", "WHU", "CDD"):
        if key in summary:
            lines.append(f"- {key}: mean {summary[key]['mean_pp']:+.3f} pp over "
                         f"{summary[key]['n']} pair(s)")
    lines += ["", "A delta below the seed noise is not an effect; the CDD row is "
                  "exploratory because Run5 has no earlier same-seed reference there."]
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
