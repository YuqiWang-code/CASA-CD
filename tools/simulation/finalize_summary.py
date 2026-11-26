#!/usr/bin/env python
"""Aggregate S9a-final / S9b / S8b-final CSV outputs into table_*.csv.

Reads per_image_*.csv + summary_*.json written by s9a_final.py (and later
s8b_partition_final.py) and produces the small tables that go into the repo:
  table_metrics.csv        arm x (seed,dataset) global scores from integer counts
  table_paired_delta.csv   arm - BASE, paired bootstrap over images
  table_budget_curve.csv   gridset x (seed,dataset) scores + token counts

No test split is touched.  Nothing here trains or mutates the runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

DATASETS = ["SYSU-CD-256", "LEVIR-CD-256", "WHU-CD-256", "CDD-CD-256"]
KEYS = ["Recall", "Precision", "OA", "F1", "IoU", "Kappa"]


def scores(tp, fp, fn, tn):
    eps = 1e-12
    recall = tp / max(tp + fn, eps)
    precision = tp / max(tp + fp, eps)
    oa = (tp + tn) / max(tp + tn + fp + fn, eps)
    f1 = 2 * tp / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    total = tp + tn + fp + fn
    po = oa
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(total * total, eps)
    kappa = (po - pe) / max(1.0 - pe, eps)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Recall=recall, Precision=precision,
                OA=oa, F1=f1, IoU=iou, Kappa=kappa)


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def agg(rows):
    tp = sum(int(r["TP"]) for r in rows)
    fp = sum(int(r["FP"]) for r in rows)
    fn = sum(int(r["FN"]) for r in rows)
    tn = sum(int(r["TN"]) for r in rows)
    out = scores(tp, fp, fn, tn)
    out["images"] = len(rows)
    return out


def boot_delta(base_rows, arm_rows, iters, seed=7):
    """Paired bootstrap over images: resample images, recompute both F1s."""
    rng = random.Random(seed)
    by_name_b = {r["file"]: r for r in base_rows}
    names = [r["file"] for r in arm_rows if r["file"] in by_name_b]
    n = len(names)
    if n == 0:
        return None
    diffs = []
    for _ in range(iters):
        tp_b = fp_b = fn_b = 0
        tp_a = fp_a = fn_a = 0
        for _ in range(n):
            nm = names[rng.randrange(n)]
            b = by_name_b[nm]
            a = next(r for r in arm_rows if r["file"] == nm)
            tp_b += int(b["TP"]); fp_b += int(b["FP"]); fn_b += int(b["FN"])
            tp_a += int(a["TP"]); fp_a += int(a["FP"]); fn_a += int(a["FN"])
        f1_b = 2 * tp_b / max(2 * tp_b + fp_b + fn_b, 1e-12)
        f1_a = 2 * tp_a / max(2 * tp_a + fp_a + fn_a, 1e-12)
        diffs.append(f1_a - f1_b)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters) - 1]
    mean = sum(diffs) / iters
    return mean, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="S9aFinal")
    ap.add_argument("--out", default="")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--include-grids", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(args.dir)
    outdir = Path(args.out) if args.out else root
    outdir.mkdir(parents=True, exist_ok=True)

    metrics, paired, budget = [], [], []
    per_image = {}
    for p in sorted(root.glob("per_image_*.csv")):
        tag = p.stem.replace("per_image_", "")
        per_image[tag] = read_rows(p)

    # ---- arms tables -------------------------------------------------
    # filenames are ambiguous (arm names contain "_"), so group by row fields.
    groups = {}
    for tag, rows in per_image.items():
        if "arms" not in tag:
            continue
        for r in rows:
            key = (r["arm_checkpoint"] if "arm_checkpoint" in r else
                   r.get("checkpoint", ""), r["seed"], r["dataset"], r["arm"])
            groups.setdefault(key, []).append(r)
    # the per-image CSV does not carry the checkpoint column: rebuild it from
    # (seed, dataset) is impossible, so use the file tag suffix instead.
    groups = {}
    for tag, rows in per_image.items():
        if "arms" not in tag:
            continue
        ck = tag.split("_arms")[0]
        ck = ck.replace("per_image_", "")
        ds = [d for d in DATASETS if d in ck]
        if not ds:
            continue
        dataset = ds[0]
        checkpoint = ck.replace("_" + dataset, "")
        seed = checkpoint.split("-")[-1]
        for r in rows:
            groups.setdefault((checkpoint, seed, dataset, r["arm"]), []).append(r)
    for (checkpoint, seed, dataset, arm), sub in groups.items():
        v = agg(sub)
        metrics.append(dict(arm=arm, checkpoint=checkpoint, seed=seed,
                            dataset=dataset, mode="arms", gridset="b224",
                            images=v["images"],
                            **{k: round(v[k], 6) for k in
                               ["TP", "FP", "FN", "TN"] + KEYS}))
    # paired bootstrap: same (checkpoint, dataset) BASE vs each arm
    ck_ck = sorted({(c, d) for (c, s_, d, a) in groups})
    for (checkpoint, dataset) in ck_ck:
        base = groups.get((checkpoint, checkpoint.split("-")[-1], dataset, "BASE"), [])
        arms_here = sorted({a for (c, s_, d, a) in groups if c == checkpoint and d == dataset})
        for arm in arms_here:
            if arm == "BASE":
                continue
            sub = groups[(checkpoint, checkpoint.split("-")[-1], dataset, arm)]
            bd = boot_delta(base, sub, args.iters)
            if bd is None:
                continue
            mean, lo, hi = bd
            paired.append(dict(arm=arm, checkpoint=checkpoint,
                               seed=checkpoint.split("-")[-1], dataset=dataset,
                               dF1=round(mean, 6), lo=round(lo, 6), hi=round(hi, 6),
                               excludes_zero=int(lo > 0 or hi < 0)))

    # ---- budget curve ------------------------------------------------
    for p in sorted(root.glob("summary_*_grids*.json")):
        s = json.loads(p.read_text())
        arm_name = s["arm"]
        seed = arm_name.split("-")[-1]
        g = dict(s["global_scores"]["BASE"])
        g.update(g.pop("counts", {}))
        budget.append(dict(gridset=s["gridset"], checkpoint=arm_name, seed=seed,
                           dataset=s["dataset"], images=s["images"],
                           regions=sum(gg * gg for gg in s["grids"]),
                           tokens=sum(2 * gg * gg for gg in s["grids"]),
                           **{k: round(g[k], 6) for k in
                              ["TP", "FP", "FN", "TN"] + KEYS}))

    def dump(name, rows, fields=None):
        if not rows:
            return
        path = outdir / name
        flds = fields or list(rows[0].keys())
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=flds)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in flds} for r in rows])
        print(f"wrote {path} ({len(rows)} rows)")

    dump("table_metrics.csv", metrics)
    dump("table_paired_delta.csv", paired)
    dump("table_budget_curve.csv", budget)

    # ---- console summary --------------------------------------------
    print("\n== global scores (arms, b224) ==")
    for dataset in ["SYSU-CD-256", "LEVIR-CD-256"]:
        arms = sorted({m["arm"] for m in metrics if m["dataset"] == dataset})
        print(f"\n[{dataset}]")
        for arm in arms:
            sub = [m for m in metrics if m["dataset"] == dataset and m["arm"] == arm]
            if not sub:
                continue
            mean_f1 = sum(m["F1"] for m in sub) / len(sub)
            print(f"  {arm:12s} F1={mean_f1:.4f}  " +
                  " ".join(f"seed{m['seed']}={m['F1']:.4f}" for m in sub))
        sub = [d for d in paired if d["dataset"] == dataset]
        if sub:
            print(f"  paired dF1 vs BASE:")
            for d in sorted(sub, key=lambda x: x["dF1"]):
                sig = "*" if d["excludes_zero"] else " "
                print(f"    {d['arm']:12s} {d['dF1']*100:+6.2f} pp "
                      f"[{d['lo']*100:+5.2f},{d['hi']*100:+5.2f}] {sig}")


if __name__ == "__main__":
    main()
