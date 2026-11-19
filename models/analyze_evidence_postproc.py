"""Zero-training post-processing test for the context evidence map ``e``.

Motivation (t12): on LEVIR the finest-scale evidence is *negatively biased*
(``e_chg`` mean -1.693, p90 -0.893) while its AUC is still 0.916.  A bias, not a
ranking failure, means the branch is switched off by the tau=0 threshold rather
than being useless.  This script asks whether three cheap post-processing rules
rescue it, without touching weights:

  base         e1 upsampled to 256, threshold 0
  cell_center  e1 - mean over each grid-8 cell (removes the per-cell offset)
  scale_center e1 - mean of the whole stage map (global offset only)
  s3s4         mean of e3 and e4, both upsampled (drop the two finest scales)

Everything is CPU, batch 1, frozen Run1 DIFF weights.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyze_arm_diagnostics import change_evidence, confusion, metrics, run_capture
from analyze_geometry_degeneracy import (DATASET_ROOT, DATASETS, MODELS, load_checkpoint,
                                          subset_loader)

TILE = 256


def stage_maps(store):
    maps = []
    for item in store["logits"]:
        evidence = change_evidence(item)  # (B, Nq)
        height = int(math.isqrt(int(item.shape[2])))
        maps.append(evidence.reshape(1, 1, height, height))
    return maps


def cell_offset(e_map, grid):
    height = e_map.shape[-1]
    cells = height // grid
    block = e_map.reshape(1, 1, grid, cells, grid, cells)
    mean = block.mean(dim=(-3, -1), keepdim=True)
    return mean.repeat_interleave(cells, -2).repeat_interleave(cells, -1).reshape(1, 1, height, height)


def up(e_map, size=TILE):
    return F.interpolate(e_map, size=(size, size), mode="bilinear", align_corners=False)


def sweep(scores, truth, lo=-4.0, hi=4.0, step=0.05):
    best = (-1.0, 0.0)
    for threshold in np.arange(lo, hi + 1e-9, step):
        f1 = metrics(confusion(scores > threshold, truth))["f1"]
        if f1 > best[0]:
            best = (float(f1), float(threshold))
    return best


def evaluate_model(model, loader, limit, grid, tag):
    variants = ("base", "cell_center", "scale_center", "s3s4")
    counts = {name: dict(tp=0, fp=0, fn=0, tn=0) for name in variants}
    pooled = {name: ([], []) for name in variants}
    stage_means = {}
    images = 0
    for step, (image, target) in enumerate(loader):
        if limit and step >= limit:
            break
        target = target.float()
        _, store = run_capture(model, image[:, :3], image[:, 3:])
        maps = stage_maps(store)
        truth = (target > 0.5).numpy()[0, 0]
        scores = {
            "base": up(maps[0]),
            "cell_center": up(maps[0] - cell_offset(maps[0], grid)),
            "scale_center": up(maps[0] - maps[0].mean()),
            "s3s4": up((maps[2] + maps[3]) / 2),
        }
        for name in variants:
            array = scores[name].numpy()[0, 0].astype(np.float32)
            for key, value in confusion(array > 0, truth).items():
                counts[name][key] += value
            pooled[name][0].append(array.ravel())
            pooled[name][1].append(truth.ravel())
        for index, item in enumerate(maps):
            stage_means.setdefault(f"stage{index + 1}", []).append(float(item.mean()))
        images += 1
    report = dict(images=images, grid=grid)
    for name in variants:
        result = metrics(counts[name])
        scores = np.concatenate(pooled[name][0])
        truth = np.concatenate(pooled[name][1])
        best_f1, best_tau = sweep(scores, truth)
        result["best_f1"], result["best_tau"] = best_f1, best_tau
        result["auc"] = float(_auc(scores, truth))
        report[name] = result
    base = report["base"]
    for name in variants[1:]:
        report[name]["delta"] = {key: round(report[name][key] - base[key], 4)
                                 for key in ("f1", "precision", "recall")}
    report["stage_mean"] = {key: round(float(np.mean(value)), 4) for key, value in stage_means.items()}
    print(f"[{tag}] " + " ".join(
        f"{name}:F1={report[name]['f1']:.2f}"
        f"({report[name].get('delta', {}).get('f1', 0):+.2f})"
        for name in variants), flush=True)
    return report


def _auc(scores, labels):
    from analyze_arm_diagnostics import auc_score
    return auc_score(scores, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="DIFF")
    parser.add_argument("--datasets", default="LEVIR,WHU")
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = {"meta": dict(tile=TILE, loader="val", images_per_dataset=args.val_limit,
                           variants=["base", "cell_center", "scale_center", "s3s4"],
                           note="thresholds at 0 unless marked best_tau (in-sample sweep)")}
    report.setdefault("runs", {})
    for name in args.models.split(","):
        spec = MODELS[name]
        for dataset in args.datasets.split(","):
            model, path = load_checkpoint(spec, dataset)
            grid = int(model.context.grid_sizes[0])
            loader, _ = subset_loader(DATASET_ROOT / DATASETS[dataset], "val",
                                      args.val_limit, workers=0)
            report["runs"].setdefault(name, {})[dataset] = evaluate_model(
                model, loader, args.val_limit, grid, f"{name}/{dataset}")
            report["runs"][name][dataset]["checkpoint"] = path
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
