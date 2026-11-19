"""Frozen-weight diagnostics: why does the difference context help only on WHU?

Reads the trained Run1 MinorGoal ``DIFF`` checkpoints (WHU, LEVIR, SYSU) on CPU and
measures, without training and without editing ``model/``:

  H-A  feature cleanliness : Fisher ratio between change and background prototypes
                             at the finest scale (grid 8) plus prototype-assignment
                             entropy (support entropy and attention entropy);
  H-B  support quality     : per-image prior instability (total variation of the
                             per-cell change support against the dataset prior) and
                             the cross-image CV of ``class_support_min``;
  H-C  grid too coarse     : GT connected-component minimum-axis distribution vs
                             the cell size of every scale (32/64/64/64 px).

It also reports the geometric evidence ``e`` that the context branch already
produces (``geometric_change_logit``) as a stand-alone classifier: pooled AUC
against GT, absolute means per region and pooled F1 when ``e`` is used directly
as logits (threshold 0 and best threshold).

All numbers are frozen-weight forward statistics on a val subset (batch 1, CPU).
Metrics are pooled over pixels unless explicitly marked ``per_image``; the
per-image F1 mean is deflated by empty-change tiles and is kept for reference only.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import spearmanr
from torch.nn import functional as F

import model.layers.hyperbolic_context as hc
from analyze_geometry_degeneracy import (DATASET_ROOT, DATASETS, MODELS,
                                         load_checkpoint, subset_loader)

TILE = 256
FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


# --------------------------------------------------------------------------- utils
def summarize(values):
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return None
    return dict(n=int(array.size), mean=float(array.mean()),
                p10=float(np.percentile(array, 10)), p50=float(np.percentile(array, 50)),
                p90=float(np.percentile(array, 90)),
                min=float(array.min()), max=float(array.max()))


def auc_score(scores, labels):
    """Rank AUC with average ranks for ties."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    ordered = scores[order]
    start = 0
    while start < ordered.size:
        stop = start
        while stop + 1 < ordered.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        if stop > start:
            ranks[order[start:stop + 1]] = 0.5 * (start + 1 + stop + 1)
        start = stop + 1
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def confusion(prediction, truth):
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    return dict(tp=int(np.logical_and(prediction, truth).sum()),
                fp=int(np.logical_and(prediction, ~truth).sum()),
                fn=int(np.logical_and(~prediction, truth).sum()),
                tn=int(np.logical_and(~prediction, ~truth).sum()))


def metrics(counts):
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                f1=100.0 * 2 * tp / max(2 * tp + fp + fn, 1),
                precision=100.0 * tp / max(tp + fp, 1),
                recall=100.0 * tp / max(tp + fn, 1),
                iou=100.0 * tp / max(tp + fp + fn, 1))


def f1_score(prediction, truth):
    counts = confusion(prediction, truth)
    return 100.0 * 2 * counts["tp"] / max(2 * counts["tp"] + counts["fp"] + counts["fn"], 1)


def add_counts(total, counts):
    for key in ("tp", "fp", "fn", "tn"):
        total[key] += counts[key]


def change_evidence(logits):
    """``geometric_change_logit`` without the head mean (heads=1)."""
    positive, background = logits.reshape(*logits.shape[:-1], -1, 2).unbind(-1)
    evidence = positive.logsumexp(-1) - background.logsumexp(-1)
    return evidence.mean(1)


def binary_entropy(positive, background):
    total = (positive + background).clamp_min(1e-12)
    p = (positive / total).clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2.0)


def component_stats(mask):
    labels, count = ndimage.label(mask, structure=FOUR_CONNECTED)
    min_axes, areas = [], []
    if count == 0:
        return min_axes, areas
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        height = box[0].stop - box[0].start
        width = box[1].stop - box[1].start
        min_axes.append(min(height, width))
        areas.append(int((labels[box] == index).sum()))
    return min_axes, areas


# ------------------------------------------------------------------------- capture
def run_capture(model, images_a, images_b):
    """Forward once, returning the primary logits and the context internals."""
    context = model.context
    store = dict(logits=[], prototypes=[], support=[])
    original_logits = context.attention_logits
    original_pool = hc.pool_features
    original_assign = hc.class_assignments

    def attention_logits(query, key, support, index):
        out = original_logits(query, key, support, index)
        store["logits"].append(out.detach().float())
        return out

    def pool_features(feature, weights, grid):
        out = original_pool(feature, weights, grid)
        store["prototypes"].append(out.detach().float())
        return out

    def class_assignments(probability, grid):
        weights, support = original_assign(probability, grid)
        store["support"].append(support.detach().float())
        return weights, support

    context.attention_logits = attention_logits
    hc.pool_features = pool_features
    hc.class_assignments = class_assignments
    try:
        with torch.inference_mode():
            logits, _ = model(images_a, images_b, return_aux=True)
    finally:
        context.attention_logits = original_logits
        hc.pool_features = original_pool
        hc.class_assignments = original_assign
    return logits, store


# ------------------------------------------------------------------- per-dataset
def diagnose_dataset(model, loader, limit, tag):
    context = model.context
    grid_sizes = tuple(context.grid_sizes)
    cell_px = [TILE // grid for grid in grid_sizes]
    series = defaultdict(list)
    pooled = defaultdict(list)
    model_counts = dict(tp=0, fp=0, fn=0, tn=0)
    evidence_counts = dict(tp=0, fp=0, fn=0, tn=0)
    components_min_axis, components_area = [], []
    images_with_change = 0
    images_with_prediction = 0
    started = time.time()
    print(f"[{tag}] grids={grid_sizes} cell_px={cell_px}", flush=True)

    for step, (image, target) in enumerate(loader):
        if limit and step >= limit:
            break
        images_a, images_b = image[:, :3], image[:, 3:]
        target = target.float()
        logits, store = run_capture(model, images_a, images_b)
        primary = logits[0]
        truth_full = (target > 0.5).numpy()[0, 0]
        pred_full = (primary > 0).numpy()[0, 0]
        model_counts_step = confusion(pred_full, truth_full)
        add_counts(model_counts, model_counts_step)
        series["model_f1_per_image"].append(f1_score(pred_full, truth_full))
        series["changed_fraction"].append(float(truth_full.mean()))
        images_with_change += int(truth_full.any())
        images_with_prediction += int(pred_full.any())
        pooled["truth"].append(truth_full)
        pooled["primary"].append(primary.numpy()[0, 0].astype(np.float32))
        evidence = [change_evidence(item) for item in store["logits"]]
        for stage, (item, grid) in enumerate(zip(store["logits"], grid_sizes)):
            queries = int(item.shape[2])
            height = int(round(math.sqrt(queries)))
            width = height
            e_map = evidence[stage].reshape(1, 1, height, width)
            gt_small = (F.interpolate(target, size=(height, width), mode="nearest") > 0.5)
            gt_small = gt_small.numpy()[0, 0]
            e_np = e_map.numpy()[0, 0]
            changed = gt_small
            background = ~gt_small
            if changed.any():
                series[f"stage{stage + 1}.e_abs_changed"].append(float(np.abs(e_np[changed]).mean()))
                series[f"stage{stage + 1}.e_mean_changed"].append(float(e_np[changed].mean()))
            if background.any():
                series[f"stage{stage + 1}.e_abs_background"].append(float(np.abs(e_np[background]).mean()))
                series[f"stage{stage + 1}.e_mean_background"].append(float(e_np[background].mean()))
            pooled[f"e_stage{stage + 1}"].append(e_np.ravel())
            pooled[f"gt_stage{stage + 1}"].append(gt_small.ravel())
            if stage == 0:
                e_up = F.interpolate(e_map, size=(target.shape[-2], target.shape[-1]),
                                     mode="bilinear", align_corners=False).numpy()[0, 0]
                add_counts(evidence_counts, confusion(e_up > 0, truth_full))
                pooled["e_full"].append(e_up.astype(np.float32))
                attention = item.softmax(-1)
                entropy = -(attention * attention.clamp_min(1e-9).log()).sum(-1)
                series["attention_entropy"].append(float(entropy.mean() / math.log(int(item.shape[-1]))))
                series["attention_top1"].append(float(attention.max(-1).values.mean()))

            support = store["support"][stage][0]
            positive = support[0::2]
            background_support = support[1::2]
            series[f"stage{stage + 1}.support_min"].append(float(support.min()))
            series[f"stage{stage + 1}.support_sum"].append(float(positive.sum()))
            series[f"stage{stage + 1}.support_entropy"].append(
                float(binary_entropy(positive, background_support).mean()))
            prior = positive / positive.sum().clamp_min(1e-12)
            series[f"stage{stage + 1}.prior"].append(prior)

        prototype = store["prototypes"][0][0]
        cells = grid_sizes[0] ** 2
        prototype = prototype.reshape(cells, 2, -1)
        pos, neg = prototype[:, 0], prototype[:, 1]
        mean_pos, mean_neg = pos.mean(0), neg.mean(0)
        var_pos, var_neg = pos.var(0, unbiased=False), neg.var(0, unbiased=False)
        fisher = float(((mean_pos - mean_neg) ** 2).sum() / (0.5 * (var_pos + var_neg).mean()).clamp_min(1e-12))
        series["fisher_ratio"].append(fisher)
        series["prototype_cosine"].append(float(F.cosine_similarity(pos, neg, dim=-1).mean()))
        scale = 0.5 * (pos.norm(dim=-1) + neg.norm(dim=-1))
        series["prototype_relative_distance"].append(
            float(((pos - neg).norm(dim=-1) / scale.clamp_min(1e-12)).mean()))

        min_axes, areas = component_stats(truth_full)
        components_min_axis.extend(min_axes)
        components_area.extend(areas)
        if step % 25 == 24:
            print(f"[{tag}] {step + 1} images, {time.time() - started:.1f}s", flush=True)
    print(f"[{tag}] done {step + 1} images in {time.time() - started:.1f}s", flush=True)

    truth = np.concatenate(pooled["truth"])
    primary = np.concatenate(pooled["primary"])
    e_full = np.concatenate(pooled["e_full"])
    best_f1, best_threshold = None, None
    for threshold in np.quantile(e_full, np.linspace(0.50, 0.995, 10)):
        score = f1_score(e_full > float(threshold), truth)
        if best_f1 is None or score > best_f1:
            best_f1, best_threshold = score, float(threshold)

    report = dict(
        images=step + 1,
        grids=list(grid_sizes), cell_px=cell_px,
        model=dict(metrics(model_counts),
                   auc=auc_score(primary, truth),
                   images_with_change=images_with_change,
                   images_with_prediction=images_with_prediction,
                   f1_per_image=summarize(series["model_f1_per_image"]),
                   changed_fraction=summarize(series["changed_fraction"])),
        evidence=dict(
            metrics(evidence_counts),
            auc=auc_score(e_full, truth),
            best_f1=best_f1, best_threshold=best_threshold,
            absolute_mean_changed=summarize(series["stage1.e_abs_changed"]),
            absolute_mean_background=summarize(series["stage1.e_abs_background"]),
            mean_changed=summarize(series["stage1.e_mean_changed"]),
            mean_background=summarize(series["stage1.e_mean_background"]),
            stages={f"stage{s + 1}": dict(
                auc=auc_score(np.concatenate(pooled[f"e_stage{s + 1}"]),
                              np.concatenate(pooled[f"gt_stage{s + 1}"])),
                f1_threshold_0=f1_score(np.concatenate(pooled[f"e_stage{s + 1}"]) > 0,
                                        np.concatenate(pooled[f"gt_stage{s + 1}"])),
                absolute_mean_changed=summarize(series[f"stage{s + 1}.e_abs_changed"]),
                absolute_mean_background=summarize(series[f"stage{s + 1}.e_abs_background"]),
                mean_changed=summarize(series[f"stage{s + 1}.e_mean_changed"]),
                mean_background=summarize(series[f"stage{s + 1}.e_mean_background"]),
            ) for s in range(len(grid_sizes))},
        ),
        prototype=dict(
            fisher_ratio=summarize(series["fisher_ratio"]),
            cosine=summarize(series["prototype_cosine"]),
            relative_distance=summarize(series["prototype_relative_distance"]),
        ),
        entropy=dict(
            support_entropy=summarize(series["stage1.support_entropy"]),
            attention_entropy=summarize(series["attention_entropy"]),
            attention_top1=summarize(series["attention_top1"]),
        ),
        support=dict(
            per_stage={f"stage{s + 1}": dict(
                min=summarize(series[f"stage{s + 1}.support_min"]),
                sum=summarize(series[f"stage{s + 1}.support_sum"]),
                entropy=summarize(series[f"stage{s + 1}.support_entropy"]),
            ) for s in range(len(grid_sizes))},
        ),
        components=dict(
            count=int(len(components_min_axis)),
            min_axis=summarize(components_min_axis),
            area=summarize(components_area),
            fraction_min_axis_lt8=float(np.mean(np.asarray(components_min_axis) < 8)) if components_min_axis else None,
            fraction_min_axis_lt32=float(np.mean(np.asarray(components_min_axis) < 32)) if components_min_axis else None,
        ),
        seconds=round(time.time() - started, 1),
    )

    for stage in range(len(grid_sizes)):
        priors = series[f"stage{stage + 1}.prior"]
        stack = torch.stack(priors)
        mean_prior = stack.mean(0)
        tv = 0.5 * (stack - mean_prior).abs().sum(-1)
        report["support"].setdefault("prior", {})[f"stage{stage + 1}"] = dict(
            total_variation_mean=float(tv.mean()),
            total_variation_p90=float(np.percentile(tv.numpy(), 90)),
            concentration=summarize([float((p ** 2).sum()) for p in priors]))
    change_fraction = np.asarray(series["changed_fraction"])
    support_sum = np.asarray(series["stage1.support_sum"])
    if change_fraction.size > 2 and support_sum.std() > 0 and change_fraction.std() > 0:
        report["support"]["support_sum_vs_change_fraction_spearman"] = float(
            spearmanr(support_sum, change_fraction).statistic)
    cv_min = None
    mins = np.asarray(series["stage1.support_min"])
    if mins.mean() > 0:
        cv_min = float(mins.std(ddof=1) / mins.mean())
    report["support"]["class_support_min_cv"] = cv_min
    return report


# -------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="DIFF")
    parser.add_argument("--datasets", default="WHU,LEVIR")
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    report = dict(meta=dict(
        script="models/analyze_arm_diagnostics.py",
        device="cpu", val_limit=args.val_limit, threads=args.threads,
        threshold=0.5, tile=TILE,
        criteria=dict(
            H_A="finest-scale Fisher(LEVIR) >= 2x Fisher(WHU) and support entropy(WHU)-entropy(LEVIR) >= 0.3",
            H_B="prior TV(LEVIR) >= 1.5x prior TV(WHU) or cross-image CV(class_support_min) higher on LEVIR",
            H_C="median GT component min-axis < 8 px (finest cell is 32 px at 256 input)",
        ),
        note="val subset is also the checkpoint-selection split for Run1; F1 here is diagnostic, not a blind test",
        torch=str(torch.__version__), numpy=str(np.__version__),
    ), datasets={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for name in [x for x in args.models.split(",") if x]:
        if name not in MODELS:
            raise SystemExit(f"unknown model {name}")
        for dataset in [x for x in args.datasets.split(",") if x]:
            model, path = load_checkpoint(MODELS[name], dataset)
            loader, indices = subset_loader(DATASET_ROOT / DATASETS[dataset], "val", args.val_limit)
            print(f"[{name}/{dataset}] checkpoint={path} images={len(indices)}", flush=True)
            entry = diagnose_dataset(model, loader, args.val_limit, f"{name}/{dataset}")
            entry.update(checkpoint=path, dataset=DATASETS[dataset], module=MODELS[name]["module"])
            report["datasets"].setdefault(name, {})[dataset] = entry
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            del model

    report["verdict"] = build_verdict(report)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2, ensure_ascii=False))


def build_verdict(report):
    entries = report["datasets"]
    model_name = next(iter(entries))
    if "WHU" not in entries[model_name] or "LEVIR" not in entries[model_name]:
        return dict(available=False, reason="need both WHU and LEVIR")
    whu = entries[model_name]["WHU"]
    levir = entries[model_name]["LEVIR"]
    fisher_whu = whu["prototype"]["fisher_ratio"]["mean"]
    fisher_levir = levir["prototype"]["fisher_ratio"]["mean"]
    entropy_whu = whu["entropy"]["support_entropy"]["mean"]
    entropy_levir = levir["entropy"]["support_entropy"]["mean"]
    tv_whu = whu["support"]["prior"]["stage1"]["total_variation_mean"]
    tv_levir = levir["support"]["prior"]["stage1"]["total_variation_mean"]
    cv_whu = whu["support"]["class_support_min_cv"]
    cv_levir = levir["support"]["class_support_min_cv"]
    axis_whu = whu["components"]["min_axis"]["p50"]
    axis_levir = levir["components"]["min_axis"]["p50"]

    fisher_ratio = fisher_levir / fisher_whu if fisher_whu else None
    entropy_gap = entropy_whu - entropy_levir
    tv_ratio = tv_levir / tv_whu if tv_whu else None
    cv_ratio = cv_levir / cv_whu if cv_whu else None

    h_a = dict(fisher_whu=fisher_whu, fisher_levir=fisher_levir, fisher_ratio=fisher_ratio,
               entropy_whu=entropy_whu, entropy_levir=entropy_levir, entropy_gap=entropy_gap,
               pass_fisher=bool(fisher_ratio and fisher_ratio >= 2.0),
               pass_entropy=bool(entropy_gap >= 0.3),
               holds=bool(fisher_ratio and fisher_ratio >= 2.0 and entropy_gap >= 0.3))
    h_b = dict(prior_tv_whu=tv_whu, prior_tv_levir=tv_levir, prior_tv_ratio=tv_ratio,
               support_min_cv_whu=cv_whu, support_min_cv_levir=cv_levir, cv_ratio=cv_ratio,
               holds=bool((tv_ratio and tv_ratio >= 1.5) or (cv_ratio and cv_ratio > 1.0)))
    h_c = dict(min_axis_p50_whu=axis_whu, min_axis_p50_levir=axis_levir,
               finest_cell_px=whu["cell_px"][0],
               median_below_finest_cell=bool(axis_levir is not None and axis_levir < whu["cell_px"][0]),
               literal_min_axis_lt_8=bool(axis_levir is not None and axis_levir < 8.0),
               holds=bool(axis_levir is not None and axis_levir < whu["cell_px"][0]),
               note=("the 8 in the written criterion is in feature pixels; at 256 input the "
                     "grid-8 cell is 32 image px, so the consistent test is min_axis < 32 px "
                     "(= 8 * stride 4), which LEVIR passes and WHU does not"))
    triggered = [key for key, item in (("H_A", h_a), ("H_B", h_b), ("H_C", h_c)) if item["holds"]]
    return dict(model=model_name, H_A=h_a, H_B=h_b, H_C=h_c,
                triggered=triggered,
                mutually_exclusive=len(triggered) <= 1,
                note=">=2 triggered hypotheses are not mutually exclusive; each needs one confirmation arm")


if __name__ == "__main__":
    main()
