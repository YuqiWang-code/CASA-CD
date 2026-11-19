"""Zero-training ablation: does prototype *selection* or *value content* matter?

Three frozen-weight interventions on the same validation images (CPU, no training):

  A ``uniform_attention``  attention weights replaced by ones/L (values unchanged)
  B ``mean_value``         the 224 value vectors replaced by their mean (attention unchanged)
  C ``no_context``         the gated context residual ``gamma_s * U_s`` set to zero

Each arm is scored at threshold 0.5 and at its own best threshold on the same
images, so the comparison is not confounded by the operating point.

The second half reproduces the suspected cause of value redundancy: when a cell
holds almost no changed pixels the class membership is close to uniform, so the
change and background prototypes are nearly the same convex combination and
their values must be similar.  ``cos(P+, P-)`` and ``cos(V+, V-)`` are therefore
binned by the cell's changed-pixel fraction.
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_geometry_degeneracy import (DATASETS, DATASET_ROOT, MODELS as BASE_MODELS,
                                         load_checkpoint, subset_loader)
from model.layers.hyperbolic_context import HyperbolicContext
from model.layers.pyramid_context import feature_regions

CHECKPOINT_ROOT = Path("/share_datasets/yqwang/checkpoints/CASA-CD")
MODELS = dict(BASE_MODELS)
MODELS["CTRL"] = dict(root=CHECKPOINT_ROOT / "CASA-MinorGoal/Run1/CTRL",
                      module="none", tar="reparam", decoder="reparam",
                      note="Run1 MinorGoal CTRL: no context module (reference)")
THRESHOLDS = np.linspace(0.05, 0.95, 91)


def counts_to_metrics(counts):
    tp, fp, fn, tn = (float(x) for x in counts)
    return dict(F1=100.0 * 2 * tp / max(2 * tp + fp + fn, 1),
                precision=100.0 * tp / max(tp + fp, 1),
                recall=100.0 * tp / max(tp + fn, 1),
                IoU=100.0 * tp / max(tp + fp + fn, 1))


@torch.inference_mode()
def run_pass(model, loader, limit):
    probabilities, targets = [], []
    for index, (image, target) in enumerate(loader):
        if index >= limit:
            break
        probabilities.append(model.predict(image[:, :3], image[:, 3:]).flatten())
        targets.append((target > 0.5).flatten())
    prob = torch.cat(probabilities)
    truth = torch.cat(targets)
    positive = ((prob > 0.5) & truth).sum()
    negative = ((prob > 0.5) & ~truth).sum()
    missed = ((prob <= 0.5) & truth).sum()
    true_negative = ((prob <= 0.5) & ~truth).sum()
    fixed = counts_to_metrics((positive, negative, missed, true_negative))
    best = dict(F1=-1.0)
    for threshold in THRESHOLDS:
        predicted = prob > float(threshold)
        counts = ((predicted & truth).sum(), (predicted & ~truth).sum(),
                  (~predicted & truth).sum(), (~predicted & ~truth).sum())
        score = counts_to_metrics(counts)
        if score["F1"] > best["F1"]:
            best = dict(score, threshold=float(threshold))
    return dict(at_0p5=fixed, best_threshold=best, images=len(targets))


@contextlib.contextmanager
def ablation(context, name):
    """Instance-level monkeypatch of one context module; nothing in models/ changes."""
    if name == "orig" or not isinstance(context, HyperbolicContext):
        yield
        return
    if name == "uniform_attention":
        original = context.attention_logits

        def uniform(query, key, support, query_scale):
            return torch.zeros_like(original(query, key, support, query_scale))

        context.attention_logits = uniform
        try:
            yield
        finally:
            context.attention_logits = original
    elif name == "rolled_attention":
        original = context.attention_logits

        def rolled(query, key, support, query_scale):
            logits = original(query, key, support, query_scale)
            return logits.roll(logits.shape[2] // 3, dims=2)

        context.attention_logits = rolled
        try:
            yield
        finally:
            context.attention_logits = original
    elif name == "mean_value":
        original = context.readout

        def mean_value(feature, probability, values, occupancy, prototype_occupancy):
            mean = values.mean(2, keepdim=True)
            return original(feature, probability, mean.expand_as(values), occupancy,
                            prototype_occupancy)

        context.readout = mean_value
        try:
            yield
        finally:
            context.readout = original
    elif name == "no_context":
        original = context.readout

        def no_context(feature, probability, values, occupancy, prototype_occupancy):
            out = original(feature, probability, values, occupancy, prototype_occupancy)
            return (torch.zeros_like(out[0]),) + tuple(out[1:])

        context.readout = no_context
        try:
            yield
        finally:
            context.readout = original
    else:
        raise ValueError(name)


@torch.inference_mode()
def capture_prototype_pairs(model, loader, limit):
    """Per-scale cos(P+, P-) and cos(V+, V-) against the cell's changed fraction."""
    context = model.context
    if not isinstance(context, HyperbolicContext):
        return None
    import model.layers.hyperbolic_context as module

    store = dict(prototypes=[], values=[], occupancy=[])
    original_pool = module.pool_features
    original_readout = context.readout

    def pool(feature, weights, grid):
        output = original_pool(feature, weights, grid)
        store["prototypes"].append(output.detach().float())
        return output

    def readout(feature, probability, values, occupancy, prototype_occupancy):
        store["values"].append(values.detach().float())
        store["occupancy"].append(occupancy.detach().float())
        return original_readout(feature, probability, values, occupancy, prototype_occupancy)

    module.pool_features = pool
    context.readout = readout
    labels = []
    try:
        for index, (image, target) in enumerate(loader):
            if index >= limit:
                break
            model.predict(image[:, :3], image[:, 3:])
            labels.append((target > 0.5).float())
    finally:
        module.pool_features = original_pool
        context.readout = original_readout

    scales = len(store["occupancy"]) // max(len(labels), 1)
    del scales
    scale_count = len(store["occupancy"]) // max(len(labels), 1)
    token_counts = [store["prototypes"][stage].shape[1] for stage in range(scale_count)]
    offsets = [sum(token_counts[:stage]) for stage in range(scale_count)]
    per_scale = []
    for stage in range(scale_count):
        cos_proto, cos_value, fraction_pred, fraction_truth = [], [], [], []
        for image in range(len(labels)):
            prototype = store["prototypes"][image * scale_count + stage]
            values = store["values"][image * scale_count + stage]
            occupancy = store["occupancy"][image * scale_count + stage]
            tokens, channels = prototype.shape[1], prototype.shape[2]
            cells = tokens // 2
            grid = int(round(cells ** 0.5))
            proto = prototype[0].reshape(cells, 2, channels)
            proto = torch.nn.functional.normalize(proto, dim=-1, eps=1e-6)
            cos_proto.append((proto[:, 0] * proto[:, 1]).sum(-1))
            value = values[0].reshape(values.shape[1], -1, values.shape[-1]).mean(0)
            value = value[offsets[stage]:offsets[stage] + tokens].reshape(cells, 2, -1)
            value = torch.nn.functional.normalize(value, dim=-1, eps=1e-6)
            cos_value.append((value[:, 0] * value[:, 1]).sum(-1))
            fraction_pred.append(
                feature_regions(occupancy, grid).squeeze(-1).mean(-1)[0])
            fraction_truth.append(
                feature_regions(labels[image], grid).squeeze(-1).mean(-1)[0])
        cos_proto = torch.cat(cos_proto).numpy()
        cos_value = torch.cat(cos_value).numpy()
        fraction_pred = torch.cat(fraction_pred).numpy()
        fraction_truth = torch.cat(fraction_truth).numpy()
        edges = [0.0, 0.01, 0.05, 0.20, 1.0]
        bins = []
        for low, high in zip(edges[:-1], edges[1:]):
            member = (fraction_truth >= low) & (fraction_truth < high)
            entry = dict(bin=f"[{low:.2f},{high:.2f})", count=int(member.sum()))
            if int(member.sum()) >= 8:
                entry.update(cos_proto_p50=float(np.percentile(cos_proto[member], 50)),
                             cos_proto_p95=float(np.percentile(cos_proto[member], 95)),
                             cos_value_p50=float(np.percentile(cos_value[member], 50)),
                             cos_value_p95=float(np.percentile(cos_value[member], 95)))
            bins.append(entry)
        per_scale.append(dict(
            scale=stage + 1, grid=grid, cells=int(cells),
            cos_proto=dict(p50=float(np.percentile(cos_proto, 50)),
                           p95=float(np.percentile(cos_proto, 95))),
            cos_value=dict(p50=float(np.percentile(cos_value, 50)),
                           p95=float(np.percentile(cos_value, 95))),
            bins_by_truth_fraction=bins,
            predicted_fraction_p50=float(np.percentile(fraction_pred, 50)),
            truth_fraction_p50=float(np.percentile(fraction_truth, 50))))
    return dict(scales=per_scale, images=len(labels),
                note="cosine of change vs background prototype (P) and value (V), "
                     "binned by the cell's ground-truth changed-pixel fraction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="DIFF,SPP,CTRL")
    parser.add_argument("--datasets", default="WHU")
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    report = dict(meta=dict(script="models/analyze_selectivity_ablation.py", device="cpu",
                            images=args.images, threshold=0.5,
                            thresholds=f"{THRESHOLDS[0]:.2f}..{THRESHOLDS[-1]:.2f}",
                            arms=list(("orig", "uniform_attention", "rolled_attention", "mean_value",
                                      "no_context")),
                            note="frozen weights; instance-level monkeypatching only"),
                  results={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in [x for x in args.models.split(",") if x]:
        spec = MODELS[name]
        for dataset in [x for x in args.datasets.split(",") if x]:
            model, path = load_checkpoint(spec, dataset)
            loader, indices = subset_loader(DATASET_ROOT / DATASETS[dataset], "val", args.images)
            entry = dict(checkpoint=path, module=spec["module"], note=spec["note"],
                         val_images=len(indices), arms={})
            applicable = isinstance(model.context, HyperbolicContext)
            entry["context_branch"] = bool(applicable)
            arms = ("orig", "uniform_attention", "rolled_attention", "mean_value",
                    "no_context") if applicable \
                else ("orig",)
            for arm in arms:
                started = time.time()
                with ablation(model.context, arm):
                    entry["arms"][arm] = run_pass(model, loader, args.images)
                entry["arms"][arm]["seconds"] = round(time.time() - started, 1)
                print(f"[{name}/{dataset}/{arm}] F1@0.5="
                      f"{entry['arms'][arm]['at_0p5']['F1']:.4f} best="
                      f"{entry['arms'][arm]['best_threshold']['F1']:.4f}", flush=True)
            if applicable:
                baseline = entry["arms"]["orig"]
                for arm in arms[1:]:
                    entry["arms"][arm]["delta_f1_at_0p5"] = \
                        entry["arms"][arm]["at_0p5"]["F1"] - baseline["at_0p5"]["F1"]
                    entry["arms"][arm]["delta_f1_best"] = \
                        entry["arms"][arm]["best_threshold"]["F1"] - baseline["best_threshold"]["F1"]
                entry["pairs"] = capture_prototype_pairs(model, loader, args.images)
            report["results"].setdefault(name, {})[dataset] = entry
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            del model
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
