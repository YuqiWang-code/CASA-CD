"""Measure named prototype classes against their retrieved GT content on val."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


def compress_ground_truth(target, weights, grid, feature_shape):
    """Apply captured regional class weights after the image-only forward."""
    from model.layers.pyramid_context import pool_features
    occupancy = F.adaptive_avg_pool2d(target, feature_shape)
    return pool_features(occupancy, weights, grid).squeeze(-1)


def retrieval_probabilities(attention, prototype_truth, feature_shape):
    """Token order is scale, region, then change/background."""
    named_change = attention[..., 0::2].sum(-1).mean(1)
    actual_change = (attention @ prototype_truth[:, None, :, None]).squeeze(-1).mean(1)
    shape = (attention.shape[0], 1, *feature_shape)
    return named_change.reshape(shape), actual_change.reshape(shape)


def verify_pooling_fixture():
    target = torch.tensor([[[[1., 0., 0., 1.], [0., 0., 1., 0.],
                              [1., 1., 0., 0.], [0., 1., 0., 1.]]]])
    weights = torch.tensor([[[[.5, .5, 0., 0.], [0., 0., 1., 0.]],
                              [[0., .25, .75, 0.], [1., 0., 0., 0.]],
                              [[0., 0., 1., 0.], [.5, .5, 0., 0.]],
                              [[0., 0., 0., 1.], [.2, .3, .5, 0.]]]])
    truth = compress_ground_truth(target, weights, 2, (4, 4))
    torch.testing.assert_close(truth, torch.tensor([[.5, 0., 1., 0., 0., 1., 1., 0.]]))
    attention = torch.zeros(1, 1, 2, 8)
    attention[0, 0, 0, 4] = 1
    attention[0, 0, 1, 5] = 1
    named, actual = retrieval_probabilities(attention, truth, (1, 2))
    torch.testing.assert_close(named.flatten(), torch.tensor([1., 0.]))
    torch.testing.assert_close(actual.flatten(), torch.tensor([0., 1.]))
    print("GT regional fixture PASS: class order and nominal/content distinction")


def verify_posterior_fixture():
    pool = dict(prior=torch.tensor([[[.3, .8, .6, .7]]]),
                posterior=torch.tensor([[[.8, .2, .4, .6]]]), grid=1)
    target = torch.tensor([[[[1., 0., 1., .5]]]])
    store = {}
    update_posterior(store, pool, target)
    values = finish_posterior(store)
    assert values["all"]["corrected"] == 2 and values["all"]["harmed"] == 1
    assert values["foreground"]["corrected"] == 1 and values["foreground"]["harmed"] == 1
    assert values["background"]["corrected"] == 1 and values["mixed"]["queries"] == 1
    for key, expected in dict(prior_mae=.525, posterior_mae=.275, prior_brier=.3325, posterior_brier=.1125).items():
        torch.testing.assert_close(torch.tensor(values["all"][key]), torch.tensor(expected))
    print("Posterior GT fixture PASS: MAE/Brier and corrected/harmed groups")


@contextmanager
def capture_retrieval(context):
    """Capture image-derived tensors without exposing labels to model code."""
    implementation = sys.modules[type(context).__module__]
    original_compress = implementation.compress_posterior
    original_logits = context.attention_logits
    captured = dict(enabled=True, pools=[], attention={}, guide=[])

    def compress(feature, guide_logits, query, initial_key, grid, temperature, distance):
        from model.layers.posterior_compression import class_assignments
        output = original_compress(feature, guide_logits, query, initial_key, grid, temperature, distance)
        if captured["enabled"]:
            initial_weights, _ = class_assignments(guide_logits.detach().sigmoid(), grid)
            captured["pools"].append(dict(weights=output[2]["weights"].detach(),
                                           initial_weights=initial_weights,
                                           prior=output[2]["prior"].detach(),
                                           posterior=output[2]["posterior"].detach(),
                                           grid=grid, shape=tuple(feature.shape[-2:])))
        return output

    def logits(query, key, support, query_scale):
        output = original_logits(query, key, support, query_scale)
        if captured["enabled"]:
            captured["attention"][query_scale] = output.detach().softmax(-1)
        return output

    def guide_hook(layer, inputs, output):
        if captured["enabled"]:
            captured["guide"].append(output.detach().sigmoid())

    implementation.compress_posterior = compress
    context.attention_logits = logits
    hook = context.guide.register_forward_hook(guide_hook)
    try:
        yield captured
    finally:
        hook.remove()
        implementation.compress_posterior = original_compress
        context.attention_logits = original_logits


def update_errors(store, prediction, target):
    groups = dict(all=torch.ones_like(target, dtype=torch.bool),
                  foreground=target >= .5, background=target < .5,
                  mixed=(target > 0) & (target < 1))
    probability = prediction.double().clamp(1e-6, 1 - 1e-6)
    truth = target.double()
    difference = prediction.double() - truth
    errors = dict(mae=difference.abs(), mse=difference.square(),
                  bce=-(truth * probability.log() + (1 - truth) * (1 - probability).log()),
                  mean_probability=prediction.double(), mean_gt_occupancy=truth)
    for name, mask in groups.items():
        item = store.setdefault(name, dict(count=0, **{key: 0.0 for key in errors}))
        item["count"] += int(mask.sum())
        for key, values in errors.items():
            item[key] += float(values[mask].sum())


def finish_errors(store):
    return {name: dict(count=item["count"],
                       **{key: value / item["count"] if item["count"] else None
                          for key, value in item.items() if key != "count"})
            for name, item in store.items()}


def update_posterior(store, pool, target):
    """Measure class correction against GT after forward; r never receives GT."""
    from model.layers.pyramid_context import feature_regions
    truth = feature_regions(target, pool["grid"]).squeeze(-1).double()
    prior, posterior = pool["prior"].double(), pool["posterior"].double()
    label = truth >= .5
    prior_correct, posterior_correct = (prior >= .5) == label, (posterior >= .5) == label
    corrected, harmed = ~prior_correct & posterior_correct, prior_correct & ~posterior_correct
    errors = dict(prior_mae=(prior - truth).abs(), posterior_mae=(posterior - truth).abs(),
                  prior_brier=(prior - truth).square(), posterior_brier=(posterior - truth).square())
    groups = dict(all=torch.ones_like(label), foreground=label, background=~label,
                  mixed=(truth > 0) & (truth < 1))
    for group, mask in groups.items():
        item = store.setdefault(group, dict(queries=0, corrected=0, harmed=0, **{name: 0. for name in errors}))
        item["queries"] += int(mask.sum())
        item["corrected"] += int(corrected[mask].sum())
        item["harmed"] += int(harmed[mask].sum())
        for name, values in errors.items():
            item[name] += float(values[mask].sum())


def finish_posterior(store):
    result = {}
    for group, item in store.items():
        result[group] = dict(queries=item["queries"], corrected=item["corrected"], harmed=item["harmed"],
                            net_corrected=item["corrected"] - item["harmed"])
        result[group].update({name: item[name] / item["queries"] if item["queries"] else None
                              for name in ("prior_mae", "posterior_mae", "prior_brier", "posterior_brier")})
    return result


def update_prototypes(store, truth):
    for name, occupancy in (("change", truth[:, 0::2]), ("background", truth[:, 1::2])):
        item = store.setdefault(name, dict(count=0, occupancy_sum=0., purity_sum=0.,
                                          nominal_agreement_sum=0., wrong_majority_count=0))
        nominal_agreement = occupancy if name == "change" else 1 - occupancy
        item["count"] += occupancy.numel()
        item["occupancy_sum"] += float(occupancy.sum())
        item["purity_sum"] += float(torch.maximum(occupancy, 1 - occupancy).sum())
        item["nominal_agreement_sum"] += float(nominal_agreement.sum())
        item["wrong_majority_count"] += int((nominal_agreement < .5).sum())


def finish_prototypes(store):
    result = {}
    for name, values in store.items():
        count = values["count"]
        result[name] = dict(count=count,
                            mean_true_change_occupancy=values["occupancy_sum"] / count,
                            mean_content_purity=values["purity_sum"] / count,
                            mean_nominal_class_agreement=values["nominal_agreement_sum"] / count,
                            wrong_majority_fraction=values["wrong_majority_count"] / count)
    count = sum(item["count"] for item in store.values())
    result["all"] = dict(count=count,
                          mean_content_purity=sum(item["purity_sum"] for item in store.values()) / count,
                          mean_nominal_class_agreement=sum(item["nominal_agreement_sum"]
                                                          for item in store.values()) / count)
    return result


def add_confusion(counts, prediction, truth):
    for index, mask in enumerate((prediction & truth, prediction & ~truth,
                                  ~prediction & truth, ~prediction & ~truth)):
        counts[index] += int(mask.sum())


def scores(counts):
    tp, fp, fn, tn = counts
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, F1=2 * tp / max(2 * tp + fp + fn, 1),
                IoU=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1))


def analyze(model, loader, device):
    stages = [dict(guide={}, nominal_change_mass={}, retrieved_gt_occupancy={},
                    nominal_vs_actual_gap_sum=0., queries=0, prototypes={}, initial_prototypes={}, posterior_correction={}) for _ in range(4)]
    counts = dict(full=[0, 0, 0, 0], context_off=[0, 0, 0, 0])
    corrections = dict(corrected_fn=0, corrected_fp=0, new_fn=0, new_fp=0)
    model.context.collect_diag = False
    with capture_retrieval(model.context) as captured, torch.inference_mode():
        for image, target in loader:
            image, target = image.to(device).float(), target.to(device).float()
            captured["enabled"] = True
            captured["pools"].clear()
            captured["attention"].clear()
            captured["guide"].clear()
            full = model.predict(image[:, :3], image[:, 3:]) > .5
            captured["enabled"] = False
            if len(captured["pools"]) != 4 or len(captured["guide"]) != 4:
                raise RuntimeError("expected four pooled scales and four guide predictions")
            # Labels enter only this post-forward measurement.
            prototype_truth = [compress_ground_truth(target, item["weights"], item["grid"], item["shape"])
                               for item in captured["pools"]]
            initial_truth = [compress_ground_truth(target, item["initial_weights"], item["grid"], item["shape"])
                             for item in captured["pools"]]
            all_prototypes = torch.cat(prototype_truth, dim=1)
            for index, (pool, probability) in enumerate(zip(captured["pools"], captured["guide"])):
                target_scale = F.adaptive_avg_pool2d(target, pool["shape"])
                named, actual = retrieval_probabilities(captured["attention"][index],
                                                        all_prototypes, pool["shape"])
                item = stages[index]
                update_posterior(item["posterior_correction"], pool, target_scale)
                update_errors(item["guide"], probability, target_scale)
                update_errors(item["nominal_change_mass"], named, target_scale)
                update_errors(item["retrieved_gt_occupancy"], actual, target_scale)
                item["nominal_vs_actual_gap_sum"] += float((named - actual).abs().sum())
                item["queries"] += target_scale.numel()
                update_prototypes(item["prototypes"], prototype_truth[index])
                update_prototypes(item["initial_prototypes"], initial_truth[index])

            gamma = model.context.gamma_context.clone()
            try:
                model.context.gamma_context.zero_()
                off = model.predict(image[:, :3], image[:, 3:]) > .5
            finally:
                model.context.gamma_context.copy_(gamma)
            truth = target > .5
            add_confusion(counts["full"], full, truth)
            add_confusion(counts["context_off"], off, truth)
            corrections["corrected_fn"] += int((truth & ~off & full).sum())
            corrections["corrected_fp"] += int((~truth & off & ~full).sum())
            corrections["new_fn"] += int((truth & off & ~full).sum())
            corrections["new_fp"] += int((~truth & ~off & full).sum())

    result_stages = {}
    for index, item in enumerate(stages):
        result_stages[f"stage{index + 1}"] = dict(
            queries=item["queries"], guide=finish_errors(item["guide"]),
            nominal_change_mass=finish_errors(item["nominal_change_mass"]),
            retrieved_gt_occupancy=finish_errors(item["retrieved_gt_occupancy"]),
            nominal_vs_actual_mae=item["nominal_vs_actual_gap_sum"] / item["queries"],
            prototypes=finish_prototypes(item["prototypes"]),
            initial_prototypes=finish_prototypes(item["initial_prototypes"]),
            posterior_correction=finish_posterior(item["posterior_correction"]))
    corrections.update(corrected=corrections["corrected_fn"] + corrections["corrected_fp"],
                       harmed=corrections["new_fn"] + corrections["new_fp"])
    return dict(stages=result_stages, segmentation={name: scores(value) for name, value in counts.items()},
                full_context_relative_to_off=corrections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--module", choices=("hyperbolic", "matched"), required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.models_dir = args.models_dir.resolve()
    sys.path.insert(0, str(args.models_dir))
    from model.casa_net import CASANet
    from train_CASA import GUIDE_OBJECTIVE, make_loader

    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CASANet(module=args.module)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model = model.to("cuda:0").eval()
    source = make_loader(args.dataset_root, "val", 1, 0).dataset
    count = min(args.samples, len(source)) if args.samples else len(source)
    indices = np.linspace(0, len(source) - 1, count, dtype=int).tolist()
    loader = DataLoader(Subset(source, indices), batch_size=1, shuffle=False, num_workers=0)
    report = analyze(model, loader, torch.device("cuda:0"))
    report.update(
        models_dir=str(args.models_dir), checkpoint=str(args.checkpoint), dataset=str(args.dataset_root),
        module=args.module, compression="within_region_geometry_class_posterior",
        guide_objective=GUIDE_OBJECTIVE, readout=model.context.readout_mode,
        split="val", sample_indices=indices, sample_names=[source.file_list[index] for index in indices],
        samples=count, batch_size=1, physical_gpu=1, threshold=.5,
        definitions=dict(
            posterior_correction="Prior p versus corrected r, grouped by GT occupancy; corrected/harmed threshold both prediction and area-averaged target at >=0.5; mixed overlaps foreground/background",
            posterior_brier="Mean squared probability error against area-averaged GT occupancy",
            initial_prototypes="Actual GT content under squared guide anchor weights; prototypes uses final first-order r weights",
            query_target="area-averaged GT at each feature resolution",
            groups="foreground: occupancy >= 0.5; background: occupancy < 0.5; mixed: 0 < occupancy < 1",
            guide="image-derived guide sigmoid at the query position",
            nominal_change_mass="attention summed over even-indexed nominal change prototypes, averaged over heads",
            retrieved_gt_occupancy="same attention applied to actual P1 posterior-weighted GT, averaged over heads",
            prototype_content_purity="max(actual change occupancy, 1 - occupancy)",
            prototype_nominal_agreement="actual change occupancy for change tokens; 1 - occupancy for background tokens",
            context_off="zero global gamma only; local branch remains enabled",
            label_use="GT enters post-forward analysis only, never model forward",
            interpretation="frozen-weight val diagnostics; not retrained ablations"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(module=args.module, compression="within_region_geometry_class_posterior",
        guide_objective=GUIDE_OBJECTIVE, readout=model.context.readout_mode,
                          segmentation=report["segmentation"],
                          corrections=report["full_context_relative_to_off"],
                          stages={name: {key: values[key]["all"]["mae"]
                                         for key in ("guide", "nominal_change_mass", "retrieved_gt_occupancy")}
                                  for name, values in report["stages"].items()})), flush=True)


if __name__ == "__main__":
    main()
