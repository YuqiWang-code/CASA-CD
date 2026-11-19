"""Zero-training probes: is the Lorentz geometry in our trained weights degenerate?

P1 (geometry degeneracy) captures the dense queries, the joint key bank and the
actual attention logits of a frozen checkpoint, then compares the squared
geodesic distance ``acosh(1 + z)**2`` with the spatial-only Euclidean distance
``||q - k||**2`` on the same pairs:

  * rank correlation (Spearman) between the two distance matrices;
  * relative deviation ``|d_geo**2 - d_euc**2| / d_geo**2``;
  * the intrinsic radii ``asinh(||z||)`` and the Lorentz norms ``||z||``;
  * the induced attention change (total variation and top-1 agreement).

P2 (inference-time geometry swap) keeps the weights fixed and replaces
``squared_distance`` by the z-only Euclidean distance used by
``MatchedEuclideanContext``. Because the softmax scale was learned in geodesic
units, two versions are reported: the raw swap (alpha = 1) and a swap whose one
free scalar ``alpha`` is fitted on the validation split by maximising val F1
at threshold 0.5 (a single scalar, no network training).

Beyond the pooled statistics, three probes separate "small radius" from
"geometry switched off": P1a shell spread (``asinh(||z||)`` percentiles and CV
for Q and K plus the Q-K radius gap), P1b per-query rank agreement over the full
224-key bank (Spearman, Kendall tau, arg-min prototype change, top-5 overlap),
P1c the affine re-scaling ``d_geo^2 ~ a*d_euc^2 + b`` with its R^2 and residual,
and P1d the learned temperature tau together with the attention entropy / top-1
mass. Squared geodesic distances are evaluated as
``4*asinh(sqrt(z/2))**2`` with ``z = cosh(d) - 1`` to avoid the cancellation of
``acosh(1 + z)`` near zero; the deviation from the model's own
``acosh1p_squared`` is recorded in ``p1.numeric``.

Everything runs on CPU so a training queue keeps the GPUs.
"""
import argparse
import json
import math
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from model.casa_net import CASANet
from model.layers.hyperbolic_context import MatchedEuclideanContext
from model.layers.lorentz_ops import pairwise_squared_distance
from train_CASA import make_loader


CHECKPOINT_ROOT = Path("/share_datasets/yqwang/checkpoints/CASA-CD")
DATASETS = {"WHU": "WHU-CD-256", "SYSU": "SYSU-CD-256", "LEVIR": "LEVIR-CD-256"}
DATASET_ROOT = Path("/share_datasets/CD")

MODELS = {
    # Run1 MinorGoal: CTRL/SPP have module="none" (no geometric context at all);
    # DIFF is hyperbolic retrieval over temporal-difference prototypes.
    "DIFF": dict(root=CHECKPOINT_ROOT / "CASA-MinorGoal/Run1/DIFF",
                 module="difference", tar="spp", decoder="reparam",
                 note="Run1 MinorGoal DIFF: hyperbolic difference-context retrieval"),
    "SPP": dict(root=CHECKPOINT_ROOT / "CASA-MinorGoal/Run1/SPP",
                module="none", tar="spp", decoder="reparam",
                note="Run1 MinorGoal SPP: no context module, so no Lorentz geometry"),
    # Run14 Pyramid: H/E are a paired hyperbolic / Euclidean-twin training.
    "H": dict(root=CHECKPOINT_ROOT / "CASA-Pyramid/Run14/H_S2333",
              module="hyperbolic", tar="reparam", decoder="reparam",
              note="Run14 hyperbolic context on fused content"),
    "E": dict(root=CHECKPOINT_ROOT / "CASA-Pyramid/Run14/E_S2333",
              module="matched", tar="reparam", decoder="reparam",
              note="Run14 Euclidean twin of H (same seed/protocol)"),
}


def subset_loader(root, split, limit, workers=0):
    loader = make_loader(root, split, 1, workers)
    data = loader.dataset
    count = len(data) if not limit else min(limit, len(data))
    indices = np.linspace(0, len(data) - 1, count, dtype=int).tolist()
    return DataLoader(Subset(data, indices), batch_size=1, shuffle=False), indices


def load_checkpoint(spec, dataset):
    path = Path(spec["root"]) / DATASETS[dataset] / "best_model.pth"
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = CASANet(None, module=spec["module"], tar_variant=spec["tar"],
                    decoder_variant=spec["decoder"])
    model.load_state_dict(state)
    return model.eval(), str(path)


@contextmanager
def euclidean_metric(context, alpha=1.0):
    """Swap the geodesic distance for ``alpha * ||q - k||**2`` without touching weights.

    The attention logits are ``-d**2 / (2 tau**2) + shared``; multiplying the
    Euclidean distance by ``2 tau**2 alpha`` therefore yields ``-alpha * d_euc**2``,
    i.e. exactly the raw swap for ``alpha = 1``.
    """
    original = context.squared_distance

    def distance(query, key):
        tau2 = F.softplus(context.raw_temperature).view(1, -1, 1, 1).square()
        return 2.0 * tau2 * alpha * MatchedEuclideanContext.squared_distance(query, key)

    context.squared_distance = distance
    try:
        yield
    finally:
        context.squared_distance = original


@contextmanager
def capture_geometry(context):
    """Record every lifted query, the joint key bank and the attention logits."""
    store = {"queries": [], "key": None, "prototype": None, "logits": []}
    original_queries, original_keys = context.project_queries, context.project_keys
    original_logits = context.attention_logits

    def project_queries(feature):
        point = original_queries(feature)
        store["queries"].append(point.detach().float())
        return point

    def project_keys(prototype):
        point, occupancy = original_keys(prototype)
        store["key"] = point.detach().float()
        store["prototype"] = prototype.detach().float()
        return point, occupancy

    def attention_logits(query, key, support, query_scale):
        logits = original_logits(query, key, support, query_scale)
        store["logits"].append(logits.detach().float())
        return logits

    context.project_queries = project_queries
    context.project_keys = project_keys
    context.attention_logits = attention_logits
    try:
        yield store
    finally:
        context.project_queries = original_queries
        context.project_keys = original_keys
        context.attention_logits = original_logits


def average_ranks(values):
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    return (cumulative - (counts - 1) / 2.0)[inverse]


def spearman(a, b):
    if a.size < 3:
        return float("nan")
    ra, rb = average_ranks(a), average_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / denominator) if denominator else float("nan")


def percentiles(values, points=(50, 90, 99)):
    if values.size == 0:
        return {f"p{p}": None for p in points}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(array, p)) for p in points}


def summarize(values, points=(50, 90, 99)):
    result = percentiles(values, points)
    array = np.asarray(values, dtype=np.float64)
    result["max"] = float(array.max()) if array.size else None
    result["mean"] = float(array.mean()) if array.size else None
    return result


def pair_sample(geo, euc, budget=40000):
    """Deterministic stride subsample of the (queries x keys) pair grid."""
    geo = geo.reshape(-1).numpy()
    euc = euc.reshape(-1).numpy()
    stride = max(1, geo.size // budget)
    return geo[::stride], euc[::stride]



def radius_stats(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    mean = float(array.mean())
    return dict(p5=float(np.percentile(array, 5)), p50=float(np.percentile(array, 50)),
                p95=float(np.percentile(array, 95)), mean=mean,
                std=float(array.std()), cv=float(array.std() / max(abs(mean), 1e-12)))


def kendall_tau(a, b):
    n = a.size
    if n < 3:
        return float("nan")
    difference_a = a[:, None] - a[None, :]
    difference_b = b[:, None] - b[None, :]
    upper = np.triu_indices(n, 1)
    product = difference_a[upper] * difference_b[upper]
    concordant = float(np.sum(product > 0))
    discordant = float(np.sum(product < 0))
    return (concordant - discordant) / max(concordant + discordant, 1.0)


def affine_fit(x, y):
    """Least-squares y = a * x + b with R^2 and residual summaries."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - (a * x + b)
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(a=float(a), b=float(b), r2=1.0 - ss_res / max(ss_tot, 1e-30),
                residual=summarize(np.abs(residual)),
                residual_over_y=summarize(np.abs(residual) / np.maximum(np.abs(y), 1e-12)))


def pairwise_terms(query, key):
    """Stable Lorentz terms for the squared distances.

    ``z = cosh(d) - 1`` from the same chord/time-difference identity the model
    uses; ``d_geo^2 = 4 * asinh(sqrt(z / 2))^2`` avoids the catastrophic
    cancellation of ``acosh(1 + z)`` near ``z = 0``. ``d_euc^2`` is the z-only
    (spatial) squared Euclidean distance, i.e. what ``MatchedEuclideanContext``
    uses.
    """
    q_space, k_space = query[..., 1:], key[..., 1:]
    inner = q_space @ k_space.transpose(-1, -2)
    # Exact Lorentz inner product: the stored time coordinate is sqrt(1 + ||s||^2),
    # so cosh(d) = t_q * t_k - q_s . k_s without any chord/time cancellation.
    cosh = query[..., :1] * key[..., :1].transpose(-1, -2) - inner
    z = (cosh - 1).clamp_min(0)
    d_geo2 = 4 * torch.asinh(torch.sqrt(z / 2)).square()
    d_euc2 = (q_space.square().sum(-1, keepdim=True) + k_space.square().sum(-1).unsqueeze(-2)
              - 2 * inner).clamp_min(0)
    return z, d_geo2, d_euc2


def attention_shape(logits):
    probability = logits.softmax(-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    return dict(entropy=float(entropy.mean()),
                max_entropy=float(math.log(logits.shape[-1])),
                top1_weight=float(probability.max(-1).values.mean()),
                uniform_weight=float(logits.shape[-1] ** -1),
                logit_range=float((logits.max(-1).values - logits.min(-1).values).mean()),
                logit_rms=float(logits.square().mean().sqrt()))


def exact_terms(query, key):
    """Float64 exact ``z = cosh(d) - 1`` and the float32 cancellation floor.

    The model builds ``z`` from ``0.5 * (chord^2 - dt^2)``, which is algebraically
    exact but subtracts two O(||s||^2) terms. With float32 the absolute noise
    floor of that subtraction is about ``2 * r^2 * 1.2e-7``.
    """
    q_space = query[..., 1:].double()
    k_space = key[..., 1:].double()
    q_time = query[..., :1].double()
    k_time = key[..., :1].double()
    cosh = q_time * k_time.transpose(-1, -2) - q_space @ k_space.transpose(-1, -2)
    z = cosh - 1
    radius_squared = 0.5 * (q_space.square().sum(-1, keepdim=True)
                            + k_space.square().sum(-1).unsqueeze(-2))
    floor = (2 * radius_squared * 1.2e-7).float()
    return z, floor


def stable_pairwise_squared_distance(query, key):
    """A/B variant: ``z = 2 sinh^2(dr/2) + sinh r_q sinh r_k (1 - cos theta)`` in float64."""
    q_space = query[..., 1:].double()
    k_space = key[..., 1:].double()
    q_norm = q_space.norm(dim=-1)
    k_norm = k_space.norm(dim=-1)
    unit_q = q_space / q_norm.clamp_min(1e-12).unsqueeze(-1)
    unit_k = k_space / k_norm.clamp_min(1e-12).unsqueeze(-1)
    one_minus_cos = 0.5 * (unit_q.unsqueeze(-2) - unit_k.unsqueeze(-3)).square().sum(-1)
    radius_q = torch.asinh(q_norm)
    radius_k = torch.asinh(k_norm)
    z = (2 * torch.sinh(0.5 * (radius_q.unsqueeze(-1) - radius_k.unsqueeze(-2))).square()
         + (q_norm.unsqueeze(-1) * k_norm.unsqueeze(-2)) * one_minus_cos)
    squared = 4 * torch.asinh(torch.sqrt(z.clamp_min(0) / 2)).square()
    return squared.float()


@contextmanager
def stable_metric(context):
    original = context.squared_distance

    def distance(query, key):
        return stable_pairwise_squared_distance(query, key)

    context.squared_distance = distance
    try:
        yield
    finally:
        context.squared_distance = original


def probe_geometry(model, loader, images, query_sample=32):
    """P1/P1a-P1d: radii, rank agreement, affine re-scaling and temperature."""
    context = model.context
    if not hasattr(context, "project_queries"):
        return dict(available=False, reason="context module has no Lorentz geometry")
    temperature = F.softplus(context.raw_temperature).detach().float()
    q_norms, k_norms, radii, k_radii = [], [], [], []
    radius_gap = []
    stage_records = []
    geo_values, euc_values = [], []
    alpha_numerator, alpha_denominator = 0.0, 0.0
    rho_rows, tau_rows, argmin_change, top5_overlap = [], [], [], []
    stable_gap = 0.0
    unresolved, total_pairs, exact_rows = 0, 0, []
    stable_argmin_change = []
    attention = []
    prototype_norms = []
    prototype_by_scale = [[] for _ in context.tokens_per_scale]
    key_radius_by_scale = [[] for _ in context.tokens_per_scale]
    query_radius_by_stage = [[] for _ in context.tokens_per_scale]
    generator = torch.Generator().manual_seed(0)
    with torch.inference_mode():
        for index, (image, _) in enumerate(loader):
            if index >= images:
                break
            with capture_geometry(context) as store:
                model.predict(image[:, :3], image[:, 3:])
            key, queries, logits = store["key"], store["queries"], store["logits"]
            assert key is not None and len(queries) == len(logits)
            q_norms.append(torch.cat([q[..., 1:].norm(dim=-1).reshape(-1) for q in queries]).numpy())
            radii.append(torch.cat([torch.asinh(q[..., 1:].norm(dim=-1)).reshape(-1)
                                    for q in queries]).numpy())
            k_norms.append(key[..., 1:].norm(dim=-1).reshape(-1).numpy())
            k_radii.append(torch.asinh(key[..., 1:].norm(dim=-1)).reshape(-1).numpy())
            prototype = store.get("prototype")
            if prototype is not None:
                token_norms = prototype.norm(dim=-1).reshape(-1).numpy()
                prototype_norms.append(token_norms)
                spatial = key[..., 1:].norm(dim=-1).reshape(-1).numpy()
                offset = 0
                for scale, count in enumerate(context.tokens_per_scale):
                    prototype_by_scale[scale].append(token_norms[offset:offset + count])
                    key_radius_by_scale[scale].append(np.arcsinh(spatial[offset:offset + count]))
                    offset += count
            for stage, item in enumerate(queries):
                if stage < len(query_radius_by_stage):
                    query_radius_by_stage[stage].append(
                        np.arcsinh(item[..., 1:].norm(dim=-1).reshape(-1).numpy()))
            for stage, (query, stage_logits) in enumerate(zip(queries, logits)):
                query = query.flatten(2, 3)  # logits are built from flattened query maps
                z, geo, euc = pairwise_terms(query, key)
                model_geo = pairwise_squared_distance(query, key)
                stable_gap = max(stable_gap, float(((model_geo - geo).abs()
                                                    / geo.clamp_min(1e-6)).max()))
                tau2 = F.softplus(context.raw_temperature).view(1, -1, 1, 1).square()
                euc_logits = stage_logits + (geo - euc) / (2 * tau2)
                p_geo = stage_logits.softmax(-1)
                p_euc = euc_logits.softmax(-1)
                tv = 0.5 * (p_geo - p_euc).abs().sum(-1)
                top1 = (p_geo.argmax(-1) == p_euc.argmax(-1)).float()
                attention.append(attention_shape(stage_logits))
                # P1b: per-query rank agreement over the full 224-key bank.
                count = int(query.shape[2])
                picked = torch.randperm(count, generator=generator)[:min(query_sample, count)]
                geo_rows = geo[0, 0, picked].numpy()
                euc_rows = euc[0, 0, picked].numpy()
                for geo_row, euc_row in zip(geo_rows, euc_rows):
                    rho_rows.append(spearman(geo_row, euc_row))
                    tau_rows.append(kendall_tau(geo_row, euc_row))
                    argmin_change.append(float(geo_row.argmin() != euc_row.argmin()))
                    order_geo = np.argsort(geo_row)[:5]
                    order_euc = np.argsort(euc_row)[:5]
                    top5_overlap.append(len(set(order_geo) & set(order_euc)) / 5.0)
                z_true, floor = exact_terms(query, key)
                unresolved += int((z_true < floor).sum())
                total_pairs += int(z_true.numel())
                exact_rows.append(pair_sample(z_true.float(), floor)[0])
                stable = stable_pairwise_squared_distance(query, key)
                stable_argmin_change.append(
                    float((stable[0, 0].argmin(-1) != geo[0, 0].argmin(-1)).float().mean()))
                geo_sample, euc_sample = pair_sample(geo, euc)
                positive = geo_sample > 0
                geo_values.append(geo_sample[positive])
                euc_values.append(euc_sample[positive])
                alpha_numerator += float((geo_sample * euc_sample).sum())
                alpha_denominator += float((euc_sample * euc_sample).sum())
                deviation = np.abs(geo_sample[positive] - euc_sample[positive]) / geo_sample[positive]
                stage_records.append(dict(
                    stage=stage + 1, queries=int(query.shape[2]), keys=int(key.shape[2]),
                    spearman=spearman(geo_sample, euc_sample),
                    relative_deviation=summarize(deviation),
                    ratio_geo_over_euc=summarize(geo_sample[positive] / euc_sample[positive]),
                    attention_total_variation=float(tv.mean()),
                    attention_top1_agreement=float(top1.mean()),
                    logit_shift_rms=float(((geo - euc) / (2 * tau2)).square().mean().sqrt()),
                    logit_rms=float(stage_logits.square().mean().sqrt()),
                ))
            # P1a: the Q-K radius gap inside each image.
            stage_gap = [abs(torch.asinh(q[..., 1:].norm(dim=-1)).reshape(-1).unsqueeze(-1)
                             - torch.asinh(key[..., 1:].norm(dim=-1)).reshape(-1).unsqueeze(0))
                         for q in queries]
            radius_gap.append(torch.cat([g.reshape(-1) for g in stage_gap]).numpy())
    geo_all = np.concatenate(geo_values)
    euc_all = np.concatenate(euc_values)
    deviation_all = np.abs(geo_all - euc_all) / geo_all
    alpha_ls = alpha_numerator / alpha_denominator if alpha_denominator else float("nan")
    fit = affine_fit(euc_all, geo_all)
    rho_rows = np.asarray(rho_rows, dtype=np.float64)
    tau_rows = np.asarray(tau_rows, dtype=np.float64)
    argmin_change = np.asarray(argmin_change, dtype=np.float64)
    top5_overlap = np.asarray(top5_overlap, dtype=np.float64)
    attention_mean = dict((key, float(np.mean([a[key] for a in attention])))
                          for key in attention[0]) if attention else {}
    return dict(
        available=True,
        images=min(images, len(loader)),
        numeric=dict(form="d_geo^2 = 4*asinh(sqrt(z/2))^2 with z = cosh(d)-1",
                     model_form="acosh1p_squared(z)",
                     max_relative_difference_vs_model=stable_gap),
        temperature=[float(x) for x in temperature.reshape(-1)],
        query_norm=summarize(np.concatenate(q_norms)),
        key_norm=summarize(np.concatenate(k_norms)),
        query_radius=summarize(np.concatenate(radii)),
        key_radius=summarize(np.concatenate(k_radii)),
        shells=dict(query_radius=radius_stats(np.concatenate(radii)),
                    key_radius=radius_stats(np.concatenate(k_radii)),
                    radius_gap=radius_stats(np.concatenate(radius_gap))),
        prototype_norm=dict(
            tokens=int(sum(context.tokens_per_scale)),
            all=(dict(radius_stats(np.concatenate(prototype_norms)),
                      max=float(np.concatenate(prototype_norms).max()))
                 if prototype_norms else None),
            per_scale=[dict(
                scale=index + 1,
                tokens=int(context.tokens_per_scale[index]),
                prototype=(dict(radius_stats(np.concatenate(rows)),
                                max=float(np.concatenate(rows).max())) if rows else None),
                key_radius=(radius_stats(np.concatenate(key_radius_by_scale[index]))
                            if key_radius_by_scale[index] else None),
                query_radius=(radius_stats(np.concatenate(query_radius_by_stage[index]))
                              if query_radius_by_stage[index] else None),
            ) for index, rows in enumerate(prototype_by_scale)]),
        ranking=dict(spearman_pairs=spearman(geo_all, euc_all),
                     top1_logit_agreement=float(np.mean([s["attention_top1_agreement"]
                                                         for s in stage_records])),
                     spearman_query_mean=float(rho_rows.mean()),
                     spearman_query_p5=float(np.percentile(rho_rows, 5)),
                     spearman_query_p50=float(np.percentile(rho_rows, 50)),
                     kendall_query_mean=float(tau_rows.mean()),
                     kendall_query_p5=float(np.percentile(tau_rows, 5)),
                     argmin_change_fraction=float(argmin_change.mean()),
                     top5_overlap_mean=float(top5_overlap.mean()),
                     queries=len(rho_rows)),
        affine=fit,
        attention=attention_mean,
        pairs=dict(geo=summarize(geo_all), euc=summarize(euc_all),
                   spearman=spearman(geo_all, euc_all),
                   relative_deviation=summarize(deviation_all),
                   fraction_deviation_gt_1e_2=float((deviation_all > 1e-2).mean()),
                   fraction_deviation_gt_1e_1=float((deviation_all > 1e-1).mean()),
                   ratio_geo_over_euc=summarize(geo_all / euc_all)),
        alpha_least_squares=float(alpha_ls),
        numerics=dict(
            radius_scale=dict(
                query=dict(mean=float(np.concatenate(q_norms).mean()),
                           p95=float(np.percentile(np.concatenate(q_norms), 95)),
                           max=float(np.concatenate(q_norms).max())),
                key=dict(mean=float(np.concatenate(k_norms).mean()),
                         p95=float(np.percentile(np.concatenate(k_norms), 95)),
                         max=float(np.concatenate(k_norms).max()))),
            z_true_p50=float(np.percentile(np.concatenate(exact_rows), 50)),
            radius_p50=float(np.median(np.concatenate(q_norms + k_norms))),
            noise_floor=float(2 * float(np.median(np.concatenate(q_norms + k_norms))) ** 2
                              * 1.2e-7),
            noise_floor_ratio=float(
                2 * (float(np.median(np.concatenate(q_norms + k_norms))) ** 2) * 1.2e-7
                / max(float(np.percentile(np.concatenate(exact_rows), 50)), 1e-30)),
            unresolved_fraction=float(unresolved / max(total_pairs, 1)),
            stable_vs_model_argmin_change=float(np.mean(stable_argmin_change))
            if stable_argmin_change else None,
            pairs=total_pairs),
        per_stage=stage_records,
    )


def probe_d3(model, loader, images):
    """D3: is the 224-token value bank redundant, and how strong is the context branch?"""
    context = model.context
    if not hasattr(context, "readout"):
        return dict(available=False, reason="context module has no readout")
    values_by_image = []
    cosine_rows = []
    gain_by_scale = {index: [] for index in range(len(context.gamma_context))}
    gate_rows = []
    original_readout = context.readout
    counter = {"calls": 0}

    def readout(feature, probability, values, occupancy, prototype_occupancy):
        result = original_readout(feature, probability, values, occupancy,
                                  prototype_occupancy)
        update, gate = result[0], result[1]
        index = counter["calls"] % len(gain_by_scale)
        counter["calls"] += 1
        gamma = context.gamma_context[index][None, :, None, None]
        gain_by_scale[index].append(float((gamma * update).norm() / feature.norm().clamp_min(1e-8)))
        gate_rows.append(float(gate.mean()))
        flat = values.detach().float()[0]
        flat = flat.reshape(-1, flat.shape[-1])
        values_by_image.append(flat)
        pairs = flat.reshape(-1, 2, flat.shape[-1])
        normalised = F.normalize(pairs, dim=-1, eps=1e-6)
        cosine_rows.append((normalised[:, 0] * normalised[:, 1]).sum(-1).numpy())
        return result

    context.readout = readout
    try:
        with torch.inference_mode():
            for index, (image, _) in enumerate(loader):
                if index >= images:
                    break
                model.predict(image[:, :3], image[:, 3:])
    finally:
        context.readout = original_readout

    cosine = np.concatenate(cosine_rows) if cosine_rows else np.zeros(1)
    bank = torch.stack(values_by_image).mean(0).numpy().astype(np.float64)
    _, singular, _ = np.linalg.svd(bank - bank.mean(0, keepdims=True), full_matrices=False)
    energy = np.cumsum(singular ** 2) / max(float((singular ** 2).sum()), 1e-30)
    rank90 = int(np.searchsorted(energy, 0.90) + 1)
    rank99 = int(np.searchsorted(energy, 0.99) + 1)
    gain = dict()
    for index, rows in gain_by_scale.items():
        if rows:
            array = np.asarray(rows)
            gain[f"scale{index + 1}"] = dict(mean=float(array.mean()),
                                             p50=float(np.percentile(array, 50)),
                                             p95=float(np.percentile(array, 95)))
    return dict(
        available=True, images=min(images, len(loader)),
        v_redundancy=dict(
            tokens=int(bank.shape[0]), dim=int(bank.shape[1]),
            cosine=dict(p5=float(np.percentile(cosine, 5)),
                        p50=float(np.percentile(cosine, 50)),
                        p95=float(np.percentile(cosine, 95)),
                        mean=float(cosine.mean())),
            effective_rank=dict(energy90=rank90, energy99=rank99,
                                singular_top8=[float(x) for x in singular[:8]]),
            independent_inputs=int(len(values_by_image)),
            criterion="redundant if cosine p50 > 0.9 or effective rank(90%) < 32"),
        context_gain=dict(per_scale=gain, gate_mean=float(np.mean(gate_rows)),
                          criterion="decorative if ||gamma*U||/||T|| < 5%"),
        note="V bank and context-branch strength, frozen weights, CPU eval",
    )


def build_d0(report):
    """D0 gate: radius CV (design knob) and top-1 prototype flip rate (verdict)."""
    gate = {}
    for name, datasets in report["p1"].items():
        for dataset, result in datasets.items():
            key = f"{name}/{dataset}"
            if not result.get("available"):
                gate[key] = dict(applicable=False, reason=result.get("reason"))
                continue
            shells, ranking = result["shells"], result["ranking"]
            flip = ranking["argmin_change_fraction"]
            scale_rows = (result.get("prototype_norm") or {}).get("per_scale", [])
            scale_rows = [row for row in scale_rows
                          if row.get("key_radius") and row.get("prototype")]
            key_p50 = [row["key_radius"]["p50"] for row in scale_rows]
            gate[key] = dict(
                radius_cv=dict(query=shells["query_radius"]["cv"], key=shells["key_radius"]["cv"]),
                radius_p50=dict(query=shells["query_radius"]["p50"], key=shells["key_radius"]["p50"]),
                flip_rate=flip,
                flip_rate_attention=1.0 - ranking["top1_logit_agreement"],
                radius_scale=result["numerics"]["radius_scale"],
                noise_floor=result["numerics"]["noise_floor"],
                noise_floor_ratio=result["numerics"]["noise_floor_ratio"],
                unresolved_frac=result["numerics"]["unresolved_fraction"],
                prototype_norm=(result.get("prototype_norm") or {}).get("all"),
                key_radius_by_scale=[dict(
                    scale=row["scale"], tokens=row["tokens"],
                    prototype_mean=row["prototype"]["mean"],
                    prototype_p50=row["prototype"]["p50"],
                    prototype_max=row["prototype"]["max"],
                    key_radius_p50=row["key_radius"]["p50"],
                    key_radius_cv=row["key_radius"]["cv"],
                    query_radius_p50=(row["query_radius"] or {}).get("p50"))
                    for row in scale_rows],
                key_radius_monotonic_in_scale=(
                    all(later >= earlier for earlier, later in zip(key_p50, key_p50[1:]))
                    if len(key_p50) > 1 else None),
                kendall_tau=ranking["kendall_query_mean"],
                verdict=("geometry off: block hyperbolic mechanisms until radii separate"
                         if flip < 0.10 else
                         "geometry active: allowed into the training queue" if flip >= 0.25 else
                         "gray zone: M2 only"),
                queries=ranking["queries"])
    return gate


def metrics(counts):
    tp, fp, fn, tn = counts
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                F1=100.0 * 2 * tp / max(2 * tp + fp + fn, 1),
                precision=100.0 * tp / max(tp + fp, 1),
                recall=100.0 * tp / max(tp + fn, 1),
                IoU=100.0 * tp / max(tp + fp + fn, 1),
                predicted_positive_fraction=100.0 * (tp + fp) / max(tp + fp + fn + tn, 1))


@torch.inference_mode()
def evaluate(model, loader, alpha=None):
    counts = [0, 0, 0, 0]
    context = euclidean_metric(model.context, alpha) if alpha is not None else nullcontext()
    with context:
        for image, target in loader:
            prediction = model.predict(image[:, :3], image[:, 3:]) > 0.5
            truth = target > 0.5
            counts[0] += int((prediction & truth).sum())
            counts[1] += int((prediction & ~truth).sum())
            counts[2] += int((~prediction & truth).sum())
            counts[3] += int((~prediction & ~truth).sum())
    return metrics(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="DIFF,H,E,SPP")
    parser.add_argument("--datasets", default="WHU,SYSU")
    parser.add_argument("--p1-images", type=int, default=8)
    parser.add_argument("--val-limit", type=int, default=256)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--alpha-grid", default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--numeric-ab", action="store_true",
                        help="A/B the model distance against the stable float64 formula on val")
    parser.add_argument("--d3", action="store_true",
                        help="measure value-bank redundancy and context-branch gain (no F1 pass)")
    parser.add_argument("--d3-images", type=int, default=200)
    parser.add_argument("--ab-limit", type=int, default=200)
    parser.add_argument("--skip-p2", action="store_true",
                        help="only run the P1 geometry probes (no F1 passes)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    grid = [float(x) for x in args.alpha_grid.split(",")]
    report = dict(meta=dict(
        script="models/analyze_geometry_degeneracy.py",
        threshold=0.5, device="cpu", p1_images=args.p1_images, val_limit=args.val_limit,
        test_limit=args.test_limit, alpha_grid=grid,
        criterion="degenerate if p99(||z||) < 0.3 or spearman > 0.99 or median relative deviation < 1e-2",
        torch=str(torch.__version__),
        models={name: {k: (str(v) if isinstance(v, Path) else v) for k, v in MODELS[name].items()}
                for name in args.models.split(",") if name in MODELS},
    ), p1={}, p2={}, d0={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name in [x for x in args.models.split(",") if x]:
        if name not in MODELS:
            raise SystemExit(f"unknown model {name}")
        spec = MODELS[name]
        for dataset in [x for x in args.datasets.split(",") if x]:
            started = time.time()
            model, path = load_checkpoint(spec, dataset)
            val_loader, val_indices = subset_loader(DATASET_ROOT / DATASETS[dataset], "val",
                                                   args.val_limit)
            test_loader, test_indices = subset_loader(DATASET_ROOT / DATASETS[dataset], "test",
                                                      args.test_limit)
            print(f"[{name}/{dataset}] loaded {path} val={len(val_indices)} "
                  f"test={len(test_indices)}", flush=True)
            entry = dict(checkpoint=path, module=spec["module"], tar_variant=spec["tar"],
                         dataset=DATASETS[dataset], note=spec["note"],
                         val_indices=val_indices if len(val_indices) < 32 else
                         [val_indices[0], val_indices[-1], len(val_indices)],
                         test_indices=test_indices if len(test_indices) < 32 else
                         [test_indices[0], test_indices[-1], len(test_indices)])
            entry["p1"] = probe_geometry(model, val_loader, args.p1_images)
            report["p1"].setdefault(name, {})[dataset] = entry["p1"]
            if args.d3:
                entry["d3"] = probe_d3(model, val_loader, args.d3_images)
                report.setdefault("d3", {}).setdefault(name, {})[dataset] = entry["d3"]
                print(f"[{name}/{dataset}] d3 cosine_p50="
                      f"{entry['d3']['v_redundancy']['cosine']['p50']:.4f} "
                      f"rank90={entry['d3']['v_redundancy']['effective_rank']['energy90']} "
                      f"gate_mean={entry['d3']['context_gain']['gate_mean']:.4f}", flush=True)
            ranking = entry["p1"].get("ranking")
            entry["p2"] = dict(ranking=None if ranking is None else dict(
                spearman_query_mean=ranking["spearman_query_mean"],
                kendall_query_mean=ranking["kendall_query_mean"],
                argmin_change_fraction=ranking["argmin_change_fraction"],
                top5_overlap_mean=ranking["top5_overlap_mean"],
                note="rank agreement between geodesic and z-only Euclidean distances over full 224-key banks"))
            if args.numeric_ab and entry["p1"].get("available"):
                ab_loader, ab_indices = subset_loader(DATASET_ROOT / DATASETS[dataset],
                                                      "val", args.ab_limit)
                baseline = evaluate(model, ab_loader)
                with stable_metric(model.context):
                    swapped = evaluate(model, ab_loader)
                entry["numerics_ab"] = dict(
                    images=len(ab_indices), current=baseline, stable=swapped,
                    delta_f1=swapped["F1"] - baseline["F1"],
                    argmin_change=entry["p1"]["numerics"]["stable_vs_model_argmin_change"])
                print(f"[{name}/{dataset}] A/B F1 current={baseline['F1']:.4f} "
                      f"stable={swapped['F1']:.4f} delta={entry['numerics_ab']['delta_f1']:+.4f}",
                      flush=True)
            if args.skip_p2:
                report["d0"] = build_d0(report)
                entry["seconds"] = round(time.time() - started, 1)
                report.setdefault("details", {}).setdefault(name, {})[dataset] = entry
                flush()
                del model
                continue
            original_val = evaluate(model, val_loader)
            entry["p2"].update(validation_original=original_val)
            alpha_fit = None
            if entry["p1"].get("available"):
                alpha_ls = entry["p1"]["alpha_least_squares"]
                candidates = sorted({round(alpha_ls * factor, 6) for factor in grid})
                curve = {}
                for alpha in candidates:
                    result = evaluate(model, val_loader, alpha)
                    curve[f"{alpha:.10g}"] = result
                    print(f"[{name}/{dataset}] val alpha={alpha:.4g} F1={result['F1']:.4f}",
                          flush=True)
                entry["p2"]["alpha_optimal_scale"] = alpha_ls
                entry["p2"]["validation_alpha_curve"] = curve
                best = max(candidates, key=lambda a: (curve[f"{a:.10g}"]["F1"], -abs(a - 1.0)))
                alpha_fit = float(best)
                entry["p2"]["alpha_optimal"] = alpha_fit
                entry["p2"]["validation_best"] = curve[f"{alpha_fit:.10g}"]
                print(f"[{name}/{dataset}] alpha*={alpha_fit:.4g} "
                      f"val F1 {original_val['F1']:.4f} -> "
                      f"{curve[f'{alpha_fit:.10g}']['F1']:.4f}", flush=True)
            else:
                entry["p2"]["geometry_probe"] = "not applicable: no hyperbolic context module"
            entry["p2"]["test_original"] = evaluate(model, test_loader)
            if alpha_fit is not None:
                entry["p2"]["test_euclidean_raw"] = evaluate(model, test_loader, 1.0)
                if abs(alpha_fit - 1.0) < 1e-9:
                    entry["p2"]["test_euclidean_fitted"] = dict(
                        entry["p2"]["test_euclidean_raw"], note="alpha* == 1, identical pass")
                else:
                    entry["p2"]["test_euclidean_fitted"] = evaluate(model, test_loader, alpha_fit)
            for key in ("test_original", "test_euclidean_raw", "test_euclidean_fitted"):
                if key in entry["p2"]:
                    value = entry["p2"][key]
                    print(f"[{name}/{dataset}] {key} F1={value['F1']:.4f} "
                          f"P={value['precision']:.4f} R={value['recall']:.4f}", flush=True)
            if "test_euclidean_raw" in entry["p2"]:
                original_f1 = entry["p2"]["test_original"]["F1"]
                entry["p2"]["deltas"] = dict(
                    raw_vs_original=entry["p2"]["test_euclidean_raw"]["F1"] - original_f1,
                    fitted_vs_original=entry["p2"]["test_euclidean_fitted"]["F1"] - original_f1)
            entry["seconds"] = round(time.time() - started, 1)
            report["p2"].setdefault(name, {})[dataset] = entry["p2"]
            report.setdefault("details", {}).setdefault(name, {})[dataset] = entry
            flush()
            del model
    report["d0"] = build_d0(report)
    for name, datasets in report.get("details", {}).items():
        for dataset, entry in datasets.items():
            if "numerics_ab" in entry:
                report["d0"][f"{name}/{dataset}"]["f1_numerics_ab"] = entry["numerics_ab"]
    report["verdict"] = build_verdict(report)
    if "d3" in report:
        report["d3_summary"] = build_d3_summary(report["d3"])
    flush()
    print("[DONE]", json.dumps(report["verdict"], indent=2)[:2000], flush=True)


def build_verdict(report):
    flags = {}
    for name, datasets in report["p1"].items():
        for dataset, result in datasets.items():
            if not result.get("available"):
                flags[f"{name}/{dataset}"] = dict(label="not applicable",
                                                  reason="no Lorentz geometry in this arm")
                continue
            ranking = result["ranking"]
            degenerate = []
            if result["query_norm"]["p99"] is not None and result["query_norm"]["p99"] < 0.3:
                degenerate.append("p99(||z||)<0.3")
            if ranking["spearman_pairs"] > 0.99:
                degenerate.append("pooled spearman>0.99")
            if result["pairs"]["relative_deviation"]["p50"] < 1e-2:
                degenerate.append("median relative deviation<1e-2")
            ranking_closed = (ranking["spearman_query_mean"] > 0.98
                              and ranking["argmin_change_fraction"] < 0.05)
            if ranking_closed:
                label = "ranking-equivalent to Euclidean"
            elif degenerate:
                label = "numerically near-Euclidean"
            else:
                label = "distinct metric"
            flags[f"{name}/{dataset}"] = dict(
                label=label, degenerate_numeric=degenerate,
                ranking_spearman_query_mean=ranking["spearman_query_mean"],
                ranking_argmin_change_fraction=ranking["argmin_change_fraction"],
                affine_r2=result["affine"]["r2"])
    return dict(per_model_dataset=flags)


def build_d3_summary(d3):
    summary = {}
    for name, datasets in d3.items():
        for dataset, result in datasets.items():
            if not result.get("available"):
                summary[f"{name}/{dataset}"] = dict(label="not applicable",
                                                    reason=result.get("reason"))
                continue
            cosine = result["v_redundancy"]["cosine"]["p50"]
            rank90 = result["v_redundancy"]["effective_rank"]["energy90"]
            gains = [value["mean"] for value in result["context_gain"]["per_scale"].values()]
            top_gain = max(gains) if gains else 0.0
            summary[f"{name}/{dataset}"] = dict(
                cosine_p50=cosine, effective_rank90=rank90,
                value_bank_redundant=bool(cosine > 0.9 or rank90 < 32),
                context_gain_max=top_gain,
                context_branch_decorative=bool(top_gain < 0.05),
                gate_mean=result["context_gain"]["gate_mean"])
    return summary


if __name__ == "__main__":
    main()
