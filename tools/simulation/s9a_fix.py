"""S9a-fix / S9b / S13a counterfactuals -- corrected addendum (V6 final closure).

This is a *copy* of ``s9a_final.py`` kept next to it; the historical script and
its CSVs are untouched.  Fixes (task book section 2.1-2.3):

* ``SCALE-OFF`` / ``COUNT-OFF`` / ``SUPPORT-OFF`` / ``PRIORONLY`` now build the
  three prior terms independently::

      learned_scale_bias   repeat(bias[:, q, s])            over source scale s
      count_prior          repeat(-log N_s)                 over source scale s
      log_support          log(support)                     per key slot

  The old ``SCALE-OFF`` did ``bias - source_bias(zeros)`` which collapses to the
  learned term (identical to ``COUNT-OFF``).  ``BASE`` still calls the frozen
  ``attention_logits``; ``recompute_full`` re-derives the very same logits with
  the explicit terms so parity of the decomposition is checked, not assumed.

* ``GUIDE`` slots now follow the module order ``(region, class)`` interleaved,
  i.e. ``[r0-change, r0-background, r1-change, r1-background, ...]`` per scale
  and scale 0 -> 3 afterwards.  The old code put every change slot before
  every background slot.

* ``UNIFORM`` means *all* attention logits are exactly zero (distance *and*
  every prior removed) and it goes through the same recording path, so its
  Jensen-Shannon divergence to ``BASE`` (nats, ``sqrt(JS)``, upper bound
  ``sqrt(log 2)``) is available.

* sampling is stratified over the *available* strata with equal quota and an
  explicit top-up; actual counts, filenames and the list SHA-256 are recorded
  in the summary JSON and the per-image CSV gains a ``stratum`` column.  The
  per-job loader list lives inside the output directory (never in the dataset
  root), so concurrent checkpoints cannot overwrite each other.

* the Euclidean distance calibration (``DECAL``) is fitted on train images
  only in float64, records the number of fitted query rows / logit pairs, and
  a missing/non-positive/non-finite calibration is a hard error.

Server usage (conda env ``casacd``)::

    python s9a_fix.py --arm R_FULL-2334 --dataset SYSU-CD-256 --images 48 \\
        --mode smoke --out .../FinalClosure/S9a
    python s9a_fix.py ... --mode fit
    python s9a_fix.py ... --mode arms
    python s9a_fix.py ... --mode grids --gridset b32
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_s7c():
    """Import the frozen runtime's capture helper without modifying the runtime."""
    candidates = [Path(__file__).resolve().parent / "s7c_capture.py",
                  Path("/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/runtime/"
                       "tools_simulation/s7c_capture.py")]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("s7c_capture", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit("s7c_capture.py not found next to this script or in the runtime")


S7 = _load_s7c()

GRID_SETS = {"b224": (8, 4, 4, 4), "b128": (4, 4, 4, 4),
             "b64": (4, 2, 2, 2), "b32": (2, 2, 2, 2)}

# name -> (distance, prior, euclid K pooling, euclid V pooling, euclid aggregation,
#          no context residual, uniform attention, guide support)
ARMS = {
    "BASE": ("geo", "full", 0, 0, 0, 0, 0, 0),
    "NOCTX": ("geo", "full", 0, 0, 0, 1, 0, 0),
    "UNIFORM": ("geo", "full", 0, 0, 0, 0, 1, 0),
    "PRIORONLY": ("none", "full", 0, 0, 0, 0, 0, 0),
    "DISTONLY": ("geo", "off", 0, 0, 0, 0, 0, 0),
    "KE": ("geo", "full", 1, 0, 0, 0, 0, 0),
    "VE": ("geo", "full", 0, 1, 0, 0, 0, 0),
    "KVE": ("geo", "full", 1, 1, 0, 0, 0, 0),
    "DE": ("euc", "full", 0, 0, 0, 0, 0, 0),
    "DECAL": ("euc_cal", "full", 0, 0, 0, 0, 0, 0),
    "AGG": ("geo", "full", 0, 0, 1, 0, 0, 0),
    "DEAGG": ("euc", "full", 0, 0, 1, 0, 0, 0),
    "SUPPORT-OFF": ("geo", "support_off", 0, 0, 0, 0, 0, 0),
    "COUNT-OFF": ("geo", "count_off", 0, 0, 0, 0, 0, 0),
    "SCALE-OFF": ("geo", "scale_off", 0, 0, 0, 0, 0, 0),
    "GUIDE": ("geo", "full", 0, 0, 0, 0, 0, 1),
}

STRATA = ("bg", "small", "mid", "large")

lift = None
source_bias = None
feature_regions = None
channel_norm = None
pool_features = None


def euclid_sq(a, b):
    a2 = (a * a).sum(-1)[..., None]
    b2 = (b * b).sum(-1)[..., None, :]
    return (a2 + b2 - 2.0 * torch.matmul(a, b.transpose(-1, -2))).clamp_min(0.0)


def js_distance(p, q):
    """sqrt(JS) in nats (natural log, upper bound ``sqrt(log 2)``)."""
    p = p.double()
    q = q.double()
    m = 0.5 * (p + q)
    tiny = torch.finfo(torch.float64).tiny
    t1 = torch.where(p > 0, p * (p.clamp_min(tiny) / m.clamp_min(tiny)).log(),
                     torch.zeros_like(p))
    t2 = torch.where(q > 0, q * (q.clamp_min(tiny) / m.clamp_min(tiny)).log(),
                     torch.zeros_like(q))
    return (0.5 * t1.sum(-1) + 0.5 * t2.sum(-1)).clamp_min(0.0).sqrt()


def entropy(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def counts_from_logits(logit, target):
    pred = logit > 0
    gt = target > 0.5
    return (int((pred & gt).sum()), int((pred & ~gt).sum()),
            int((~pred & gt).sum()), int((~pred & ~gt).sum()))


def scores_from_counts(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / max(n, 1)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n * n, 1)
    return {
        "OA": po,
        "Recall": tp / max(tp + fn, 1),
        "Precision": tp / max(tp + fp, 1),
        "F1": 2 * tp / max(2 * tp + fp + fn, 1),
        "IoU": tp / max(tp + fp + fn, 1),
        "Kappa": (po - pe) / (1 - pe) if abs(1 - pe) > 1e-9 else float("nan"),
    }


def repeat_by_scale(value, counts):
    """(heads, n_sources) -> (heads, M): repeat column s exactly ``counts[s]``."""
    parts = [value[:, s:s + 1].expand(-1, int(c)) for s, c in enumerate(counts)]
    return torch.cat(parts, dim=-1)


def prior_terms(scale_bias, query_scale, counts, support):
    """Return (log_support, learned, count) broadcastable to (B, h, Nq, M)."""
    heads = scale_bias.shape[0]
    dtype, device = scale_bias.dtype, scale_bias.device
    log_support = support.clamp_min(1e-6).log()[:, None, None]          # (B,1,1,M)
    learned = repeat_by_scale(scale_bias[:, query_scale, :], counts)    # (h,M)
    learned = learned[None, :, None, :]
    neg_log = torch.tensor([-math.log(float(c)) for c in counts],
                           dtype=dtype, device=device).view(1, -1).expand(heads, -1)
    count = repeat_by_scale(neg_log, counts)[None, :, None, :]
    return log_support, learned, count


def stratified_pick(val_rows, images):
    """Equal quota over the *available* strata, deterministic spread, top-up.

    Returns (picked, counts).  ``images`` is a target, not a guarantee: the
    pool may be smaller (recorded in ``counts`` and the summary JSON).
    """
    strata = {}
    for r in val_rows:
        strata.setdefault(r["stratum"], []).append(r)
    avail = [k for k in STRATA if strata.get(k)]
    if not avail:
        raise SystemExit("no stratified val rows in samples.json")
    base = images // len(avail)
    rem = images % len(avail)
    quota = {k: base + (1 if i < rem else 0) for i, k in enumerate(avail)}
    picked, counts = [], {}
    for k in avail:
        rows = strata[k]
        take = min(quota[k], len(rows))
        step = max(1, len(rows) // max(take, 1))
        sel = rows[::step][:take]
        picked += sel
        counts[k] = len(sel)
    for k in avail:                       # top-up if a stratum ran out
        need = images - len(picked)
        if need <= 0:
            break
        extra = strata[k][counts[k]:]
        add = extra[:need]
        picked += add
        counts[k] += len(add)
    picked.sort(key=lambda r: (r.get("scene_id", ""), r["filename"]))
    return picked, counts


class Patcher:
    """Single-operator overrides installed on one ManifoldContext instance."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.orig_logits = ctx.attention_logits
        self.orig_pool = ctx.manifold_pool
        self.orig_aggregate = ctx.aggregate
        self.orig_readout = ctx.readout
        self.orig_support = ctx.support_term
        self.orig_grids = ctx.grid_sizes
        self.orig_tokens = ctx.tokens_per_scale
        self.calls = 0
        self.record = None
        self.recompute_full = False

    # ---------------- operators ----------------
    def pool_euclid(self, points, weights):
        summed = torch.einsum("brtp,bhrpd->bhrtd", weights, points)
        mass = weights.sum(-1)[:, :, :, None].unsqueeze(1)
        return lift(summed / mass.clamp_min(1e-12)).flatten(2, 3)

    def support_guide(self, sources, support):
        """Per-region class mean of the raw guide probability, module slot order.

        Module order is ``(region, class)`` interleaved per scale; the old code
        produced all change slots first and all background slots afterwards.
        """
        pooled = []
        for x, grid in zip(sources, self.ctx.grid_sizes):
            prob = self.ctx.guide(x).sigmoid()
            pos = F.adaptive_avg_pool2d(prob, grid).flatten(1)      # (B,R)
            pooled.append(torch.stack((pos, 1.0 - pos), dim=-1).flatten(1))
        return torch.cat(pooled, dim=1).clamp_min(1e-6)

    def make_logits(self, dist, prior, calib, uniform=False):
        ctx = self.ctx
        orig = self.orig_logits

        def logits(query, key, support, query_scale):
            tau2 = ctx.scale_temperature(query_scale).square()
            b, h, qn, kn = query.shape[0], ctx.heads, query.shape[-2], key.shape[-2]
            counts = ctx.tokens_per_scale
            if uniform:
                out = torch.zeros(b, h, qn, kn, dtype=query.dtype, device=query.device)
            else:
                if dist == "geo":
                    base = -ctx.squared_distance(query, key) / (2 * tau2)
                elif dist == "none":
                    base = torch.zeros(b, h, qn, kn, dtype=query.dtype,
                                       device=query.device)
                else:
                    scale = 1.0 if dist == "euc" else float(calib)
                    base = -scale * euclid_sq(query[..., 1:], key[..., 1:]) / (2 * tau2)
                log_support, learned, count = prior_terms(
                    ctx.scale_bias, query_scale, counts, support)
                if prior == "full" and dist == "geo" and not self.recompute_full:
                    out = orig(query, key, support, query_scale)
                elif prior == "full":
                    out = base + log_support + learned + count
                elif prior == "off":
                    out = base
                elif prior == "support_off":
                    out = base + learned + count
                elif prior == "count_off":
                    out = base + log_support + learned
                elif prior == "scale_off":
                    out = base + log_support + count
                else:
                    raise ValueError(prior)
            if self.record is not None:
                with torch.no_grad():
                    prob = out.softmax(-1)
                    self.record.append((query_scale, prob.detach(),
                                        float(entropy(prob).mean())))
            return out
        return logits

    # ---------------- lifecycle ----------------
    def apply(self, arm, calib=1.0, record=None, grids=None, recompute_full=False):
        ctx = self.ctx
        dist, prior, ke, ve, agge, noctx, uniform, guide_sup = ARMS[arm]
        self.record = record
        self.calls = 0
        self.recompute_full = recompute_full
        orig_pool = self.orig_pool

        def pooled(points, weights):
            want = ke if self.calls % 2 == 0 else ve
            self.calls += 1
            return self.pool_euclid(points, weights) if want else orig_pool(points, weights)

        ctx.manifold_pool = pooled if (ke or ve) else orig_pool
        ctx.aggregate = (lambda prob, values: prob @ values) if agge else self.orig_aggregate
        ctx.attention_logits = self.make_logits(dist, prior, calib, uniform=bool(uniform))
        ctx.support_term = self.support_guide if guide_sup else self.orig_support
        if noctx:
            def readout(*a, **kw):
                out = self.orig_readout(*a, **kw)
                return (out[0] * 0.0,) + tuple(out[1:])
            ctx.readout = readout
        else:
            ctx.readout = self.orig_readout
        if grids is not None:
            ctx.grid_sizes = tuple(grids)
            ctx.tokens_per_scale = tuple(2 * g * g for g in grids)
        else:
            ctx.grid_sizes = self.orig_grids
            ctx.tokens_per_scale = self.orig_tokens

    def restore(self):
        ctx = self.ctx
        ctx.manifold_pool = self.orig_pool
        ctx.aggregate = self.orig_aggregate
        ctx.attention_logits = self.orig_logits
        ctx.support_term = self.orig_support
        ctx.readout = self.orig_readout
        ctx.grid_sizes = self.orig_grids
        ctx.tokens_per_scale = self.orig_tokens
        self.recompute_full = False


def main():
    global lift, source_bias, feature_regions, channel_norm, pool_features

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R_FULL-2334")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--images", type=int, default=48)
    ap.add_argument("--train-images", type=int, default=16)
    ap.add_argument("--mode", default="arms", choices=["smoke", "fit", "arms", "grids"])
    ap.add_argument("--gridset", default="b224")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("S7B_THREADS", "6")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(S7.SEED)

    C, CASANet, Dataset, transforms, T = S7.bootstrap_paths()
    from model.layers.lorentz_ops import lift as _lift
    from model.layers.pyramid_context import (source_bias as _sb, feature_regions as _fr,
                                              channel_norm as _cn, pool_features as _pf)
    lift, source_bias, feature_regions, channel_norm, pool_features = _lift, _sb, _fr, _cn, _pf

    outdir = Path(args.out) if args.out else (S7.OUT_ROOT.parent / "S9aFinal")
    outdir.mkdir(parents=True, exist_ok=True)
    root = S7.DATA_ROOT / args.dataset
    tag = f"{args.arm}_{args.dataset}"

    manifest = S7.OUT_ROOT / tag / "samples.json"
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest}; run s7c_capture.py first")
    m = json.loads(manifest.read_text())
    picked, sampled = stratified_pick(m["val"], args.images)
    train_names = [r["filename"] for r in m["train"]][: args.train_images]

    tf = transforms.Compose([transforms.Scale(256, 256),
                             transforms.Normalize(T.MEAN, T.STD), transforms.ToTensor()])
    listing = outdir / f"_list_{tag}_{args.mode}.txt"
    list_sha = hashlib.sha256()

    def loader(names):
        listing.write_text("\n".join(names) + "\n")
        list_sha.update("\n".join(names).encode())
        ds = Dataset(str(root), str(listing), transform=tf)
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = CASANet(None, **C.context_kwargs(args.arm)).eval()
    model.load_state_dict(torch.load(str(S7.CKPT_ROOT / args.arm / args.dataset / "best_model.pth"),
                                     map_location="cpu", weights_only=True), strict=True)
    ctx = model.context
    for p in model.parameters():
        p.requires_grad_(False)
    patcher = Patcher(ctx)

    def encode(a, b):
        with torch.no_grad():
            fa, fb = model.encoder(a), model.encoder(b)
            res = model.tar(fa[1:], fb[1:])
            tar, difference = (res[0], res[1]) if len(res) >= 2 else (res, None)
            stem = torch.cat((fa[0], fb[0]), 1)
            detail = model.detail(stem) if model.detail is not None else None
        return dict(tar=tar, difference=difference, stem=stem, detail=detail)

    def decode(enc, arm, record=None, grids=None, calib=1.0, recompute_full=False):
        patcher.apply(arm, calib=calib, record=record, grids=grids,
                      recompute_full=recompute_full)
        try:
            with torch.no_grad():
                enhanced, _ = ctx(enc["tar"], enc["difference"], return_aux=False)
                pyramid = model.decoder.forward_pyramid(enhanced)
                refined = model.fine(model._resize(model.head(pyramid[0]),
                                                   enc["stem"].shape[-2:]), enc["stem"])
                primary = refined if enc["detail"] is None else refined + enc["detail"]
                logit = model._resize(primary, (256, 256))
        finally:
            patcher.restore()
        return logit

    # ---------------- parity smoke (+ decomposition parity) ----------------
    if args.mode in ("smoke", "fit"):
        rows = []
        for rec in picked[:4]:
            name = rec["filename"]
            for image, target in loader([name]):
                enc = encode(image[:, :3], image[:, 3:])
                mine = decode(enc, "BASE")
                recomp = decode(enc, "BASE", recompute_full=True)
                with torch.no_grad():
                    ref = model(image[:, :3], image[:, 3:], return_aux=True)[0][0]
                rows.append(dict(file=name,
                                 max_abs_diff=float((mine - ref).abs().max()),
                                 binary_diff=int(((mine > 0) != (ref > 0)).sum()),
                                 recompute_abs_diff=float((mine - recomp).abs().max()),
                                 recompute_binary_diff=int(
                                     ((mine > 0) != (recomp > 0)).sum())))
                print(f"[parity] {name} max|d|={rows[-1]['max_abs_diff']:.3e} "
                      f"bin={rows[-1]['binary_diff']} "
                      f"recomp={rows[-1]['recompute_abs_diff']:.3e}", flush=True)
        (outdir / f"parity_{tag}.json").write_text(json.dumps(dict(rows=rows), indent=2))
        bad = [r for r in rows if r["max_abs_diff"] > 2e-4 or r["binary_diff"] != 0]
        bad += [r for r in rows if r["recompute_abs_diff"] > 2e-4]
        if bad:
            raise SystemExit(f"parity failed: {bad[:2]}")
        if args.mode == "smoke":
            return

        num = torch.zeros((), dtype=torch.float64)
        den = torch.zeros((), dtype=torch.float64)
        pairs = 0
        queries = 0
        for name in train_names:
            for image, target in loader([name]):
                enc = encode(image[:, :3], image[:, 3:])
                with torch.no_grad():
                    normalized = [channel_norm(ctx.norm, x) for x in enc["tar"]]
                    sources = ctx.prototype_features(normalized, enc["difference"])
                    occ = [ctx.guide(x).sigmoid() for x in sources]
                    comp = [ctx.class_support(p, g) for p, g in zip(occ, ctx.grid_sizes)]
                    if ctx.prototype_pooling == "lorentz":
                        key, _ = ctx.manifold_keys_and_values(sources, comp)
                    else:
                        proto = [pool_features(x, w, g) for x, (w, _), g in
                                 zip(sources, comp, ctx.grid_sizes)]
                        key, _ = ctx.project_keys(torch.cat(proto, 1))
                    for index in ctx.retrieval_scales:
                        q = ctx.project_queries(normalized[index]).flatten(2, 3)[:, :, ::7, :]
                        tau2 = ctx.scale_temperature(index).square()
                        h = (ctx.squared_distance(q, key) / (2 * tau2)).double()
                        e = (euclid_sq(q[..., 1:], key[..., 1:]) / (2 * tau2)).double()
                        h = h - h.mean(-1, keepdim=True)
                        e = e - e.mean(-1, keepdim=True)
                        num += (h * e).sum()
                        den += (e * e).sum()
                        pairs += h.numel()
                        queries += q.shape[0] * q.shape[-2]
        a = float(num / den) if float(den) > 0 else float("nan")
        (outdir / f"calib_{tag}.json").write_text(json.dumps(
            dict(a=a, n_pairs=pairs, n_queries=queries, train_images=len(train_names),
                 arm=args.arm, dataset=args.dataset, fitted_on="train"), indent=2))
        print(f"[calib] a={a:.4f} queries={queries} pairs={pairs}", flush=True)
        return

    # ---------------- counterfactual arms ----------------
    needs_calib = any(ARMS[a][0] == "euc_cal" for a in ARMS)
    calib = 1.0
    cp = outdir / f"calib_{tag}.json"
    if cp.is_file():
        payload = json.loads(cp.read_text())
        calib = float(payload.get("a", float("nan")))
    if needs_calib and not (math.isfinite(calib) and calib > 0):
        raise SystemExit(f"missing/invalid Euclidean calibration at {cp}")
    print(f"[calib] a={calib:.4f}", flush=True)

    if args.mode == "arms":
        arm_list = [a.strip() for a in args.only.split(",") if a.strip()] or list(ARMS.keys())
        grids = None
    else:
        arm_list = ["BASE"]
        grids = GRID_SETS[args.gridset]

    order = ["BASE"] + [a for a in arm_list if a != "BASE"]
    per_image, attn = [], []
    for rec in picked:
        name = rec["filename"]
        for image, target in loader([name]):
            enc = encode(image[:, :3], image[:, 3:])
            base_prob = {}
            for arm in order:
                record = [] if (arm == "BASE" or arm in arm_list) else None
                t0 = time.time()
                logit = decode(enc, arm, record=record, grids=grids, calib=calib)
                dt = time.time() - t0
                if arm == "BASE":
                    base_prob = {i: (p, h) for i, p, h in record}
                if arm not in arm_list:
                    continue
                tp, fp, fn, tn = counts_from_logits(logit, target)
                row = dict(checkpoint=args.arm, file=name, stratum=rec.get("stratum", ""),
                           scene_id=rec.get("scene_id", ""),
                           sample_id=rec.get("sample_id", -1), arm=arm,
                           seed=args.arm.split("-")[-1], dataset=args.dataset,
                           split="val", TP=tp, FP=fp, FN=fn, TN=tn,
                           GT_pixels=int((target > 0.5).sum()),
                           pred_pixels=int((logit > 0).sum()),
                           output_shape="256x256", seconds=round(dt, 3),
                           gridset=args.gridset)
                row.update({k: round(float(v), 6)
                            for k, v in scores_from_counts(tp, fp, fn, tn).items()})
                per_image.append(row)
                if arm != "BASE":
                    for i, p, h in record or []:
                        if i not in base_prob:
                            continue
                        b, bh = base_prob[i]
                        attn.append(dict(checkpoint=args.arm, file=name, arm=arm,
                                         seed=args.arm.split("-")[-1],
                                         dataset=args.dataset, scale=i,
                                         js=round(float(js_distance(b, p).mean()), 6),
                                         tv=round(float(0.5 * (b.double() - p.double())
                                                       .abs().sum(-1).mean()), 6),
                                         entropy=round(float(h), 6),
                                         base_entropy=round(float(bh), 6)))
                print(f"[arm] {name} {arm:11s} F1={row['F1']:.4f} ({dt:.2f}s)", flush=True)

    def dump(path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump(outdir / f"per_image_{tag}_{args.mode}{args.gridset}.csv", per_image)
    if attn:
        dump(outdir / f"attn_{tag}_{args.mode}{args.gridset}.csv", attn)

    glob = {}
    for arm in arm_list:
        sub = [r for r in per_image if r["arm"] == arm]
        tp = sum(r["TP"] for r in sub); fp = sum(r["FP"] for r in sub)
        fn = sum(r["FN"] for r in sub); tn = sum(r["TN"] for r in sub)
        glob[arm] = dict(counts=dict(TP=tp, FP=fp, FN=fn, TN=tn),
                         **scores_from_counts(tp, fp, fn, tn))
    summary = dict(arm=args.arm, dataset=args.dataset, mode=args.mode,
                   images=len(picked), sampled=sampled, calib_a=calib,
                   gridset=args.gridset,
                   grids=list(grids) if grids else list(ctx.grid_sizes),
                   tokens=list(ctx.tokens_per_scale),
                   arms=arm_list, global_scores=glob,
                   list_sha256=list_sha.hexdigest(),
                   uniform_definition="all attention logits are exactly zero",
                   js_definition="sqrt(0.5 KL(p||m)+0.5 KL(q||m)) in nats")
    (outdir / f"summary_{tag}_{args.mode}{args.gridset}.json").write_text(
        json.dumps(summary, indent=1))
    for arm, v in glob.items():
        print(f"[global] {arm:11s} F1={v['F1']:.4f} IoU={v['IoU']:.4f} "
              f"P={v['Precision']:.4f} R={v['Recall']:.4f} Kappa={v['Kappa']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
