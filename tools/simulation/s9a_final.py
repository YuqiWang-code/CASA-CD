"""S9a-final / S9b / S13a-final: same-checkpoint counterfactuals scored with the
frozen evaluator's decision rule.

Fixes relative to the V5 ``s9a_counterfactual.py``:

* metrics are integer TP/FP/FN/TN per image from ``outputs[0] > 0`` (the frozen
  ``train_CASA.evaluate`` rule), aggregated to a global confusion matrix before
  recall/precision/F1/IoU/Kappa -- no per-image F1 averaging, no interpolation;
* Jensen-Shannon is ``sqrt(0.5 KL(p||m) + 0.5 KL(q||m))`` in nats, computed per
  image against that image's BASE attention (the V5 code multiplied by an extra
  0.5 and only kept the last image);
* the Euclidean distance calibration is fitted on *train* images with row
  centering in float64, never on val;
* each arm replaces exactly one operator (K pooling / V pooling / distance /
  V aggregation / context write-back / one prior component) and BASE is the
  untouched module, verified to be bitwise identical to the plain forward;
* budget arms re-use the same weights by overriding ``grid_sizes`` so 224 -> 32
  prototypes change only the partition (8/4/4/4, 4/4/4/4, 4/2/2/2, 2/2/2/2).

Server usage (conda env ``casacd``), e.g.::

    python tools/simulation/s9a_final.py --arm R_FULL-2334 --dataset SYSU-CD-256 \
        --images 48 --mode smoke --out .../analysis/simulation/S9aFinal
    python tools/simulation/s9a_final.py ... --mode fit
    python tools/simulation/s9a_final.py ... --mode arms
    python tools/simulation/s9a_final.py ... --mode grids --gridset b32
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
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

    # ---------------- operators ----------------
    def pool_euclid(self, points, weights):
        summed = torch.einsum("brtp,bhrpd->bhrtd", weights, points)
        mass = weights.sum(-1)[:, :, :, None].unsqueeze(1)
        return lift(summed / mass.clamp_min(1e-12)).flatten(2, 3)

    def support_guide(self, sources, support):
        """区域类均值(guide 概率) 代替 support_head：尺度内逐 cell 平均 p 与 1-p。"""
        pooled = []
        for x, grid in zip(sources, self.ctx.grid_sizes):
            prob = self.ctx.guide(x).sigmoid()
            pos = F.adaptive_avg_pool2d(prob, grid).flatten(2).transpose(1, 2).reshape(prob.shape[0], -1)
            pooled.append(torch.cat((pos, 1.0 - pos), dim=1))
        return torch.cat(pooled, dim=1).clamp_min(1e-6)

    def make_logits(self, dist, prior, calib):
        ctx = self.ctx
        orig = self.orig_logits

        def logits(query, key, support, query_scale):
            tau2 = ctx.scale_temperature(query_scale).square()
            b, h, qn, kn = query.shape[0], ctx.heads, query.shape[-2], key.shape[-2]
            if dist == "geo":
                base = -ctx.squared_distance(query, key) / (2 * tau2)
            elif dist == "none":
                base = torch.zeros(b, h, qn, kn, dtype=query.dtype, device=query.device)
            else:
                scale = 1.0 if dist == "euc" else float(calib)
                base = -scale * euclid_sq(query[..., 1:], key[..., 1:]) / (2 * tau2)
            if prior == "full" and dist == "geo":
                out = orig(query, key, support, query_scale)
            elif prior == "full":
                out = base + support.log()[:, None, None] + source_bias(
                    ctx.scale_bias, query_scale, ctx.tokens_per_scale)
            elif prior == "off":
                out = base
            elif prior == "support_off":
                out = base + source_bias(ctx.scale_bias, query_scale, ctx.tokens_per_scale)
            elif prior == "count_off":
                bias = source_bias(ctx.scale_bias, query_scale, ctx.tokens_per_scale)
                add = torch.cat([torch.full((n,), math.log(n), dtype=query.dtype)
                                 for n in ctx.tokens_per_scale])
                out = base + support.log()[:, None, None] + bias + add[None, None, None, :]
            elif prior == "scale_off":
                bias = source_bias(ctx.scale_bias, query_scale, ctx.tokens_per_scale)
                sub = source_bias(torch.zeros_like(ctx.scale_bias), query_scale,
                                  ctx.tokens_per_scale)
                out = base + support.log()[:, None, None] + bias - sub
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
    def apply(self, arm, calib=1.0, record=None, grids=None):
        ctx = self.ctx
        dist, prior, ke, ve, agge, noctx, uniform, guide_sup = ARMS[arm]
        self.record = record
        self.calls = 0
        orig_pool = self.orig_pool

        def pooled(points, weights):
            want = ke if self.calls % 2 == 0 else ve
            self.calls += 1
            return self.pool_euclid(points, weights) if want else orig_pool(points, weights)

        ctx.manifold_pool = pooled if (ke or ve) else orig_pool
        ctx.aggregate = (lambda prob, values: prob @ values) if agge else self.orig_aggregate
        if uniform:
            ctx.attention_logits = lambda q, k, s, qi: torch.zeros(
                q.shape[0], ctx.heads, q.shape[-2], k.shape[-2], dtype=q.dtype,
                device=q.device)
        else:
            ctx.attention_logits = self.make_logits(dist, prior, calib)
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
    strata = {}
    for r in m["val"]:
        strata.setdefault(r["stratum"], []).append(r)
    per = max(1, args.images // 4)
    picked = []
    for k in ("bg", "small", "mid", "large"):
        picked += strata.get(k, [])[:per]
    picked = picked[: args.images]
    train_names = [r["filename"] for r in m["train"]][: args.train_images]

    tf = transforms.Compose([transforms.Scale(256, 256),
                             transforms.Normalize(T.MEAN, T.STD), transforms.ToTensor()])

    def loader(names):
        tmp = root / "_s9afinal_list.txt"
        tmp.write_text("\n".join(names) + "\n")
        ds = Dataset(str(root), str(tmp), transform=tf)
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

    def decode(enc, arm, record=None, grids=None, calib=1.0):
        patcher.apply(arm, calib=calib, record=record, grids=grids)
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

    # ---------------- parity smoke ----------------
    if args.mode in ("smoke", "fit"):
        rows = []
        for rec in picked[:4]:
            name = rec["filename"]
            for image, target in loader([name]):
                enc = encode(image[:, :3], image[:, 3:])
                mine = decode(enc, "BASE")
                with torch.no_grad():
                    ref = model(image[:, :3], image[:, 3:], return_aux=True)[0][0]
                rows.append(dict(file=name,
                                 max_abs_diff=float((mine - ref).abs().max()),
                                 binary_diff=int(((mine > 0) != (ref > 0)).sum())))
                print(f"[parity] {name} max|d|={rows[-1]['max_abs_diff']:.3e} "
                      f"bin={rows[-1]['binary_diff']}", flush=True)
        (outdir / f"parity_{tag}.json").write_text(json.dumps(dict(rows=rows), indent=2))
        if args.mode == "smoke":
            return

        num = torch.zeros((), dtype=torch.float64)
        den = torch.zeros((), dtype=torch.float64)
        n = 0
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
                        n += h.numel()
        a = float(num / den) if float(den) > 0 else float("nan")
        (outdir / f"calib_{tag}.json").write_text(json.dumps(
            dict(a=a, samples=n, arm=args.arm, dataset=args.dataset), indent=2))
        print(f"[calib] a={a:.4f} samples={n}", flush=True)
        return

    # ---------------- counterfactual arms ----------------
    calib = 1.0
    cp = outdir / f"calib_{tag}.json"
    if cp.is_file():
        calib = float(json.loads(cp.read_text())["a"])
    print(f"[calib] a={calib:.4f}", flush=True)

    if args.mode == "arms":
        arm_list = [args.only] if args.only else list(ARMS.keys())
        grids = None
    else:
        arm_list = ["BASE"]
        grids = GRID_SETS[args.gridset]

    per_image, attn = [], []
    for rec in picked:
        name = rec["filename"]
        for image, target in loader([name]):
            enc = encode(image[:, :3], image[:, 3:])
            base_prob = {}
            for arm in arm_list:
                record = []
                t0 = time.time()
                logit = decode(enc, arm, record=record, grids=grids, calib=calib)
                dt = time.time() - t0
                tp, fp, fn, tn = counts_from_logits(logit, target)
                row = dict(file=name, scene_id=rec.get("scene_id", ""),
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
                if arm == "BASE":
                    base_prob = {i: (p, h) for i, p, h in record}
                else:
                    for i, p, h in record:
                        if i not in base_prob:
                            continue
                        b, bh = base_prob[i]
                        attn.append(dict(file=name, arm=arm, scale=i,
                                         js=round(float(js_distance(b, p).mean()), 6),
                                         tv=round(float(0.5 * (b.double() - p.double())
                                                       .abs().sum(-1).mean()), 6),
                                         entropy=round(float(h), 6),
                                         base_entropy=round(float(bh), 6)))
                print(f"[arm] {name} {arm:11s} F1={row['F1']:.4f} ({dt:.1f}s)", flush=True)

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
                   images=len(picked), calib_a=calib, gridset=args.gridset,
                   grids=list(grids) if grids else list(ctx.grid_sizes),
                   arms=arm_list, global_scores=glob)
    (outdir / f"summary_{tag}_{args.mode}{args.gridset}.json").write_text(
        json.dumps(summary, indent=2))
    for arm, v in glob.items():
        print(f"[global] {arm:11s} F1={v['F1']:.4f} IoU={v['IoU']:.4f} "
              f"P={v['Precision']:.4f} R={v['Recall']:.4f} Kappa={v['Kappa']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
