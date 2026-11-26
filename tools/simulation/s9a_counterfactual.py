"""S9a: on-model counterfactuals for the compression/geometry operators.

Task book: docs/temporary/CASA-CD_V4仿真复核与下一轮实验执行方案.md  §5 (S9a)

Only operators are replaced; every other parameter (guide, TAR, DCR, support,
temperature, scale bias, readout) stays the same as the frozen checkpoint.
The script never trains and never touches ``test``.

Arms
----
BASE          unmodified forward (parity check against a second plain forward)
NOCTX         ``gamma_context * update`` forced to zero (local/detail kept)
UNIFORM       attention logits forced to 0 (no distance, no prior, no bias)
PRIORONLY     distance term removed (prior + scale bias only)
DISTONLY      prior + scale bias removed (distance only)
KE / VE / KVE pooling geometry replaced by a Euclidean mean for K, V, or both
DE            geodesic squared distance replaced by Euclidean squared in z
DECAL         same, with one scalar ``a`` fitted on train (a*DE2 ~= DH2)
AGG           V aggregation replaced by the Euclidean weighted mean
DEAGG         DE + AGG

Outputs (server): <analysis>/simulation/S9a/<arm>_<dataset>/
    s9a_arms.csv, s9a_attn.csv, s9a_summary.json
"""

import argparse
import csv as _csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7c_capture as S7  # noqa: E402

LIFT = None


def js_distance(p, q, eps=1e-12):
    """Square root of the Jensen-Shannon divergence (natural log)."""
    p = p.double().clamp_min(eps)
    q = q.double().clamp_min(eps)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)
    return (0.5 * js).clamp_min(0).sqrt()


class OperatorPatcher:
    """Temporarily install instance-level operator overrides on the context."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.orig_logits = ctx.attention_logits
        self.orig_pool = ctx.manifold_pool
        self.orig_aggregate = ctx.aggregate
        self.orig_gamma = ctx.gamma_context.detach().clone()
        self.pool_calls = 0
        self.pending = None  # (h, e) lists for calibration

    # ---- operators -----------------------------------------------------
    def pool_euclidean(self, points, weights):
        summed = torch.einsum("brtp,bhrpd->bhrtd", weights, points)  # (B,h,R,2,D)
        mass = weights.sum(-1)[:, :, :, None].unsqueeze(1)           # (B,1,R,2,1)
        mean = summed / mass.clamp_min(1e-12)
        return LIFT(mean).flatten(2, 3)

    def aggregate_euclidean(self, probability, values):
        return probability @ values

    def make_pool(self, e_k, e_v):
        def pooled(points, weights):
            idx = self.pool_calls
            self.pool_calls += 1
            use_e = e_k if idx % 2 == 0 else e_v
            return self.pool_euclidean(points, weights) if use_e else self.orig_pool(points, weights)
        return pooled

    def make_logits(self, mode, calib=1.0, record=None):
        orig = self.orig_logits
        ctx = self.ctx

        def logits(query, key, support, query_scale):
            full = orig(query, key, support, query_scale)
            tau2 = ctx.scale_temperature(query_scale).square()
            if mode in ("de", "decal", "fit"):
                dh2 = ctx.squared_distance(query, key)
                qz, kz = query[..., 1:], key[..., 1:]
                de2 = ((qz * qz).sum(-1)[..., None] + (kz * kz).sum(-1)[..., None, :]
                       - 2.0 * torch.matmul(qz, kz.transpose(-1, -2))).clamp_min(0.0)
                if mode == "fit":
                    self.pending.setdefault("h", []).append(dh2.reshape(-1).double())
                    self.pending.setdefault("e", []).append(de2.reshape(-1).double())
                    return full
                a = 1.0 if mode == "de" else calib
                out = full + (dh2 - a * de2) / (2 * tau2)
            elif mode == "uniform":
                out = torch.zeros_like(full)
            elif mode == "prior":
                out = full + ctx.squared_distance(query, key) / (2 * tau2)
            elif mode == "dist":
                out = -ctx.squared_distance(query, key) / (2 * tau2)
            else:
                out = full
            if record is not None:
                record[query_scale] = out.softmax(-1).detach()
            return out
        return logits

    # ---- lifecycle -----------------------------------------------------
    MODE = {"base": "base", "noctx": "base", "uniform": "uniform",
            "prioronly": "prior", "distonly": "dist", "de": "de",
            "decal": "decal", "deagg": "de", "ke": "base", "ve": "base",
            "kve": "base", "agg": "base"}

    def apply(self, arm, calib=1.0, record=None):
        ctx = self.ctx
        self.pool_calls = 0
        ctx.manifold_pool = self.make_pool(arm in ("ke", "kve"), arm in ("ve", "kve"))
        ctx.aggregate = self.aggregate_euclidean if "agg" in arm else self.orig_aggregate
        ctx.attention_logits = self.make_logits(self.MODE[arm], calib=calib, record=record)
        if arm == "noctx":
            ctx.gamma_context.data.zero_()

    def restore(self):
        ctx = self.ctx
        ctx.manifold_pool = self.orig_pool
        ctx.aggregate = self.orig_aggregate
        ctx.attention_logits = self.orig_logits
        ctx.gamma_context.data.copy_(self.orig_gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R_FULL-2334")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--calib-images", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    global LIFT
    torch.set_num_threads(int(os.environ.get("S7B_THREADS", "6")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(S7.SEED)

    C, CASANet, Dataset, transforms, T = S7.bootstrap_paths()
    from model.layers.lorentz_ops import lift as _lift
    LIFT = _lift

    ckpt = S7.CKPT_ROOT / args.arm / args.dataset / "best_model.pth"
    outdir = S7.OUT_ROOT.parent / "S9a" / f"{args.arm}_{args.dataset}"
    outdir.mkdir(parents=True, exist_ok=True)
    root = S7.DATA_ROOT / args.dataset

    s7c_dir = S7.OUT_ROOT / f"{args.arm}_{args.dataset}"
    spath = s7c_dir / "samples.json"
    if spath.is_file():
        samples = json.loads(spath.read_text())
        val_names = [r["filename"] for r in samples["val"]][: args.images]
        train_names = [r["filename"] for r in samples["train"]][: args.calib_images]
    else:
        val_names = [ln.strip() for ln in (root / "list" / "val.txt").read_text().splitlines()
                     if ln.strip()][: args.images]
        train_names = [ln.strip() for ln in (root / "list" / "train.txt").read_text().splitlines()
                       if ln.strip()][: args.calib_images]

    eval_transform = transforms.Compose([transforms.Scale(256, 256),
                                         transforms.Normalize(T.MEAN, T.STD),
                                         transforms.ToTensor()])

    def loader_for(names, tag):
        listing = outdir / f"_list_{tag}.txt"
        listing.write_text("\n".join(names) + "\n")
        ds = Dataset(str(root), str(listing), transform=eval_transform)
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                           num_workers=0, drop_last=False)

    model = CASANet(None, **C.context_kwargs(args.arm)).eval()
    model.load_state_dict(S7.load_state(ckpt), strict=True)
    ctx = model.context
    dev = torch.device(args.device)
    model.to(dev)
    patcher = OperatorPatcher(ctx)

    # 0. parity: plain vs base-patched
    image = next(iter(loader_for(val_names[:2], "parity")))[0].to(dev).float()
    with torch.no_grad():
        plain, _ = model(image[:, :3], image[:, 3:], return_aux=True)
        patcher.apply("base")
        patched, _ = model(image[:, :3], image[:, 3:], return_aux=True)
        patcher.restore()
    parity = max(float((plain[k] - patched[k]).abs().max()) for k in range(len(plain)))
    print(f"[parity] base max|dlogit| = {parity:.3e}", flush=True)

    # 1. calibration scalar a from TRAIN images (a*DE2 ~= DH2, row-centred)
    patcher.pending = {}
    ctx.attention_logits = patcher.make_logits("fit")
    try:
        for image, _ in loader_for(train_names, "calib"):
            image = image.to(dev).float()
            with torch.no_grad():
                model(image[:, :3], image[:, 3:], return_aux=False)
    finally:
        ctx.attention_logits = patcher.orig_logits
    h = torch.cat(patcher.pending["h"])
    e = torch.cat(patcher.pending["e"])
    patcher.pending = None
    h = h - h.mean()
    e = e - e.mean()
    calib = float((h * e).sum() / (e * e).sum().clamp_min(1e-30))
    calib_resid = float((h - calib * e).abs().max())
    print(f"[calib] a={calib:.4f} resid={calib_resid:.3e}", flush=True)

    # 2. arms
    arms = ["BASE", "NOCTX", "UNIFORM", "PRIORONLY", "DISTONLY",
            "KE", "VE", "KVE", "DE", "DECAL", "AGG", "DEAGG"]
    rows, attn_rows = [], []
    base_logit, base_prob = {}, {}

    for arm in arms:
        rec = {}
        patcher.apply(arm.lower() if arm != "BASE" else "base", calib=calib, record=rec)
        for idx, (image, target) in enumerate(loader_for(val_names, "val")):
            image = image.to(dev).float()
            target = target.to(dev).float()
            with torch.no_grad():
                out, _ = model(image[:, :3], image[:, 3:], return_aux=True)
            prob256 = F.interpolate(out[0].sigmoid(), size=(256, 256), mode="bilinear",
                                    align_corners=False)
            gt = (target > 0.5).float()
            pred = (prob256 > 0.5).float()
            tp = int(((pred > 0.5) & (gt > 0.5)).sum())
            fp = int(((pred > 0.5) & (gt <= 0.5)).sum())
            fn = int(((pred <= 0.5) & (gt > 0.5)).sum())
            cur = out[0].detach().cpu()
            if arm == "BASE":
                base_logit[idx] = cur
                base_prob = {k: v.detach().clone() for k, v in rec.items()}
                dmax = dmean = 0.0
                flip = 0.0
            else:
                ref = base_logit[idx]
                dmax = float((cur - ref).abs().max())
                dmean = float((cur - ref).abs().mean())
                flip = float((((cur > 0) != (ref > 0))).float().mean())
            rows.append(dict(arm=arm, sample_id=idx, tp=tp, fp=fp, fn=fn,
                             f1=2 * tp / max(2 * tp + fp + fn, 1),
                             recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                             pred_px=int(pred.sum()), gt_px=int(gt.sum()),
                             dlogit_max=dmax, dlogit_mean=dmean, flip=flip))
        js_vals, ent_vals = [], []
        if arm != "BASE":
            for scale in rec:
                if scale not in base_prob:
                    continue
                p = rec[scale].reshape(-1, rec[scale].shape[-1])
                q = base_prob[scale].reshape(-1, base_prob[scale].shape[-1])
                js_vals.append(float(js_distance(p, q).mean()))
                ent_vals.append(float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean()))
        attn_rows.append(dict(arm=arm, js=float(np.mean(js_vals)) if js_vals else 0.0,
                              entropy=float(np.mean(ent_vals)) if ent_vals else 0.0))
        patcher.restore()
        print(f"[arm] {arm} done", flush=True)

    def write_csv(name, data):
        with open(outdir / name, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            for d in data:
                w.writerow(d)

    write_csv("s9a_arms.csv", rows)
    write_csv("s9a_attn.csv", attn_rows)
    summary = dict(arm=args.arm, dataset=args.dataset, images=len(val_names),
                   calib_a=calib, calib_resid=calib_resid, parity=parity, arms=arms)
    (outdir / "s9a_summary.json").write_text(json.dumps(summary, indent=1))
    print("[done]", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
