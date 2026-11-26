"""S7c: real-feature diagnostics for Run10 ManifoldContext (capture, stats, probes).

Fixes required by the execution plan section 4:

1. sample lists are written to disk before any run (sample id + scene group id);
2. train and val use the SAME deterministic evaluation preprocessing
   (no crop / flip / rotate / exchange, no shuffle, no drop_last);
3. ``strict=True`` state-dict load plus a manifest check that ``best`` comes
   from validation;
4. train and val probe pixels are collected separately (never merged);
5. pool rows carry ``source_scale`` and ``class_id`` (the previous version
   wrote a class id into a field named ``scale_index``);
6. the recorder is optional and always forwards the original return values;
7. per-checkpoint parity on >= 4 images: four-level outputs, aux, binary diff,
   prototype K/V, support and attention statistics.

Everything is inference only, FP32, TF32 disabled.  No test split is loaded.

    python s7c_capture.py --arm R_FULL-2334 --dataset SYSU-CD-256 --val-limit 120
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

RUNTIME = Path("/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/runtime")
SCRIPTS = RUNTIME / "train_scripts" / "CASA-MinorGoal" / "Run10"
CKPT_ROOT = Path("/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run10")
DATA_ROOT = Path("/share_datasets/CD")
OUT_ROOT = Path("/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/analysis/simulation/S7c")

SEED = 2334


def bootstrap_paths():
    sys.path.insert(0, str(RUNTIME / "models"))
    sys.path.insert(0, str(SCRIPTS))
    import config as C
    from dataset.dataset import Dataset
    from dataset import casa_transforms as transforms
    from model.casa_net import CASANet
    import train_CASA as T
    return C, CASANet, Dataset, transforms, T


def scene_id(name):
    """Group id for spatial grouping: strip the trailing '_k' crop index."""
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) <= 2:
        return "_".join(parts[:-1])
    return stem


def load_state(path):
    obj = torch.load(path, map_location="cpu")
    for key in ("model", "state_dict", "net", "model_state"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {k.replace("module.", "", 1): v for k, v in obj.items()}


def desc(values):
    a = np.asarray(values, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0)
    q = np.percentile(a, [5, 50, 95])
    mean = float(a.mean())
    return dict(n=int(a.size), mean=mean, median=float(q[1]),
                std=float(a.std(ddof=0)),
                cv=float(a.std(ddof=0) / mean) if abs(mean) > 1e-12 else float("nan"),
                p05=float(q[0]), p95=float(q[2]),
                vmin=float(a.min()), vmax=float(a.max()))


def radius_of(space):
    """Intrinsic Lorentz radius of the ACTUAL geometry coordinate z.

    ``space`` is the spatial part of a Lorentz point that was already scaled
    by the module (Q/K divided by sqrt(d) once, V not).  Dividing by
    ``sqrt(d)`` again is the S7b bug: it produced radii smaller by that factor.
    """
    return torch.asinh(space.double().norm(dim=-1))


def radius_legacy(space):
    """S7b (wrong) convention, kept only to quantify the old bias."""
    return torch.asinh(space.double().norm(dim=-1) / math.sqrt(space.shape[-1]))


def radius_from_time(time):
    """Independent cross-check: rho = acosh(t) for a point on the hyperboloid."""
    return torch.acosh(time.double().clamp_min(1.0))


def auc_rank(score, label):
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label, dtype=np.int64).ravel()
    pos = label == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    ordered = score[order]
    i = 0
    while i < score.size:
        j = i
        while j + 1 < score.size and ordered[j + 1] == ordered[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def aupr(score, label):
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label, dtype=np.int64).ravel()
    if label.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    lab = label[order]
    tp = np.cumsum(lab)
    precision = tp / np.arange(1, lab.size + 1)
    return float((precision * lab).sum() / lab.sum())


def fit_logistic(x, y, l2=1e-3, steps=600, lr=0.25):
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


class Recorder:
    """Optional wrapper: records pooling inputs/outputs, queries, support, guide."""

    def __init__(self, context, enabled=True):
        self.ctx = context
        self.enabled = enabled
        self.q = {}
        self.support = {}
        self.pool = []
        self.guide = []
        self.logits = {}
        self.tau = []
        self._saved = {}

    def __enter__(self):
        if not self.enabled:
            return self
        ctx = self.ctx
        self._saved["pool"] = ctx.manifold_pool
        self._saved["queries"] = ctx.project_queries
        self._saved["logits"] = ctx.attention_logits

        def pool(points, weights):
            out = self._saved["pool"](points, weights)
            self.pool.append((points.detach(), weights.detach(), out.detach()))
            return out

        def queries(feature):
            q = self._saved["queries"](feature)
            self.q[int(feature.shape[-1])] = q.detach()
            return q

        def logits(query, key, support, query_scale):
            out = self._saved["logits"](query, key, support, query_scale)
            self.support[int(query_scale)] = support.detach()
            self.logits[int(query_scale)] = out.detach()
            return out

        ctx.manifold_pool = pool
        ctx.project_queries = queries
        ctx.attention_logits = logits
        self._handle = ctx.guide.register_forward_hook(
            lambda mod, inp, out: self.guide.append(out.detach()))
        return self

    def __exit__(self, *exc):
        if not self.enabled:
            return False
        self.ctx.manifold_pool = self._saved["pool"]
        self.ctx.project_queries = self._saved["queries"]
        self.ctx.attention_logits = self._saved["logits"]
        self._handle.remove()
        return False


def pool_rows_for(points, weights, out, scale, kind):
    """Per-class pooling statistics for one manifold_pool call.

    Frozen-runtime layout (verified by s7b probes): ``points`` is
    (B, h, R, P, D) (already scaled the way the module scales it), ``weights``
    is (B, R, 2, P) and ``out`` is (B, h, 2R, D+1) with prototype order
    ``[region0-class0, region0-class1, region1-class0, ...]``.
    """
    assert points.dim() == 5 and out.dim() == 4, (points.shape, out.shape)
    b, h, r, p, d = points.shape
    if weights.dim() == 4:
        weights = weights[:, None].expand(b, h, weights.shape[1], weights.shape[2], weights.shape[3])
    assert weights.shape[1] == h and weights.shape[2] == r and weights.shape[4] == p
    assert out.shape[2] == 2 * r and out.shape[-1] == d + 1
    pts = points.double().reshape(b * h, r, p, d)
    proto = out.double().reshape(b * h, 2 * r, d + 1)
    t_pts = torch.sqrt(1.0 + (pts ** 2).sum(-1)).reshape(b * h, r, p)
    unit = pts / pts.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    mem_rho = radius_of(points).reshape(b * h, r, p)
    mem_legacy = radius_legacy(points).reshape(b * h, r, p)
    rows = []
    n_cls = 2
    for cls in range(n_cls):
        idx = torch.arange(r, device=out.device) * 2 + cls
        if idx.numel() and int(idx.max()) >= proto.shape[1]:
            raise RuntimeError(
                f"pool idx out of range: proto={tuple(proto.shape)} r={r} "
                f"idx_max={int(idx.max())} weights={tuple(weights.shape)} "
                f"points={tuple(points.shape)} out={tuple(out.shape)}")
        sel = proto[:, idx, :]
        t_c = sel[..., 0].reshape(b * h, r)
        z_c = sel[..., 1:].reshape(b * h, r, d)
        w = weights[:, :, :, cls, :].double().reshape(b * h, r, p)
        wsum = w.sum(-1, keepdim=True)
        wnorm = w / wsum.clamp_min(1e-12)
        rho_member = (wnorm * mem_rho).sum(-1)
        ess = 1.0 / (wnorm ** 2).sum(-1)
        a_dir = (wnorm.unsqueeze(-1) * unit).sum(-2).norm(dim=-1)
        rho_proto = radius_of(sel[..., 1:].reshape(b * h, r, d))
        rho_proto_time = radius_from_time(t_c)
        cosh_d = (t_pts * t_c.unsqueeze(-1)
                  - (pts * z_c.unsqueeze(-2)).sum(-1)).clamp_min(1.0)
        d_hyp2 = torch.acosh(cosh_d).pow(2)
        d_euc = (pts - z_c.unsqueeze(-2)).pow(2).sum(-1)
        rows.append(dict(
            kind=kind, source_scale=int(scale), class_id=int(cls),
            **{"mem_" + k: v for k, v in desc(rho_member.numpy()).items()},
            **{"memraw_" + k: v for k, v in desc(mem_rho.numpy()).items()},
            **{"memlegacy_" + k: v for k, v in desc(mem_legacy.numpy()).items()},
            **{"proto_" + k: v for k, v in desc(rho_proto.numpy()).items()},
            time_check=float((rho_proto_time - rho_proto).abs().max()),
            ratio=float((w * rho_proto.unsqueeze(-1)).sum(-1).div(
                (w * mem_rho).sum(-1).clamp_min(1e-12)).mean()),
            cv_within_cell=float((mem_rho.std(-1, unbiased=False) /
                                  mem_rho.mean(-1).clamp_min(1e-12)).mean()),
            cv_across_cells=float(rho_member.std(unbiased=False) /
                                  rho_member.mean().clamp_min(1e-12)),
            ess=float(ess.mean()), pixel_count=float(p),
            ess_frac=float(ess.mean() / p),
            a_dir=float(a_dir.mean()),
            dist_euc=float((wnorm * d_euc).sum(-1).mean()),
            dist_hyp2=float((wnorm * d_hyp2).sum(-1).mean()),
            chord=float((wnorm * 2.0 * (cosh_d - 1.0)).sum(-1).mean()),
            mass_raw=float(wsum.mean()),
            mass_norm=float((wsum / wsum.sum(-1, keepdim=True)).mean())))
    return rows


def dump_tensors(path, rec, scale_of_size, tau, gt, pred, prob):
    payload = dict(gt=gt.astype(np.uint8), pred=pred.astype(np.uint8),
                   prob=prob.astype(np.float32))
    for sidx, q in rec.q.items():
        s = scale_of_size.get(sidx, -1)
        payload[f"q_z_s{s}"] = q[..., 1:].float().numpy().astype(np.float32)
        payload[f"q_t_s{s}"] = q[..., 0].float().numpy().astype(np.float32)
    for s, sup in rec.support.items():
        payload[f"support_s{s}"] = sup.float().numpy().astype(np.float32)
    for s, lg in rec.logits.items():
        payload[f"logits_s{s}"] = lg.float().numpy().astype(np.float32)
    for i, (pts, wts, outs) in enumerate(rec.pool):
        kind = "K" if i % 2 == 0 else "V"
        s = i // 2
        payload[f"{kind}_member_z_s{s}"] = pts[..., :].float().numpy().astype(np.float32)
        payload[f"{kind}_member_w_s{s}"] = wts.float().numpy().astype(np.float32)
        payload[f"{kind}_proto_s{s}"] = outs.float().numpy().astype(np.float32)
    payload["tau"] = np.asarray(tau, dtype=np.float32)
    np.savez_compressed(path, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R_FULL-2334")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--val-limit", type=int, default=120)
    ap.add_argument("--train-limit", type=int, default=48)
    ap.add_argument("--dump-images", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parity", type=int, default=4)
    ap.add_argument("--probe-cap", type=int, default=60000)
    args = ap.parse_args()

    threads = int(os.environ.get("S7B_THREADS", "8"))
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    C, CASANet, Dataset, transforms, T = bootstrap_paths()
    ckpt = CKPT_ROOT / args.arm / args.dataset / "best_model.pth"
    if not ckpt.is_file():
        print(f"[skip] missing checkpoint {ckpt}")
        return
    outdir = OUT_ROOT / f"{args.arm}_{args.dataset}"
    outdir.mkdir(parents=True, exist_ok=True)
    root = DATA_ROOT / args.dataset

    # ---- 1. sample lists first (sample id + scene group id) ----
    samples = {}
    for split, limit in (("val", args.val_limit), ("train", args.train_limit)):
        names = [ln.strip() for ln in (root / "list" / f"{split}.txt").read_text().splitlines()
                 if ln.strip()]
        rng = np.random.default_rng(SEED + (0 if split == "val" else 1))
        pool_idx = np.sort(rng.choice(len(names), size=min(len(names), max(limit * 6, 300)),
                                      replace=False))
        info = []
        for i in pool_idx:
            name = names[int(i)]
            mask = cv2.imread(str(root / "label" / name), 0)
            change_px = int((mask >= 128).sum())
            frac = change_px / float(mask.size)
            info.append(dict(list_index=int(i), filename=name, change_px=change_px,
                             change_frac=frac, scene_id=scene_id(name)))
        # stratified pick by change fraction quantiles, deterministic order
        info.sort(key=lambda r: (r["change_frac"], r["filename"]))
        n = len(info)
        edges = [0.0, 0.005, 0.05, 0.20, 1.01]
        strata = {"bg": [], "small": [], "mid": [], "large": []}
        for r in info:
            key = ("bg" if r["change_frac"] <= edges[1] else
                   "small" if r["change_frac"] <= edges[2] else
                   "mid" if r["change_frac"] <= edges[3] else "large")
            strata[key].append(r)
        quota = {"bg": 0.25, "small": 0.30, "mid": 0.25, "large": 0.20}
        chosen = []
        for key, rows_k in strata.items():
            take = int(round(limit * quota[key])) if rows_k else 0
            step = max(1, len(rows_k) // max(take, 1))
            chosen.extend(rows_k[::step][:take])
        chosen = chosen[:limit]
        chosen.sort(key=lambda r: (r["scene_id"], r["filename"]))
        rows = []
        for k, r in enumerate(chosen):
            rows.append(dict(split=split, sample_id=k, list_index=r["list_index"],
                             filename=r["filename"], scene_id=r["scene_id"],
                             change_px=r["change_px"], change_frac=r["change_frac"],
                             stratum=("bg" if r["change_frac"] <= edges[1] else
                                      "small" if r["change_frac"] <= edges[2] else
                                      "mid" if r["change_frac"] <= edges[3] else "large")))
        samples[split] = rows
        print(f"[samples] {split}: {len(rows)} picked from {len(names)}")
    with open(outdir / "samples.json", "w") as fh:
        json.dump(samples, fh, indent=1)

    # ---- 2. deterministic evaluation preprocessing for both splits ----
    eval_transform = transforms.Compose([transforms.Scale(256, 256),
                                         transforms.Normalize(T.MEAN, T.STD),
                                         transforms.ToTensor()])

    def make_split_loader(split):
        names = [r["filename"] for r in samples[split]]
        listing = outdir / f"_list_{split}.txt"
        listing.write_text("\n".join(names) + "\n")
        ds = Dataset(str(root), str(listing), transform=eval_transform)
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                           num_workers=0, drop_last=False)

    # ---- 3. strict load + manifest check ----
    model = CASANet(None, **C.context_kwargs(args.arm)).eval()
    state = load_state(ckpt)
    report = model.load_state_dict(state, strict=True)
    print(f"[ckpt] strict load ok: {report} threads={threads}")
    manifest = {}
    mpath = ckpt.parent / "run_manifest.json"
    if mpath.is_file():
        manifest = json.loads(mpath.read_text())
    selection = manifest.get("selection_split", "unknown")
    print(f"[ckpt] selection_split={selection} selection={manifest.get('selection')} "
          f"independent_test={manifest.get('independent_test')} protocol={manifest.get('protocol')}")
    ctx = model.context
    device = torch.device(args.device)
    model.to(device)
    scales = list(map(int, ctx.retrieval_scales))
    scale_of_size = {256 // (4 * (2 ** k)): k for k in range(len(ctx.grid_sizes))}
    tau = []
    try:
        for s_i in range(len(ctx.grid_sizes)):
            value = ctx.scale_temperature(s_i)
            if torch.is_tensor(value):
                value = float(value.detach().reshape(-1)[0])
            tau.append(float(value))
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[warn] could not read per-scale temperature: {exc}")

    # ---- 4. parity ----
    val_loader = make_split_loader("val")
    def flat_aux(obj, prefix=""):
        out = {}
        if torch.is_tensor(obj):
            out[prefix] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                out.update(flat_aux(v, f"{prefix}/{k}"))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                out.update(flat_aux(v, f"{prefix}[{i}]"))
        return out

    parity = {"logit": 0.0, "logit_levels": 0.0, "aux": 0.0, "binary": 0.0,
              "kv": 0.0, "support": 0.0, "n_aux": 0, "n_kv": 0}
    for i, (image, _) in enumerate(val_loader):
        if i >= args.parity:
            break
        image = image.to(device).float()
        with torch.no_grad():
            plain, aux_plain = model(image[:, :3], image[:, 3:], return_aux=True)
            prev = ctx.collect_diag
            ctx.collect_diag = True
            try:
                rec_a = Recorder(ctx)
                with rec_a:
                    inst_a, aux_a = model(image[:, :3], image[:, 3:], return_aux=True)
                rec_b = Recorder(ctx)
                with rec_b:
                    inst_b, aux_b = model(image[:, :3], image[:, 3:], return_aux=True)
            finally:
                ctx.collect_diag = prev
        parity["logit"] = max(parity["logit"], float((plain[0] - inst_a[0]).abs().max()))
        for lvl in range(1, len(plain)):
            parity["logit_levels"] = max(parity["logit_levels"],
                                         float((plain[lvl] - inst_a[lvl]).abs().max()))
        parity["binary"] = max(parity["binary"],
                               float(((plain[0] > 0) != (inst_a[0] > 0)).float().mean()))
        fa, fb = flat_aux(aux_a), flat_aux(aux_b)
        for k in fa:
            if k in fb and fa[k].shape == fb[k].shape:
                parity["aux"] = max(parity["aux"], float((fa[k] - fb[k]).abs().max()))
                parity["n_aux"] += 1
        for (pa, wa, oa), (pb, wb, ob) in zip(rec_a.pool, rec_b.pool):
            parity["kv"] = max(parity["kv"], float((oa - ob).abs().max()),
                               float((wa - wb).abs().max()))
            parity["n_kv"] += 1
        for s, sup in rec_a.support.items():
            if s in rec_b.support:
                parity["support"] = max(parity["support"],
                                        float((sup - rec_b.support[s]).abs().max()))
        print(f"[parity] {i} logit={parity['logit']:.3e} kv={parity['kv']:.3e} "
              f"support={parity['support']:.3e} aux={parity['aux']:.3e}")
    print("[parity]", json.dumps(parity))
    with open(outdir / "parity.json", "w") as fh:
        json.dump(parity, fh, indent=1)

    radius_rows, pool_rows, per_image = [], [], []
    probe = {split: {s: {"logit": [], "p": [], "r": [], "raw": [], "dir": [], "z": [],
                         "y": [], "sid": []}
                     for s in scales} for split in ("train", "val")}
    dumped = 0

    def run(split, limit):
        nonlocal dumped
        loader = make_split_loader(split)
        for idx, (image, target) in enumerate(loader):
            if idx >= limit:
                break
            image = image.to(device).float()
            target = target.to(device).float()
            ctx.collect_diag = True
            rec = Recorder(ctx)
            with torch.no_grad(), rec:
                outputs, _ = model(image[:, :3], image[:, 3:], return_aux=True)
            prob = outputs[0].sigmoid()
            prob256 = F.interpolate(prob, size=(256, 256), mode="bilinear", align_corners=False)
            gt = (target > 0.5).float()
            pred = (prob256 > 0.5).float()
            tp = int(((pred > 0.5) & (gt > 0.5)).sum())
            fp = int(((pred > 0.5) & (gt <= 0.5)).sum())
            fn = int(((pred <= 0.5) & (gt > 0.5)).sum())
            tn = int(((pred <= 0.5) & (gt <= 0.5)).sum())
            per_image.append(dict(split=split, sample_id=idx, tp=tp, fp=fp, fn=fn, tn=tn,
                                  change_px=int(gt.sum()), pred_px=int(pred.sum()),
                                  prob_change=float(((prob256 > 0.5) & (gt > 0.5)).sum() /
                                                    max(int(gt.sum()), 1))))
            if dumped < args.dump_images and split == "val":
                dump_tensors(outdir / f"dump_{split}_{idx:04d}.npz", rec, scale_of_size, tau,
                             gt[0, 0].cpu().numpy(), pred[0, 0].cpu().numpy(),
                             prob256[0, 0].cpu().numpy())
                dumped += 1

            for size, q in rec.q.items():
                s = scale_of_size.get(size, -1)
                qz = q[..., 1:]
                r_map = radius_of(qz).mean(1)
                radius_rows.append(dict(split=split, kind="Q", source_scale=s, sample_id=idx,
                                        **desc(r_map.cpu().numpy())))
                hh, ww = q.shape[2], q.shape[3]
                gt_small = F.adaptive_avg_pool2d(gt, (hh, ww))
                y = (gt_small[0, 0] > 0.5).flatten().cpu().numpy().astype(np.int64)
                p_map = rec.guide[s].sigmoid().mean(1) if s < len(rec.guide) else None
                if p_map is not None and y.min() != y.max() and s in probe[split]:
                    r_vec = r_map.flatten().cpu().numpy()
                    p_vec = p_map.flatten().cpu().numpy()
                    raw = qz.norm(dim=-1).mean(1).flatten().cpu().numpy()
                    unit = qz / qz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    wgt = (p_map[:, None] - 0.5).unsqueeze(-1)
                    mdir = (unit * wgt).sum((1, 2, 3), keepdim=True)
                    mdir = mdir / mdir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    d_vec = (unit * mdir).sum(-1).flatten().cpu().numpy()
                    if split == "val":
                        per_image.append(dict(split=split, sample_id=idx, source_scale=s,
                                              auc_radius=auc_rank(r_vec, y),
                                              auc_guide=auc_rank(p_vec, y)))
                    bkt = probe[split][s]
                    bkt["logit"].append(np.log(np.clip(p_vec, 1e-6, 1 - 1e-6) /
                                               np.clip(1 - p_vec, 1e-6, 1 - 1e-6)))
                    bkt["p"].append(p_vec)
                    bkt["r"].append(r_vec)
                    bkt["raw"].append(raw)
                    bkt["dir"].append(d_vec)
                    zz = unit.squeeze(1) if unit.dim() == 5 else unit
                    bkt["z"].append(zz.permute(0, 2, 3, 1).reshape(-1, zz.shape[-1])
                                    .cpu().numpy().astype(np.float32))
                    bkt["y"].append(y)
                    bkt["sid"].append(np.full(y.size, idx, dtype=np.int32))

            for s, sup in rec.support.items():
                radius_rows.append(dict(split=split, kind="support", source_scale=int(s),
                                        sample_id=idx, **desc(sup.cpu().numpy())))
            for i, (pts, wts, outs) in enumerate(rec.pool):
                kind = "K" if i % 2 == 0 else "V"
                try:
                    rows_i = pool_rows_for(pts, wts, outs, i // 2, kind)
                except Exception:
                    print(f"[pool-shape] i={i} kind={kind} points={tuple(pts.shape)} "
                          f"weights={tuple(wts.shape)} out={tuple(outs.shape)}")
                    raise
                for row in rows_i:
                    row.update(split=split, sample_id=idx)
                    pool_rows.append(row)
            if idx % 10 == 0:
                print(f"[{split}] {idx} q_scales={sorted(rec.q)} pools={len(rec.pool)}")

    run("val", args.val_limit)
    run("train", args.train_limit)

    # ---- 5. probes: fit on train, evaluate on val (never merged) ----
    def stack(split, s, key):
        vals = probe[split][s][key]
        return np.concatenate(vals) if vals else None

    def boot_ci(sid, y, s_score, b_score, n_boot=2000, seed=0):
        rng = np.random.default_rng(seed)
        uniq = np.unique(sid)
        d = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.choice(uniq, uniq.size, replace=True)
            idx = np.concatenate([np.where(sid == u)[0] for u in pick]) if uniq.size < 40 else None
            if idx is None:
                w = rng.multinomial(uniq.size, np.full(uniq.size, 1.0 / uniq.size))
                keep = np.isin(sid, uniq[w > 0])
                idx = np.where(keep)[0]
            d[i] = auc_rank(s_score[idx], y[idx]) - auc_rank(b_score[idx], y[idx])
        return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))

    probes, paired = [], []
    for s in scales:
        y_tr = stack("train", s, "y")
        y_va = stack("val", s, "y")
        if y_tr is None or y_va is None:
            print(f"[probe] scale={s} skipped (missing split)")
            continue
        if y_tr.min() == y_tr.max() or y_va.min() == y_va.max():
            print(f"[probe] scale={s} skipped (single-class split)")
            continue
        rng = np.random.default_rng(SEED + s)
        arrays = {}
        for split in ("train", "val"):
            cols = {k: stack(split, s, k) for k in ("logit", "p", "r", "raw", "dir", "z", "y", "sid")}
            n = cols["y"].size
            if n > args.probe_cap:
                keep = np.sort(rng.choice(n, args.probe_cap, replace=False))
                cols = {k: v[keep] for k, v in cols.items()}
            arrays[split] = cols
        # independent direction feature fitted on TRAIN pixels only
        ztr, ytr = arrays["train"]["z"], arrays["train"]["y"]
        c1 = ztr[ytr == 1].mean(0)
        c0 = ztr[ytr == 0].mean(0)
        c1 /= max(np.linalg.norm(c1), 1e-12)
        c0 /= max(np.linalg.norm(c0), 1e-12)
        for split in ("train", "val"):
            zz = arrays[split]["z"]
            arrays[split]["dirfit"] = zz @ c1 - zz @ c0
        x_tr = {k: arrays["train"][k] for k in ("logit", "p", "r", "raw", "dir", "dirfit")}
        x_va = {k: arrays["val"][k] for k in ("logit", "p", "r", "raw", "dir", "dirfit")}
        y_tr, y_va = arrays["train"]["y"], arrays["val"]["y"]
        sid_va = arrays["val"]["sid"]
        mu = {k: (float(x_tr[k].mean()), float(x_tr[k].std() + 1e-12)) for k in x_tr}
        zs = {k: ((x - mu[k][0]) / mu[k][1])[:, None] for k, x in x_tr.items()}
        zv = {k: ((x - mu[k][0]) / mu[k][1])[:, None] for k, x in x_va.items()}
        feats = {
            "P0_guide_prob": (zs["p"], zv["p"]),
            "P1_guide_logit": (zs["logit"], zv["logit"]),
            "P2_radius": (zs["r"], zv["r"]),
            "P3_logit_radius": (np.hstack([zs["logit"], zs["r"]]),
                                np.hstack([zv["logit"], zv["r"]])),
            "P4_logit_rawnorm": (np.hstack([zs["logit"], zs["raw"]]),
                                 np.hstack([zv["logit"], zv["raw"]])),
            "P5_dir_indep": (zs["dirfit"], zv["dirfit"]),
            "P5b_dir_guide_only": (zs["dir"], zv["dir"]),
            "P6_logit_dir_indep": (np.hstack([zs["logit"], zs["dirfit"]]),
                                   np.hstack([zv["logit"], zv["dirfit"]])),
        }
        scores = {}
        for name, (xtr, xva) in feats.items():
            w, b0 = fit_logistic(xtr, y_tr)
            score = (torch.as_tensor(xva, dtype=torch.float64) @ w + b0).numpy()
            scores[name] = score
            probes.append(dict(source_scale=s, probe=name, n_train=int(y_tr.size),
                               n_val=int(y_va.size), n_images=int(np.unique(sid_va).size),
                               pos_train=float(y_tr.mean()), pos_val=float(y_va.mean()),
                               auc=auc_rank(score, y_va), ap=aupr(score, y_va)))
            print(f"[probe] s={s} {name} auc={probes[-1]['auc']:.4f} ap={probes[-1]['ap']:.4f}")
        for name in ("P2_radius", "P3_logit_radius", "P4_logit_rawnorm",
                     "P5_dir_indep", "P6_logit_dir_indep"):
            lo, hi = boot_ci(sid_va, y_va, scores[name], scores["P1_guide_logit"], seed=SEED + s)
            paired.append(dict(source_scale=s, contrast=f"{name}-P1_guide_logit",
                               delta_auc=auc_rank(scores[name], y_va) -
                               auc_rank(scores["P1_guide_logit"], y_va),
                               ci_lo=lo, ci_hi=hi, n_boot=2000))
            print(f"[paired] s={s} {paired[-1]['contrast']} "
                  f"d={paired[-1]['delta_auc']:+.4f} CI=({lo:+.4f},{hi:+.4f})")

    def write_csv(path, rows):
        if not rows:
            return
        keys = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            for row in rows:
                wr.writerow(row)

    write_csv(outdir / "radius_stats.csv", radius_rows)
    write_csv(outdir / "pool_stats.csv", pool_rows)
    write_csv(outdir / "probe_scores.csv", probes)
    write_csv(outdir / "probe_paired_ci.csv", paired)
    write_csv(outdir / "per_image.csv", per_image)
    summary = dict(arm=args.arm, dataset=args.dataset, ckpt=str(ckpt),
                   radius_convention="asinh(||z||) on the actual geometry coordinate",
                   selection_split=selection, independent_test=manifest.get("independent_test"),
                   protocol=manifest.get("protocol"), parity=parity, retrieval_scales=scales,
                   grid_sizes=list(map(int, ctx.grid_sizes)), tau=tau,
                   val_images=args.val_limit, train_images=args.train_limit,
                   dump_images=dumped, scratch=str(outdir))
    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print("[done]", json.dumps(summary))


if __name__ == "__main__":
    main()
