"""S7: real-feature diagnostics for the Run10 ManifoldContext arms.

Runs on the training server against the frozen Run10 runtime, inference only:

    /home/yqwang/miniforge3/envs/casacd/bin/python s7_capture.py \
        --arm R_FULL-2334 --dataset SYSU-CD-256 --limit 40

No training, no writes into the frozen runtime, no test labels: only ``val``
(plus a disjoint ``train`` slice used to fit the diagnostic probes).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RUNTIME = Path("/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/runtime")
SCRIPTS = RUNTIME / "train_scripts" / "CASA-MinorGoal" / "Run10"
CKPT_ROOT = Path("/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run10")
DATA_ROOT = Path("/share_datasets/CD")
OUT_ROOT = Path("/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/analysis/simulation/S7")


def bootstrap_paths():
    sys.path.insert(0, str(RUNTIME / "models"))
    sys.path.insert(0, str(SCRIPTS))
    import config as C  # noqa: E402
    from model.casa_net import CASANet  # noqa: E402
    from train_CASA import make_loader  # noqa: E402
    return C, CASANet, make_loader


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
    q = np.percentile(a, [5, 25, 75, 95])
    mean = float(a.mean())
    return dict(n=int(a.size), mean=mean, median=float(np.median(a)),
                std=float(a.std(ddof=0)),
                cv=float(a.std(ddof=0) / mean) if abs(mean) > 1e-12 else float("nan"),
                p05=float(q[0]), p25=float(q[1]), p75=float(q[2]), p95=float(q[3]),
                vmin=float(a.min()), vmax=float(a.max()))


def radius_of(space):
    return torch.asinh(space.norm(dim=-1))


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


def fit_logistic(x, y, l2=1e-3, steps=800, lr=0.2):
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
    """Wrap one context instance: record queries, support, logits, pooling."""

    def __init__(self, context):
        self.ctx = context
        self.q = {}
        self.support = {}
        self.pool = {}
        self.guide = []
        self._saved = {}

    def __enter__(self):
        ctx = self.ctx
        self._saved["pool"] = ctx.manifold_pool
        self._saved["queries"] = ctx.project_queries
        self._saved["logits"] = ctx.attention_logits

        def pool(points, weights):
            out = self._saved["pool"](points, weights)
            self.pool.setdefault("points", []).append(points.detach())
            self.pool.setdefault("weights", []).append(weights.detach())
            self.pool.setdefault("out", []).append(out.detach())
            return out

        def queries(feature):
            q = self._saved["queries"](feature)
            self.q[feature.shape[-1]] = q.detach()
            return q

        def logits(query, key, support, query_scale):
            self.support[query_scale] = support.detach()
            return self._saved["logits"](query, key, support, query_scale)

        ctx.manifold_pool = pool
        ctx.project_queries = queries
        ctx.attention_logits = logits
        self._handle = ctx.guide.register_forward_hook(
            lambda mod, inp, out: self.guide.append(out.detach()))
        return self

    def __exit__(self, *exc):
        self.ctx.manifold_pool = self._saved["pool"]
        self.ctx.project_queries = self._saved["queries"]
        self.ctx.attention_logits = self._saved["logits"]
        self._handle.remove()
        return False


def pool_stats(points, weights, out):
    """points: (B,h,R,P,D) in the coordinate the module lifts (already scaled)."""
    b, h, r, p, d = points.shape
    lifted = torch.cat((torch.sqrt(1.0 + (points ** 2).sum(-1, keepdim=True)), points), -1).double()
    rows = []
    for cls in range(weights.shape[2]):
        w = weights[:, :, cls, :].double()
        wsum = w.sum(-1, keepdim=True)
        wnorm = (w / wsum.clamp_min(1e-12)).unsqueeze(1)          # (B,1,R,P)
        mem = radius_of(points.double())
        rho_mean = (wnorm * mem).sum(-1)
        ess = 1.0 / (wnorm ** 2).sum(-1)
        unit = points.double() / points.double().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        a_dir = (wnorm.unsqueeze(-1) * unit).sum(-2).norm(dim=-1)
        proto_index = 2 * torch.arange(r) + cls
        proto = out[:, :, proto_index, :].double()                 # (B,h,R,d+1)
        t_c, c_z = proto[..., 0], proto[..., 1:]
        rho_proto = radius_of(c_z)
        t_i = lifted[..., 0:1]
        t_c5 = t_c[:, :, :, None, None]
        cosh_d = (t_i * t_c5 - (points.double() * c_z[:, :, :, None, :]).sum(-1, keepdim=True)).clamp_min(1.0).squeeze(-1)
        d_hyp2 = torch.acosh(cosh_d).pow(2)
        d_euc = (points.double() - c_z[:, :, :, None, :]).pow(2).sum(-1)
        rows.append(dict(
            cls=cls,
            member_radius=desc(rho_mean.flatten().numpy()),
            proto_radius=desc(rho_proto.flatten().numpy()),
            ratio=float((rho_proto / rho_mean.clamp_min(1e-12)).mean()),
            cv_within=float((mem.std(-1, unbiased=False) / mem.mean(-1).clamp_min(1e-12)).mean()),
            ess=float(ess.mean()),
            a_dir=float(a_dir.mean()),
            dist_euc=float((wnorm * d_euc).sum(-1).mean()),
            dist_hyp2=float((wnorm * d_hyp2).sum(-1).mean()),
            chord=float((wnorm * 2.0 * (cosh_d - 1.0)).sum(-1).mean()),
            mass=float(wsum.mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R_FULL-2334")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--train-limit", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parity", type=int, default=4)
    ap.add_argument("--probe-cap", type=int, default=120000)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    C, CASANet, make_loader = bootstrap_paths()
    base_arm = args.arm.split("-")[0]
    ckpt = CKPT_ROOT / args.arm / args.dataset / "best_model.pth"
    if not ckpt.is_file():
        print(f"[skip] missing checkpoint {ckpt}")
        return
    outdir = OUT_ROOT
    outdir.mkdir(parents=True, exist_ok=True)

    model = CASANet(None, **C.context_kwargs(args.arm)).eval()
    missing, unexpected = model.load_state_dict(load_state(ckpt), strict=False)
    ctx = model.context
    device = torch.device(args.device)
    model.to(device)
    scales = list(map(int, ctx.retrieval_scales))
    size_to_scale = {}
    for k in range(len(ctx.grid_sizes)):
        size_to_scale[256 // (4 * (2 ** k))] = k
    print(f"[cfg] arm={args.arm} ds={args.dataset} device={device} scales={scales} "
          f"grid={list(map(int, ctx.grid_sizes))} ckpt={ckpt}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    root = DATA_ROOT / args.dataset
    val_loader = make_loader(root, "val", batch_size=1, workers=0)
    train_loader = make_loader(root, "train", batch_size=1, workers=0)

    parity = 0.0
    for i, (image, _) in enumerate(val_loader):
        if i >= args.parity:
            break
        image = image.to(device).float()
        with torch.no_grad():
            plain, _ = model(image[:, :3], image[:, 3:], return_aux=True)
            with Recorder(ctx):
                inst, _ = model(image[:, :3], image[:, 3:], return_aux=True)
        d = float((plain[0] - inst[0]).abs().max())
        parity = max(parity, d)
        print(f"[parity] {i}: max|Δlogit|={d:.3e}")
    print(f"[parity] worst={parity:.3e}")

    radius_rows, pool_rows, per_image = [], [], []
    probe = {s: {"r": [], "p": [], "y": []} for s in scales}

    def run(loader, split, limit, collect_probe):
        for idx, (image, target) in enumerate(loader):
            if idx >= limit:
                break
            image = image.to(device).float()
            target = target.to(device).float()
            rec = Recorder(ctx)
            with torch.no_grad(), rec:
                outputs, _ = model(image[:, :3], image[:, 3:], return_aux=True)
            prob = outputs[0].sigmoid()
            prob256 = F.interpolate(prob, size=(256, 256), mode="bilinear", align_corners=False)
            gt = (target > 0.5).float()
            pred = (prob256 > 0.5).float()
            row = dict(split=split, index=idx,
                       tp=int(((pred > 0.5) & (gt > 0.5)).sum()),
                       fp=int(((pred > 0.5) & (gt <= 0.5)).sum()),
                       fn=int(((pred <= 0.5) & (gt > 0.5)).sum()),
                       change_px=int(gt.sum()), pred_px=int(pred.sum()))
            per_image.append(row)

            for s_index, q in rec.q.items():
                r_map = radius_of(q[..., 1:]).mean(1)                  # (B,H,W)
                radius_rows.append(dict(split=split, scale="Q", scale_index=size_to_scale.get(s_index, -1),
                                        size=s_index, index=idx, **desc(r_map.cpu().numpy())))
                hh, ww = q.shape[2], q.shape[3]
                gt_small = F.adaptive_avg_pool2d(gt, (hh, ww))
                sidx = size_to_scale.get(s_index, -1)
                p_map = None
                if 0 <= sidx < len(rec.guide):
                    p_map = rec.guide[sidx].sigmoid().mean(1)          # (B,H,W)
                y = (gt_small > 0.5).flatten().cpu().numpy().astype(np.int64)
                if y.sum() > 0 and (y == 0).sum() > 0:
                    per_image.append(dict(split=split, index=idx, scale_index=sidx,
                                          auc_radius=auc_rank(r_map.flatten().cpu().numpy(), y),
                                          auc_guide=auc_rank(p_map.flatten().cpu().numpy(), y) if p_map is not None else float("nan")))
                if collect_probe and p_map is not None and sidx in probe:
                    probe[sidx]["r"].append(r_map.flatten().cpu().numpy())
                    probe[sidx]["p"].append(p_map.flatten().cpu().numpy())
                    probe[sidx]["y"].append(y)

            for s_index, sup in rec.support.items():
                radius_rows.append(dict(split=split, scale="support",
                                        scale_index=size_to_scale.get(s_index, -1),
                                        index=idx, **desc(sup.cpu().numpy())))
            for s_i, (pts, wts, outs) in enumerate(zip(rec.pool.get("points", []),
                                                      rec.pool.get("weights", []),
                                                      rec.pool.get("out", []))):
                kind = "K" if s_i % 2 == 0 else "V"
                for k, row in enumerate(pool_stats(pts, wts, outs)):
                    pool_rows.append(dict(split=split, index=idx, pool_scale=s_i // 2,
                                          scale_index=k, kind=kind, **row))
            print(f"[{split}] {idx} q_scales={sorted(rec.q)} guide={len(rec.guide)}")

    run(val_loader, "val", args.limit, collect_probe=False)
    run(train_loader, "train", args.train_limit, collect_probe=True)

    # ---- S7-A probes: radius vs guide probability vs both ----
    probes = []
    for s in scales:
        chunk = probe[s]
        if not chunk["r"]:
            continue
        r = np.concatenate(chunk["r"])
        p = np.concatenate(chunk["p"])
        y = (np.concatenate(chunk["y"]) > 0.5).astype(np.int64)
        if y.sum() == 0 or (y == 0).sum() == 0:
            continue
        if y.size > args.probe_cap:
            pick = np.random.default_rng(7).choice(y.size, args.probe_cap, replace=False)
            r, p, y = r[pick], p[pick], y[pick]
        mu_r, sd_r = r.mean(), r.std() + 1e-12
        mu_p, sd_p = p.mean(), p.std() + 1e-12
        feats = {
            "radius": (r[:, None] - mu_r) / sd_r,
            "guide_p": (p[:, None] - mu_p) / sd_p,
            "p_plus_radius": np.stack([(p - mu_p) / sd_p, (r - mu_r) / sd_r], 1),
        }
        half = y.size // 2
        for name, x in feats.items():
            w, b0 = fit_logistic(x[:half], y[:half])
            logit = torch.as_tensor(x[half:], dtype=torch.float64) @ w + b0
            s_val = logit.cpu().numpy()
            probes.append(dict(scale=s, probe=name, n=int(y[half:].size),
                               train_n=int(half), pos=float(y[half:].mean()),
                               auc=auc_rank(s_val, y[half:]), ap=aupr(s_val, y[half:])))
        print(f"[probe] scale={s} n={y.size} pos={y.mean():.3f}")

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

    tag = f"{args.arm}_{args.dataset}"
    write_csv(outdir / f"s7_radius_stats_{tag}.csv", radius_rows)
    write_csv(outdir / f"s7_pool_stats_{tag}.csv", pool_rows)
    write_csv(outdir / f"s7_probe_results_{tag}.csv", probes)
    write_csv(outdir / f"s7_per_image_{tag}.csv", per_image)
    summary = dict(arm=args.arm, dataset=args.dataset, ckpt=str(ckpt),
                   base_arm=base_arm, retrieval_scales=scales,
                   grid_sizes=list(map(int, ctx.grid_sizes)),
                   parity_max_abs=parity, val_images=args.limit,
                   train_images=args.train_limit, device=str(device),
                   n_radius_rows=len(radius_rows), n_pool_rows=len(pool_rows),
                   n_points=len(per_image))
    with open(outdir / f"s7_summary_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print("[done]", json.dumps(summary))


if __name__ == "__main__":
    main()
