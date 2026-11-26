#!/usr/bin/env python
"""S8b-final: same-budget adaptive prototype partition on real features.

Only the *representation fidelity* layer (S8b-A) is computed here; the
downstream (S9b) layer is reported by s9a_final.py's grids mode, which already
shows that FIX-32 ~ FIX-224 at the task level.

What this script adds over the V5 `s8b_partition.py`:

* membership is rebuilt from the online **raw guide probability** p (and 1-p),
  never from a previously normalised grid weight;
* the split objective uses a globally fixed per-scale class mass
  ``M_s,c = sum over the whole scale`` so that ``gain = J(R) - J(R1) - J(R2)``
  is additive and comparable across regions;
* the quantity that is optimised (J with the E-optimal or H-optimal centre) is
  reported separately from the quantity that is finally evaluated (the *actual*
  Lorentz prototype that the module would emit);
* every split candidate is a dyadic midline cut, so the fixed FIX grid is
  itself a member of the candidate set -- the adaptive arm therefore cannot
  lose to FIX inside its own objective, which is used as a sanity check;
* a dynamic-programming partition with the same leaf budget gives the optimum
  of that objective on the same tree and bounds what greedy could reach.

Arms: FIX / RAND / ADA-K-E / ADA-K-H / ADA-V-E / ADA-V-H  (+ DP bound).
Both K and V sides are evaluated with the same metrics for every arm.

Usage (server, conda env casacd):
  python s8b_final.py --arm R_FULL-2334 --dataset SYSU-CD-256 --images 8 \
      --out ./S8bFinal
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
SIM_DIR = HERE.parent
SPEC = importlib.util.spec_from_file_location("s9a_final", str(SIM_DIR / "s9a_final.py"))
S9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S9)          # reuse model loading / dataset helpers
S7 = S9.S7

torch.set_grad_enabled(False)

ARMS = ["FIX", "RAND", "ADA-K-E", "ADA-K-H", "ADA-V-E", "ADA-V-H", "DP-K-E", "DP-K-H"]


# ---------------------------------------------------------------- geometry --
def lift_np(z):
    t2 = (z * z).sum(-1, keepdim=True)
    return torch.cat((torch.sqrt(t2 + 1.0), z), dim=-1)


def lorentz_center(z, m):
    """OEM alpha=0 centre (module's manifold_pool) as a spatial vector."""
    s = (m[:, None] * lift_np(z)).sum(0)
    n2 = (s[0] * s[0] - (s[1:] * s[1:]).sum()).clamp_min(1e-9)
    return s[1:] / torch.sqrt(n2)


def hypo_sq(z, c):
    """squared geodesic distance from members ``z`` to centre ``c``."""
    tz = (z * z).sum(-1).clamp_min(0)
    tc = (c * c).sum().clamp_min(0)
    cos = (z * c).sum(-1) / torch.sqrt((tz + 1.0) * (tc + 1.0)).clamp_min(1e-9)
    ch = torch.sqrt(1.0 + tz) * math.sqrt(1.0 + float(tc)) - torch.sqrt(tz) * \
        torch.sqrt(tc) * cos
    return torch.acosh(ch.clamp_min(1.0)) ** 2


def metrics(z, m, c):
    """weighted fidelity of members ``z`` (against centre ``c``)."""
    tot = m.sum().clamp_min(1e-9)
    w = m / tot
    d2 = ((z - c) ** 2).sum(-1)
    h2 = hypo_sq(z, c)
    chord = 2.0 * (torch.cosh(torch.sqrt(h2)) - 1.0)
    cosang = (z * c).sum(-1) / torch.sqrt(
        ((z * z).sum(-1) + 1.0) * ((c * c).sum() + 1.0)).clamp_min(1e-9)
    return dict(sse=float((w * d2).sum()), hyp2=float((w * h2).sum()),
                chord=float((w * chord).sum()), cos=float((w * cosang).sum()))


# ---------------------------------------------------------------- splits ----
def is_split_ok(r, min_side):
    x0, x1, y0, y1 = r
    return (x1 - x0) >= 2 * min_side and (y1 - y0) >= 2 * min_side


def splits(r, min_side):
    x0, x1, y0, y1 = r
    out = []
    if (x1 - x0) >= 2 * min_side:
        xm = (x0 + x1) // 2
        out.append(((x0, xm, y0, y1), (xm, x1, y0, y1)))
    if (y1 - y0) >= 2 * min_side:
        ym = (y0 + y1) // 2
        out.append(((x0, x1, y0, ym), (x0, x1, ym, y1)))
    return out


class ScaleData:
    """members + raw p for one scale, plus O(1) rectangle statistics."""

    def __init__(self, zk, zv, p, h, w):
        self.zk, self.zv, self.p = zk, zv, p
        self.h, self.w = h, w
        self.shape = h * w

    def idx(self, r):
        x0, x1, y0, y1 = r
        yy = torch.arange(y0, y1)
        xx = torch.arange(x0, x1)
        return (yy[:, None] * self.w + xx[None, :]).reshape(-1)

    def class_stats(self, r, cls):
        idx = self.idx(r)
        m = self.p[idx] if cls == 0 else 1.0 - self.p[idx]
        return idx, m

    def cost(self, r, cls, side, kind):
        """J for one rectangle/class: smaller is better, additive in the split."""
        idx, m = self.class_stats(r, cls)
        M = m.sum()
        if float(M) <= 1e-8:
            return 0.0, M
        z = self.zk[idx] if side == "K" else self.zv[idx]
        if kind == "E":
            B = (m[:, None] * z).sum(0)
            C = (m * (z * z).sum(-1)).sum()
            return float(C - (B * B).sum() / M), M
        S = (m[:, None] * lift_np(z)).sum(0)
        a2 = (S[0] * S[0] - (S[1:] * S[1:]).sum()).clamp_min(0.0)
        a = torch.sqrt(a2)
        return float(2.0 * (a - M)), M


def partition_costs(sd, rects, side, kind):
    return [sd.cost(r, c, side, kind) for r in rects for c in (0, 1)]


def objective(sd, rects, side, kind, mass_ref):
    """sum over rects/classes of J / M_s,c  (mass_ref fixed for the scale)."""
    tot = 0.0
    for r in rects:
        for c in (0, 1):
            j, _ = sd.cost(r, c, side, kind)
            tot += j / max(mass_ref[c], 1e-9)
    return tot


def greedy_partition(sd, leaves, side, kind):
    rects = [(0, sd.w, 0, sd.h)]
    while len(rects) < leaves:
        best = None
        for i, r in enumerate(rects):
            for r1, r2 in splits(r, max(2, min(sd.h, sd.w) // 8)):
                d = (sd.cost(r, 0, side, kind)[0] + sd.cost(r, 1, side, kind)[0]) \
                    - (sd.cost(r1, 0, side, kind)[0] + sd.cost(r1, 1, side, kind)[0]) \
                    - (sd.cost(r2, 0, side, kind)[0] + sd.cost(r2, 1, side, kind)[0])
                if best is None or d > best[0] + 1e-12:
                    best = (d, i, r1, r2)
        if best is None:
            break
        _, i, r1, r2 = best
        rects = rects[:i] + [r1, r2] + rects[i + 1:]
    return rects


def random_partition(sd, leaves, rng, min_side):
    rects = [(0, sd.w, 0, sd.h)]
    guard = 0
    while len(rects) < leaves and guard < 1000:
        guard += 1
        cand = [i for i, r in enumerate(rects) if splits(r, min_side)]
        if not cand:
            break
        i = cand[rng.randrange(len(cand))]
        sp = splits(rects[i], min_side)
        r1, r2 = sp[rng.randrange(len(sp))]
        rects = rects[:i] + [r1, r2] + rects[i + 1:]
    return rects


def dp_partition(sd, leaves, side, kind):
    """Optimal dyadic partition with `leaves` leaves (diagnostic bound)."""
    from functools import lru_cache

    min_side = max(2, min(sd.h, sd.w) // 8)

    @lru_cache(maxsize=None)
    def best(r, k):
        j = sum(sd.cost(r, c, side, kind)[0] for c in (0, 1))
        if k == 1:
            return j, (r,)
        out = (float("inf"), None)
        for r1, r2 in splits(r, min_side):
            for k1 in range(1, k):
                v1, p1 = best(r1, k1)
                v2, p2 = best(r2, k - k1)
                if v1 + v2 < out[0]:
                    out = (v1 + v2, p1 + p2)
        return out

    val, rects = best((0, sd.w, 0, sd.h), leaves)
    return list(rects)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R_FULL-2334")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--leaves", type=int, default=4)
    ap.add_argument("--fixgrid", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("S7B_THREADS", "6")))
    torch.manual_seed(S7.SEED)
    rng = __import__("random").Random(args.seed)

    C, CASANet, Dataset, transforms, T = S7.bootstrap_paths()
    from model.layers.pyramid_context import channel_norm, pool_features  # noqa
    outdir = Path(args.out) if args.out else (S7.OUT_ROOT.parent / "S8bFinal")
    outdir.mkdir(parents=True, exist_ok=True)
    root = S7.DATA_ROOT / args.dataset
    tag = f"{args.arm}_{args.dataset}"

    manifest = S7.OUT_ROOT / tag / "samples.json"
    m = json.loads(manifest.read_text())
    strata = {}
    for r in m["val"]:
        strata.setdefault(r["stratum"], []).append(r)
    per = max(1, args.images // 4)
    picked = []
    for k in ("bg", "small", "mid", "large"):
        picked += strata.get(k, [])[:per]
    picked = picked[: args.images]

    tf = transforms.Compose([transforms.Scale(256, 256),
                             transforms.Normalize(T.MEAN, T.STD), transforms.ToTensor()])

    def loader(names):
        tmp = root / "_s8bfinal_list.txt"
        tmp.write_text("\n".join(names) + "\n")
        ds = Dataset(str(root), str(tmp), transform=tf)
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = CASANet(None, **C.context_kwargs(args.arm)).eval()
    model.load_state_dict(torch.load(
        str(S7.CKPT_ROOT / args.arm / args.dataset / "best_model.pth"),
        map_location="cpu", weights_only=True), strict=True)
    ctx = model.context
    for p_ in model.parameters():
        p_.requires_grad_(False)

    d = ctx.head_dim
    rows = []

    def analyse(name, enc):
        sources = ctx.prototype_features([channel_norm(ctx.norm, t) for t in enc["tar"]],
                                        enc["difference"])
        for si, (x, grid) in enumerate(zip(sources, ctx.grid_sizes)):
            q = ctx.qk(x)
            v_ = ctx.v(x)
            if q.dim() == 5:      # (B, heads, d, H, W)
                q = q[:, 0]
                v_ = v_[:, 0]
            zk = q / math.sqrt(d)                          # (1, d, H, W)
            zv = v_
            prob = ctx.guide(x).sigmoid()
            if prob.dim() == 4:   # (B, 1, H, W)
                prob = prob[:, :1]
            _, _, H, W = zk.shape
            zk = zk[0].permute(1, 2, 0).reshape(H * W, d)
            zv = zv[0].permute(1, 2, 0).reshape(H * W, d)
            pr = prob[0, 0].reshape(H * W)
            sd = ScaleData(zk, zv, pr, H, W)
            mass_ref = [float(sd.p.sum()), float((1.0 - sd.p).sum())]

            # fixed grid with the SAME leaf budget as the adaptive arms
            gx, gy = args.fixgrid, args.fixgrid
            fx = [round(i * W / gx) for i in range(gx + 1)]
            fy = [round(i * H / gy) for i in range(gy + 1)]
            fix = [(fx[i], fx[i + 1], fy[j], fy[j + 1])
                   for j in range(gy) for i in range(gx)]
            parts = {"FIX": fix}
            for side in ("K", "V"):
                for kind in ("E", "H"):
                    key = f"ADA-{side}-{kind}"
                    parts[key] = greedy_partition(sd, args.leaves, side, kind)
                    parts[f"DP-{side}-{kind}"] = dp_partition(sd, args.leaves, side, kind)
            parts["RAND"] = random_partition(sd, args.leaves, rng,
                                            max(2, min(H, W) // 8))

            for arm, rects in parts.items():
                side_opt = None
                kind_opt = None
                if arm.startswith(("ADA-", "DP-")):
                    _, side_opt, kind_opt = arm.split("-")
                rec = dict(file=name, scale=si, arm=arm, leaves=len(rects),
                           H=H, W=W, J=float("nan"))
                for side in ("K", "V"):
                    z = sd.zk if side == "K" else sd.zv
                    # actual emitted centre (Lorentz) and its fidelity
                    tot_m = 0.0
                    agg = dict(sse=0.0, hyp2=0.0, chord=0.0, cos=0.0)
                    for r in rects:
                        for c in (0, 1):
                            idx, mm = sd.class_stats(r, c)
                            if float(mm.sum()) <= 1e-8:
                                continue
                            cc = lorentz_center(z[idx], mm)
                            mt = metrics(z[idx], mm, cc)
                            tot_m += float(mm.sum())
                            for k in agg:
                                agg[k] += mt[k] * float(mm.sum())
                    for k in agg:
                        rec[f"{side}_{k}"] = agg[k] / max(tot_m, 1e-9)
                    if side == side_opt:
                        rec["J"] = objective(sd, rects, side, kind_opt, mass_ref)
                rows.append(rec)
            print(f"[s8b] {name} s{si} done", flush=True)

    for rec in picked:
        for image, _ in loader([rec["filename"]]):
            with torch.no_grad():
                fa, fb = model.encoder(image[:, :3]), model.encoder(image[:, 3:])
                res = model.tar(fa[1:], fb[1:])
                tar, diff = (res[0], res[1]) if len(res) >= 2 else (res, None)
                enc = dict(tar=tar, difference=diff)
            analyse(rec["filename"], enc)

    path = outdir / f"s8b_final_{tag}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")

    # ---- aggregate by arm ------------------------------------------------
    print(f"\n== {tag}: mean over images/scales (lower better; cos higher better)")
    print(f"{'arm':11s} {'K_sse':>9s} {'K_hyp2':>9s} {'K_cos':>8s} "
          f"{'V_sse':>9s} {'V_hyp2':>9s} {'V_cos':>8s}")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        def mean(k):
            return sum(r[k] for r in sub) / len(sub)
        print(f"{arm:11s} {mean('K_sse'):9.4f} {mean('K_hyp2'):9.4f} "
              f"{mean('K_cos'):8.4f} {mean('V_sse'):9.4f} "
              f"{mean('V_hyp2'):9.4f} {mean('V_cos'):8.4f}")


if __name__ == "__main__":
    main()
