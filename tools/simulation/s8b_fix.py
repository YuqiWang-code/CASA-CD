#!/usr/bin/env python
"""S8b-fix: same-budget adaptive prototype partition on real features (addendum).

Copy of ``s8b_final.py`` with the three defects reported in the closure review
fixed; the historical script and its CSVs are untouched.

Fixes
-----
1. **hyperbolic distance / direction cosine**
   The old code computed

       ch = sqrt(1+|z|^2) sqrt(1+|c|^2) - |z| |c| cos(ambient)

   with ``cos(ambient) = z.c / sqrt((1+|z|^2)(1+|c|^2))``, i.e. the spatial inner
   product was divided by both time components.  The correct Minkowski inner
   product is ``<L(z),L(c)>_L = t_z t_c - z.c`` and

       d_H^2 = acosh(<L(z),L(c)>_L)^2,   chord^2 = 2(<L(z),L(c)>_L - 1).

   The **direction cosine** is the cosine between the two geodesic tangents *at
   the origin*; with ``u(z) = z / t_z`` the time factors cancel, so it is the
   plain spatial cosine ``z.c / (|z| |c|)``.  The old column was
   ``cos_spatial * tanh|z| * tanh|c|`` and is now reported separately as
   ``*_cos_ambient`` for audit only.

2. **one normalization target for greedy / DP / objective**
   All three now optimise the same quantity
   ``sum_{R,c} J(R,c) / M_{s,c}`` with the scale-wide class mass ``M_{s,c}``
   fixed before the partition is built.  Previously greedy and DP used raw
   ``J`` while the reported objective divided by ``M_{s,c}``.

3. **sampling and provenance**
   Strata are selected with equal quota over the strata that actually exist
   (SYSU has no ``bg`` stratum), a top-up pass fills the target when possible,
   and the realised counts, filenames and list SHA-256 are written next to the
   CSV.  The temporary loader list lives in the output directory, not in the
   dataset root.

All ten arms (DP-V-E / DP-V-H included) are emitted, so nothing is silently
dropped from the aggregate table again.

Usage (server, conda env casacd)::

    python s8b_fix.py --arm R_FULL-2334 --dataset SYSU-CD-256 --images 12 \\
        --leaves 4 --fixgrid 2 --out .../FinalClosure/S8b
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
SIM_DIR = HERE.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, str(SIM_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S9 = _load("s9a_fix", "s9a_fix.py")
S7 = S9.S7
stratified_pick = S9.stratified_pick

torch.set_grad_enabled(False)

ARMS = ["FIX", "RAND",
        "ADA-K-E", "ADA-K-H", "ADA-V-E", "ADA-V-H",
        "DP-K-E", "DP-K-H", "DP-V-E", "DP-V-H"]

lift = None
acosh1p_squared = None


# ------------------------------------------------------------------ geometry --
def lorentz_center(z, m):
    """Module-identical OEM barycentre (``manifold_pool``) as a spatial vector."""
    m = m.double()
    z = z.double()
    s = (m[:, None] * lift(z)).sum(0)
    n2 = (s[0] * s[0] - (s[1:] * s[1:]).sum()).clamp_min(1e-9)
    return s[1:] / torch.sqrt(n2)


def metrics(z, m, c):
    """Mass-weighted fidelity of members ``z`` against centre ``c`` (float64)."""
    z = z.double()
    c = c.double()
    tot = m.double().sum().clamp_min(1e-12)
    w = m.double() / tot
    sse = ((z - c) ** 2).sum(-1)                                  # ambient Euclidean
    inner = lift(z)[:, 0] * lift(c)[0] - (z * c).sum(-1)          # <L(z),L(c)>_L
    h2 = acosh1p_squared((inner - 1.0).clamp_min(0.0))            # geodesic squared
    chord = (2.0 * (inner - 1.0)).clamp_min(0.0)                  # ambient chord^2
    zn = z.norm(dim=-1)
    cn = c.norm().clamp_min(1e-12)
    cos_dir = (z * c).sum(-1) / (zn * cn).clamp_min(1e-12)        # origin tangent
    cos_amb = (z * c).sum(-1) / (
        torch.sqrt((zn ** 2 + 1.0) * (cn ** 2 + 1.0))).clamp_min(1e-12)
    return dict(sse=float((w * sse).sum()), hyp2=float((w * h2).sum()),
                chord=float((w * chord).sum()), cos=float((w * cos_dir).sum()),
                cos_ambient=float((w * cos_amb).sum()))


# ---------------------------------------------------------------- splits ----
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
    """members + raw guide probability for one scale, O(1) rectangle statistics."""

    def __init__(self, zk, zv, p, h, w, mass_ref):
        self.zk, self.zv, self.p = zk.double(), zv.double(), p.double()
        self.h, self.w = h, w
        self.mass_ref = mass_ref            # [M_change, M_background] for the scale

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
        """J for one rectangle/class and the (fixed) class mass of the scale."""
        idx, m = self.class_stats(r, cls)
        M = m.sum()
        if float(M) <= 1e-10:
            return 0.0, M
        z = self.zk[idx] if side == "K" else self.zv[idx]
        if kind == "E":
            b = (m[:, None] * z).sum(0)
            c = (m * (z * z).sum(-1)).sum()
            return float(c - (b * b).sum() / M), M
        s = (m[:, None] * lift(z)).sum(0)
        a2 = (s[0] * s[0] - (s[1:] * s[1:]).sum()).clamp_min(0.0)
        return float(2.0 * (torch.sqrt(a2) - M)), M


def objective(sd, rects, side, kind):
    """Normalised objective: sum_{R,c} J(R,c) / M_{s,c} (smaller is better)."""
    total = 0.0
    for r in rects:
        for c in (0, 1):
            j, _ = sd.cost(r, c, side, kind)
            total += j / max(sd.mass_ref[c], 1e-9)
    return total


def greedy_partition(sd, leaves, side, kind):
    rects = [(0, sd.w, 0, sd.h)]
    min_side = max(2, min(sd.h, sd.w) // 8)
    while len(rects) < leaves:
        base = sum(sd.cost(r, c, side, kind)[0] / max(sd.mass_ref[c], 1e-9)
                   for r in rects for c in (0, 1))
        best = None
        for i, r in enumerate(rects):
            for r1, r2 in splits(r, min_side):
                gain = base
                for c in (0, 1):
                    mr = max(sd.mass_ref[c], 1e-9)
                    gain -= (sd.cost(r1, c, side, kind)[0]
                             + sd.cost(r2, c, side, kind)[0]) / mr
                if best is None or gain > best[0] + 1e-12:
                    best = (gain, i, r1, r2)
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
    """Optimum of the *same* normalised objective on the same dyadic tree."""
    from functools import lru_cache

    min_side = max(2, min(sd.h, sd.w) // 8)

    def rect_cost(r):
        return sum(sd.cost(r, c, side, kind)[0] / max(sd.mass_ref[c], 1e-9)
                   for c in (0, 1))

    @lru_cache(maxsize=None)
    def best(r, k):
        j = rect_cost(r)
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

    _, rects = best((0, sd.w, 0, sd.h), leaves)
    return list(rects)


def fix_grid(h, w, gx, gy):
    fx = [round(i * w / gx) for i in range(gx + 1)]
    fy = [round(i * h / gy) for i in range(gy + 1)]
    return [(fx[i], fx[i + 1], fy[j], fy[j + 1]) for j in range(gy) for i in range(gx)]


# ------------------------------------------------------------------ main ----
def main():
    global lift, acosh1p_squared

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R_FULL-2334")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--leaves", type=int, default=4)
    ap.add_argument("--fixgrid", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("S7B_THREADS", "6")))
    torch.manual_seed(S7.SEED)
    rng = random.Random(args.seed)

    C, CASANet, Dataset, transforms, T = S7.bootstrap_paths()
    from model.layers.lorentz_ops import lift as _lift, acosh1p_squared as _a1p
    from model.layers.pyramid_context import channel_norm
    lift, acosh1p_squared = _lift, _a1p

    outdir = Path(args.out) if args.out else (S7.OUT_ROOT.parent / "S8bFinal")
    outdir.mkdir(parents=True, exist_ok=True)
    root = S7.DATA_ROOT / args.dataset
    tag = f"{args.arm}_{args.dataset}"

    manifest = S7.OUT_ROOT / tag / "samples.json"
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest}; run s7c_capture.py first")
    picked, sampled = stratified_pick(json.loads(manifest.read_text())["val"], args.images)

    tf = transforms.Compose([transforms.Scale(256, 256),
                             transforms.Normalize(T.MEAN, T.STD), transforms.ToTensor()])

    names = [r["filename"] for r in picked]
    listing = outdir / f"_list_{tag}.txt"
    listing.write_text("\n".join(names) + "\n")
    list_sha = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()

    def loader():
        ds = Dataset(str(root), str(listing), transform=tf)
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
    sanity = []          # DP must never be worse than FIX on its own objective
    greedy = []          # greedy is myopic: reported separately, not an assertion

    def analyse(name, enc):
        sources = ctx.prototype_features([channel_norm(ctx.norm, t) for t in enc["tar"]],
                                        enc["difference"])
        for si, (x, grid) in enumerate(zip(sources, ctx.grid_sizes)):
            q = ctx.qk(x)
            v_ = ctx.v(x)
            if q.dim() == 5:
                q = q[:, 0]
                v_ = v_[:, 0]
            zk = q / math.sqrt(d)
            zv = v_
            prob = ctx.guide(x).sigmoid()
            if prob.dim() == 4:
                prob = prob[:, :1]
            _, _, h, w = zk.shape
            zk = zk[0].permute(1, 2, 0).reshape(h * w, d)
            zv = zv[0].permute(1, 2, 0).reshape(h * w, d)
            pr = prob[0, 0].reshape(h * w)

            m_change = float(pr.double().sum())
            m_bg = float((1.0 - pr.double()).sum())
            sd = ScaleData(zk, zv, pr, h, w, [m_change, m_bg])

            parts = {"FIX": fix_grid(h, w, args.fixgrid, args.fixgrid)}
            parts["RAND"] = random_partition(sd, args.leaves, rng,
                                            max(2, min(h, w) // 8))
            for side in ("K", "V"):
                for kind in ("E", "H"):
                    parts[f"ADA-{side}-{kind}"] = greedy_partition(
                        sd, args.leaves, side, kind)
                    parts[f"DP-{side}-{kind}"] = dp_partition(
                        sd, args.leaves, side, kind)

            # sanity: every dyadic arm must be no worse than FIX on its own objective
            for side in ("K", "V"):
                for kind in ("E", "H"):
                    fix_obj = objective(sd, parts["FIX"], side, kind)
                    for arm in (f"ADA-{side}-{kind}", f"DP-{side}-{kind}"):
                        obj = objective(sd, parts[arm], side, kind)
                        if obj > fix_obj + 1e-6:
                            (sanity if arm.startswith("DP-") else greedy).append(
                                dict(file=name, scale=si, arm=arm, side=side,
                                     kind=kind, obj=round(obj, 6),
                                     fix=round(fix_obj, 6)))

            for arm, rects in parts.items():
                side_opt = kind_opt = None
                if arm.startswith(("ADA-", "DP-")):
                    _, side_opt, kind_opt = arm.split("-")
                rec = dict(checkpoint=args.arm, dataset=args.dataset, file=name,
                           scale=si, arm=arm, leaves=len(rects), H=h, W=w,
                           mass_change=round(m_change, 3), mass_bg=round(m_bg, 3),
                           J=float("nan"))
                for side in ("K", "V"):
                    z = sd.zk if side == "K" else sd.zv
                    tot_m = 0.0
                    agg = dict(sse=0.0, hyp2=0.0, chord=0.0, cos=0.0, cos_ambient=0.0)
                    for r in rects:
                        for c in (0, 1):
                            idx, mm = sd.class_stats(r, c)
                            if float(mm.sum()) <= 1e-10:
                                continue
                            cc = lorentz_center(z[idx], mm)
                            mt = metrics(z[idx], mm, cc)
                            tot_m += float(mm.sum())
                            for k in agg:
                                agg[k] += mt[k] * float(mm.sum())
                    for k in agg:
                        rec[f"{side}_{k}"] = round(agg[k] / max(tot_m, 1e-12), 6)
                    if side == side_opt:
                        rec["J"] = round(objective(sd, rects, side_opt, kind_opt), 6)
                for side in ("K", "V"):
                    for kind in ("E", "H"):
                        rec[f"J_{side}{kind}"] = round(objective(sd, rects, side, kind), 6)
                rows.append(rec)
            print(f"[s8b] {name} s{si} done", flush=True)

    for i, (image, _target) in enumerate(loader()):
        with torch.no_grad():
            fa, fb = model.encoder(image[:, :3]), model.encoder(image[:, 3:])
            res = model.tar(fa[1:], fb[1:])
            tar, diff = (res[0], res[1]) if len(res) >= 2 else (res, None)
            enc = dict(tar=tar, difference=diff)
        analyse(names[i], enc)

    path = outdir / f"s8b_fix_{tag}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = dict(checkpoint=args.arm, dataset=args.dataset, images=len(picked),
                   sampled=sampled, list_sha256=list_sha, leaves=args.leaves,
                   fixgrid=args.fixgrid, arms=ARMS, rows=len(rows),
                   sanity_violations=len(sanity),
                   greedy_violations=len(greedy),
                   greedy_examples=greedy[:5],
                   leaves_note="FIX and adaptive arms all use the same leaf budget",
                   metrics_note=("sse=ambient Euclidean squared error; hyp2=squared "
                                 "Lorentz geodesic distance; chord=ambient chord^2; "
                                 "cos=origin-tangent direction cosine (pure spatial); "
                                 "cos_ambient=old, tanh-damped variant for audit only"),
                   J_note="sum_{R,c} J(R,c)/M_{s,c} with scale-wide M fixed")
    (outdir / f"s8b_fix_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"[sanity] DP violations={len(sanity)} greedy violations={len(greedy)}", flush=True)
    if sanity:
        print(f"[sanity] first DP violation={sanity[0]}", flush=True)
    else:
        print("[sanity] no violations (adaptive <= FIX on its own objective)", flush=True)

    print(f"\n== {tag}: mean over images/scales (lower better; cos higher better)")
    print(f"{'arm':11s} {'K_sse':>9s} {'K_hyp2':>9s} {'K_cos':>8s} "
          f"{'V_sse':>9s} {'V_hyp2':>9s} {'V_cos':>8s}")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        mean = lambda k: sum(r[k] for r in sub) / len(sub)
        print(f"{arm:11s} {mean('K_sse'):9.4f} {mean('K_hyp2'):9.4f} "
              f"{mean('K_cos'):8.4f} {mean('V_sse'):9.4f} "
              f"{mean('V_hyp2'):9.4f} {mean('V_cos'):8.4f}")


if __name__ == "__main__":
    main()
