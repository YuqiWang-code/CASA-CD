"""S8b-A: same-budget partition fidelity on REAL captured K/V tensors.

Task book §6: strictly the same budget, both cost families, identical candidate
tree for E and H, class-balanced normalisation, K and V driven separately.

For every captured (image, scale, kind=K/V) we compare

    FIX   the checkpoint's own uniform grid  (g x g regions)
    ADA-E greedy dyadic split minimising weighted Euclidean SSE
    ADA-H greedy dyadic split minimising the Lorentz chord cost 2(a - M)
    RAND  random dyadic split (same number of regions)

and report each partition's distortion under BOTH metrics (Euclidean SSE and
chord cost), class-balanced, so a rule cannot win by only optimising its own
objective. Distortions are computed per member with the guide weights, exactly
like the module pools them.

Usage (server):  python -u s8b_partition.py --arm R_FULL-2334 --dataset SYSU-CD-256 --images 6
Output: <analysis>/simulation/S8b/<arm>_<dataset>/{s8b_regions.csv,s8b_summary.json}
"""

import argparse
import csv as _csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7c_capture as S7  # noqa: E402


# ---------------------------------------------------------------- partitions
def grid_boxes(height, width, grid):
    ch, cw = height // grid, width // grid
    boxes = []
    for gy in range(grid):
        for gx in range(grid):
            boxes.append((gy * ch, (gy + 1) * ch, gx * cw, (gx + 1) * cw))
    return boxes


def build_masks(height, width, boxes, yx):
    """Map member index -> region id for a box list (yx is (N,2) row/col)."""
    ids = np.full(len(yx), -1, dtype=np.int64)
    for ridx, (y0, y1, x0, x1) in enumerate(boxes):
        m = (yx[:, 0] >= y0) & (yx[:, 0] < y1) & (yx[:, 1] >= x0) & (yx[:, 1] < x1)
        ids[m] = ridx
    return ids


def region_stats(z, t, w, yx, boxes):
    """Per-region, per-class sums needed by both costs (vectorised by class)."""
    n_cls = w.shape[1]
    ids = build_masks(int(yx[:, 0].max()) + 1, int(yx[:, 1].max()) + 1, boxes, yx)
    keep = ids >= 0
    ids = ids[keep]
    z, t, w = z[keep], t[keep], w[keep]
    stats = []
    quad_all = (z * z).sum(1)
    for c in range(n_cls):
        wc = w[:, c]
        szt = np.zeros((len(boxes), z.shape[1]))
        np.add.at(szt, ids, z * wc[:, None])
        wsum = np.zeros(len(boxes))
        np.add.at(wsum, ids, wc)
        tsum = np.zeros(len(boxes))
        np.add.at(tsum, ids, t * wc)
        qsum = np.zeros(len(boxes))
        np.add.at(qsum, ids, quad_all * wc)
        stats.append((wsum, szt, tsum, qsum))
    return stats


def total_cost(z, t, w, yx, boxes, balanced=True, eps=1e-9):
    stats = region_stats(z, t, w, yx, boxes)
    n_reg = len(boxes)
    e = np.zeros(n_reg)
    h = np.zeros(n_reg)
    mass = [s[0].sum() for s in stats]
    for c, (wsum, szt, tsum, qsum) in enumerate(stats):
        for r in range(n_reg):
            wv = float(wsum[r])
            if wv <= 0:
                continue
            zs = szt[r]
            ts = float(tsum[r])
            zz = float((zs * zs).sum())
            sse_r = max(float(qsum[r]) - zz / wv, 0.0)
            a = np.sqrt(max(ts * ts - zz, 0.0))
            e[r] += sse_r / (mass[c] + eps) if balanced else sse_r
            h[r] += 2.0 * max(a - wv, 0.0) / (mass[c] + eps) if balanced else \
                2.0 * max(a - wv, 0.0)
    return float(e.sum()), float(h.sum())


def split_gain(z, t, w, yx, box, metric, balanced, eps=1e-9):
    y0, y1, x0, x1 = box
    if (y1 - y0) < 2 and (x1 - x0) < 2:
        return None
    cands = []
    if y1 - y0 >= 2:
        my = (y0 + y1) // 2
        cands.append([(y0, my, x0, x1), (my, y1, x0, x1)])
    if x1 - x0 >= 2:
        mx = (x0 + x1) // 2
        cands.append([(y0, y1, x0, mx), (y0, y1, mx, x1)])
    best = None
    for pair in cands:
        e0, h0 = total_cost(z, t, w, yx, [box], balanced, eps)
        e1, h1 = total_cost(z, t, w, yx, pair, balanced, eps)
        gain_e = e0 - e1
        gain_h = h0 - h1
        use = gain_e if metric == "e" else gain_h
        if best is None or use > best[0]:
            best = (use, pair)
    return best


def adaptive(z, t, w, yx, r_target, metric="e", balanced=True):
    height = int(yx[:, 0].max()) + 1
    width = int(yx[:, 1].max()) + 1
    boxes = [(0, height, 0, width)]
    while len(boxes) < r_target:
        best = (-1e18, None, None)
        for bi, box in enumerate(boxes):
            got = split_gain(z, t, w, yx, box, metric, balanced)
            if got is None:
                continue
            gain, pair = got
            if gain > best[0] + 1e-12:
                best = (gain, bi, pair)
        if best[1] is None:
            break
        bi, pair = best[1], best[2]
        boxes = boxes[:bi] + list(pair) + boxes[bi + 1:]
    return boxes


def random_boxes(height, width, r_target, rng):
    boxes = [(0, height, 0, width)]
    while len(boxes) < r_target:
        bi = int(rng.integers(len(boxes)))
        y0, y1, x0, x1 = boxes[bi]
        axes = []
        if y1 - y0 >= 2:
            axes.append("y")
        if x1 - x0 >= 2:
            axes.append("x")
        if not axes:
            break
        ax = axes[int(rng.integers(len(axes)))]
        if ax == "y":
            my = (y0 + y1) // 2
            pair = [(y0, my, x0, x1), (my, y1, x0, x1)]
        else:
            mx = (x0 + x1) // 2
            pair = [(y0, y1, x0, mx), (y0, y1, mx, x1)]
        boxes = boxes[:bi] + pair + boxes[bi + 1:]
    return boxes


def members_of(points, weights, grid):
    """Flatten the module's (1,h,R,P,D)/(1,R,2,P) into pixel order + spatial index."""
    b, h, g, p, d = points.shape
    height = width = int(round(np.sqrt(g * p)))
    gsz = int(round(np.sqrt(g)))
    ch = height // gsz
    z = points.reshape(-1, d).numpy()
    w = weights.reshape(g, 2, p).permute(0, 2, 1).reshape(g * p, 2).numpy()
    yy = []
    xx = []
    for r in range(g):
        for pp in range(p):
            yy.append((r // gsz) * ch + pp // ch)
            xx.append((r % gsz) * ch + pp % ch)
    yx = np.stack([np.array(yy, dtype=np.int64), np.array(xx, dtype=np.int64)], 1)
    t = np.sqrt(1.0 + (z * z).sum(1))
    return z, t, w, yx, height, width, gsz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R_FULL-2334")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--images", type=int, default=6)
    ap.add_argument("--scales", default="0,2")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("S7B_THREADS", "6")))
    torch.backends.cuda.matmul.allow_tf32 = False
    C, CASANet, Dataset, transforms, T = S7.bootstrap_paths()

    ckpt = S7.CKPT_ROOT / args.arm / args.dataset / "best_model.pth"
    outdir = S7.OUT_ROOT.parent / "S8b" / f"{args.arm}_{args.dataset}"
    outdir.mkdir(parents=True, exist_ok=True)
    root = S7.DATA_ROOT / args.dataset
    samples = json.loads((S7.OUT_ROOT / f"{args.arm}_{args.dataset}" / "samples.json").read_text())
    names = [r["filename"] for r in samples["val"]][: args.images]

    eval_transform = transforms.Compose([transforms.Scale(256, 256),
                                         transforms.Normalize(T.MEAN, T.STD),
                                         transforms.ToTensor()])
    listing = outdir / f"_list_{len(names)}.txt"
    listing.write_text("\n".join(names) + "\n")
    ds = Dataset(str(root), str(listing), transform=eval_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=0, drop_last=False)

    model = CASANet(None, **C.context_kwargs(args.arm)).eval()
    model.load_state_dict(S7.load_state(ckpt), strict=True)
    ctx = model.context
    model.to(torch.device(args.device))
    scale_of_size = {256 // (4 * (2 ** k)): k for k in range(len(ctx.grid_sizes))}
    want = [int(s) for s in args.scales.split(",")]

    rows = []
    rng = np.random.default_rng(S7.SEED)
    for idx, (image, _) in enumerate(loader):
        image = image.to(args.device).float()
        rec = S7.Recorder(ctx)
        with torch.no_grad(), rec:
            model(image[:, :3], image[:, 3:], return_aux=False)
        for call, (pts, wts, _out) in enumerate(rec.pool):
            kind = "K" if call % 2 == 0 else "V"
            # recover the scale from the member count: g*p = H*W
            npix = pts.shape[2] * pts.shape[3]
            side = int(round(np.sqrt(npix)))
            scale = {64: 0, 32: 1, 16: 2, 8: 3}.get(side)
            if scale is None or scale not in want:
                continue
            z, t, w, yx, height, width, gsz = members_of(pts, wts, ctx.grid_sizes[scale])
            r_target = gsz * gsz
            fix = grid_boxes(height, width, gsz)
            ada_e = adaptive(z, t, w, yx, r_target, "e")
            ada_h = adaptive(z, t, w, yx, r_target, "h")
            rand = random_boxes(height, width, r_target, rng)
            for name, boxes in (("FIX", fix), ("ADA-E", ada_e),
                                ("ADA-H", ada_h), ("RAND", rand)):
                e_bal, h_bal = total_cost(z, t, w, yx, boxes, True)
                e_raw, h_raw = total_cost(z, t, w, yx, boxes, False)
                rows.append(dict(arm=args.arm, dataset=ds_name(args.dataset), kind=kind,
                                 scale=scale, sample_id=idx, partition=name,
                                 n_regions=len(boxes), r_target=r_target,
                                 e_bal=e_bal, h_bal=h_bal, e_raw=e_raw, h_raw=h_raw))
        print(f"[image] {idx} done", flush=True)

    with open(outdir / "s8b_regions.csv", "w", newline="", encoding="utf-8") as fh:
        wr = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    (outdir / "s8b_summary.json").write_text(json.dumps(
        dict(arm=args.arm, dataset=args.dataset, images=len(names),
             scales=want, rows=len(rows)), indent=1))
    print("[done]", len(rows), "rows", flush=True)


def ds_name(x):
    return x.split("-")[0]


if __name__ == "__main__":
    main()
