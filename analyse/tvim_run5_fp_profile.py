"""Run5 §10 止损只读诊断：SYSU 小预测假阳性（FP-like）空间画像。

**触发条件**：Run5 S1 前测 `NO_80K`（val 初筛 `BOTH_SCREENED_OUT`）。按 Run5 设计文档
§8.4 D1 / §10，**唯一立即动作**是只读 FP 空间画像：不增模型、不改损失、不跑 80K。

画像对象（均原生 256²、4 连通）：
  A `fp_like_unmatched_small`：预测连通域，面积 [1,255]，IoU≥0.10 一次性最大权重匹配后未匹配；
  B `gt_small_matched`：面积 [1,255] 的 GT 对象，且已被匹配；
  C `gt_small_missed`：面积 [1,255] 的 GT 对象，像素 recall `r<0.25`；
  D `background_control`：与 A 面积匹配、落在 `~GT` 且远离任何预测的背景窗口（校准基线）。

四项子指标（§10）：
  (i) 与 GT 边界距离：`EDT(~GT)` 在对象上的最小值、≤1/2/4px 像素比例；
  (ii) 变化图与 A/B 单时相边缘的重叠：逐图确定性边缘场（三通道均值的 Sobel 幅值 ≥ 该图 90 分位），
       对象像素落在 A/B 边缘 1px 邻域内的比例（并给 A-only / B-only / both 分解）；
  (iii) 面积与纹理能量：`mean|∇I_A|`、`mean|∇I_B|`、`mean|A−B|`；
  (iv) 与错位邻域关系：在 ±3px 平移搜索下与 GT 对象的最大 IoU（`best_shift_iou`）、
       取得该 IoU 的非零平移、`shifted_matched = best_shift_iou>=0.5 且平移非零`、
       以及到最近 GT 对象质心的距离。

统计：**图像级**有放回 bootstrap 1000 次、seed 16，每次重算计数比/均值，报 95%CI。
本脚本只读：模型 `.eval()` + `requires_grad_(False)` + `torch.no_grad()`；不训练、不重拟合、
不写回任何历史产物；`--out-dir` 必须不存在或为空。

    python analyse/tvim_run5_fp_profile.py --device cuda:0 --batch-size 16 --num-workers 4 \
      --out-dir /home/yqwang/outputs/CASA-CD/diagnostics/TViM-Run5-FP-PROFILE-20261011
"""
import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvim_diag_common import (  # noqa: E402
    build_model, env_info, pick_best_ckpt, read_list_names, set_eval_numerics,
    sha256_file, sha256_text_lines, write_json,
)
from tvim_object_metrics import (  # noqa: E402
    CONN4, GROUP_ALIAS, bin_index, label_components, match_objects, object_iou_matrix,
)

PROFILE_VERSION = "run5_fp_profile_v1"
M1_SYSU_SHA = "45d688a01af22fe22521f4dfb4579e80827ddcaa886e07bafc778fc288731457"
SYSU_TEST_LIST_SHA = "5f4122c9d32cb6db475f4f361618b2ac8bb10938df93d9550e0bb9a422a45797"
SMALL_ALIAS = set(GROUP_ALIAS["small"])
SOBEL_ST = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
DIL3 = np.ones((3, 3), dtype=bool)
EDGE_Q = 0.90
MAX_SHIFT = 3
SHIFT_MATCH_IOU = 0.50
BOOTSTRAP = 1000


# ---------------------------------------------------------------------------- 纯函数
def edge_field(img3):
    """三通道均值的 Sobel 幅值 → 该图 90 分位阈值边缘场。`img3` 为 (3,256,256)。

    边界用 `mode="reflect"`（SciPy 默认）：常量场梯度恒为 0，不会像 `mode="nearest"`
    那样在图像首末行/列注入虚假边缘。
    """
    from scipy import ndimage as ndi
    g = img3.astype(np.float64).mean(axis=0)
    gx = ndi.sobel(g, axis=1, mode="reflect")
    gy = ndi.sobel(g, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    thr = float(np.quantile(mag, EDGE_Q))
    # 稀疏梯度场的退化保护：若 90 分位为 0（>10% 像素零梯度），用严格 >0。否则
    # `mag >= 0` 会把整幅图判成边缘。
    return mag, (mag > 0.0 if thr <= 0.0 else mag >= thr)


def dilate1(mask):
    from scipy import ndimage as ndi
    return ndi.binary_dilation(mask, structure=DIL3, iterations=1)


def dist_to_gt(gt):
    """EDT：每个像素到最近 GT 像素的欧氏距离（GT 像素自身为 0）。"""
    from scipy import ndimage as ndi
    if not gt.any():
        return np.full(gt.shape, np.inf)
    return ndi.distance_transform_edt(~gt)


def best_shift_iou(comp, gt_lab, n_gt, gt_areas, comp_bbox, max_shift=MAX_SHIFT):
    """在 ±max_shift 平移下，comp 与任一 GT 对象的最大 IoU 及取得它的平移。"""
    if n_gt == 0:
        return 0.0, (0, 0), None
    y0 = max(0, comp_bbox[0].start - max_shift)
    y1 = min(comp.shape[0], comp_bbox[0].stop + max_shift)
    x0 = max(0, comp_bbox[1].start - max_shift)
    x1 = min(comp.shape[1], comp_bbox[1].stop + max_shift)
    cand = np.unique(gt_lab[y0:y1, x0:x1])
    cand = cand[cand > 0]
    if cand.size == 0:
        return 0.0, (0, 0), None
    ca = int(np.count_nonzero(comp))
    best, best_sh, best_k = 0.0, (0, 0), None
    for k in cand:
        gtk = (gt_lab == int(k))
        ga = int(gt_areas[int(k) - 1])
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                s = np.roll(gtk, (dy, dx), axis=(0, 1))
                if dy > 0:
                    s[:dy, :] = False
                elif dy < 0:
                    s[dy:, :] = False
                if dx > 0:
                    s[:, :dx] = False
                elif dx < 0:
                    s[:, dx:] = False
                inter = int(np.count_nonzero(comp & s))
                if inter == 0:
                    continue
                iou = inter / (ca + ga - inter)
                if iou > best:
                    best, best_sh, best_k = iou, (dy, dx), int(k)
    return float(best), best_sh, best_k


def object_metrics(comp, gt, gt_lab, n_gt, gt_areas, gt_objs, bbox, magA, magB, edgeA_d,
                   edgeB_d, dgt, chg, want_shift=True):
    """单个对象的四项子指标（comp 为全尺寸 bool mask）。"""
    m = {"area": int(np.count_nonzero(comp))}
    dm = dgt[comp]
    dmin = float(dm.min()) if dm.size else None
    m["dist_to_gt_min_px"] = dmin if (dmin is not None and np.isfinite(dmin)) else None
    m["frac_px_within_1px_of_gt"] = float((dm <= 1).mean()) if dm.size else None
    m["frac_px_within_2px_of_gt"] = float((dm <= 2).mean()) if dm.size else None
    m["frac_px_within_4px_of_gt"] = float((dm <= 4).mean()) if dm.size else None
    ea, eb = edgeA_d[comp], edgeB_d[comp]
    m["frac_px_on_A_edge_1px"] = float(ea.mean()) if ea.size else None
    m["frac_px_on_B_edge_1px"] = float(eb.mean()) if eb.size else None
    both = ea & eb
    m["frac_px_on_any_edge_1px"] = float((ea | eb).mean()) if ea.size else None
    m["frac_px_on_both_edges_1px"] = float(both.mean()) if both.size else None
    m["gradA_mean"] = float(magA[comp].mean()) if m["area"] else None
    m["gradB_mean"] = float(magB[comp].mean()) if m["area"] else None
    m["change_abs_mean"] = float(chg[comp].mean()) if m["area"] else None
    if want_shift:
        iou, sh, k = best_shift_iou(comp, gt_lab, n_gt, gt_areas, bbox)
        m["best_shift_iou"] = iou
        m["best_shift_dy"] = int(sh[0])
        m["best_shift_dx"] = int(sh[1])
        m["best_shift_gt_id"] = k
        m["shifted_matched"] = int(iou >= SHIFT_MATCH_IOU and (sh[0], sh[1]) != (0, 0))
        m["matched_at_zero_shift"] = int(iou >= SHIFT_MATCH_IOU and (sh[0], sh[1]) == (0, 0))
    else:
        m.update({"best_shift_iou": None, "best_shift_dy": None, "best_shift_dx": None,
                  "best_shift_gt_id": None, "matched_at_zero_shift": None,
                  "shifted_matched": None})
    return m


def background_control(comp_area, pred_d, gt, rng, tries=24):
    """与 FP 组件面积匹配、落在 `~GT` 且远离预测的背景方形窗口（校准基线）。"""
    side = max(1, int(round(np.sqrt(comp_area))))
    h, w = gt.shape
    if side >= h or side >= w:
        return None
    for _ in range(tries):
        y = int(rng.integers(0, h - side))
        x = int(rng.integers(0, w - side))
        win = np.zeros_like(gt)
        win[y:y + side, x:x + side] = True
        if (win & gt).any() or (win & pred_d).any():
            continue
        return win
    return None


def bootstrap_ratio(num, den, n_boot=BOOTSTRAP, seed=16):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    obs = float(num.sum() / den.sum()) if den.sum() > 0 else None
    rng = np.random.default_rng(seed)
    n = num.size
    vals = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = den[idx].sum()
        if d > 0:
            vals[b] = num[idx].sum() / d
    ok = np.isfinite(vals)
    if not ok.any():
        return obs, None, None, 0
    lo, hi = np.percentile(vals[ok], [2.5, 97.5])
    return obs, float(lo), float(hi), int(ok.sum())


def bootstrap_paired(a, b, n_boot=BOOTSTRAP, seed=16):
    """图像级配对均值差 a−b 的 bootstrap（每图一个已配对的均值差）。"""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None, None, None, 0
    rng = np.random.default_rng(seed)
    vals = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), int(d.size)


# ---------------------------------------------------------------------------- 主流程
def _loader(dataset_root, list_path, batch_size, num_workers):
    import torch
    from dataset import dataset as myDataLoader
    from dataset import Transforms as myTransforms
    mean6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    tf = myTransforms.Compose([myTransforms.Normalize(mean=mean6, std=std6),
                               myTransforms.Scale(256, 256), myTransforms.ToTensor()])
    ds = myDataLoader.Dataset(file_root=dataset_root, list_path=list_path, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True), ds


def run(args):
    t0 = time.time()
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"[P0_INVALID] out-dir not empty: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    import hashlib

    import torch
    from scipy import ndimage as ndi
    set_eval_numerics()

    ck, meta = pick_best_ckpt("Run1", "M1_FULL", args.dataset, ckpt_root=args.ckpt_root)
    sha = sha256_file(ck)
    if sha != M1_SYSU_SHA:
        raise SystemExit(f"[P0_INVALID] M1 sha256 {sha} != {M1_SYSU_SHA}")
    lst_sha = sha256_text_lines(args.test_list)
    if lst_sha != SYSU_TEST_LIST_SHA:
        raise SystemExit(f"[P0_INVALID] test list sha256_text_lines {lst_sha} != "
                         f"{SYSU_TEST_LIST_SHA}")
    print(f"[FP-PROFILE] {PROFILE_VERSION} ckpt={ck} sha={sha[:16]}", flush=True)

    model, info = build_model("M1_FULL", device=args.device, ckpt_path=ck, strict=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    loader, ds = _loader(args.dataset_root, args.test_list, args.batch_size,
                         args.num_workers)
    names = list(ds.file_list)
    print(f"[FP-PROFILE] test images={len(names)}", flush=True)

    rows = []
    per_image = []
    seen = 0
    with torch.no_grad():
        for img, label in loader:
            pre = img[:, 0:3].to(args.device).float()
            post = img[:, 3:6].to(args.device).float()
            out = model(pre, post)
            prob = out[:, 0].float().cpu().numpy()
            gts = label.numpy()[:, 0] > 0.5
            ab = img.numpy()
            for i in range(gts.shape[0]):
                nm = names[seen] if seen < len(names) else f"idx{seen}"
                seen += 1
                gt = gts[i]
                pred = prob[i] > 0.5
                if not pred.any() and not gt.any():
                    per_image.append({"image": nm, "n_fp": 0, "n_gt_small": 0,
                                      "n_fp_near_gt4": 0, "n_fp_shift_matched": 0,
                                      "sum_fp_edge": 0.0, "n_ctl": 0,
                                      "sum_ctl_edge": 0.0})
                    continue
                magA, edgeA = edge_field(ab[i, 0:3])
                magB, edgeB = edge_field(ab[i, 3:6])
                eAd, eBd = dilate1(edgeA), dilate1(edgeB)
                chg = np.abs(ab[i, 0:3].mean(axis=0) - ab[i, 3:6].mean(axis=0))
                dgt = dist_to_gt(gt)
                pred_d = dilate1(pred)
                cache = object_iou_matrix(pred, gt, 4)
                mt = match_objects(pred, gt, 4, cache=cache)
                matched_pred = set(mt.get("matched_pred_idx", []))
                # 稳定 per-image RNG（禁止用 Python hash()，它随进程随机化）
                rng = np.random.default_rng(
                    int(hashlib.sha1(nm.encode()).hexdigest()[:8], 16))

                rec = {"image": nm, "n_fp": 0, "n_gt_small": 0, "n_fp_near_gt4": 0,
                       "n_fp_shift_matched": 0, "sum_fp_edge": 0.0,
                       "n_ctl": 0, "sum_ctl_edge": 0.0}

                # A: FP-like
                for j in range(cache["n_pred"]):
                    if j in matched_pred:
                        continue
                    area = int(cache["pred_areas"][j])
                    if bin_index(area) not in SMALL_ALIAS:
                        continue
                    comp = (cache["pr_lab"] == j + 1)
                    m = object_metrics(comp, gt, cache["gt_lab"], cache["n_gt"],
                                       cache["gt_areas"], cache["gt_objs"],
                                       cache["pr_objs"][j], magA, magB, eAd, eBd, dgt, chg)
                    m.update({"image": nm, "group": "fp_like_unmatched_small", "obj_id": j})
                    rows.append(m)
                    rec["n_fp"] += 1
                    rec["n_fp_near_gt4"] += int(m["frac_px_within_4px_of_gt"] >= 0.5)
                    rec["n_fp_shift_matched"] += int(m["shifted_matched"] == 1)
                    rec["sum_fp_edge"] += float(m["frac_px_on_any_edge_1px"])
                    # D: 面积匹配背景对照
                    ctl = background_control(area, pred_d, gt, rng)
                    if ctl is not None:
                        sl = ndi.find_objects(ctl.astype(np.int64))[0]
                        mc = object_metrics(ctl, gt, cache["gt_lab"], cache["n_gt"],
                                            cache["gt_areas"], cache["gt_objs"], sl,
                                            magA, magB, eAd, eBd, dgt, chg,
                                            want_shift=False)
                        mc.update({"image": nm, "group": "background_control",
                                   "obj_id": None})
                        rows.append(mc)
                        rec["n_ctl"] += 1
                        rec["sum_ctl_edge"] += float(mc["frac_px_on_any_edge_1px"])

                # B/C: GT small objects
                if cache["n_gt"]:
                    counts = cache["gt_areas"]
                    tp_map = pred & gt
                    tpc = (np.bincount(cache["gt_lab"][tp_map].ravel(),
                                       minlength=cache["n_gt"] + 1)
                           if tp_map.any() else np.zeros(cache["n_gt"] + 1, dtype=np.int64))
                    matched_gt = set(int(x) for x in mt.get("matched_gt_idx", []))
                    for j in range(1, cache["n_gt"] + 1):
                        area = int(counts[j - 1])
                        if bin_index(area) not in SMALL_ALIAS:
                            continue
                        comp = (cache["gt_lab"] == j)
                        r = int(tpc[j]) / area
                        grp = ("gt_small_matched" if (j - 1) in matched_gt
                               else ("gt_small_missed" if r < 0.25 else "gt_small_unmatched_hit"))
                        m = object_metrics(comp, gt, cache["gt_lab"], cache["n_gt"],
                                           cache["gt_areas"], cache["gt_objs"],
                                           cache["gt_objs"][j - 1], magA, magB, eAd, eBd,
                                           dgt, chg, want_shift=False)
                        m.update({"image": nm, "group": grp, "obj_id": j,
                                  "pixel_recall": float(r)})
                        rows.append(m)
                        rec["n_gt_small"] += 1
                per_image.append(rec)
            if seen % 400 == 0:
                print(f"[FP-PROFILE] {seen} images elapsed={time.time() - t0:.0f}s", flush=True)
            if args.limit and seen >= args.limit:
                break

    elapsed = time.time() - t0

    def grp(g):
        return [r for r in rows if r["group"] == g]

    fp, ctl = grp("fp_like_unmatched_small"), grp("background_control")
    gm, gmiss = grp("gt_small_matched"), grp("gt_small_missed")

    def mean_of(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    def frac_of(rs, key, thr):
        v = [r[key] for r in rs if r.get(key) is not None]
        return float(np.mean([x >= thr for x in v])) if v else None

    n_fp = len(fp)
    n_fp_near4 = sum(1 for r in fp if r["frac_px_within_4px_of_gt"] >= 0.5)
    n_fp_shift = sum(1 for r in fp if r["shifted_matched"] == 1)
    n_fp_zero = sum(1 for r in fp if r["matched_at_zero_shift"] == 1)
    img = per_image
    ci = {}
    ci["fp_frac_near_gt4"] = bootstrap_ratio([r["n_fp_near_gt4"] for r in img],
                                             [r["n_fp"] for r in img], args.bootstrap,
                                             args.seed)
    ci["fp_frac_shift_matched"] = bootstrap_ratio([r["n_fp_shift_matched"] for r in img],
                                                  [r["n_fp"] for r in img], args.bootstrap,
                                                  args.seed)
    ci["fp_edge_overlap"] = bootstrap_ratio([r["sum_fp_edge"] for r in img],
                                            [r["n_fp"] for r in img], args.bootstrap,
                                            args.seed)
    ci["ctl_edge_overlap"] = bootstrap_ratio([r["sum_ctl_edge"] for r in img],
                                             [r["n_ctl"] for r in img], args.bootstrap,
                                             args.seed)
    # 图像级配对：每图 FP 与对照的边缘重叠差
    pair_a, pair_b = [], []
    for r in img:
        if r["n_fp"] and r["n_ctl"]:
            pair_a.append(r["sum_fp_edge"] / r["n_fp"])
            pair_b.append(r["sum_ctl_edge"] / r["n_ctl"])
    ci["fp_minus_ctl_edge_overlap"] = bootstrap_paired(pair_a, pair_b, args.bootstrap,
                                                       args.seed)

    summary = {
        "profile_version": PROFILE_VERSION,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset, "run": "Run1", "variant": "M1_FULL",
        "checkpoint": {"path": ck, "sha256": sha, "candidates": meta["candidates"]},
        "m1_sha256_expected": M1_SYSU_SHA,
        "test_list": args.test_list, "test_list_sha256_kind": "sha256_text_lines",
        "test_list_sha256": lst_sha, "test_list_sha256_expected": SYSU_TEST_LIST_SHA,
        "n_images": seen, "elapsed_s": round(elapsed, 1),
        "edge_field": {"definition": "三通道均值 Sobel 幅值 ≥ 该图 90 分位",
                       "quantile": EDGE_Q, "dilation_for_overlap_px": 1},
        "shift_search": {"max_shift_px": MAX_SHIFT, "match_iou": SHIFT_MATCH_IOU},
        "object_counts": {"fp_like_unmatched_small": n_fp,
                          "background_control": len(ctl),
                          "gt_small_matched": len(gm),
                          "gt_small_missed": len(gmiss)},
        "A_fp_like": {
            "n": n_fp,
            "frac_objects_with_majority_within_4px_of_gt": (n_fp_near4 / n_fp) if n_fp else None,
            "frac_objects_with_majority_within_4px_of_gt_ci95": ci["fp_frac_near_gt4"][1:3],
            "mean_frac_px_within_4px_of_gt": mean_of(fp, "frac_px_within_4px_of_gt"),
            "mean_frac_px_within_2px_of_gt": mean_of(fp, "frac_px_within_2px_of_gt"),
            "mean_frac_px_within_1px_of_gt": mean_of(fp, "frac_px_within_1px_of_gt"),
            "mean_dist_to_gt_min_px": mean_of(fp, "dist_to_gt_min_px"),
            "frac_objects_touching_gt_1px": frac_of(fp, "frac_px_within_1px_of_gt", 1e-9),
            "frac_objects_shift_matched": (n_fp_shift / n_fp) if n_fp else None,
            "frac_objects_shift_matched_ci95": ci["fp_frac_shift_matched"][1:3],
            "frac_objects_matched_at_zero_shift": (n_fp_zero / n_fp) if n_fp else None,
            "mean_best_shift_iou": mean_of(fp, "best_shift_iou"),
            "mean_frac_px_on_A_edge_1px": mean_of(fp, "frac_px_on_A_edge_1px"),
            "mean_frac_px_on_B_edge_1px": mean_of(fp, "frac_px_on_B_edge_1px"),
            "mean_frac_px_on_any_edge_1px": mean_of(fp, "frac_px_on_any_edge_1px"),
            "mean_frac_px_on_any_edge_1px_ci95": ci["fp_edge_overlap"][1:3],
            "mean_frac_px_on_both_edges_1px": mean_of(fp, "frac_px_on_both_edges_1px"),
            "mean_gradA": mean_of(fp, "gradA_mean"), "mean_gradB": mean_of(fp, "gradB_mean"),
            "mean_change_abs": mean_of(fp, "change_abs_mean"),
        },
        "D_background_control": {
            "n": len(ctl),
            "mean_frac_px_on_any_edge_1px": mean_of(ctl, "frac_px_on_any_edge_1px"),
            "mean_frac_px_on_any_edge_1px_ci95": ci["ctl_edge_overlap"][1:3],
            "mean_gradA": mean_of(ctl, "gradA_mean"), "mean_gradB": mean_of(ctl, "gradB_mean"),
            "mean_change_abs": mean_of(ctl, "change_abs_mean"),
        },
        "fp_minus_control_edge_overlap": {
            "obs": ci["fp_minus_ctl_edge_overlap"][0],
            "ci95": list(ci["fp_minus_ctl_edge_overlap"][1:3]),
            "n_images_paired": ci["fp_minus_ctl_edge_overlap"][3],
        },
        "B_gt_small_matched": {
            "n": len(gm), "mean_gradA": mean_of(gm, "gradA_mean"),
            "mean_gradB": mean_of(gm, "gradB_mean"),
            "mean_change_abs": mean_of(gm, "change_abs_mean"),
            "mean_frac_px_on_any_edge_1px": mean_of(gm, "frac_px_on_any_edge_1px"),
        },
        "C_gt_small_missed": {
            "n": len(gmiss), "mean_gradA": mean_of(gmiss, "gradA_mean"),
            "mean_gradB": mean_of(gmiss, "gradB_mean"),
            "mean_change_abs": mean_of(gmiss, "change_abs_mean"),
            "mean_frac_px_on_any_edge_1px": mean_of(gmiss, "frac_px_on_any_edge_1px"),
        },
        "bootstrap": {"unit": "image", "n": args.bootstrap, "seed": args.seed},
        "env": env_info(),
        "not_a_gate": ("本文件是 §10 的只读止损诊断，不构成 Run5 的 PASS/FAIL 门，"
                       "也不得用于事后放宽 §5.4 的 G0–G8。"),
    }
    if n_fp:
        summary["A_fp_like"]["change_to_control_ratio_edge"] = (
            None if not summary["D_background_control"]["mean_frac_px_on_any_edge_1px"]
            else summary["A_fp_like"]["mean_frac_px_on_any_edge_1px"]
            / summary["D_background_control"]["mean_frac_px_on_any_edge_1px"])

    write_json(os.path.join(args.out_dir, "fp_profile.json"), summary)
    write_json(os.path.join(args.out_dir, "bootstrap.json"), {
        "unit": "image", "n": args.bootstrap, "seed": args.seed,
        **{k: {"obs": v[0], "ci95": [v[1], v[2]], "valid": v[3]} for k, v in ci.items()},
    })
    cols = ["image", "group", "obj_id", "area", "dist_to_gt_min_px",
            "frac_px_within_1px_of_gt", "frac_px_within_2px_of_gt",
            "frac_px_within_4px_of_gt", "frac_px_on_A_edge_1px", "frac_px_on_B_edge_1px",
            "frac_px_on_any_edge_1px", "frac_px_on_both_edges_1px", "gradA_mean",
            "gradB_mean", "change_abs_mean", "best_shift_iou", "best_shift_dy",
            "best_shift_dx", "best_shift_gt_id", "shifted_matched",
            "matched_at_zero_shift"]
    with open(os.path.join(args.out_dir, "per_object.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.out_dir, "per_image.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "n_fp", "n_gt_small", "n_fp_near_gt4",
                                          "n_fp_shift_matched", "sum_fp_edge",
                                          "n_ctl", "sum_ctl_edge"])
        w.writeheader()
        for r in per_image:
            w.writerow(r)
    write_json(os.path.join(args.out_dir, "gate.json"), {
        "stage": "run5_fp_profile", "status": "COMPLETED",
        "profile_version": PROFILE_VERSION, "n_images": seen,
        "n_objects": len(rows), "elapsed_s": round(elapsed, 1),
        "not_a_gate": summary["not_a_gate"],
    })
    lines = []
    for fn in sorted(os.listdir(args.out_dir)):
        p = os.path.join(args.out_dir, fn)
        if os.path.isfile(p) and fn != "SHA256SUMS.txt":
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            lines.append(f"{h}  {fn}")
    with open(os.path.join(args.out_dir, "SHA256SUMS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[FP-PROFILE] images={seen} objects={len(rows)} elapsed={elapsed:.0f}s", flush=True)
    print(f"[FP-PROFILE] A fp-like n={n_fp} near_gt4={summary['A_fp_like']['frac_objects_with_majority_within_4px_of_gt']} "
          f"shift_matched={summary['A_fp_like']['frac_objects_shift_matched']}", flush=True)
    print(f"[FP-PROFILE] edge overlap: fp={summary['A_fp_like']['mean_frac_px_on_any_edge_1px']} "
          f"ctl={summary['D_background_control']['mean_frac_px_on_any_edge_1px']} "
          f"diff_ci={summary['fp_minus_control_edge_overlap']['ci95']}", flush=True)
    print(f"[FP-PROFILE] -> {os.path.join(args.out_dir, 'fp_profile.json')}", flush=True)
    return 0 if seen == len(names) and not args.limit else 2


def main():
    ap = argparse.ArgumentParser(description="Run5 §10 只读 SYSU FP 空间画像")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--ckpt-root",
                    default="/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM")
    ap.add_argument("--dataset-root", default="/share_datasets/CD/SYSU-CD-256")
    ap.add_argument("--test-list", default="/share_datasets/CD/SYSU-CD-256/list/test.txt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug only；非零时 rc=2")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
