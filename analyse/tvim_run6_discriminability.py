"""Run6 §4 只读可分辨性前测：冻结 M1 的既有特征里，能否把 FP-like 小预测与真实 small 变化分开？

**这是 Run6 的唯一开门依据，且必须先于任何 `models/` 改动执行。**
背景（Run6 设计文档 §1）：SYSU 小目标失败的形态是「判别失败」而非「证据缺失」——
FP 区域的原始变化能量（0.9730）**高于**真实漏检 small 变化（0.8626），且手工证据通道
（边缘重叠 / 变化幅值 / 纹理能量，含逐图背景归一化）对二者的判别 AUC 上限只有 **0.5992**。
因此 Run6 不新增证据通道，而是问：**既有学习到的多尺度表征里是否已存在可分离信息？**

对象口径（与 Run5 前测 / F14 判别力分析**逐字一致**）：
  * 正样本（真实 small 变化）= GT small 对象（面积 [1,255] 且 4 连通）中
    `IoU>=0.10 已匹配` **或** `像素 recall r<0.25` 者（test 上 n=327）；
  * 负样本（FP-like）= 预测 4 连通域（面积 [1,255]）中一次性最大权重匹配后**未匹配**者（test 上 n=2430）。

拟合/评估分离（防泄漏，Run6 §4.1）：
  * 拟合 = `train.txt` 的 D3 SHA1-hash **90% train** 部分（该 split 已固定，非新 seed）；
  * 评估 = `test.txt` 全量 4000 张；**test 不参与拟合、不选特征、不选阈值**；
  * 拟合图像是模型训练过的，分布偏乐观 ⇒ 门只认 **test 侧**数值。

特征（全部只依赖冻结模型输出或 A/B 图像，**不依赖 GT**）：见 `FEATURE_GROUPS`。
两候选：**C1 跨尺度一致性**、**C2 对象内部/边界一致性**；基线必须包含"模型自身置信度"与
"手工证据通道"，候选必须在基线之上**再**有增量（Run6 §9 R2）。

    python analyse/tvim_run6_discriminability.py --device cuda:0 --batch-size 16 \
      --num-workers 4 --limit 0 --budget-sec 900 \
      --out-dir /home/yqwang/outputs/CASA-CD/diagnostics/TViM-Run6-DISCRIMINABILITY-20261011
"""
import argparse
import csv
import datetime
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvim_diag_common import (  # noqa: E402
    build_model, env_info, pick_best_ckpt, read_list_names, set_eval_numerics,
    sha256_file, sha256_text_lines, write_json,
)
from tvim_linear_probe import _hash_split, _resolve_module  # noqa: E402
from tvim_object_metrics import (  # noqa: E402
    CONN4, GROUP_ALIAS, bin_index, label_components, match_objects, object_iou_matrix,
)
from tvim_run5_fp_profile import dilate1, edge_field  # noqa: E402

PROFILE_VERSION = "run6_discriminability_v1"
PROTOCOL_TAG = "run6_scale_gate_exact80k_v1"
M1_SYSU_SHA = "45d688a01af22fe22521f4dfb4579e80827ddcaa886e07bafc778fc288731457"
SYSU_TEST_LIST_SHA = "5f4122c9d32cb6db475f4f361618b2ac8bb10938df93d9550e0bb9a422a45797"
SMALL_ALIAS = set(GROUP_ALIAS["small"])
BOOTSTRAP = 1000
BUDGET_S = 900.0
EDGE_Q = 0.90
TAU_EPS = 1e-6

# Run6 §4.2 硬门（数值在见结果之前冻结）
GATES = {
    "K1_n_pos_min": 300,
    "K1_n_neg_min": 2000,
    "K2_test_auc_min": 0.80,
    "K2_test_auc_ci_low_min": 0.75,
    "K3_test_ap_min": 0.35,
    "K3_test_ap_ci_low_min": 0.25,
    "K4_auc_over_handcrafted_min": 0.15,
    "K4_handcrafted_baseline_auc": 0.5992,
    "K4b_auc_over_full_baseline_min": 0.05,
    "K5_auc_over_amplitude_min": 0.10,
}

# 特征组（列名 → 见下方 compute_features）
FEATURE_GROUPS = {
    "amp": ["magY_s1", "magY_s2", "magY_s3", "magY_s4", "magT_s1", "magT_s2",
            "magT_s3", "magT_s4", "refine_mag"],
    "hand": ["edge_any", "edge_both", "change_abs", "gradA"],
    "head": ["logit_mean", "logit_std", "logit_ring", "prob_mean", "prob_std", "prob_max"],
    "C1": ["Y_ratio_s1", "Y_ratio_s2", "Y_ratio_s3", "Y_ratio_s4",
           "Y_range", "Y_cv", "Y_entropy", "Y_spatstd_mean"],
    "C2": ["refine_contrast", "refine_std", "logit_contrast", "logit_inside_std"],
}
BASELINE_AMP = FEATURE_GROUPS["amp"]
BASELINE_HAND = FEATURE_GROUPS["hand"]
BASELINE_HEAD = FEATURE_GROUPS["head"]
BASELINE_FULL = BASELINE_AMP + BASELINE_HAND + BASELINE_HEAD
PROBE_SETS = {
    "baseline_amp": BASELINE_AMP,
    "baseline_hand": BASELINE_HAND,
    "baseline_head": BASELINE_HEAD,
    "baseline_full": BASELINE_FULL,
    "C1_features": FEATURE_GROUPS["C1"],
    "C2_features": FEATURE_GROUPS["C2"],
    "candidate_C1": BASELINE_FULL + FEATURE_GROUPS["C1"],
    "candidate_C2": BASELINE_FULL + FEATURE_GROUPS["C2"],
}
LAMBDAS = (1e-3, 1e-2, 1e-1)
ALL_FEATURES = BASELINE_FULL + FEATURE_GROUPS["C1"] + FEATURE_GROUPS["C2"]


# ============================================================================ 纯函数
def pool_map(map2d, frac):
    """面积平均池化：`frac` 为 256² 掩码按 map2d 分辨率做 INTER_AREA 后的覆盖率。"""
    w = float(frac.sum())
    if w <= 0.0:
        return None
    return float((np.asarray(map2d, dtype=np.float64) * frac).sum() / w)


def resize_frac(mask256, size):
    """256² 掩码 → size² 覆盖率（INTER_AREA）。"""
    import cv2
    return cv2.resize(mask256.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def ring_mask(obj, gt, iterations=4):
    """`R=(dilate_4(O)\\O)∩(~GT)`（与 Run5 FP 画像同口径）。"""
    from scipy import ndimage as ndi
    d = ndi.binary_dilation(obj, structure=CONN4, iterations=iterations)
    return d & ~obj & ~gt


def cross_scale_features(e_by_scale):
    """C1：跨尺度一致性统计量（`e_s` = 各尺度对象内平均变化证据）。"""
    e = np.asarray([max(v, 0.0) for v in e_by_scale], dtype=np.float64)
    tot = e.sum() + TAU_EPS
    ratio = e / tot
    rng = float(e.max() - e.min())
    cv = float(e.std() / (e.mean() + TAU_EPS))
    p = ratio / (ratio.sum() + TAU_EPS)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return {"Y_ratio_s1": float(ratio[0]), "Y_ratio_s2": float(ratio[1]),
            "Y_ratio_s3": float(ratio[2]), "Y_ratio_s4": float(ratio[3]),
            "Y_range": rng, "Y_cv": cv, "Y_entropy": ent}


def auc_score(y, s):
    """Mann-Whitney AUC（分数越大越倾向 y=1）。"""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="stable")
    ss = s[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    counts = np.diff(np.r_[starts, s.size])
    avg = starts + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ap_score(y, s):
    """Average Precision（按分数降序的 P-R 阶梯积分）。"""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    pos = int(y.sum())
    if pos == 0:
        return None
    order = np.argsort(-s, kind="stable")
    yy = y[order]
    tp = np.cumsum(yy)
    k = np.arange(1, yy.size + 1)
    prec = tp / k
    rec = tp / pos
    prev = 0.0
    ap = 0.0
    for i in range(yy.size):
        if yy[i]:
            ap += prec[i] * (rec[i] - prev)
            prev = rec[i]
    return float(ap)


def fit_logistic(X, y, lam, steps=1000, lr=0.5):
    """标准化 + L2 正则 logistic（全批梯度下降，确定性；无 sklearn 依赖）。"""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = np.c_[np.ones(X.shape[0]), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    yv = y.astype(np.float64)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - yv) / len(yv)
        g[1:] += lam * w[1:] / len(yv)
        w -= lr * g
    return w, mu, sd


def _impute(X_fit, X_other):
    """用拟合集列中位数填补 NaN（ring 为空等导致的缺失），避免整行被丢弃。"""
    med = np.nanmedian(np.where(np.isfinite(X_fit), X_fit, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    def _fill(X):
        Y = X.copy()
        idx = np.where(~np.isfinite(Y))
        if idx[0].size:
            Y[idx] = np.take(med, idx[1])
        return Y
    return _fill(X_fit), _fill(X_other), med


def apply_logistic(w, mu, sd, X):
    Z = np.c_[np.ones(X.shape[0]), (X - mu) / sd]
    return Z @ w


def boot_ci(y, s, images, n_boot=BOOTSTRAP, seed=16, metric="auc"):
    """图像级有放回 bootstrap 的 95%CI（同图对象同进同出）。"""
    fn = auc_score if metric == "auc" else ap_score
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    groups = {}
    for i, im in enumerate(images):
        groups.setdefault(im, []).append(i)
    keys = sorted(groups)
    idx_of = [np.asarray(groups[k]) for k in keys]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idx_of[p] for p in pick])
        v = fn(y[sel], s[sel])
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None, 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


# ============================================================================ 数据
def _loader(dataset_root, list_path, batch_size, num_workers, names_filter=None):
    import torch
    from dataset import dataset as myDataLoader
    from dataset import Transforms as myTransforms
    mean6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    tf = myTransforms.Compose([myTransforms.Normalize(mean=mean6, std=std6),
                               myTransforms.Scale(256, 256), myTransforms.ToTensor()])
    ds = myDataLoader.Dataset(file_root=dataset_root, list_path=list_path, transform=tf)
    if names_filter is not None:
        keep = [i for i, n in enumerate(ds.file_list) if names_filter(n)]
        ds.file_list = [ds.file_list[i] for i in keep]
        ds.pre_images = [ds.pre_images[i] for i in keep]
        ds.post_images = [ds.post_images[i] for i in keep]
        ds.gts = [ds.gts[i] for i in keep]
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True), ds


def _resolve(model, key):
    kind = {"Y_s1": ("tar", "stage1.temporal"), "Y_s2": ("tar", "stage2.temporal"),
            "Y_s3": ("tar", "stage3.temporal"), "Y_s4": ("tar", "stage4.temporal"),
            "T_s1": ("tar", "stage1"), "T_s2": ("tar", "stage2"),
            "T_s3": ("tar", "stage3"), "T_s4": ("tar", "stage4")}
    if key in kind:
        _k, attr = kind[key]
        obj = model.tar
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj
    if key == "refine":
        return model.decoder.refine
    if key == "head":
        return model.head
    raise KeyError(key)


def _weighted_mean_std(map2d, w):
    tot = float(w.sum())
    if tot <= 0.0:
        return None, None
    mu = float((np.asarray(map2d, dtype=np.float64) * w).sum() / tot)
    var = float((w * (np.asarray(map2d, dtype=np.float64) - mu) ** 2).sum() / tot)
    return mu, float(np.sqrt(max(var, 0.0)))


def compute_row(cap, i, obj, gt, prob_map, chg, magA, eAd, eBd):
    """单对象特征行（全部不依赖 GT；obj 为 256² bool）。"""
    row = {}
    for s in range(1, 5):
        y = cap[f"Y_s{s}"][i].float()
        t = cap[f"T_s{s}"][i].float()
        row[f"magY_s{s}"] = pool_map(y.abs().mean(dim=0).cpu().numpy(),
                                     resize_frac(obj, y.shape[-2]))
        row[f"magT_s{s}"] = pool_map(t.abs().mean(dim=0).cpu().numpy(),
                                     resize_frac(obj, t.shape[-2]))
    e_scales = [row[f"magY_s{s}"] if row[f"magY_s{s}"] is not None else 0.0
                for s in range(1, 5)]
    row.update(cross_scale_features(e_scales))
    y1 = cap["Y_s1"][i].float().abs().mean(dim=0).cpu().numpy()
    _mu, _sd = _weighted_mean_std(y1, resize_frac(obj, y1.shape[-2]))
    row["Y_spatstd_s1"] = _sd
    row["Y_spatstd_mean"] = _sd

    rmag = cap["refine"][i].float().abs().mean(dim=0).cpu().numpy()
    fr = resize_frac(obj, rmag.shape[-2])
    ring = ring_mask(obj, gt)
    fr_ring = resize_frac(ring, rmag.shape[-2]) if ring.any() else np.zeros_like(fr)
    row["refine_mag"], row["refine_std"] = _weighted_mean_std(rmag, fr)
    row["refine_ring_mag"] = (_weighted_mean_std(rmag, fr_ring)[0]
                              if fr_ring.sum() > 0 else None)
    row["refine_contrast"] = (None if row["refine_mag"] is None
                              or row["refine_ring_mag"] is None
                              else row["refine_mag"] - row["refine_ring_mag"])

    lg = cap["head"][i].float()[0].cpu().numpy()
    fl = resize_frac(obj, lg.shape[-2])
    row["logit_mean"], row["logit_inside_std"] = _weighted_mean_std(lg, fl)
    row["logit_std"] = row["logit_inside_std"]
    row["logit_ring"] = (_weighted_mean_std(lg, fr_ring)[0]
                         if fr_ring.sum() > 0 else None)
    row["logit_contrast"] = (None if row["logit_mean"] is None
                             or row["logit_ring"] is None
                             else row["logit_mean"] - row["logit_ring"])

    w = resize_frac(obj, prob_map.shape[-1])
    row["prob_mean"], row["prob_std"] = _weighted_mean_std(prob_map, w)
    row["prob_max"] = float(prob_map[obj].max()) if obj.any() else None
    row["edge_any"] = pool_map(eAd | eBd, w)
    row["edge_both"] = pool_map(eAd & eBd, w)
    row["change_abs"] = pool_map(chg, w)
    row["gradA"] = pool_map(magA, w)
    return row


def run_pass(model, loader, device, args, tag, t_start, positive_rule):
    """一次只读前向，返回 (rows, seen, timed_out)。positive_rule(gt_objs, mt, tpc) → 标签。"""
    import torch

    cap = {}
    handles = []
    for key in ("Y_s1", "Y_s2", "Y_s3", "Y_s4", "T_s1", "T_s2", "T_s3", "T_s4",
                "refine", "head"):
        handles.append(_resolve(model, key).register_forward_hook(
            (lambda k: (lambda m, i, o: cap.__setitem__(k, o)))(key)))

    rows = []
    seen = 0
    timed_out = False
    names = list(loader.dataset.file_list)
    try:
        with torch.no_grad():
            for img, label in loader:
                cap.clear()
                pre = img[:, 0:3].to(device).float()
                post = img[:, 3:6].to(device).float()
                out = model(pre, post)
                prob = out[:, 0].float().cpu().numpy()
                gts = label.numpy()[:, 0] > 0.5
                ab = img.numpy()
                B = pre.shape[0]
                for i in range(B):
                    nm = names[seen] if seen < len(names) else f"idx{seen}"
                    seen += 1
                    gt = gts[i]
                    p = prob[i]
                    pred = p > 0.5
                    magA, edgeA = edge_field(ab[i, 0:3])
                    magB, edgeB = edge_field(ab[i, 3:6])
                    eAd, eBd = dilate1(edgeA), dilate1(edgeB)
                    chg = np.abs(ab[i, 0:3].mean(axis=0) - ab[i, 3:6].mean(axis=0))
                    cache = object_iou_matrix(pred, gt, 4)
                    mt = match_objects(pred, gt, 4, cache=cache)

                    if cache["n_gt"]:
                        tp_map = pred & gt
                        tpc = (np.bincount(cache["gt_lab"][tp_map].ravel(),
                                           minlength=cache["n_gt"] + 1)
                               if tp_map.any() else np.zeros(cache["n_gt"] + 1,
                                                             dtype=np.int64))
                        matched_gt = set(int(x) for x in mt.get("matched_gt_idx", []))
                        for j in range(1, cache["n_gt"] + 1):
                            area = int(cache["gt_areas"][j - 1])
                            if bin_index(area) not in SMALL_ALIAS:
                                continue
                            label_ = positive_rule(j - 1, matched_gt,
                                                   int(tpc[j]) / area)
                            if label_ is None:
                                continue
                            obj = (cache["gt_lab"] == j)
                            row = compute_row(cap, i, B, obj, gt, p, chg, magA, eAd, eBd,
                                              p.shape[0])
                            row.update({"image": nm, "label": label_,
                                        "kind": "gt_small", "area": area})
                            rows.append(row)
                    matched_pred = set(mt.get("matched_pred_idx", []))
                    for j in range(cache["n_pred"]):
                        if j in matched_pred:
                            continue
                        area = int(cache["pred_areas"][j])
                        if bin_index(area) not in SMALL_ALIAS:
                            continue
                        obj = (cache["pr_lab"] == j + 1)
                        row = compute_row(cap, i, B, obj, gt, p, chg, magA, eAd, eBd,
                                          p.shape[0])
                        row.update({"image": nm, "label": 0, "kind": "fp_like",
                                    "area": area})
                        rows.append(row)
                if args.limit and seen >= args.limit:
                    break
                if seen % 400 == 0:
                    print(f"[{tag}] {seen} images elapsed={time.time() - t_start:.0f}s",
                          flush=True)
                if time.time() - t_start > args.budget_sec:
                    print(f"[{tag}][TIMEOUT] budget exceeded at {seen} images", flush=True)
                    timed_out = True
                    break
    finally:
        for h in handles:
            h.remove()
    return rows, seen, timed_out


def _matrix(rows, cols):
    X = np.array([[np.nan if r.get(c) is None else r.get(c) for c in cols] for r in rows],
                 dtype=np.float64)
    return X


def probe_report(rows_tr, rows_te, cols, seed, n_boot):
    """train 拟合（5 折按图选 λ）→ test AUC/AP + 图像级 bootstrap CI。"""
    ytr = np.array([r["label"] for r in rows_tr], dtype=bool)
    yte = np.array([r["label"] for r in rows_te], dtype=bool)
    Xtr_raw, Xte_raw = _matrix(rows_tr, cols), _matrix(rows_te, cols)
    if Xtr_raw.shape[0] < 50 or Xte_raw.shape[0] < 50 or ytr.sum() < 10:
        return {"insufficient": True, "n_train": int(Xtr_raw.shape[0]),
                "n_test": int(Xte_raw.shape[0])}
    Xtr, Xte, _med = _impute(Xtr_raw, Xte_raw)
    imgs = sorted({r["image"] for r in rows_tr})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(imgs))
    fold_of = {imgs[perm[i]]: i % 5 for i in range(len(imgs))}
    fold = np.array([fold_of[r["image"]] for r in rows_tr])
    best_lam, best_cv = LAMBDAS[0], -1.0
    for lam in LAMBDAS:
        oof = np.full(len(rows_tr), np.nan)
        for k in range(5):
            tr = fold != k
            te = fold == k
            if ytr[tr].sum() < 5 or (~ytr[tr]).sum() < 5 or te.sum() == 0:
                continue
            w, mu, sd = fit_logistic(Xtr[tr], ytr[tr], lam)
            oof[te] = apply_logistic(w, mu, sd, Xtr[te])
        m = np.isfinite(oof)
        cv = auc_score(ytr[m], oof[m]) if m.sum() > 10 else None
        if cv is not None and cv > best_cv:
            best_lam, best_cv = lam, cv
    w, mu, sd = fit_logistic(Xtr, ytr, best_lam)
    sc = apply_logistic(w, mu, sd, Xte)
    imgs_te = [r["image"] for r in rows_te]
    auc = auc_score(yte, sc)
    ap = ap_score(yte, sc)
    a_lo, a_hi, a_v = boot_ci(yte, sc, imgs_te, n_boot, seed, "auc")
    p_lo, p_hi, p_v = boot_ci(yte, sc, imgs_te, n_boot, seed, "ap")
    return {"insufficient": False, "lambda": best_lam, "train_cv_auc": best_cv,
            "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
            "n_pos_test": int(yte.sum()), "n_neg_test": int((~yte).sum()),
            "test_auc": auc, "test_auc_ci95": [a_lo, a_hi], "auc_valid_boot": a_v,
            "test_ap": ap, "test_ap_ci95": [p_lo, p_hi], "ap_valid_boot": p_v}


def main_run(args):
    t0 = time.time()
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"[P0_INVALID] out-dir not empty: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    set_eval_numerics()
    ck, meta = pick_best_ckpt("Run1", "M1_FULL", args.dataset, ckpt_root=args.ckpt_root)
    sha = sha256_file(ck)
    lst_sha = sha256_text_lines(args.test_list)
    tr_sha = sha256_text_lines(args.train_list)
    if sha != M1_SYSU_SHA:
        raise SystemExit(f"[P0_INVALID] ckpt sha256 {sha} != {M1_SYSU_SHA}")
    if lst_sha != SYSU_TEST_LIST_SHA:
        raise SystemExit(f"[P0_INVALID] test list sha256_text_lines {lst_sha} != "
                         f"{SYSU_TEST_LIST_SHA}")
    print(f"[R6-DISC] {PROFILE_VERSION} ckpt={os.path.basename(ck)} sha={sha[:16]}", flush=True)

    model, info = build_model("M1_FULL", device=args.device, ckpt_path=ck, strict=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]

    tr_loader, tr_ds = _loader(args.dataset_root, args.train_list, args.batch_size,
                               args.num_workers,
                               names_filter=lambda n: _hash_split(n, 0.10) == "train")
    te_loader, te_ds = _loader(args.dataset_root, args.test_list, args.batch_size,
                               args.num_workers)
    print(f"[R6-DISC] fit images={len(tr_ds.file_list)}  eval images={len(te_ds.file_list)}",
          flush=True)

    def train_rule(gt_index, matched_gt, r):
        if (gt_index in matched_gt) or (r < 0.25):
            return 1
        return None

    tr_rows, n_tr, tr_to = run_pass(model, tr_loader, args.device, args, "FIT", t0, train_rule)
    te_rows, n_te, te_to = run_pass(model, te_loader, args.device, args, "EVAL", t0,
                                    lambda gi, mg, r: 1 if ((gi in mg) or (r < 0.25)) else None)
    elapsed = time.time() - t0

    n_pos = sum(1 for r in te_rows if r["label"] == 1)
    n_neg = sum(1 for r in te_rows if r["label"] == 0)
    print(f"[R6-DISC] fit objects={len(tr_rows)} eval objects={len(te_rows)} "
          f"(pos={n_pos} neg={n_neg}) elapsed={elapsed:.0f}s", flush=True)

    results = {}
    for name, cols in PROBE_SETS.items():
        results[name] = probe_report(tr_rows, te_rows, cols, args.seed, args.bootstrap)
        r = results[name]
        if r.get("insufficient"):
            print(f"  {name:16s} INSUFFICIENT {r}", flush=True)
        else:
            print(f"  {name:16s} trainCV={r['train_cv_auc']:.4f} "
                  f"testAUC={r['test_auc']:.4f} CI[{r['test_auc_ci95'][0]:.4f},"
                  f"{r['test_auc_ci95'][1]:.4f}] AP={r['test_ap']:.4f} "
                  f"CI[{r['test_ap_ci95'][0]:.4f},{r['test_ap_ci95'][1]:.4f}]", flush=True)

    # ---- K7：按 train-CV AUC 选一（平局 ≤0.005 取参数更少者 = C1）
    cand = {"C1": results["candidate_C1"], "C2": results["candidate_C2"]}
    usable = {k: v for k, v in cand.items() if not v.get("insufficient")}
    if not usable:
        selected, tie = None, None
    else:
        ranked = sorted(usable, key=lambda k: -usable[k]["train_cv_auc"])
        if len(ranked) == 1:
            selected, tie = ranked[0], "only_usable"
        else:
            a, b = usable[ranked[0]]["train_cv_auc"], usable[ranked[1]]["train_cv_auc"]
            selected = ranked[0] if (a - b) > 0.005 else "C1"
            tie = ("train_cv_auc" if (a - b) > 0.005 else "tie<=0.005 -> fewer params (C1)")
    sel_res = cand.get(selected, {}) if selected else {}
    base_full = results["baseline_full"]
    base_hand = results["baseline_hand"]
    base_amp = results["baseline_amp"]

    g = {}
    g["K0"] = _k("K0", True, {"ckpt_sha256": sha, "test_list_sha256": lst_sha,
                              "train_list_sha256": tr_sha,
                              "fitted_on": "train_hash90", "test_used_for_fit": False},
                 "SHA 严格匹配 + 输出目录全新 + test 未参与拟合")
    g["K1"] = _k("K1", (n_pos >= GATES["K1_n_pos_min"] and n_neg >= GATES["K1_n_neg_min"]),
                 {"n_pos": n_pos, "n_neg": n_neg},
                 {"n_pos_min": GATES["K1_n_pos_min"], "n_neg_min": GATES["K1_n_neg_min"]})
    auc = sel_res.get("test_auc")
    ap = sel_res.get("test_ap")
    a_lo = (sel_res.get("test_auc_ci95") or [None])[0]
    p_lo = (sel_res.get("test_ap_ci95") or [None])[0]
    g["K2"] = _k("K2", (auc is not None and auc >= GATES["K2_test_auc_min"]
                        and a_lo is not None and a_lo > GATES["K2_test_auc_ci_low_min"]),
                 {"test_auc": auc, "ci_low": a_lo}, GATES["K2_test_auc_min"])
    g["K3"] = _k("K3", (ap is not None and ap >= GATES["K3_test_ap_min"]
                        and p_lo is not None and p_lo > GATES["K3_test_ap_ci_low_min"]),
                 {"test_ap": ap, "ci_low": p_lo}, GATES["K3_test_ap_min"])
    hand_auc = base_hand.get("test_auc")
    g["K4"] = _k("K4", (auc is not None and hand_auc is not None
                        and (auc - hand_auc) >= GATES["K4_auc_over_handcrafted_min"]),
                 {"winner_auc": auc, "handcrafted_auc": hand_auc,
                  "delta": None if auc is None or hand_auc is None else auc - hand_auc},
                 {"delta_min": GATES["K4_auc_over_handcrafted_min"]})
    full_auc = base_full.get("test_auc")
    g["K4b"] = _k("K4b", (auc is not None and full_auc is not None
                          and (auc - full_auc) >= GATES["K4b_auc_over_full_baseline_min"]),
                  {"winner_auc": auc, "full_baseline_auc": full_auc,
                   "delta": None if auc is None or full_auc is None else auc - full_auc},
                  {"delta_min": GATES["K4b_auc_over_full_baseline_min"]})
    amp_auc = base_amp.get("test_auc")
    k5_delta = None if auc is None or amp_auc is None else auc - amp_auc
    g["K5"] = _k("K5", (k5_delta is not None and k5_delta >= GATES["K5_auc_over_amplitude_min"]),
                 {"winner_auc": auc, "amplitude_auc": amp_auc, "delta": k5_delta},
                 {"delta_min": GATES["K5_auc_over_amplitude_min"]})
    g["K6"] = _k("K6", True, {"same_object_set": True, "both_candidates_archived": True},
                 "两候选同一对象集合/probe/bootstrap")
    g["K7"] = _k("K7", selected in ("C1", "C2"),
                 {"selected": selected, "tie_breaker": tie}, "按 train-CV AUC 选一；test 不参与")

    hard = [g["K2"]["pass"], g["K3"]["pass"], g["K4"]["pass"], g["K4b"]["pass"]]
    dry = bool(args.limit)
    if dry:
        status = "DRY_ONLY"
    elif not g["K1"]["pass"]:
        status = "INCONCLUSIVE"
    elif all(hard):
        status = "PASS"
    else:
        status = "FAIL"
    if status == "PASS" and not g["K5"]["pass"]:
        gate_stat = "amplitude"
    else:
        gate_stat = "consistency"
    gate_impl = (f"c1_scale_consistency" if selected == "C1" else
                 ("c2_interior_consistency" if selected == "C2" else "none"))

    inputs = {
        "profile_version": PROFILE_VERSION, "protocol_tag": PROTOCOL_TAG,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset, "run": "Run1", "variant": "M1_FULL",
        "repository_sha": os.environ.get("CASA_REPO_SHA"),
        "ckpt_path": ck, "ckpt_sha256": sha, "ckpt_sha256_expected": M1_SYSU_SHA,
        "train_list": args.train_list, "train_list_sha256_kind": "sha256_text_lines",
        "train_list_sha256": tr_sha,
        "test_list": args.test_list, "test_list_sha256_kind": "sha256_text_lines",
        "test_list_sha256": lst_sha, "test_list_sha256_expected": SYSU_TEST_LIST_SHA,
        "n_fit_images": n_tr, "n_eval_images": n_te,
        "n_fit_objects": len(tr_rows), "n_eval_objects": len(te_rows),
        "n_pos_eval": n_pos, "n_neg_eval": n_neg,
        "fit_split": "train.txt D3 sha1-hash 10% val excluded -> 90% train used for fitting",
        "positive_rule": "GT small (area 1..255, 4-conn) with (IoU>=0.10 matched) OR (r<0.25)",
        "negative_rule": "unmatched predicted CC, area 1..255, one-shot max-weight matching",
        "pooling": "area-averaged mask coverage (cv2.INTER_AREA) at each feature map resolution",
        "edge_field": {"definition": "三通道均值 Sobel 幅值 >= 该图 90 分位 (mode=reflect)",
                       "quantile": EDGE_Q, "dilation_for_overlap_px": 1},
        "probe": "standardized linear logistic, L2 selected by image-grouped 5-fold CV on FIT",
        "lambdas": list(LAMBDAS), "bootstrap": {"unit": "image", "n": args.bootstrap,
                                                "seed": args.seed},
        "feature_groups": FEATURE_GROUPS, "probe_sets": {k: v for k, v in PROBE_SETS.items()},
        "gates_frozen": GATES,
        "caveat": "拟合图像是模型训练过的，分布偏乐观；门只认 test 侧数值",
        "env": env_info(),
    }
    write_json(os.path.join(args.out_dir, "inputs.json"), inputs)

    for tag, rows in (("features_fit", tr_rows), ("features_test", te_rows)):
        with open(os.path.join(args.out_dir, f"{tag}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            cols = ["image", "label", "kind", "area"] + ALL_FEATURES
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_json(os.path.join(args.out_dir, "probe_results.json"), {
        "results": results, "selection": {"selected": selected, "tie_breaker": tie,
                                          "gate_impl": gate_impl, "gate_stat": gate_stat},
    })
    write_json(os.path.join(args.out_dir, "bootstrap.json"), {
        "unit": "image", "n": args.bootstrap, "seed": args.seed,
        "per_probe": {k: {"test_auc_ci95": v.get("test_auc_ci95"),
                          "test_ap_ci95": v.get("test_ap_ci95"),
                          "auc_valid_boot": v.get("auc_valid_boot"),
                          "ap_valid_boot": v.get("ap_valid_boot")}
                      for k, v in results.items()},
    })
    gate = {
        "stage": "run6_discriminability", "status": status,
        "val_screen_status": None,
        "preflight_version": PROFILE_VERSION, "protocol_tag": PROTOCOL_TAG,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "selected": selected, "gate_impl": gate_impl, "gate_stat": gate_stat,
        "gates": g, "n_pos_eval": n_pos, "n_neg_eval": n_neg,
        "n_fit_objects": len(tr_rows), "n_eval_objects": len(te_rows),
        "winner_test_auc": auc, "winner_test_ap": ap,
        "handcrafted_baseline_auc": hand_auc, "full_baseline_auc": full_auc,
        "amplitude_baseline_auc": amp_auc,
        "elapsed_s": round(elapsed, 1), "budget_sec": args.budget_sec,
        "timed_out": bool(tr_to or te_to),
        "decision": ("80K ALLOWED (K0-K7 PASS)" if status == "PASS" else
                     "NO_80K (preflight not PASS) — 不新增证据通道、不改阈值重跑"),
    }
    write_json(os.path.join(args.out_dir, "gate.json"), gate)
    lines = []
    for fn in sorted(os.listdir(args.out_dir)):
        p = os.path.join(args.out_dir, fn)
        if os.path.isfile(p) and fn != "SHA256SUMS.txt":
            lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {fn}")
    with open(os.path.join(args.out_dir, "SHA256SUMS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[R6-DISC] selected={selected} impl={gate_impl} stat={gate_stat}", flush=True)
    print(f"[R6-DISC] winner AUC={auc} AP={ap} | hand={hand_auc} full={full_auc} "
          f"amp={amp_auc}", flush=True)
    for k in sorted(g):
        print(f"[GATE] {k} pass={g[k]['pass']} value={g[k]['value']}", flush=True)
    print(f"[R6-DISC] STATUS={status} -> {os.path.join(args.out_dir, 'gate.json')}",
          flush=True)
    return 0 if status == "PASS" else 2


def _k(name, ok, value, threshold, note=""):
    return {"pass": bool(ok), "value": value, "threshold": threshold, "note": note}


def main():
    ap = argparse.ArgumentParser(
        description="Run6 §4 只读可分辨性前测（冻结 M1 特征 → FP vs 真实 small 变化）")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--ckpt-root",
                    default="/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM")
    ap.add_argument("--dataset-root", default="/share_datasets/CD/SYSU-CD-256")
    ap.add_argument("--train-list", default="/share_datasets/CD/SYSU-CD-256/list/train.txt")
    ap.add_argument("--test-list", default="/share_datasets/CD/SYSU-CD-256/list/test.txt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug only；非零时状态恒为 DRY_ONLY，永不 PASS")
    ap.add_argument("--budget-sec", type=float, default=BUDGET_S)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    sys.exit(main_run(args))


if __name__ == "__main__":
    main()
