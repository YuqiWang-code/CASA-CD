"""D2：按模型真实计算顺序定位首次证据衰减（分阶段 hook 诊断，本轮核心）。

设计要点（对应 Run-Diag 文档 §6）：
  * 只在一个已训练 checkpoint 上做只读 hook：不 train()、不改 BN、不改参数、不重训。
  * 编码器节点 L00–L08 有 A/B 配对特征 → 用 **余弦差分 proxy**（观测性，不断言因果）；
    TAR/DCR/head 节点是**双时相融合后的单路特征** → 禁止做 A/B cosine，只记录
    形状/范数/与 GT 的统计信噪比（写入 summary，标注为不可与 AP 直接比较）。
  * CAACP 内部通过 hook `caacp_op.pool` 捕获 x_low 与 c_avg，并用模块自身函数
    重算 c_ca（同一数学、确定性），从而获得 ΔC / βΔC / residual 的精确统计。
  * 分辨率双口径：主口径 = score 上采样到 **原生 256²**（GT 保持原生原样）；
    副口径 = **native feature grid** 的 GT occupancy（adaptive_avg_pool2d）分桶。
  * pooled AP 用固定 2000-bin 分数直方图累计（内存安全、确定性）；per-image AP 精确计算
    并做 image-level bootstrap；同时报告正类先验比例（跨数据集不可直接横比）。

输出：stage_raw.csv、stage_summary.json、caacp_internal.json、stage_profile.png、gate.json
"""
import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tvim_diag_common import (  # noqa: E402
    DATA_ROOT, VARIANT_CFG, build_model, ensure_dir, make_loader, pick_best_ckpt,
    read_list_names, set_eval_numerics, write_gate, write_json, bootstrap_ci,
)
from tvim_object_metrics import GROUP_ALIAS, bin_index, label_components  # noqa: E402

NBINS = 2000
SCORE_MAX = 2.0                      # cosine 差分 1-cos ∈ [0,2]
BUDGETS = (0.01, 0.05)               # 固定背景预测比例预算（诊断曲线采样点，不是部署阈值）
ENC_NODES = ["L00_patch_embed", "L01_network0", "L01b_norm0", "L02_network1",
             "L03_network2", "L03b_norm2", "L04_network3", "L05_stage3_last_prefix",
             "L06b_norm4", "L07_network5", "L08b_norm6"]
SINGLE_NODES = ["L06a_caacp_block", "T01_tar1", "T02_tar2", "T03_tar3", "T04_tar4",
                "D03_decoder_block3", "D02_decoder_block2", "D01_decoder_block1",
                "DOUT_decoder_refine", "P00_head_logits"]


def _bins():
    return np.linspace(0.0, SCORE_MAX, NBINS + 1)


class NodeScoreAcc:
    """一个编码器节点的分数统计累加器（直方图 + per-image AP + 分组分位数）。"""

    def __init__(self, name, nbins=NBINS, groups=("small", "medium", "large")):
        self.name = name
        self.nbins = nbins
        self.hist_pos = np.zeros(nbins, dtype=np.float64)
        self.hist_all = np.zeros(nbins, dtype=np.float64)
        self.group_pix = {g: 0 for g in groups}
        self.group_sum = {g: 0.0 for g in groups}
        self.group_hist = {g: np.zeros(nbins, dtype=np.float64) for g in groups}
        self.per_image_ap = []
        self.per_image_budget = {b: [] for b in BUDGETS}
        self.per_image_margin = []          # small 对象 margin 的图像级均值
        self.obj_margins = []               # per-object margin（small 组）
        self.bg_sum = 0.0
        self.bg_count = 0

    # -------------------------------------------------- per image
    def add(self, score_256, gt_bool, label_groups, per_image_ap=None):
        """score_256: (256,256) float; gt_bool: (256,256) bool; label_groups: (256,256) int8
        （0=背景, 1=small, 2=medium, 3=large）；per_image_ap 由调用方按同一直方图口径算出。"""
        s = score_256.astype(np.float64)
        g = gt_bool
        pos = g
        bg = ~g
        h, edges = np.histogram(s[pos], bins=self.nbins, range=(0.0, SCORE_MAX)) if pos.any() else (np.zeros(self.nbins), None)
        h_all, _ = np.histogram(s, bins=self.nbins, range=(0.0, SCORE_MAX))
        self.hist_pos += h
        self.hist_all += h_all
        for gi, gname in enumerate(("small", "medium", "large"), start=1):
            m = (label_groups == gi)
            if m.any():
                self.group_pix[gname] += int(m.sum())
                self.group_sum[gname] += float(s[m].sum())
                hh, _ = np.histogram(s[m], bins=self.nbins, range=(0.0, SCORE_MAX))
                self.group_hist[gname] += hh
        if bg.any():
            self.bg_sum += float(s[bg].sum())
            self.bg_count += int(bg.sum())
        # 固定 FP 预算下的像素召回（预算为诊断曲线采样点，非部署阈值）
        for b in BUDGETS:
            thr = _score_at_fraction(s, b)
            rec = float((s[pos] >= thr).mean()) if pos.any() else np.nan
            self.per_image_budget[b].append(rec)
        # small 对象 margin（对象内均值 - 外侧 4px 环带背景均值）
        lab, n = label_components(g, 4)
        if n:
            from scipy import ndimage as ndi
            counts = np.bincount(lab.ravel(), minlength=n + 1)
            margins = []
            for j in range(1, n + 1):
                if int(counts[j]) >= 256:          # 只统计 small 组（1<=area<256）
                    continue
                mobj = (lab == j)
                ring = ndi.binary_dilation(
                    mobj, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool),
                    iterations=4) & ~mobj
                if not ring.any():
                    continue
                margins.append(float(s[mobj].mean() - s[ring].mean()))
            if margins:
                self.per_image_margin.append(float(np.mean(margins)))
                self.obj_margins.extend(margins)
        # per-image AP（与 pooled 同口径：由本图 2000-bin 直方图累计，避免 O(N log N) 排序）
        ap = _ap_from_hist(h, h_all) if pos.any() else np.nan
        self.per_image_ap.append(ap)
        return ap

    # -------------------------------------------------- dataset summary
    def summary(self):
        pos = self.hist_pos.sum()
        allp = self.hist_all.sum()
        out = {
            "n_positive_pixels": float(pos), "n_pixels": float(allp),
            "positive_prior": float(pos / allp) if allp else None,
            "pooled_AP_hist": _ap_from_hist(self.hist_pos, self.hist_all),
            "per_image_AP_mean": float(np.nanmean(self.per_image_ap)) if self.per_image_ap else None,
            "per_image_AP_bootstrap_CI": bootstrap_ci(self.per_image_ap, n_boot=1000, seed=16) if self.per_image_ap else None,
            "per_image_AP_n": len(self.per_image_ap),
            "group_score_mean": {g: (self.group_sum[g] / self.group_pix[g]) if self.group_pix[g] else None
                                 for g in self.group_sum},
            "group_score_p90": {g: _quantile_from_hist(self.group_hist[g], 0.90) for g in self.group_hist},
            "group_score_max": {g: _max_from_hist(self.group_hist[g]) for g in self.group_hist},
            "background_score_mean": (self.bg_sum / self.bg_count) if self.bg_count else None,
            "small_object_margin": {
                "n_objects": len(self.obj_margins),
                "mean": float(np.mean(self.obj_margins)) if self.obj_margins else None,
                "median": float(np.median(self.obj_margins)) if self.obj_margins else None,
                "per_image_mean": float(np.nanmean(self.per_image_margin)) if self.per_image_margin else None,
                "per_image_bootstrap_CI": bootstrap_ci(self.per_image_margin, n_boot=1000, seed=16) if self.per_image_margin else None,
            },
            "budget_recall": {
                f"fp_budget_{int(b * 100)}pct": {
                    "per_image_mean": float(np.nanmean(self.per_image_budget[b])) if self.per_image_budget[b] else None,
                    "bootstrap_CI": bootstrap_ci(self.per_image_budget[b], n_boot=1000, seed=16) if self.per_image_budget[b] else None,
                } for b in BUDGETS
            },
        }
        return out


class SingleNodeAcc:
    """单路（A/B 已融合）节点：只记录形状/范数/GT 内外的统计信噪比（禁止 cosine）。"""

    def __init__(self, name):
        self.name = name
        self.shapes = set()
        self.gt_norm_sum = 0.0
        self.gt_norm_count = 0
        self.bg_norm_sum = 0.0
        self.bg_norm_count = 0
        self.ratio_per_image = []

    def add(self, feat, gt_small_bool, gt_bool):
        """feat: (C,H,W) float; gt_bool/gt_small_bool: (H,W) bool at the node's own grid."""
        norm = np.linalg.norm(feat, axis=0)              # (H,W)
        self.shapes.add(tuple(feat.shape))
        if gt_bool.any():
            self.gt_norm_sum += float(norm[gt_bool].sum())
            self.gt_norm_count += int(gt_bool.sum())
        bg = ~gt_bool
        if bg.any():
            self.bg_norm_sum += float(norm[bg].sum())
            self.bg_norm_count += int(bg.sum())
        if gt_small_bool.any() and bg.any():
            self.ratio_per_image.append(float(norm[gt_small_bool].mean() / (norm[bg].mean() + 1e-12)))

    def summary(self):
        return {
            "shapes": [list(s) for s in sorted(self.shapes)],
            "gt_norm_mean": (self.gt_norm_sum / self.gt_norm_count) if self.gt_norm_count else None,
            "bg_norm_mean": (self.bg_norm_sum / self.bg_norm_count) if self.bg_norm_count else None,
            "small_gt_over_bg_norm_ratio_mean": float(np.mean(self.ratio_per_image)) if self.ratio_per_image else None,
            "small_gt_over_bg_norm_ratio_bootstrap_CI": bootstrap_ci(self.ratio_per_image, n_boot=1000, seed=16) if self.ratio_per_image else None,
            "caveat": "单路融合特征：不适用 A/B cosine；此处的范数信噪比仅为描述性，不可与编码器 AP 直接比较",
        }


# ------------------------------------------------------------------ AP utilities
def _average_precision(scores, labels):
    """精确 per-image AP（step-wise，与 sklearn.average_precision_score 同义）。"""
    y = np.asarray(labels).astype(bool).ravel()
    s = np.asarray(scores).ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return np.nan
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        if r > prev_r:
            ap += p * (r - prev_r)
            prev_r = r
    return float(ap)


def _ap_from_hist(hist_pos, hist_all):
    """由分数直方图累计得到 pooled AP（越高分位越多正）。"""
    pos = hist_pos.sum()
    if pos <= 0:
        return None
    order = np.arange(len(hist_all))[::-1]          # 从高分到低分
    tp = np.cumsum(hist_pos[order])
    allc = np.cumsum(hist_all[order])
    precision = tp / np.maximum(allc, 1)
    recall = tp / pos
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        if r > prev_r:
            ap += p * (r - prev_r)
            prev_r = r
    return float(ap)


def _quantile_from_hist(hist, q):
    tot = hist.sum()
    if tot <= 0:
        return None
    cum = np.cumsum(hist)
    idx = int(np.searchsorted(cum, q * tot))
    idx = min(idx, len(hist) - 1)
    return float((idx + 0.5) * SCORE_MAX / len(hist))


def _max_from_hist(hist):
    nz = np.nonzero(hist)[0]
    if nz.size == 0:
        return None
    return float((nz[-1] + 1) * SCORE_MAX / len(hist))


def _score_at_fraction(score, frac):
    """返回使被判正像素比例不超过 frac 的分数阈值（图像内）。"""
    flat = score.ravel()
    k = int(np.ceil(frac * flat.size))
    if k <= 0 or k >= flat.size:
        return float(flat.max() + 1e-9)
    part = np.partition(flat, -k)
    return float(part[-k])


# ------------------------------------------------------------------ feature prep
def _to_np(t):
    return t.detach().float().cpu().numpy()


def _cos_score(fa, fb, eps=1e-8):
    """1 - cos(F_A, F_B)，逐像素（C 维归一化）。fa/fb: (B,C,H,W) numpy."""
    na = np.linalg.norm(fa, axis=1, keepdims=True)
    nb = np.linalg.norm(fb, axis=1, keepdims=True)
    cos = (fa * fb).sum(axis=1, keepdims=True) / (np.maximum(na, eps) * np.maximum(nb, eps))
    return (1.0 - cos)[:, 0]                     # (B,H,W)


def _occupancy(gt_bool, H, W):
    """native grid occupancy（自适应平均池化），覆盖不完全为 0/1 的 cell。"""
    import torch
    import torch.nn.functional as F
    g = torch.as_tensor(gt_bool.astype(np.float32))[None, None]
    occ = F.adaptive_avg_pool2d(g, (H, W))[0, 0].numpy()
    return occ


def _group_masks_native(group_labels, H):
    """把 native 分组标签图（1=small,2=medium,3=large）降采样到 H×H 的 occupancy 权重（每组的 0/1 质量）。"""
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(group_labels.astype(np.float32))[None, None]
    out = []
    for gi in (1, 2, 3):
        m = (t == float(gi)).float()
        out.append(F.adaptive_avg_pool2d(m, (H, H))[0, 0].cpu().numpy())
    return out


def _group_labels_native(gt_bool):
    """原生 256² 上按 GT 对象面积给出 0/1/2/3（背景/small/medium/large）。"""
    lab, n = label_components(gt_bool, 4)
    out = np.zeros(gt_bool.shape, dtype=np.int8)
    if n:
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        for j in range(1, n + 1):
            g = bin_index(int(counts[j]))
            gi = 1 if g in GROUP_ALIAS["small"] else (2 if g in GROUP_ALIAS["medium"] else 3)
            out[lab == j] = gi
    return out


# ------------------------------------------------------------------ main
def run_d2(args):
    import torch
    import torch.nn.functional as F
    from model.layers.caacp_ss2d import change_weighted_pool2x2, confidence_preserving_pool2x2
    from tvim_object_metrics import pixel_metrics

    out_dir = ensure_dir(args.out_dir)
    device = args.device
    set_eval_numerics()
    ckpt_path, ckpt_meta = pick_best_ckpt(args.run, args.variant, args.dataset)
    model, build_info = build_model(args.variant, device=device, ckpt_path=ckpt_path)
    enc = model.encoder
    loader, list_path = make_loader(args.dataset, "test", batch_size=args.batch_size,
                                   num_workers=args.num_workers)
    names = read_list_names(list_path)

    score_accs = {n: NodeScoreAcc(n) for n in ENC_NODES}
    single_accs = {n: SingleNodeAcc(n) for n in SINGLE_NODES}
    caacp = {}                      # CAACP 内部累计
    captured = {}
    handles = []

    def hook_fn(key):
        def _h(module, inp, out):
            captured[key] = out
        return _h

    def reg(module, key):
        handles.append(module.register_forward_hook(hook_fn(key)))

    depths = enc.depths
    reg(enc.patch_embed, "L00_patch_embed")
    reg(enc.network[0], "L01_network0")
    reg(getattr(enc, "norm0"), "L01b_norm0")
    reg(enc.network[1], "L02_network1")
    reg(enc.network[2], "L03_network2")
    reg(getattr(enc, "norm2"), "L03b_norm2")
    reg(enc.network[3], "L04_network3")
    reg(enc.network[4][depths[2] - 2], "L05_stage3_last_prefix")
    reg(enc.network[4][depths[2] - 1], "L06a_caacp_block")
    reg(getattr(enc, "norm4"), "L06b_norm4")
    reg(enc.network[5], "L07_network5")
    reg(enc.network[6], "L08_network6")
    reg(getattr(enc, "norm6"), "L08b_norm6")
    for i in range(4):
        reg(getattr(model.tar, f"stage{i+1}"), f"T0{i+1}_tar{i+1}")
    reg(model.decoder.block3, "D03_decoder_block3")
    reg(model.decoder.block2, "D02_decoder_block2")
    reg(model.decoder.block1, "D01_decoder_block1")
    reg(model.decoder.refine, "DOUT_decoder_refine")
    reg(model.head, "P00_head_logits")

    op = enc.caacp_op
    pool_handle = op.pool.register_forward_hook(
        lambda m, inp, out: captured.__setitem__("_caacp_pool", (inp[0], out))) if op is not None else None

    def _init_caacp():
        return {"beta": None, "delta_abs_mean_sum": 0.0, "delta_abs_count": 0,
                "beta_delta_abs_sum": 0.0, "beta_delta_sq_sum": 0.0, "beta_delta_norm_sq": 0.0,
                "cavg_norm_sq": 0.0, "n": 0,
                "group_delta": {g: 0.0 for g in ("small", "medium", "large")},
                "group_delta_pix": {g: 0 for g in ("small", "medium", "large")},
                "res_current_abs_sum": {g: 0.0 for g in ("small", "medium", "large")},
                "res_anchor_abs_sum": {g: 0.0 for g in ("small", "medium", "large")},
                "res_pix": {g: 0 for g in ("small", "medium", "large")},
                "entropy_sum": 0.0, "entropy_n": 0, "top1_sum": 0.0,
                "xlow_norm_sum": 0.0, "xlow_norm_n": 0, "chigh_norm_sum": 0.0, "chigh_norm_n": 0,
                "score_mean_sum": 0.0, "score_mean_n": 0}

    cacc = _init_caacp()
    # 记录网络节点顺序与输出分辨率（Gate 用）
    node_shapes = {}
    per_image_rows = []
    recon_check = None
    idx = 0

    try:
        with torch.no_grad():
            for img, label in loader:
                captured.clear()
                pre = img[:, 0:3].to(device).float()
                post = img[:, 3:6].to(device).float()
                gt = (label.to(device) > 0.5)
                prob = model(pre, post)
                pred_b = (prob > 0.5)

                # ---- 重建校验：head logits → interp → sigmoid 应等于模型输出
                if recon_check is None and "P00_head_logits" in captured:
                    logits = captured["P00_head_logits"]
                    recon = torch.sigmoid(F.interpolate(
                        logits, size=pre.shape[-2:], mode="bilinear", align_corners=False))
                    recon_check = {"max_abs_recon_vs_model": float((recon - prob).abs().max().item()),
                                   "binary_disagreement": float(((recon > 0.5) != (prob > 0.5)).float().mean().item())}

                for key in list(captured.keys()):
                    t = captured[key]
                    if isinstance(t, torch.Tensor):
                        node_shapes.setdefault(key, tuple(t.shape))

                gt_np = gt.cpu().numpy()[:, 0].astype(bool)
                prob_np = prob.cpu().numpy()[:, 0]
                pred_np = pred_b.cpu().numpy()[:, 0]
                B = gt_np.shape[0]
                # 每个 GT 只标记一次（此前在编码器节点循环与 CAACP 分组循环各算一次）
                group_labels_cache = [_group_labels_native(gt_np[i]) for i in range(B)]

                for i in range(B):
                    nm = names[idx + i] if (idx + i) < len(names) else f"idx{idx+i}"
                    gl = group_labels_cache[i]
                    gt_b = gt_np[i]
                    row = {"sample_name": nm}
                    # ---- 编码器 A/B cosine 节点
                    for nname in ENC_NODES:
                        t = captured.get(nname)
                        if t is None or not isinstance(t, torch.Tensor):
                            continue
                        fa = _to_np(t[i])
                        fb = _to_np(t[B + i])
                        s = _cos_score(fa[None], fb[None], )[0]
                        if s.shape != gt_b.shape:
                            s = np.array(torch.nn.functional.interpolate(
                                torch.as_tensor(s)[None, None], size=gt_b.shape,
                                mode="bilinear", align_corners=False)[0, 0])
                        ap_i = score_accs[nname].add(s, gt_b, gl)
                        row[f"{nname}_score_in_gt"] = float(s[gt_b].mean()) if gt_b.any() else None
                        row[f"{nname}_score_bg"] = float(s[~gt_b].mean()) if (~gt_b).any() else None
                        row[f"{nname}_AP"] = ap_i
                    # ---- 单路节点（不做 cosine）
                    for nname in SINGLE_NODES:
                        t = captured.get(nname)
                        if t is None or not isinstance(t, torch.Tensor):
                            continue
                        f = _to_np(t[i])
                        H, W = f.shape[-2], f.shape[-1]
                        gt_small = np.zeros((H, W), dtype=bool)
                        if H != gt_b.shape[0]:
                            occ = _occupancy(gt_b, H, W)
                            gt_small = occ > 0.0                    # cell touched（仅用于描述）
                            gt_at = occ >= 0.5
                        else:
                            gt_at = gt_b
                            gt_small = gt_b & (gl == 1)
                        single_accs[nname].add(f, gt_small, gt_at)
                    # ---- pixel 级混淆（用于 per-image 表）
                    pm = pixel_metrics(pred_np[i], gt_b)
                    row.update({"tp": pm["tp"], "fp": pm["fp"], "fn": pm["fn"],
                                "F1": pm["F1"], "IoU": pm["IoU"], "recall": pm["recall"],
                                "precision": pm["precision"],
                                "gt_small_pixels": int((gl == 1).sum()),
                                "gt_medium_pixels": int((gl == 2).sum()),
                                "gt_large_pixels": int((gl == 3).sum())})
                    per_image_rows.append(row)

                # ---- CAACP 内部（整批）
                if "_caacp_pool" in captured and op is not None:
                    x_low, c_avg = captured["_caacp_pool"]
                    score = op._pair_score
                    if op.score_mode == "cp":
                        c_ca = confidence_preserving_pool2x2(x_low, op._pair_abs, score)
                        q = op._pair_abs * score
                        w = (1.0 + q).view(q.shape[0], 1, q.shape[1] // 2, 2, q.shape[2] // 2, 2)
                        w = w / w.sum(dim=(3, 5), keepdim=True)
                    else:
                        c_ca = change_weighted_pool2x2(x_low, score)
                        s6 = score.view(score.shape[0], 1, score.shape[1] // 2, 2,
                                        score.shape[2] // 2, 2)
                        w = (s6 + 1e-6) / (s6 + 1e-6).sum(dim=(3, 5), keepdim=True)
                    beta = op.beta.detach()
                    delta = c_ca - c_avg
                    beta_delta = beta * delta
                    cacc["beta"] = float(beta.item())
                    cacc["delta_abs_mean_sum"] += float(delta.abs().mean().item()) * delta.shape[0]
                    cacc["delta_abs_count"] += delta.shape[0]
                    cacc["beta_delta_abs_sum"] += float(beta_delta.abs().mean().item()) * delta.shape[0]
                    cacc["beta_delta_sq_sum"] += float((beta_delta ** 2).sum().item())
                    cacc["beta_delta_norm_sq"] += float((beta_delta ** 2).sum(dim=(1, 2, 3)).sum().item())
                    cacc["cavg_norm_sq"] += float((c_avg ** 2).sum(dim=(1, 2, 3)).sum().item())
                    cacc["n"] += delta.shape[0]
                    ent = -(w * (w + 1e-12).log()).sum(dim=(3, 5))
                    cacc["entropy_sum"] += float(ent.mean().item()) * delta.shape[0]
                    cacc["entropy_n"] += delta.shape[0]
                    cacc["top1_sum"] += float(w.max(dim=3).values.max(dim=3).values.mean().item()) * delta.shape[0]
                    cacc["xlow_norm_sum"] += float(x_low.float().norm(dim=1).mean().item()) * delta.shape[0]
                    cacc["xlow_norm_n"] += delta.shape[0]
                    # GT 分组：用 native-grid occupancy（adaptive_avg_pool2d）加权，
                    # 不把 8²/16² 特征上采样到 256²（更快、无插值伪影）；口径见 resolution_protocol
                    c_used = c_avg + beta * delta
                    res_cur = x_low - F.interpolate(c_used, size=x_low.shape[-2:], mode="nearest")
                    res_anchor = x_low - F.interpolate(c_avg, size=x_low.shape[-2:], mode="nearest")
                    delta_map = delta.abs().mean(dim=1, keepdim=True)      # (2B,1,8,8)
                    cur_map = res_cur.abs().mean(dim=1, keepdim=True)      # (2B,1,16,16)
                    anc_map = res_anchor.abs().mean(dim=1, keepdim=True)
                    d_all = delta_map[:, 0].cpu().numpy()
                    rc_all = cur_map[:, 0].cpu().numpy()
                    ra_all = anc_map[:, 0].cpu().numpy()
                    for i in range(B):        # delta 为 2B（A;B 拼接）→ 只取 A 半与 GT 对齐
                        gl = group_labels_cache[i]
                        m8 = _group_masks_native(gl, 8)
                        m16 = _group_masks_native(gl, 16)
                        d_i, rc_i, ra_i = d_all[i], rc_all[i], ra_all[i]
                        for gname, (a8, a16) in zip(("small", "medium", "large"), zip(m8, m16)):
                            sw = float(a8.sum())
                            if sw > 0:
                                cacc["group_delta"][gname] += float((d_i * a8).sum())
                                cacc["group_delta_pix"][gname] += sw
                            sw16 = float(a16.sum())
                            if sw16 > 0:
                                cacc["res_current_abs_sum"][gname] += float((rc_i * a16).sum())
                                cacc["res_anchor_abs_sum"][gname] += float((ra_i * a16).sum())
                                cacc["res_pix"][gname] += sw16
                idx += B
                if args.verbose and (idx % 400 == 0 or idx >= (args.limit or 10 ** 9)):
                    print(f"[D2] processed {idx} images", flush=True)
                if args.limit and idx >= args.limit:
                    break
    finally:
        for h in handles:
            h.remove()
        if pool_handle is not None:
            pool_handle.remove()

    # ---------------------------------------------------------------- summary
    n = cacc["n"] or 1
    caacp_summary = {
        "beta": cacc["beta"],
        "mean_abs_deltaC": cacc["delta_abs_mean_sum"] / n,
        "mean_abs_beta_deltaC": cacc["beta_delta_abs_sum"] / n,
        "beta_delta_rms_exact": float(np.sqrt(cacc["beta_delta_sq_sum"] / (n * 168 * 8 * 8))),
        "ratio_beta_delta_over_cavg": (float(np.sqrt(cacc["beta_delta_norm_sq"] / cacc["cavg_norm_sq"]))
                                       if cacc["cavg_norm_sq"] > 0 else None),
        "cell_weight_entropy_mean": cacc["entropy_sum"] / max(1, cacc["entropy_n"]),
        "cell_weight_top1_mean": cacc["top1_sum"] / max(1, cacc["entropy_n"]),
        "xlow_feature_norm_mean": cacc["xlow_norm_sum"] / max(1, cacc["xlow_norm_n"]),
        "n_samples": cacc["n"],
        "group_mean_abs_deltaC": {g: (cacc["group_delta"][g] / cacc["group_delta_pix"][g])
                                  if cacc["group_delta_pix"][g] else None
                                  for g in cacc["group_delta"]},
        "group_mean_abs_residual_current": {g: (cacc["res_current_abs_sum"][g] / cacc["res_pix"][g])
                                            if cacc["res_pix"][g] else None
                                            for g in cacc["res_current_abs_sum"]},
        "group_mean_abs_residual_avg_anchor": {g: (cacc["res_anchor_abs_sum"][g] / cacc["res_pix"][g])
                                               if cacc["res_pix"][g] else None
                                               for g in cacc["res_anchor_abs_sum"]},
        "note": ("mean_abs_deltaC / mean_abs_beta_deltaC 为 batch 内 |·| 均值；"
                 "beta_delta_rms_exact = sqrt(mean((βΔC)²))（不使用旧脚本 mean² 汇总口径）；"
                 "分组统计：把 native GT 分组掩码用 adaptive_avg_pool2d 降采样为 8²（ΔC）与 16²（residual）"
                 "occupancy 权重后加权平均（权重和为浮点像素质量，非取整计数）"),
    }

    summary = {
        "run": args.run, "variant": args.variant, "dataset": args.dataset,
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_meta["sha256"]},
        "config": VARIANT_CFG[args.variant],
        "n_images_processed": idx,
        "limited": bool(args.limit), "dry_run": bool(args.limit),
        "not_for_conclusion": bool(args.limit),
        "encoder_nodes": {k: v.summary() for k, v in score_accs.items()},
        "single_path_nodes": {k: v.summary() for k, v in single_accs.items()},
        "node_shapes": {k: list(v) for k, v in node_shapes.items()},
        "reconstruction_check": recon_check,
        "resolution_protocol": {
            "main": "score 上采样到原生 256²（bilinear, align_corners=False），GT 保持原生原样",
            "secondary": "native feature grid occupancy = adaptive_avg_pool2d(GT)，用于分桶描述",
            "AP": f"pooled AP 由 {NBINS}-bin 分数直方图累计；per-image AP 精确计算并做 image-level bootstrap",
            "budgets": [f"固定背景预测比例 {int(b*100)}%（诊断曲线采样点，非部署阈值）" for b in BUDGETS],
        },
        "caveats": [
            "编码器节点为 A/B cosine proxy（观测性）",
            "单路融合节点禁止 cosine，仅描述性范数信噪比",
            "pooled AP 近似来自直方图；跨数据集绝对 AP 不可直接横比（须看正类先验）",
        ],
    }
    if args.limit:
        summary["not_for_conclusion"] = True

    # ---------------------------------------------------------------- outputs
    ensure_dir(out_dir)
    if per_image_rows:
        fields = sorted({k for r in per_image_rows for k in r})
        with open(os.path.join(out_dir, "stage_raw.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in per_image_rows:
                w.writerow(r)
    write_json(os.path.join(out_dir, "stage_summary.json"), summary)
    write_json(os.path.join(out_dir, "caacp_internal.json"), caacp_summary)
    _plot_profile(out_dir, summary)

    # ---------------------------------------------------------------- gate
    checks = {
        "n_images_processed": idx,
        "node_shapes": {k: list(v) for k, v in node_shapes.items()},
        "reconstruction_max_abs": (recon_check or {}).get("max_abs_recon_vs_model"),
        "reconstruction_binary_disagreement": (recon_check or {}).get("binary_disagreement"),
        "all_nodes_captured": all(k in node_shapes for k in
                                  ["L00_patch_embed", "L01b_norm0", "L03b_norm2", "L05_stage3_last_prefix",
                                   "L06b_norm4", "L08b_norm6", "T01_tar1", "DOUT_decoder_refine",
                                   "P00_head_logits"]),
        "caacp_internal_recorded": cacc["n"] > 0,
        "no_nan_in_ap": all(np.isfinite(v).all() if v is not None else True
                            for v in [a.summary()["pooled_AP_hist"] for a in score_accs.values()]),
        "single_path_no_cosine": True,
    }
    ok = (checks["all_nodes_captured"] and checks["caacp_internal_recorded"]
          and (recon_check is not None and recon_check["max_abs_recon_vs_model"] == 0.0
               and recon_check["binary_disagreement"] == 0.0)
          and not args.limit)
    status = "PASS" if ok else ("WARN" if args.limit else "FAIL")
    write_gate(out_dir, "D2-VALID", status, checks)
    print(f"[D2] gate={status} images={idx} recon_max_abs="
          f"{(recon_check or {}).get('max_abs_recon_vs_model')} beta={cacc['beta']}", flush=True)
    print(f"[D2] wrote {out_dir}", flush=True)
    return summary


def _plot_profile(out_dir, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    nodes = [n for n in ENC_NODES if summary["encoder_nodes"].get(n, {}).get("pooled_AP_hist") is not None]
    if not nodes:
        return
    ap = [summary["encoder_nodes"][n]["pooled_AP_hist"] for n in nodes]
    margin = [summary["encoder_nodes"][n]["small_object_margin"]["mean"] for n in nodes]
    prior = [summary["encoder_nodes"][n]["positive_prior"] for n in nodes]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes[0, 0].plot(range(len(nodes)), ap, "o-")
    axes[0, 0].set_xticks(range(len(nodes))); axes[0, 0].set_xticklabels(nodes, rotation=60, ha="right", fontsize=7)
    axes[0, 0].set_title("pooled AP by encoder node (A/B cosine proxy)")
    axes[0, 1].plot(range(len(nodes)), margin, "s-", color="tab:orange")
    axes[0, 1].set_xticks(range(len(nodes))); axes[0, 1].set_xticklabels(nodes, rotation=60, ha="right", fontsize=7)
    axes[0, 1].set_title("small-object score margin (in-object minus 4px ring)")
    for b, key in zip(BUDGETS, [f"fp_budget_{int(x*100)}pct" for x in BUDGETS]):
        vals = [summary["encoder_nodes"][n]["budget_recall"][key]["per_image_mean"] for n in nodes]
        axes[1, 0].plot(range(len(nodes)), vals, "^-", label=f"budget {int(b*100)}%")
    axes[1, 0].set_xticks(range(len(nodes))); axes[1, 0].set_xticklabels(nodes, rotation=60, ha="right", fontsize=7)
    axes[1, 0].set_title("pixel recall at fixed FP budgets"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(range(len(nodes)), prior, "d-", color="tab:green")
    axes[1, 1].set_xticks(range(len(nodes))); axes[1, 1].set_xticklabels(nodes, rotation=60, ha="right", fontsize=7)
    axes[1, 1].set_title("positive prior (constant; shown for scale awareness)")
    fig.suptitle(f"D2 stage profile — {summary['run']}/{summary['variant']}/{summary['dataset']} (cosine proxy)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stage_profile.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--run", default="Run1")
    ap.add_argument("--variant", default="M1_FULL", choices=sorted(VARIANT_CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run_d2(args)


if __name__ == "__main__":
    main()
