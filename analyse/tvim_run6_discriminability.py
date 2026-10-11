"""Run6 §4 只读可分辨性前测：冻结 M1 的既有特征里，能否把 FP-like 小预测与真实 small 变化分开？

**这是 Run6 的唯一开门依据，且必须先于任何 `models/` 改动执行。**
背景（Run6 设计文档 §1）：SYSU 小目标失败的形态是「判别失败」而非「证据缺失」——
FP 区域的原始变化能量（0.9730）**高于**真实漏检 small 变化（0.8626），且手工证据通道
（边缘重叠 / 变化幅值 / 纹理能量，含逐图背景归一化）对二者的判别 AUC 上限只有 **0.5992**。
因此 Run6 不新增证据通道，而是问：**既有学习到的多尺度表征里是否已存在可分离信息？**

对象口径（与 Run5 前测 / F14 判别力分析**逐字一致**）：
  * 正样本（真实 small 变化）= GT small 对象（面积 [1,255] 且 4 连通）中
    `IoU>=0.10 已匹配` **或** `像素 recall r<0.25` 者（test 上 n=327）；
  * 负样本（FP-like）= 预测 4 连通域（面积 [1,255]）中一次性最大权重匹配后**未匹配**者
    （test 上 n=2430）。

拟合/评估分离（防泄漏，Run6 §4.1）：
  * 拟合 = `train.txt` 的 D3 SHA1-hash **90% train** 部分（该 split 已固定，非新 seed）；
  * 评估 = `test.txt` 全量 4000 张；**test 不参与拟合、不选特征、不选阈值**；
  * 拟合图像是模型训练过的，分布偏乐观 ⇒ 门只认 **test 侧**数值。

特征（全部只依赖冻结模型输出或 A/B 图像，**不依赖 GT**）：见 `FEATURE_GROUPS`。
两候选：**C1 跨尺度一致性**、**C2 对象内部/边界一致性**；基线必须包含"模型自身置信度"与
"手工证据通道"，候选必须在基线之上**再**有增量（Run6 §9 R2）。

口径实现说明（P0，写进 `inputs.json`，见 Run6 README §2.2）：

1. **手工证据通道在与 F14 / Run5 FP 画像完全相同的像素空间计算**，即
   `Normalize → Scale → ToTensor` 之后的**归一化张量**（DataLoader 直接给出的 6 通道张量，
   通道 0:3 = A、3:6 = B）。这不是随意选择：冻结读数 `change_abs` FP=0.9730、
   `gradA` FP=1.8008 本身就是**归一化口径**下的数值（原始 [0,255] 下均值绝对差不可能 <2），
   ⇒ 只有同口径重算才能与冻结的手工基线 `frac_px_on_both_edges_1px` AUC=0.5992 直接对账。
   因此 `edge_field` / `change_abs` / `gradA` 与 Run5 `tvim_run5_fp_profile.py` 逐字同口径。
2. **C2 的"外环"不用 GT**：`ring = dilate_4px(O) \\ O`（4 邻域结构元）。Run5 FP 画像里的
   `R=(dilate(O)\\O)∩(~GT)` 只用于空间画像；作为**特征**会引入 GT 依赖（正样本的外环被
   系统性削去 GT 邻域）⇒ 本前测不采用，改用无 GT 的纯几何环 + 内部腐蚀 1px。
3. **C1 的"对象内平均变化证据"**按 Run6 文档 §5.1 的统计量口径：`f_s` 是 TAR 已融合的
   单路特征（无 A/B 之分），故用 `f_s.abs().mean(dim=1)` 的固定统计量，不新增张量入口。

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
from tvim_linear_probe import _hash_split  # noqa: E402
from tvim_object_metrics import (  # noqa: E402
    CONN4, GROUP_ALIAS, bin_index, match_objects, object_iou_matrix,
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
VAL_FRAC = 0.10                      # D3 SHA1-hash 10% probe-val 被排除，取 90% train 拟合

# ImageNet 统计 ×2（BGR 顺序，与 train.py / tvim_diag_common.MEAN/STD 一致）
MEAN6 = (0.406, 0.456, 0.485, 0.406, 0.456, 0.485)
STD6 = (0.225, 0.224, 0.229, 0.225, 0.224, 0.229)

# 冻结参考值（Diag1 / Run5；只用于对账，不参与任何门限计算）
FROZEN_REF = {
    "n_pos_expected": 327,           # gt_small_matched 35 + gt_small_missed 292
    "n_neg_expected": 2430,          # Diag1 冻结 "2430 unmatched <256px"
    "handcrafted_best_single_auc": 0.5992,   # frac_px_on_both_edges_1px（F14）
    "handcrafted_change_abs_auc": 0.4779,    # change_abs（F14，方向相反）
    "fp_change_abs_mean": 0.9730,    # FP 画像 (iii)
    "gt_missed_change_abs_mean": 0.8626,
}

# Run6 §4.2 硬门（数值在见结果之前冻结；本脚本不得修改）
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
K7_TIE_EPS = 0.005
K7_TIE_EPS_SLACK = 1e-12      # 仅抵消 IEEE754 表示误差，不改变 0.005 的语义

# 特征组（列名 → 见 compute_row）。注意：`mag_s_mean` 是 §4.1 要求的"‖Δ‖ 的 4 尺度平均"，
# 由 4 个尺度均值派生，不是新的证据通道。
FEATURE_GROUPS = {
    "amp": ["magY_s1", "magY_s2", "magY_s3", "magY_s4",
            "magT_s1", "magT_s2", "magT_s3", "magT_s4",
            "refine_mag", "mag_s_mean"],
    "hand": ["edge_any", "edge_both", "change_abs", "gradA"],
    "head": ["logit_mean", "logit_std", "logit_ring", "prob_mean", "prob_std", "prob_max"],
    "C1": ["Y_ratio_s1", "Y_ratio_s2", "Y_ratio_s3", "Y_ratio_s4",
           "Y_range", "Y_cv", "Y_entropy", "Y_spatstd_s1", "Y_spatstd_s2",
           "Y_spatstd_s3", "Y_spatstd_s4", "Y_spatstd_mean"],
    "C2": ["refine_interior_mag", "refine_interior_std", "refine_ring_mag",
           "refine_ring_std", "refine_contrast",
           "logit_interior_mean", "logit_interior_std", "logit_bnd_mean",
           "logit_bnd_std", "logit_contrast"],
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
# 需要在同一分辨率下池化的"逐对象固定统计量"输入的键（用于每图一次预计算）
_AMP_KEYS = [f"Y_s{s}" for s in range(1, 5)] + [f"T_s{s}" for s in range(1, 5)]


# ============================================================================ 纯函数
def channel_mean3(x3):
    """(3,H,W) → (H,W) 三通道均值，**不做任何反归一化**。

    手工证据通道必须与 F14 / Run5 FP 画像同口径（见模块 docstring 口径说明第 1 条）：
    DataLoader 给出的 `img` 已是 `Normalize → Scale → ToTensor` 之后的张量。
    """
    a = np.asarray(x3, dtype=np.float64)
    if a.ndim != 3 or a.shape[0] != 3:
        raise ValueError(f"[P0] expected (3,H,W), got {a.shape}")
    return a.mean(axis=0)


def check_finite(raw, name=""):
    """有限性自检（P0）：NaN/Inf 一旦进入特征矩阵就会静默污染 probe。"""
    f = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(f).all():
        raise ValueError(f"[P0] non-finite feature input in {name}")
    return float(f.min()), float(f.max())


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


def ring_mask(obj, iterations=4):
    """外环 `R = dilate_4px(O) \\ O`（4 邻域结构元，**不使用 GT**，Run6 README §2.2 第 2 条）。"""
    from scipy import ndimage as ndi
    d = ndi.binary_dilation(obj, structure=CONN4, iterations=iterations)
    return d & ~obj


def interior_mask(obj):
    """内部 `I = erode_1px(O)`（4 邻域结构元）。"""
    from scipy import ndimage as ndi
    return ndi.binary_erosion(obj, structure=CONN4, iterations=1)


def cross_scale_features(e_by_scale):
    """C1：跨尺度一致性统计量（`e_s` = 各尺度对象内平均变化证据）。"""
    e = np.asarray([max(float(v), 0.0) for v in e_by_scale], dtype=np.float64)
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
    """标准化 + L2 正则 logistic（全批梯度下降，确定性；无 sklearn 依赖）。

    L2 惩罚用 `lam * ||w||²/(2n)` 口径：`g[1:] += lam * w[1:] / n`（截距不正则）。
    """
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
    """用**拟合集**列中位数填补 NaN（ring 为空等导致的缺失），避免整行被丢弃。"""
    import warnings
    with warnings.catch_warnings():
        # 整列全 NaN（例如所有对象 ring 都为空）⇒ nanmedian 报警并返回 NaN，随后兜底为 0
        warnings.simplefilter("ignore", RuntimeWarning)
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


def _weighted_mean_std(map2d, w):
    tot = float(w.sum())
    if tot <= 0.0:
        return None, None
    mu = float((np.asarray(map2d, dtype=np.float64) * w).sum() / tot)
    var = float((w * (np.asarray(map2d, dtype=np.float64) - mu) ** 2).sum() / tot)
    return mu, float(np.sqrt(max(var, 0.0)))


# ============================================================================ 数据
def _loader(dataset_root, list_path, batch_size, num_workers, names_filter=None):
    import torch
    from dataset import dataset as myDataLoader
    from dataset import Transforms as myTransforms

    tf = myTransforms.Compose([myTransforms.Normalize(mean=list(MEAN6), std=list(STD6)),
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
    """取 `run_pass` 需要 hook 的模块（TAR 四级 temporal / refine / head）。"""
    kind = {"Y_s1": "stage1.temporal", "Y_s2": "stage2.temporal",
            "Y_s3": "stage3.temporal", "Y_s4": "stage4.temporal",
            "T_s1": "stage1", "T_s2": "stage2", "T_s3": "stage3", "T_s4": "stage4"}
    if key in kind:
        obj = model.tar
        for part in kind[key].split("."):
            obj = getattr(obj, part)
        return obj
    if key == "refine":
        return model.decoder.refine
    if key == "head":
        return model.head
    raise KeyError(key)


def _obj_frac(ctx, obj, size):
    """对象掩码在该分辨率下的覆盖率（每图每分辨率只算一次）。"""
    key = ("obj", size)
    fr = ctx["w_full"].get(key)
    if fr is None:
        fr = resize_frac(obj, size)
        ctx["w_full"][key] = fr
    return fr


def precompute_object_fracs(ctx, obj):
    """每个对象一次：把该对象在所有用到的分辨率上的覆盖率算好并写回缓存。

    分辨率集合 = {Y_s*(s=1..4), T_s*(s=1..4), refine 64², head 64², prob 256²,
    hand 256², interior/ring @64²}。这样 `compute_row` 里不再有任何 resize，
    只有小的加权求和（每图 0.7 个对象的真实规模下，resize 是主要开销）。
    """
    sizes = {ctx["shape"][f"Y_s{s}"] for s in range(1, 5)}
    sizes |= {ctx["shape"][f"T_s{s}"] for s in range(1, 5)}
    sizes |= {ctx["shape"]["refine"], ctx["shape"]["logit"], ctx["prob"].shape[-1], 256}
    for size in sizes:
        _obj_frac(ctx, obj, int(size))
    inter = interior_mask(obj)
    ring = ring_mask(obj, iterations=4)
    for tag, mask in (("interior", inter), ("ring", ring)):
        if mask.any():
            for size in {ctx["shape"]["refine"], ctx["shape"]["logit"]}:
                _mask_frac(ctx, mask, tag, int(size))
    return inter, ring


def _mask_frac(ctx, mask, tag, size):
    """对象派生掩码（interior / ring）在该分辨率下的覆盖率（每图每分辨率只算一次）。"""
    key = (tag, size)
    fr = ctx["w_full"].get(key)
    if fr is None:
        fr = resize_frac(mask, size)
        ctx["w_full"][key] = fr
    return fr


def image_context(cap, i, ab_img, prob):
    """每图一次预计算：与对象无关的逐分辨率覆盖图与逐像素统计量。

    `ab_img` = DataLoader 给出的 6 通道张量（numpy），`prob` = 256² 预测概率。
    A = `ab_img[i, 0:3]`、B = `ab_img[i, 3:6]`，**直接使用归一化张量**：
    与 F14 / Run5 FP 画像（`tvim_run5_fp_profile.py` 的 `edge_field(ab[i, 0:3])`）
    逐字同口径，冻结读数 `change_abs`=0.9730 / `gradA`=1.8008 即在此口径下取得。
    `ToTensor` 做过 `image[:, :, ::-1]`，故通道 0:3 为 BGR 顺序——而 `edge_field`
    与 `channel_mean3` 都是对三通道取均值，通道顺序不影响结果。
    """
    ctx = {"maps": {}, "shape": {}, "w_full": {}, "prob": prob}
    A = np.asarray(ab_img[i, 0:3], dtype=np.float64)
    B = np.asarray(ab_img[i, 3:6], dtype=np.float64)
    check_finite(A, name=f"A[{i}]")
    check_finite(B, name=f"B[{i}]")
    magA, edgeA = edge_field(A)
    _magB, edgeB = edge_field(B)
    ctx["gradA"] = magA
    # 1px 邻域布尔场只存两份布尔图（每个对象池化时再 OR/AND），避免额外的 256² 副本
    ctx["edgeA_d"] = dilate1(edgeA)
    ctx["edgeB_d"] = dilate1(edgeB)
    ctx["chg"] = np.abs(channel_mean3(A) - channel_mean3(B))
    for key in _AMP_KEYS:
        t = cap[key][i]
        ctx["maps"][key] = t.float().abs().mean(dim=0).cpu().numpy()
        ctx["shape"][key] = int(t.shape[-1])
    ctx["maps"]["refine"] = cap["refine"][i].float().abs().mean(dim=0).cpu().numpy()
    ctx["maps"]["logit"] = cap["head"][i].float()[0].cpu().numpy()
    ctx["shape"]["refine"] = int(cap["refine"][i].shape[-1])
    ctx["shape"]["logit"] = int(cap["head"][i].shape[-1])
    return ctx


def compute_row(ctx, obj):
    """单对象特征行（全部不依赖 GT；obj 为 256² bool）。"""
    row = {}
    for s in range(1, 5):
        for pre, key in (("magY", f"Y_s{s}"), ("magT", f"T_s{s}")):
            row[f"{pre}_s{s}"] = pool_map(ctx["maps"][key],
                                          _obj_frac(ctx, obj, ctx["shape"][key]))
    e_scales = [row[f"magY_s{s}"] if row[f"magY_s{s}"] is not None else 0.0
                for s in range(1, 5)]
    row.update(cross_scale_features(e_scales))
    for s in range(1, 5):
        row[f"Y_spatstd_s{s}"] = _weighted_mean_std(
            ctx["maps"][f"Y_s{s}"], _obj_frac(ctx, obj, ctx["shape"][f"Y_s{s}"]))[1]
    ok_scales = [row[f"magY_s{s}"] for s in range(1, 5) if row[f"magY_s{s}"] is not None]
    row["mag_s_mean"] = float(np.mean(ok_scales)) if ok_scales else None
    row["n_scales_missing"] = int(4 - len(ok_scales))
    st = [row[f"Y_spatstd_s{s}"] for s in range(1, 5) if row[f"Y_spatstd_s{s}"] is not None]
    row["Y_spatstd_mean"] = float(np.mean(st)) if st else None

    # ---- C2：对象内部（erode 1px）与外环（dilate 4px \ O），均不使用 GT
    inter, ring = precompute_object_fracs(ctx, obj)
    row["interior_empty"] = int(not inter.any())
    row["ring_empty"] = int(not ring.any())
    r_mu = r_sd = l_mu = l_sd = None
    i_mu = i_sd = None
    l_ring = l_ring_sd = None
    if inter.any():
        i_mu, i_sd = _weighted_mean_std(
            ctx["maps"]["refine"],
            _mask_frac(ctx, inter, "interior", ctx["shape"]["refine"]))
        l_mu, l_sd = _weighted_mean_std(
            ctx["maps"]["logit"],
            _mask_frac(ctx, inter, "interior", ctx["shape"]["logit"]))
    if ring.any():
        r_mu, r_sd = _weighted_mean_std(
            ctx["maps"]["refine"],
            _mask_frac(ctx, ring, "ring", ctx["shape"]["refine"]))
        l_ring, l_ring_sd = _weighted_mean_std(
            ctx["maps"]["logit"],
            _mask_frac(ctx, ring, "ring", ctx["shape"]["logit"]))
    row["refine_interior_mag"], row["refine_interior_std"] = i_mu, i_sd
    row["refine_ring_mag"], row["refine_ring_std"] = r_mu, r_sd
    row["refine_contrast"] = None if (i_mu is None or r_mu is None) else i_mu - r_mu
    row["logit_interior_mean"], row["logit_interior_std"] = l_mu, l_sd
    # 列名不得与 head 组的 `logit_ring` / `logit_std` 撞名（否则 CSV 列重复、probe 静默混用）
    row["logit_bnd_mean"], row["logit_bnd_std"] = l_ring, l_ring_sd
    row["logit_contrast"] = None if (l_mu is None or l_ring is None) else l_mu - l_ring

    # ---- confound / hand：模型置信度 + 手工证据通道（原始像素口径）
    w = _obj_frac(ctx, obj, ctx["prob"].shape[-1])
    row["prob_mean"], row["prob_std"] = _weighted_mean_std(ctx["prob"], w)
    row["prob_max"] = float(ctx["prob"][obj].max()) if obj.any() else None
    wf = _obj_frac(ctx, obj, 256)
    ea, eb = ctx["edgeA_d"], ctx["edgeB_d"]
    row["edge_any"] = pool_map((ea | eb).astype(np.float32), wf)
    row["edge_both"] = pool_map((ea & eb).astype(np.float32), wf)
    row["change_abs"] = pool_map(ctx["chg"], wf)
    row["gradA"] = pool_map(ctx["gradA"], wf)
    return row


def run_pass(model, loader, device, args, tag, t_start, positive_rule, deadline_s):
    """一次只读前向，返回 (rows, seen, timed_out, per_image)。

    `positive_rule(gt_index0, matched_gt_idx0_set, pixel_recall) → 1|None`。
    """
    import torch

    cap = {}
    handles = []
    for key in ("Y_s1", "Y_s2", "Y_s3", "Y_s4", "T_s1", "T_s2", "T_s3", "T_s4",
                "refine", "head"):
        handles.append(_resolve(model, key).register_forward_hook(
            (lambda k: (lambda m, i, o: cap.__setitem__(k, o)))(key)))

    rows = []
    per_image = []
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
                    cache = object_iou_matrix(pred, gt, 4)
                    mt = match_objects(pred, gt, 4, cache=cache)
                    rec = {"image": nm, "n_pos": 0, "n_neg": 0}

                    # ---- 先定对象集合（便宜），再决定是否付 image_context 的固定开销。
                    # 真实 SYSU test 每个对象的中位数为 0（4000 图仅 2794 个对象），
                    # 因此"无合格对象的图直接跳过特征提取"既不改任何数值，
                    # 又把 CPU 预算砍掉一半以上（见 Run6 README §2.3）。
                    items = []
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
                            r = int(tpc[j]) / area
                            label_ = positive_rule(j - 1, matched_gt, r)
                            if label_ is None:
                                continue
                            items.append((label_, "gt_small", area,
                                          cache["gt_lab"] == j, float(r)))
                    matched_pred = set(mt.get("matched_pred_idx", []))
                    for j in range(cache["n_pred"]):
                        if j in matched_pred:
                            continue
                        area = int(cache["pred_areas"][j])
                        if bin_index(area) not in SMALL_ALIAS:
                            continue
                        items.append((0, "fp_like", area, cache["pr_lab"] == j + 1, None))

                    if items:
                        ctx = image_context(cap, i, ab, p)
                        for label_, kind_, area, obj, r in items:
                            row = compute_row(ctx, obj)
                            row.update({"image": nm, "label": label_, "kind": kind_,
                                        "area": area, "pixel_recall": r})
                            rows.append(row)
                            if label_ == 1:
                                rec["n_pos"] += 1
                            else:
                                rec["n_neg"] += 1
                    per_image.append(rec)
                cap.clear()
                if args.limit and seen >= args.limit:
                    break
                if seen % 400 == 0:
                    print(f"[{tag}] {seen} images elapsed={time.time() - t_start:.0f}s",
                          flush=True)
                if time.time() - t_start > deadline_s:
                    print(f"[{tag}][TIMEOUT] budget exceeded at {seen} images", flush=True)
                    timed_out = True
                    break
    finally:
        for h in handles:
            h.remove()
    return rows, seen, timed_out, per_image


def _matrix(rows, cols):
    return np.array([[np.nan if r.get(c) is None else r.get(c) for c in cols]
                     for r in rows], dtype=np.float64)


def single_feature_aucs(rows, cols):
    """逐特征 test AUC（诊断用；不参与任何门）。"""
    y = np.array([r["label"] for r in rows], dtype=bool)
    out = {}
    for c in cols:
        out[c] = auc_score(y, _matrix(rows, [c])[:, 0])
    return out


def probe_report(rows_tr, rows_te, cols, seed, n_boot):
    """train 拟合（5 折按图选 λ）→ test AUC/AP + 图像级 bootstrap CI。"""
    ytr = np.array([r["label"] for r in rows_tr], dtype=bool)
    yte = np.array([r["label"] for r in rows_te], dtype=bool)
    Xtr_raw, Xte_raw = _matrix(rows_tr, cols), _matrix(rows_te, cols)
    if Xtr_raw.shape[0] < 50 or Xte_raw.shape[0] < 50 or ytr.sum() < 10:
        return {"insufficient": True, "n_train": int(Xtr_raw.shape[0]),
                "n_test": int(Xte_raw.shape[0])}
    Xtr, Xte, med = _impute(Xtr_raw, Xte_raw)
    imgs = sorted({r["image"] for r in rows_tr})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(imgs))
    fold_of = {imgs[perm[i]]: i % 5 for i in range(len(imgs))}
    fold = np.array([fold_of[r["image"]] for r in rows_tr])
    ytr = np.array([r["label"] for r in rows_tr], dtype=bool)
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
            "test_ap": ap, "test_ap_ci95": [p_lo, p_hi], "ap_valid_boot": p_v,
            "n_missing_imputed_train": int((~np.isfinite(Xtr_raw)).sum()),
            "n_missing_imputed_test": int((~np.isfinite(Xte_raw)).sum()),
            "impute_median_used": [float(x) for x in med]}


def _k(name, ok, value, threshold, note=""):
    return {"pass": bool(ok), "value": value, "threshold": threshold, "note": note}


def k7_select(usable):
    """Run6 §4.2 K7：按 train-CV AUC 降序；相差 ≤0.005 取参数更少者（C1）；再平局字典序。

    返回 (selected, tie_breaker)。规则在见结果之前冻结，禁止按 test 选。
    """
    if not usable:
        return None, None
    ranked = sorted(usable, key=lambda k: (-usable[k]["train_cv_auc"], k))
    if len(ranked) == 1:
        return ranked[0], "only_usable"
    a = usable[ranked[0]]["train_cv_auc"]
    b = usable[ranked[1]]["train_cv_auc"]
    # "相差 ≤0.005" ⇒ 差值 > 0.005 才算领先；+1e-12 只抵消浮点表示误差
    if (a - b) > K7_TIE_EPS + K7_TIE_EPS_SLACK:
        return ranked[0], "train_cv_auc"
    return "C1", "tie<=0.005 -> fewer deploy params (C1)"


def _budget_split(budget_s, n_fit_img, n_eval_img):
    """按图像数比例切分总预算（拟合 pass / 评估 pass）。"""
    tot = max(1, n_fit_img + n_eval_img)
    fit = budget_s * n_fit_img / tot
    return max(60.0, fit), max(60.0, budget_s - fit)


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
                               names_filter=lambda n: _hash_split(n, VAL_FRAC) == "train")
    te_loader, te_ds = _loader(args.dataset_root, args.test_list, args.batch_size,
                               args.num_workers)
    tr_names = read_list_names(args.train_list)
    te_names = read_list_names(args.test_list)
    if args.limit:
        n_fit_plan, n_eval_plan = args.limit, args.limit
    else:
        n_fit_plan, n_eval_plan = len(tr_ds.file_list), len(te_ds.file_list)
    fit_budget, eval_budget = _budget_split(args.budget_sec, n_fit_plan, n_eval_plan)
    print(f"[R6-DISC] fit images={len(tr_ds.file_list)}/{len(tr_names)} "
          f"eval images={len(te_ds.file_list)}/{len(te_names)} "
          f"budget_fit={fit_budget:.0f}s budget_eval={eval_budget:.0f}s", flush=True)

    def positive_rule(gt_index0, matched_gt, r):
        return 1 if ((gt_index0 in matched_gt) or (r < 0.25)) else None

    tr_rows, n_tr, tr_to, _pi_tr = run_pass(
        model, tr_loader, args.device, args, "FIT", t0, positive_rule,
        t0 + fit_budget)
    t_fit = time.time()
    te_rows, n_te, te_to, _pi_te = run_pass(
        model, te_loader, args.device, args, "EVAL", t_fit, positive_rule,
        t_fit + eval_budget)
    elapsed = time.time() - t0

    n_pos = sum(1 for r in te_rows if r["label"] == 1)
    n_neg = sum(1 for r in te_rows if r["label"] == 0)
    n_pos_matched = sum(1 for r in te_rows if r["kind"] == "gt_small"
                        and r["pixel_recall"] is not None and r["pixel_recall"] >= 0.25)
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

    sf_auc = single_feature_aucs(te_rows, ALL_FEATURES)

    # ---- K7：按 train-CV AUC 选一（平局 ≤0.005 取参数更少者 = C1）
    cand = {"C1": results["candidate_C1"], "C2": results["candidate_C2"]}
    usable = {k: v for k, v in cand.items() if not v.get("insufficient")}
    selected, tie = k7_select(usable)
    sel_res = cand.get(selected, {}) if selected else {}
    base_full = results["baseline_full"]
    base_hand = results["baseline_hand"]
    base_amp = results["baseline_amp"]

    recomputed_hand_best = max((v for v in (sf_auc.get(c) for c in BASELINE_HAND)
                                if v is not None), default=None)
    recon = {
        "n_pos": n_pos, "n_pos_expected": FROZEN_REF["n_pos_expected"],
        "n_pos_match": n_pos == FROZEN_REF["n_pos_expected"],
        "n_neg": n_neg, "n_neg_expected": FROZEN_REF["n_neg_expected"],
        "n_neg_match": n_neg == FROZEN_REF["n_neg_expected"],
        "n_pos_from_matched_rule": n_pos_matched,
        "handcrafted_best_single_auc_recomputed": recomputed_hand_best,
        "handcrafted_best_single_auc_frozen": FROZEN_REF["handcrafted_best_single_auc"],
        "change_abs_auc_recomputed": sf_auc.get("change_abs"),
        "change_abs_auc_frozen": FROZEN_REF["handcrafted_change_abs_auc"],
    }
    print(f"[R6-DISC] reconciliation n_pos={n_pos}/{FROZEN_REF['n_pos_expected']} "
          f"n_neg={n_neg}/{FROZEN_REF['n_neg_expected']} "
          f"hand_best_recomputed={recomputed_hand_best} "
          f"change_abs_auc={sf_auc.get('change_abs')}", flush=True)

    auc = sel_res.get("test_auc")
    ap = sel_res.get("test_ap")
    a_lo = (sel_res.get("test_auc_ci95") or [None])[0]
    p_lo = (sel_res.get("test_ap_ci95") or [None])[0]
    hand_auc = base_hand.get("test_auc")
    full_auc = base_full.get("test_auc")
    amp_auc = base_amp.get("test_auc")
    k4_const = GATES["K4_handcrafted_baseline_auc"]
    k5_delta = None if auc is None or amp_auc is None else auc - amp_auc

    g = {}
    g["K0"] = _k("K0", True, {"ckpt_sha256": sha, "test_list_sha256": lst_sha,
                              "train_list_sha256": tr_sha,
                              "fitted_on": "train_hash90", "test_used_for_fit": False,
                              "out_dir_fresh": True},
                 "SHA 严格匹配 + 输出目录全新 + test 未参与拟合")
    g["K1"] = _k("K1", (n_pos >= GATES["K1_n_pos_min"] and n_neg >= GATES["K1_n_neg_min"]),
                 {"n_pos": n_pos, "n_neg": n_neg},
                 {"n_pos_min": GATES["K1_n_pos_min"], "n_neg_min": GATES["K1_n_neg_min"]})
    g["K2"] = _k("K2", (auc is not None and auc >= GATES["K2_test_auc_min"]
                        and a_lo is not None and a_lo > GATES["K2_test_auc_ci_low_min"]),
                 {"test_auc": auc, "ci_low": a_lo},
                 {"auc_min": GATES["K2_test_auc_min"],
                  "ci_low_min_exclusive": GATES["K2_test_auc_ci_low_min"]})
    g["K3"] = _k("K3", (ap is not None and ap >= GATES["K3_test_ap_min"]
                        and p_lo is not None and p_lo > GATES["K3_test_ap_ci_low_min"]),
                 {"test_ap": ap, "ci_low": p_lo},
                 {"ap_min": GATES["K3_test_ap_min"],
                  "ci_low_min_exclusive": GATES["K3_test_ap_ci_low_min"]})
    g["K4"] = _k("K4", (auc is not None and (auc - k4_const) >= GATES["K4_auc_over_handcrafted_min"]),
                 {"winner_auc": auc, "handcrafted_baseline_auc_constant": k4_const,
                  "delta_vs_constant": None if auc is None else auc - k4_const,
                  "handcrafted_baseline_auc_recomputed": hand_auc,
                  "delta_vs_recomputed": (None if auc is None or hand_auc is None
                                          else auc - hand_auc)},
                 {"delta_min": GATES["K4_auc_over_handcrafted_min"],
                  "baseline_source": "冻结常数 0.5992（F14 最佳单项）；重算值并列报告但不进门"})
    g["K4b"] = _k("K4b", (auc is not None and full_auc is not None
                          and (auc - full_auc) >= GATES["K4b_auc_over_full_baseline_min"]),
                  {"winner_auc": auc, "full_baseline_auc": full_auc,
                   "delta": None if auc is None or full_auc is None else auc - full_auc},
                  {"delta_min": GATES["K4b_auc_over_full_baseline_min"]})
    g["K5"] = _k("K5", (k5_delta is not None
                        and k5_delta >= GATES["K5_auc_over_amplitude_min"]),
                 {"winner_auc": auc, "amplitude_auc": amp_auc, "delta": k5_delta},
                 {"delta_min": GATES["K5_auc_over_amplitude_min"]})
    g["K6"] = _k("K6", True, {"same_object_set": True, "same_probe": True,
                              "same_bootstrap": True, "both_candidates_archived": True},
                 "两候选同一对象集合/probe/bootstrap")
    g["K7"] = _k("K7", selected in ("C1", "C2"),
                 {"selected": selected, "tie_breaker": tie,
                  "train_cv_auc": {k: v.get("train_cv_auc") for k, v in cand.items()},
                  "deploy_params": {"C1": 8, "C2": 961},
                  "tie_eps": K7_TIE_EPS, "tie_eps_slack": K7_TIE_EPS_SLACK},
                 "按 train-CV AUC 选一；相差≤0.005 取参数更少者；test 不参与")

    hard = [g["K2"]["pass"], g["K3"]["pass"], g["K4"]["pass"], g["K4b"]["pass"]]
    dry = bool(args.limit)
    timed_out = bool(tr_to or te_to)
    if dry:
        status = "DRY_ONLY"
    elif timed_out:
        status = "INCOMPLETE"
    elif not g["K1"]["pass"]:
        status = "INCONCLUSIVE"
    elif all(hard):
        status = "PASS"
    else:
        status = "FAIL"
    gate_stat = "amplitude" if (status == "PASS" and not g["K5"]["pass"]) else "consistency"
    gate_impl = ("c1_scale_consistency" if selected == "C1" else
                 ("c2_interior_consistency" if selected == "C2" else "none"))

    inputs = {
        "profile_version": PROFILE_VERSION, "protocol_tag": PROTOCOL_TAG,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset, "run": "Run1", "variant": "M1_FULL",
        "repository_sha": os.environ.get("CASA_REPO_SHA"),
        "ckpt_path": ck, "ckpt_sha256": sha, "ckpt_sha256_expected": M1_SYSU_SHA,
        "ckpt_candidates": meta.get("candidates"),
        "train_list": args.train_list, "train_list_sha256_kind": "sha256_text_lines",
        "train_list_sha256": tr_sha,
        "test_list": args.test_list, "test_list_sha256_kind": "sha256_text_lines",
        "test_list_sha256": lst_sha, "test_list_sha256_expected": SYSU_TEST_LIST_SHA,
        "n_fit_images": n_tr, "n_eval_images": n_te,
        "n_fit_objects": len(tr_rows), "n_eval_objects": len(te_rows),
        "n_pos_eval": n_pos, "n_neg_eval": n_neg,
        "n_pos_eval_from_matched_rule": n_pos_matched,
        "fit_split": (f"train.txt D3 sha1-hash {VAL_FRAC:.2f} val excluded -> "
                      f"{1 - VAL_FRAC:.2f} train used for fitting"),
        "val_frac": VAL_FRAC,
        "positive_rule": "GT small (area 1..255, 4-conn) with (IoU>=0.10 matched) OR (r<0.25)",
        "negative_rule": "unmatched predicted CC, area 1..255, one-shot max-weight matching",
        "pooling": "area-averaged mask coverage (cv2.INTER_AREA) at each feature map resolution",
        "edge_field": {"definition": "三通道均值 Sobel 幅值 >= 该图 90 分位 (mode=reflect)",
                       "quantile": EDGE_Q, "dilation_for_overlap_px": 1},
        "hand_feature_pixel_space": (
            "Normalize→Scale→ToTensor 之后的归一化张量通道（DataLoader 直接给出，"
            "不做反归一化）——与 F14 / Run5 FP 画像逐字同口径；冻结读数 "
            "change_abs=0.9730 / gradA=1.8008 即在此口径下取得"),
        "hand_reference_frozen": {
            "edge_both_best_single_auc": FROZEN_REF["handcrafted_best_single_auc"],
            "change_abs_auc": FROZEN_REF["handcrafted_change_abs_auc"],
            "change_abs_mean_fp": FROZEN_REF["fp_change_abs_mean"],
            "change_abs_mean_gt_missed": FROZEN_REF["gt_missed_change_abs_mean"],
        },
        "c2_ring_definition": "dilate_4px(O) \\ O（4 邻域结构元），**不使用 GT**",
        "c1_statistic": "f_s.abs().mean(dim=1)（TAR 已融合单路特征的固定统计量）",
        "probe": "standardized linear logistic, L2 selected by image-grouped 5-fold CV on FIT",
        "lambdas": list(LAMBDAS), "bootstrap": {"unit": "image", "n": args.bootstrap,
                                                "seed": args.seed},
        "feature_groups": FEATURE_GROUPS, "probe_sets": {k: v for k, v in PROBE_SETS.items()},
        "gates_frozen": GATES, "frozen_reference": FROZEN_REF,
        "budget": {"total_sec": args.budget_sec, "fit_sec": fit_budget,
                   "eval_sec": eval_budget},
        "caveat": "拟合图像是模型训练过的，分布偏乐观；门只认 test 侧数值",
        "env": env_info(),
    }
    write_json(os.path.join(args.out_dir, "inputs.json"), inputs)

    cols = ["image", "label", "kind", "area", "pixel_recall", "n_scales_missing",
            "interior_empty", "ring_empty"] + ALL_FEATURES
    for tag, rows in (("features_train", tr_rows), ("features_test", te_rows)):
        with open(os.path.join(args.out_dir, f"{tag}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    write_json(os.path.join(args.out_dir, "probe_results.json"), {
        "results": results, "single_feature_auc_test": sf_auc,
        "frozen_reference_reconciliation": recon,
        "selection": {"selected": selected, "tie_breaker": tie,
                      "gate_impl": gate_impl, "gate_stat": gate_stat,
                      "rule": "train-CV AUC desc; tie<=0.005 -> fewer params (C1); "
                              "then lexicographic; test never used"},
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
        "preflight_version": PROFILE_VERSION, "protocol_tag": PROTOCOL_TAG,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "selected": selected, "gate_impl": gate_impl, "gate_stat": gate_stat,
        "gates": g, "frozen_reference_reconciliation": recon,
        "n_pos_eval": n_pos, "n_neg_eval": n_neg,
        "n_fit_objects": len(tr_rows), "n_eval_objects": len(te_rows),
        "winner_test_auc": auc, "winner_test_ap": ap,
        "handcrafted_baseline_auc": hand_auc, "full_baseline_auc": full_auc,
        "amplitude_baseline_auc": amp_auc,
        "elapsed_s": round(elapsed, 1), "budget_sec": args.budget_sec,
        "timed_out": timed_out,
        "decision": ("80K ALLOWED (K0-K7 PASS)" if status == "PASS" else
                     "NO_80K (preflight not PASS) — 不新增证据通道、不改阈值重跑"),
        "test_used_for_fit": False,
    }
    write_json(os.path.join(args.out_dir, "gate.json"), gate)
    lines = []
    for fn in sorted(os.listdir(args.out_dir)):
        p = os.path.join(args.out_dir, fn)
        if os.path.isfile(p) and fn != "SHA256SUMS.txt":
            with open(p, "rb") as fh:
                lines.append(f"{hashlib.sha256(fh.read()).hexdigest()}  {fn}")
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
