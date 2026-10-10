"""Run5 §5 只读前测：原生中间高频（`local_conv(x_high)`）→ TAR 同尺度时相投影的配对选择与 G0–G8 硬门。

**本脚本是 Run5 唯一的“开门”依据，必须先于任何模型改动执行**（Run5 设计文档 §11.1 步骤 2–3）。
唯一问题：TinyViM 内部、在 `cat(x_low, x_high)` 与 `out_proj` **混合之前**的
`local_conv(x_high)` 张量，对 M1 已漏检的 SYSU small 对象（1–255px）是否提供
**同尺度旧 tap（norm2/norm4 冻结 probe）之外**的增量判别证据，且该证据对
未匹配小预测连通域（FP-like）的响应显著低于对漏检对象的响应。

两个候选插点（**互斥，最终只选一个**）：
  * `Pair-2`：`encoder.network[2][-1].op.local_conv`（1/8，2B×32×32²）→ `TAR.stage2.temporal`（同尺度 tap = `L03b_norm2`）
  * `Pair-3`：`encoder.network[4][-1].op.local_conv`（1/16，2B×84×16²，CAACP block）→ `TAR.stage3.temporal`（同尺度 tap = `L06b_norm4`）

流程（严格按 §5.3）：
  1. P0：输出目录必须全新；SYSU Run1 `M1_FULL` best SHA256 与 Diag1 锁定值一致；
     test list SHA256 一致；两个同尺度 tap probe 存在、协议 `concat`、`C_in==3C`；
     `selective_scan_cuda_oflex` 可导入；`local_conv` 输出 shape 与契约一致。任一不满足 → 退出 2。
  2. **只用 train.txt 的 D3 SHA1-hash 10% probe-val** 做候选选择（一次前向同时 hook 两对）；
     val 需 `n_missed>=30`、`n_fp_like>=50`，再排除 `gap<0.05` 或 `novel_rescue<0.05` 的候选。
  3. 固定选择规则：`val_novel_rescue` 降序 → 相差 ≤0.01 用 `val_gap` 降序 → 仍平局取 deploy MAC
     更低者（Pair-3）→ 仍平局按字典序。写 `selection_val.json`（`locked=true`）。
  4. **只对已锁定的那一对**跑一次 SYSU 全量 test（4000 张）前向，评价 G1–G7；test 不参与选层。
  5. 总耗时（val + test + 统计）上界 `--budget-sec`（默认 600），超时记 `TIMEOUT`。
     `--limit>0` 永远只输出 `DRY_ONLY`，不得 PASS。

与 Run4 前测的关键差别（§5.4 已预注册，禁止事后回调）：
  * `G5` gap 门 0.10 → **0.15**，CI 下界 `>0` → **>0.03**；
  * 新增 `G6` **对旧 tap 的独有率** `novel_rescue>=0.10` 且 CI 下界 `>0.02`；
  * 新增 `G7` **同尺度 rank margin 增量** `mean(margin_HF)-mean(margin_TAP)>=0.02` 且配对 CI 下界 `>0`。

只读纪律：模型与 probe 全部 `.eval()` + `requires_grad_(False)` + `torch.no_grad()`；
不训练、不重拟合、不写回任何 Diag1 / Run1–4 产物；输出目录独立且必须为空。

M1 二值来源：**train graph eval**（`prob > 0.5`），与 Diag1 `tvim_small_error_audit.py` 同口径；
另以 `--deploy-crosscheck-n`（默认前 64 张真实图）核验 train/deploy 折叠二值一致（必须 0 flip），
作为 G0 的 P0 证据，从而保证对象集合与 deploy 图一致。

    python analyse/tvim_run5_hf_pair_preflight.py \
      --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
      --ckpt-root /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM \
      --dataset-root /share_datasets/CD/SYSU-CD-256 \
      --train-list /share_datasets/CD/SYSU-CD-256/list/train.txt \
      --test-list  /share_datasets/CD/SYSU-CD-256/list/test.txt \
      --probe-dir /home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1/D3_concat/M1_FULL \
      --device cuda:0 --batch-size 16 --num-workers 4 \
      --split-hash sha1_10pct --seed 16 --bootstrap 1000 \
      --limit 0 --budget-sec 600 \
      --out-dir /home/yqwang/outputs/CASA-CD/diagnostics/TViM-Run5-HF-PREFLIGHT-20261010
"""
import argparse
import copy
import csv
import datetime
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvim_diag_common import (  # noqa: E402
    DIAG_CODE_FILES, build_model, code_identity, env_info, pick_best_ckpt,
    read_list_names, set_eval_numerics, sha256_file, sha256_text_lines, write_json,
)
from tvim_linear_probe import _encoder_probe_input, _hash_split, _resolve_module  # noqa: E402
from tvim_object_metrics import (  # noqa: E402
    CONN4, GROUP_ALIAS, bin_index, match_objects, object_iou_matrix,
)
from tvim_stage_recoverability import NBINS, _ap_from_hist  # noqa: E402

PREFLIGHT_VERSION = "run5_hf_pair_preflight_v1"
PROTOCOL_TAG = "run5_hf_tar_exact80k_v1"

# ---------------------------------------------------------------------------- 预注册常数
MARGIN_THRESHOLD = 0.03          # §5.2.6 诊断读出阈值；不是分割阈值、不是训练超参
BOOTSTRAP = 1000
BUDGET_S = 600.0                 # §5.3.4 / A06
EXPECT_N_TEST = 4000             # G1
VAL_MIN_MISSED = 30              # §5.3.2
VAL_MIN_FP = 50                  # §5.3.2
VAL_SCREEN_GAP_MIN = 0.05        # §5.3.2 初筛排除线
VAL_SCREEN_NOVEL_MIN = 0.05      # §5.3.2 初筛排除线
TIE_EPS = 0.01                   # §5.3.3

GATES = {
    "G2_n_missed_small_min": 100,
    "G3_n_fp_like_min": 100,
    "G4_rescue_rate_min": 0.25,
    "G5_gap_min": 0.15,
    "G5_gap_ci_low_min": 0.03,
    "G6_novel_rescue_min": 0.10,
    "G6_novel_ci_low_min": 0.02,
    "G7_margin_delta_min": 0.02,
    "G7_margin_ci_low_min": 0.0,
}

# Diag1 锁定输入（§5.3 读取锁；不匹配即 P0_INVALID，禁止自动重训/替换素材）
# 注：list 锁是 Diag1 manifest 的 `sha256_text_lines`（行内容归一化：rstrip CRLF + 每行补 \n），
# **不是** raw-bytes sha256；两者不同，必须用 `tvim_diag_common.sha256_text_lines` 复算。
M1_SYSU_SHA = "45d688a01af22fe22521f4dfb4579e80827ddcaa886e07bafc778fc288731457"
SYSU_TEST_LIST_SHA = "5f4122c9d32cb6db475f4f361618b2ac8bb10938df93d9550e0bb9a422a45797"
SYSU_TEST_LIST_SHA_KIND = "sha256_text_lines"

PAIR_SPECS = {
    "Pair-2": {
        "stage": "2",
        "hf_attr": "encoder.network[2][-1].op.local_conv",
        "hf_channels": 32,
        "hf_spatial": 32,
        "tap_key": "L03b_norm2",
        "tap_channels": 64,
        "deploy_delta_params": 6144,
        "deploy_mac": 32 * 32 * 96 * 64,
        "contract_shape": (32, 32, 32),
    },
    "Pair-3": {
        "stage": "3",
        "hf_attr": "encoder.network[4][-1].op.local_conv",
        "hf_channels": 84,
        "hf_spatial": 16,
        "tap_key": "L06b_norm4",
        "tap_channels": 168,
        "deploy_delta_params": 16128,
        "deploy_mac": 16 * 16 * 96 * 168,
        "contract_shape": (84, 16, 16),
    },
}
PAIR_NAMES = ["Pair-2", "Pair-3"]
SMALL_ALIAS = set(GROUP_ALIAS["small"])
ST5 = CONN4
GALIAS_CODE = {"small": 1, "medium": 2, "large": 3}
CODE_GALIAS = {v: k for k, v in GALIAS_CODE.items()}
MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


# ============================================================================ 基础工具
def _avg_rank01(x):
    """确定性稳定平均秩 → [0,1]（tie 用平均秩；与 label 无关；向量化）。"""
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    if n <= 1:
        return np.zeros_like(a)
    order = np.argsort(a, kind="stable")
    sa = a[order]
    starts = np.flatnonzero(np.r_[True, sa[1:] != sa[:-1]])
    counts = np.diff(np.r_[starts, n])
    avg = starts + (counts - 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(avg, counts)
    return ranks / (n - 1.0)


def _rank_tensor(t, size=256):
    """(C,h,w)/(h,w) 分数张量 → 双线性上采样 size² → 逐图 rank，返回 (C,size,size) float64。"""
    import torch
    import torch.nn.functional as F

    if t.dim() == 2:
        t = t[None]
    if tuple(t.shape[-2:]) != (size, size):
        t = F.interpolate(t[None].float(), size=(size, size), mode="bilinear",
                          align_corners=False)[0]
    else:
        t = t.float()
    arr = t.detach().cpu().numpy()
    out = np.empty((arr.shape[0], size, size), dtype=np.float64)
    for c in range(arr.shape[0]):
        out[c] = _avg_rank01(arr[c]).reshape(size, size)
    return out


def _ring_margin(score, obj, gt, sl, iterations=4):
    """margin = mean(S[O]) − mean(S[R])，`R=(dilate_4(O)\\O)∩(~GT)`；R 为空 → (None, False)。"""
    from scipy import ndimage as ndi
    pad = iterations + 1
    y0 = max(0, sl[0].start - pad)
    y1 = min(obj.shape[0], sl[0].stop + pad)
    x0 = max(0, sl[1].start - pad)
    x1 = min(obj.shape[1], sl[1].stop + pad)
    o = obj[y0:y1, x0:x1]
    g = gt[y0:y1, x0:x1]
    if not o.any():
        return None, False
    d = ndi.binary_dilation(o, structure=ST5, iterations=iterations)
    ring = d & ~o & ~g
    if not ring.any():
        return None, False
    s = score[y0:y1, x0:x1]
    return float(s[o].mean() - s[ring].mean()), True


def _small_ap_hist(score, small_mask, gt):
    """§5.2.5：正样本仅 small 组 GT 像素；负样本所有 GT==0；其它 GT 对象像素不进样本。"""
    sv = score.ravel()
    pm = small_mask.ravel()
    bg = (~gt).ravel()
    sel = pm | bg
    hp, _ = np.histogram(sv[sel & pm], bins=NBINS, range=(0.0, 1.0))
    ha, _ = np.histogram(sv[sel], bins=NBINS, range=(0.0, 1.0))
    return hp, ha


# ============================================================================ 数据 / probe
def _make_loader(dataset_root, list_path, batch_size, num_workers, names_filter=None):
    """与 `tvim_diag_common.make_loader` 同一条 eval 变换链；根/list 由 CLI 显式给出。"""
    import torch
    from dataset import dataset as myDataLoader
    from dataset import Transforms as myTransforms

    transform = myTransforms.Compose([
        myTransforms.Normalize(mean=MEAN6, std=STD6),
        myTransforms.Scale(256, 256),
        myTransforms.ToTensor(),
    ])
    ds = myDataLoader.Dataset(file_root=dataset_root, list_path=list_path, transform=transform)
    if names_filter is not None:
        keep = [i for i, n in enumerate(ds.file_list) if names_filter(n)]
        ds.file_list = [ds.file_list[i] for i in keep]
        ds.pre_images = [ds.pre_images[i] for i in keep]
        ds.post_images = [ds.post_images[i] for i in keep]
        ds.gts = [ds.gts[i] for i in keep]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False)
    return loader, ds


def _hf_module(model, pair_name):
    spec = PAIR_SPECS[pair_name]
    if spec["stage"] == "2":
        return model.encoder.network[2][-1].op.local_conv
    if spec["stage"] == "3":
        return model.encoder.network[4][-1].op.local_conv
    raise KeyError(pair_name)


def _load_frozen_probe(probe_dir, key, tap_channels, device):
    """严格加载 D3 `concat` 冻结 probe；协议不符即 P0 退出（禁止重拟合）。"""
    import torch
    path = os.path.join(probe_dir, f"probe_{key}_PROBE_ONLY.pth")
    if not os.path.isfile(path):
        raise SystemExit(f"[P0_INVALID] missing frozen probe: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc_input = (ck.get("protocol") or {}).get("encoder_input")
    c_in = int(ck["C_in"])
    if enc_input != "concat":
        raise SystemExit(f"[P0_INVALID] probe {key} encoder_input={enc_input!r} != 'concat'")
    if c_in != 3 * tap_channels:
        raise SystemExit(f"[P0_INVALID] probe {key} C_in={c_in} != 3*{tap_channels}")
    module = torch.nn.Conv2d(c_in, 1, 1).to(device)
    module.load_state_dict(ck["probe"])
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module, ck, path


def _probe_prob(module, fa, fb):
    """冻结 probe 概率场（sigmoid 后双线性上采样到 256²）——与 D3 `evaluate_probe` 同口径。"""
    import torch
    import torch.nn.functional as F
    feat = _encoder_probe_input(fa.float(), fb.float(), "concat")[None]
    p = torch.sigmoid(module(feat))[0, 0]
    if tuple(p.shape) != (256, 256):
        p = F.interpolate(p[None, None], size=(256, 256), mode="bilinear",
                          align_corners=False)[0, 0]
    return p


def _is_hit(o, key):
    v = o["margins"].get(key)
    return v is not None and v >= MARGIN_THRESHOLD


def classify_val_screen(counts_ok, timed_out, eligible):
    """§5.3.2 两种终止分支必须区分（决定相同 = NO_80K，但记录不可混同）。"""
    if timed_out:
        return "VAL_TIMEOUT"
    if not counts_ok:
        return "INSUFFICIENT_COUNTS"
    return "PASSED" if eligible else "BOTH_SCREENED_OUT"


def select_pair(val_cand, eligible):
    """§5.3.3 固定选择规则（纯函数，禁止依据 test 结果；便于单测）。

    `val_novel_rescue` 降序 → 相差 ≤TIE_EPS 时按 `val_gap` 降序 → 仍平局取 deploy MAC 更低者
    （Pair-3）→ 仍平局按字典序。返回 (selected_pair, tie_breaker)。
    """
    if not eligible:
        return None, None
    ranked = sorted(eligible, key=lambda p: -val_cand[p]["novel_rescue"])
    top = val_cand[ranked[0]]["novel_rescue"]
    tied = [p for p in ranked if top - val_cand[p]["novel_rescue"] <= TIE_EPS]
    if len(tied) == 1:
        return tied[0], "novel_rescue"
    best_gap = max(val_cand[p]["gap"] for p in tied)
    tied2 = [p for p in tied if best_gap - val_cand[p]["gap"] <= TIE_EPS]
    if len(tied2) == 1:
        return tied2[0], "novel_rescue>gap"
    min_mac = min(val_cand[p]["deploy_mac"] for p in tied2)
    tied3 = sorted([p for p in tied2 if val_cand[p]["deploy_mac"] == min_mac])
    return tied3[0], "novel_rescue>gap>deploy_mac>lexicographic"


# ============================================================================ 单张图记录
def _image_records(prob, gt, scores, cache, mt):
    """返回 (object_rows, gt_rows, rec, small_mask)。

    object_rows：missed_small + fp_like 逐对象（含每个 score 的 rank margin）。
    gt_rows：全部 GT 对象（含组别、面积、pixel_recall、每个 score 的 margin），供组统计/AP。
    rec：逐图计数（n_missed/n_fp/n_hit_*/n_fp_hit_*/n_hf_only/n_tap_only/...）。
    small_mask：small 组（1–255px）GT 像素并集，供 small-vs-background AP。
    """
    score_names = list(scores.keys())
    rec = {"n_missed": 0, "n_missed_pre_ring": 0, "n_fp": 0,
           "n_skipped_ring": 0, "n_skipped_ring_small": 0, "n_hf_only": 0,
           "n_tap_only": 0, "sum_margin_diff_pair": 0.0, "n_margin_pairs": 0,
           "small_n": 0, "small_px": 0, "small_tp": 0, "small_hit25": 0}
    for name in score_names:
        rec[f"n_hit_{name}"] = 0
        rec[f"n_fp_hit_{name}"] = 0

    pred_b = prob > 0.5
    gt_lab = cache["gt_lab"]
    n_gt = cache["n_gt"]
    iou = cache["iou"]
    gt_rows = []
    small_mask = np.zeros_like(gt)

    if n_gt:
        counts = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
        tp_map = pred_b & gt
        tp_counts = (np.bincount(gt_lab[tp_map].ravel(), minlength=n_gt + 1)
                     if tp_map.any() else np.zeros(n_gt + 1, dtype=np.int64))
        lut = np.zeros(n_gt + 1, dtype=np.uint8)
        for j in range(1, n_gt + 1):
            gi = bin_index(int(counts[j]))
            galias = ("small" if gi in SMALL_ALIAS
                      else ("medium" if gi == "medium_256_1023" else "large"))
            lut[j] = GALIAS_CODE[galias]
            if galias == "small":
                small_mask |= (gt_lab == j)
        for j in range(1, n_gt + 1):
            area = int(counts[j])
            gi = bin_index(area)
            galias = CODE_GALIAS[int(lut[j])]
            mobj = (gt_lab == j)
            r = int(tp_counts[j]) / area
            if galias == "small":
                # 与 Diag1/Run4 冻结口径对拍用的参考量（§5.2.1）
                rec["small_n"] += 1
                rec["small_px"] += area
                rec["small_tp"] += int(tp_counts[j])
                rec["small_hit25"] += int(r >= 0.25)
            margins, ring_ok = {}, {}
            for name, score in scores.items():
                m, ok = _ring_margin(score, mobj, gt, cache["gt_objs"][j - 1])
                margins[name], ring_ok[name] = m, ok
            gt_rows.append({"obj_id": j, "kind": "gt_object", "area": area, "group": gi,
                            "galias": galias, "pixel_recall": float(r),
                            "margins": margins, "ring_ok": ring_ok, "missed": r < 0.25,
                            "iou_max_any_pred": (float(iou[j - 1].max())
                                                 if iou.size else 0.0),
                            "fp_like_id": None})

    object_rows = []
    for o in gt_rows:
        if o["galias"] != "small":
            # §5.2.1：missed_small **只**限 GT 面积 [1,255]（tiny_1_15 ∪ tiny_16_63 ∪
            # small_64_255）。medium/large 对象即使 r<0.25 也不属于 small 组。
            continue
        if not o["missed"]:
            continue
        rec["n_missed_pre_ring"] += 1
        if not all(o["ring_ok"].values()):
            rec["n_skipped_ring"] += 1
            rec["n_skipped_ring_small"] += 1
            continue
        rec["n_missed"] += 1
        for name in score_names:
            if _is_hit(o, name):
                rec[f"n_hit_{name}"] += 1
        if "HF" in scores and "TAP" in scores:
            hf_hit, tap_hit = _is_hit(o, "HF"), _is_hit(o, "TAP")
            rec["n_hf_only"] += int(hf_hit and not tap_hit)
            rec["n_tap_only"] += int(tap_hit and not hf_hit)
            if o["margins"]["HF"] is not None and o["margins"]["TAP"] is not None:
                rec["sum_margin_diff_pair"] += float(o["margins"]["HF"] - o["margins"]["TAP"])
                rec["n_margin_pairs"] += 1
        object_rows.append(dict(o, kind="missed_small"))

    # ---- FP-like：未匹配预测 4 连通域，面积 [1,255]（§5.2.2） --------------------
    pr_lab = cache["pr_lab"]
    matched = set(mt.get("matched_pred_idx", []))
    for j in range(cache["n_pred"]):
        if j in matched:
            continue
        area = int(cache["pred_areas"][j])
        if bin_index(area) not in SMALL_ALIAS:
            continue
        mobj = (pr_lab == j + 1)
        margins, ring_ok = {}, {}
        for name, score in scores.items():
            m, ok = _ring_margin(score, mobj, gt, cache["pr_objs"][j])
            margins[name], ring_ok[name] = m, ok
        if not all(ring_ok.values()):
            rec["n_skipped_ring"] += 1
            continue
        rec["n_fp"] += 1
        for name in score_names:
            if margins.get(name) is not None and margins[name] >= MARGIN_THRESHOLD:
                rec[f"n_fp_hit_{name}"] += 1
        object_rows.append({"obj_id": None, "kind": "fp_like", "area": area,
                            "group": bin_index(area), "galias": None, "pixel_recall": None,
                            "margins": margins, "ring_ok": ring_ok, "missed": None,
                            "iou_max_any_pred": None, "fp_like_id": j})
    return object_rows, gt_rows, rec, small_mask


# ============================================================================ bootstrap
def _boot_ratio(num, den, n_boot, seed):
    """按图像有放回重采样，每次重算计数比。返回 (obs, lo, hi, valid)。"""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    obs = float(num.sum() / den.sum()) if den.sum() > 0 else None
    n = num.size
    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = den[idx].sum()
        if d > 0:
            vals[b] = num[idx].sum() / d
    ok = np.isfinite(vals)
    if ok.sum() == 0:
        return obs, None, None, 0
    lo, hi = np.percentile(vals[ok], [2.5, 97.5])
    return obs, float(lo), float(hi), int(ok.sum())


def _boot_gap(num_hit, num_fp_hit, den_missed, den_fp, n_boot, seed):
    """gap = rescue − fp_like 的图像级 bootstrap（同一重采样索引，保持配对）。"""
    nh = np.asarray(num_hit, dtype=np.float64)
    nf = np.asarray(num_fp_hit, dtype=np.float64)
    dm = np.asarray(den_missed, dtype=np.float64)
    df = np.asarray(den_fp, dtype=np.float64)
    obs = None
    if dm.sum() > 0 and df.sum() > 0:
        obs = float(nh.sum() / dm.sum() - nf.sum() / df.sum())
    n = nh.size
    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m, f = dm[idx].sum(), df[idx].sum()
        if m > 0 and f > 0:
            vals[b] = nh[idx].sum() / m - nf[idx].sum() / f
    ok = np.isfinite(vals)
    if ok.sum() == 0:
        return obs, None, None, 0
    lo, hi = np.percentile(vals[ok], [2.5, 97.5])
    return obs, float(lo), float(hi), int(ok.sum())


def _boot_mean_pair(diff_sum, n_pairs, n_boot, seed):
    """按图配对的 mean(HF − TAP) bootstrap：每图贡献 (Σdiff, n)，重采样后取比值。"""
    s = np.asarray(diff_sum, dtype=np.float64)
    k = np.asarray(n_pairs, dtype=np.float64)
    obs = float(s.sum() / k.sum()) if k.sum() > 0 else None
    n = s.size
    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        kk = k[idx].sum()
        if kk > 0:
            vals[b] = s[idx].sum() / kk
    ok = np.isfinite(vals)
    if ok.sum() == 0:
        return obs, None, None, 0
    lo, hi = np.percentile(vals[ok], [2.5, 97.5])
    return obs, float(lo), float(hi), int(ok.sum())


# ============================================================================ 单次 pass
def _run_pass(model, model_deploy, loader, names, device, args, pair_names, probes,
              evidence_keys, tag, t_start, deploy_crosscheck_n=0, keep_maps=False):
    """对给定 pairs 做一次只读前向。val 时同时 hook 两对（§5.3.2）；test 只 hook 锁定对。"""
    import torch

    cap = {}
    hooks = []

    def _hf_hook(name):
        def _h(mod, inp, out):
            cap[name] = out
        return _h

    def _ev_hook(name, store):
        def _h(mod, inp, out):
            store[name] = {"shape": [int(x) for x in out.shape],
                           "mean": float(out.float().mean().item()),
                           "std": float(out.float().std().item())}
        return _h

    for pn in pair_names:
        hooks.append(_hf_module(model, pn).register_forward_hook(_hf_hook(f"HFRAW_{pn}")))
    for pn in pair_names:
        key = PAIR_SPECS[pn]["tap_key"]
        hooks.append(_resolve_module(model, key).register_forward_hook(
            _hf_hook(f"TAPRAW_{pn}")))

    ev = {}
    for key in evidence_keys:
        hooks.append(_resolve_module(model, key).register_forward_hook(_ev_hook(key, ev)))

    score_names = ([f"HF_{pn}" for pn in pair_names] + [f"TAP_{pn}" for pn in pair_names]
                   if len(pair_names) > 1 else ["HF", "TAP"])

    per_object, per_image, kept = [], [], []
    group_margin = {}
    hist = {}
    n_seen = 0
    deploy_flips = 0
    deploy_checked = 0
    timed_out = False

    def _acc(name, galias, val):
        d = group_margin.setdefault(name, {}).setdefault(galias, {"sum": 0.0, "n": 0})
        d["sum"] += float(val)
        d["n"] += 1

    try:
        with torch.no_grad():
            for img, label in loader:
                cap.clear()
                pre = img[:, 0:3].to(device).float()
                post = img[:, 3:6].to(device).float()
                out = model(pre, post)
                B = pre.shape[0]
                if deploy_crosscheck_n and deploy_checked < deploy_crosscheck_n:
                    k = min(B, deploy_crosscheck_n - deploy_checked)
                    out_d = model_deploy(pre[:k], post[:k])
                    deploy_flips += int(((out_d > 0.5) != (out[:k] > 0.5)).sum().item())
                    deploy_checked += k

                gts = (label.numpy()[:, 0] > 0.5)
                probs = out[:, 0].float().cpu().numpy()
                for i in range(B):
                    nm = names[n_seen] if n_seen < len(names) else f"idx{n_seen}"
                    gt = gts[i]
                    prob = probs[i]
                    pred_b = prob > 0.5
                    cache = object_iou_matrix(pred_b, gt, 4)
                    mt = match_objects(pred_b, gt, 4, cache=cache)

                    scores = {}
                    for pn in pair_names:
                        spec = PAIR_SPECS[pn]
                        hf2b = cap[f"HFRAW_{pn}"]
                        ha, hb = hf2b[i].float(), hf2b[B + i].float()
                        if tuple(ha.shape) != spec["contract_shape"]:
                            raise SystemExit(
                                f"[P0_INVALID_HOOK] {pn} local_conv shape {tuple(ha.shape)}"
                                f" != {spec['contract_shape']}")
                        u = (hb - ha).abs().mean(dim=0)
                        r_small = _avg_rank01(u.detach().cpu().numpy()).reshape(
                            spec["hf_spatial"], spec["hf_spatial"])
                        hf_rank = _rank_tensor(torch.from_numpy(r_small)[None])[0]
                        tap2b = cap[f"TAPRAW_{pn}"]
                        p_tap = _probe_prob(probes[spec["tap_key"]], tap2b[i], tap2b[B + i])
                        tap_rank = _rank_tensor(p_tap[None])[0]
                        if len(pair_names) > 1:
                            scores[f"HF_{pn}"] = hf_rank
                            scores[f"TAP_{pn}"] = tap_rank
                        else:
                            scores["HF"] = hf_rank
                            scores["TAP"] = tap_rank

                    object_rows, gt_rows, rec, small_mask = _image_records(
                        prob, gt, scores, cache, mt)
                    rec["image"] = nm
                    for r in gt_rows:
                        for name in score_names:
                            if r["ring_ok"].get(name) and r["margins"].get(name) is not None:
                                _acc(name, r["galias"], r["margins"][name])
                    if small_mask.any():
                        for name in score_names:
                            hp, ha = _small_ap_hist(scores[name], small_mask, gt)
                            h = hist.setdefault(name, {"pos": np.zeros(NBINS),
                                                       "all": np.zeros(NBINS)})
                            h["pos"] += hp
                            h["all"] += ha
                    for r in object_rows:
                        r["image"] = nm
                        per_object.append(r)
                    per_image.append(rec)
                    if keep_maps and ("HF" in scores):
                        why = None
                        for r in object_rows:
                            if (r["kind"] == "missed_small" and _is_hit(r, "HF")
                                    and not _is_hit(r, "TAP")):
                                why = "hf_only_missed_small"
                            elif r["kind"] == "fp_like" and _is_hit(r, "HF"):
                                why = why or "fp_like_hf_hit"
                        if why:
                            kept.append({"image": nm, "why": why,
                                         "prob": (prob * 255).astype(np.uint8),
                                         "hf": (scores["HF"] * 255).astype(np.uint8),
                                         "tap": (scores["TAP"] * 255).astype(np.uint8)})
                    n_seen += 1

                if args.limit and n_seen >= args.limit:
                    break
                if n_seen % 400 == 0:
                    print(f"[{tag}] {n_seen} images elapsed={time.time() - t_start:.0f}s",
                          flush=True)
                if time.time() - t_start > args.budget_sec:
                    print(f"[{tag}][TIMEOUT] budget {args.budget_sec:.0f}s exceeded at "
                          f"{n_seen} images", flush=True)
                    timed_out = True
                    break
    finally:
        for h in hooks:
            h.remove()

    agg = {}
    for name in score_names:
        gm = group_margin.get(name, {})
        agg[name] = {
            "n_hit": int(sum(r.get(f"n_hit_{name}", 0) for r in per_image)),
            "n_fp_hit": int(sum(r.get(f"n_fp_hit_{name}", 0) for r in per_image)),
            "margins": {g: (gm[g]["sum"] / gm[g]["n"] if gm.get(g, {}).get("n") else None)
                        for g in ("small", "medium", "large")},
            "n_margin": {g: gm.get(g, {}).get("n", 0) for g in ("small", "medium", "large")},
            "small_ap": (_ap_from_hist(hist[name]["pos"], hist[name]["all"])
                         if name in hist and hist[name]["pos"].sum() > 0 else None),
        }
    return {"per_object": per_object, "per_image": per_image, "n_seen": n_seen,
            "timed_out": timed_out, "agg": agg, "evidence": ev, "kept_maps": kept,
            "hook_shapes": {pn: [int(x) for x in PAIR_SPECS[pn]["contract_shape"]]
                            for pn in pair_names},
            "deploy_crosscheck": {"n_checked": deploy_checked, "n_flip": deploy_flips}}


# ============================================================================ 输出
def _write_sums(out_dir):
    lines = []
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p) and name != "SHA256SUMS.txt":
            lines.append(f"{sha256_file(p)}  {name}")
    with open(os.path.join(out_dir, "SHA256SUMS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_figures(out_dir, args, kept):
    """§5.5：32 张规则抽样定性图（前 16 missed-small 且 HF-only，后 16 FP-like 且 HF-hit）。"""
    if not kept:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cv2
    except Exception as e:  # noqa: BLE001
        print(f"[PREFLIGHT][WARN] figures skipped: {type(e).__name__}: {e}", flush=True)
        return 0
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    key = lambda n: hashlib.sha1(n.encode()).hexdigest()  # noqa: E731
    g1 = sorted({k["image"] for k in kept if k["why"] == "hf_only_missed_small"}, key=key)[:16]
    g2 = sorted({k["image"] for k in kept if k["why"] == "fp_like_hf_hit"}, key=key)[:16]
    order = g1 + g2
    picked, seen = [], set()
    for k in kept:
        if k["image"] in order and k["image"] not in seen:
            picked.append(k)
            seen.add(k["image"])
    written = 0
    for k in picked:
        nm = k["image"]
        try:
            a = cv2.imread(os.path.join(args.dataset_root, "A", nm))
            b = cv2.imread(os.path.join(args.dataset_root, "B", nm))
            g = cv2.imread(os.path.join(args.dataset_root, "label", nm), 0)
            if a is None or b is None or g is None:
                continue
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_NEAREST) >= 128
            mask = (k["prob"] > 127).astype(np.uint8)
            vis = a.copy()
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255, 0, 0), 1)
            fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
            ax[0, 0].imshow(a); ax[0, 0].set_title(f"A  {nm}")
            ax[0, 1].imshow(b); ax[0, 1].set_title("B")
            ax[0, 2].imshow(g, cmap="gray"); ax[0, 2].set_title("GT (gray>=128)")
            ax[1, 0].imshow(vis); ax[1, 0].set_title("M1 pred (p>0.5) outline")
            ax[1, 1].imshow(k["hf"], cmap="inferno",
                            vmin=0, vmax=255); ax[1, 1].set_title("S_HF (rank)")
            ax[1, 2].imshow(k["tap"], cmap="inferno",
                            vmin=0, vmax=255); ax[1, 2].set_title("S_TAP (rank)")
            for r in ax.ravel():
                r.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"{written:02d}_{nm}.png"), dpi=100)
            plt.close(fig)
            written += 1
        except Exception:  # noqa: BLE001
            continue
    return written


def _gate(name, ok, value, threshold, note=""):
    return {"pass": bool(ok), "value": value, "threshold": threshold, "note": note}


# ============================================================================ 主流程
def run(args):
    t_start = time.time()
    out_dir = args.out_dir
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            raise SystemExit(f"[P0_INVALID] out-dir exists and is not a directory: {out_dir}")
        if os.listdir(out_dir):
            raise SystemExit(f"[P0_INVALID] out-dir is not empty (refuse to overwrite): "
                             f"{out_dir}")
    os.makedirs(out_dir, exist_ok=False)

    print(f"[PREFLIGHT] {PREFLIGHT_VERSION} dataset={args.dataset} run={args.run} "
          f"variant={args.variant}", flush=True)

    import torch
    import selective_scan_cuda_oflex  # noqa: F401  (§5.3.1 CUDA kernel 必须可用)
    set_eval_numerics()

    # ---------------------------------------------------------------- P0 输入身份
    if args.dataset != "SYSU-CD-256":
        raise SystemExit("[P0_INVALID] Run5 前测只允许 SYSU-CD-256")
    if args.split_hash != "sha1_10pct":
        raise SystemExit("[P0_INVALID] --split-hash 只允许 sha1_10pct")
    if args.limit:
        print("[PREFLIGHT][DRY] --limit>0：本次只做链路自检，状态恒为 DRY_ONLY", flush=True)

    ckpt_path, ckpt_meta = pick_best_ckpt(args.run, args.variant, args.dataset,
                                          ckpt_root=args.ckpt_root)
    m1_sha = sha256_file(ckpt_path)
    test_list_sha = sha256_text_lines(args.test_list)
    test_list_sha_raw = sha256_file(args.test_list)
    train_list_sha = sha256_text_lines(args.train_list)
    if m1_sha != M1_SYSU_SHA:
        raise SystemExit(f"[P0_INVALID] M1 ckpt sha256 mismatch: {m1_sha} != {M1_SYSU_SHA}")
    if test_list_sha != SYSU_TEST_LIST_SHA:
        raise SystemExit(f"[P0_INVALID] test list {SYSU_TEST_LIST_SHA_KIND} mismatch: "
                         f"{test_list_sha} != {SYSU_TEST_LIST_SHA} "
                         f"(raw={test_list_sha_raw})")
    test_names = read_list_names(args.test_list)
    train_names = read_list_names(args.train_list)
    if len(test_names) != EXPECT_N_TEST:
        raise SystemExit(f"[P0_INVALID] test list has {len(test_names)} entries != "
                         f"{EXPECT_N_TEST}")

    probes, probe_info = {}, {}
    for pn in PAIR_NAMES:
        spec = PAIR_SPECS[pn]
        mod, ck, path = _load_frozen_probe(args.probe_dir, spec["tap_key"],
                                           spec["tap_channels"], args.device)
        probes[spec["tap_key"]] = mod
        probe_info[spec["tap_key"]] = {
            "path": path, "sha256": sha256_file(path), "C_in": int(ck["C_in"]),
            "encoder_input": (ck.get("protocol") or {}).get("encoder_input"),
            "variant": ck.get("variant"), "tag": ck.get("tag"),
        }
        print(f"[PREFLIGHT] probe {spec['tap_key']} "
              f"sha256={probe_info[spec['tap_key']]['sha256']} C_in={ck['C_in']}", flush=True)

    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pretrain = os.path.join(project, "pretrained_weight", "tinyvim_s_1000e.pth")
    pretrain_sha = sha256_file(pretrain) if os.path.isfile(pretrain) else None

    inputs = {
        "preflight_version": PREFLIGHT_VERSION,
        "protocol_tag": PROTOCOL_TAG,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "repository_sha": os.environ.get("CASA_REPO_SHA"),
        "dataset": args.dataset,
        "run": args.run,
        "variant": args.variant,
        "ckpt_root": args.ckpt_root,
        "ckpt_path": ckpt_path,
        "ckpt_sha256": m1_sha,
        "ckpt_sha256_expected": M1_SYSU_SHA,
        "arch_sidecar": ckpt_meta.get("arch"),
        "train_list": args.train_list,
        "train_list_sha256": train_list_sha,
        "test_list": args.test_list,
        "test_list_sha256": test_list_sha,
        "test_list_sha256_kind": SYSU_TEST_LIST_SHA_KIND,
        "test_list_sha256_expected": SYSU_TEST_LIST_SHA,
        "test_list_raw_sha256": test_list_sha_raw,
        "n_train_list": len(train_names),
        "n_test_list": len(test_names),
        "probe_dir": args.probe_dir,
        "probes": probe_info,
        "pretrained_path": pretrain,
        "pretrained_sha256": pretrain_sha,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "split_hash": args.split_hash,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "limit": args.limit,
        "budget_sec": args.budget_sec,
        "deploy_crosscheck_n": args.deploy_crosscheck_n,
        "margin_threshold": MARGIN_THRESHOLD,
        "gates": GATES,
        "val_screen": {"min_missed": VAL_MIN_MISSED, "min_fp_like": VAL_MIN_FP,
                       "gap_min": VAL_SCREEN_GAP_MIN, "novel_min": VAL_SCREEN_NOVEL_MIN,
                       "tie_eps": TIE_EPS},
        "pairs": {pn: dict(PAIR_SPECS[pn]) for pn in PAIR_NAMES},
        "binary_source": ("train_graph_eval；另以 deploy 折叠图在前 N 张真实图做二值一致性核验"),
        "code_identity": code_identity(),
        "source_sha256": {rel: sha256_file(os.path.join(project, rel))
                          for rel in DIAG_CODE_FILES
                          + ["analyse/tvim_run5_hf_pair_preflight.py"]},
        "env": env_info(),
        "rules": {
            "gt_connectivity": 4,
            "small_area": "[1,255] at native 256^2；missed_small = r_i = TP_i/area_i < 0.25",
            "label_rule": "gray>=128 已由 Transforms.Normalize 应用；不二次以 0.5 灰度分割",
            "pred_rule": "prob > 0.5（严格大于，与 train.py 一致）",
            "fp_like": "预测 4 连通域 IoU>=0.10 一次性最大权重匹配后剩余、面积 [1,255]",
            "score_rule": ("HF: u=mean_c|hB-hA| 在 h*w 内稳定平均秩→双线性上采样 256²→再 rank；"
                           "TAP: sigmoid(probe([F_A,F_B,|F_A-F_B|])) 上采样 256²→同样 rank"),
            "ring": "R=(dilate_4(O)\\O)∩(~GT)，4-邻接 structure，iterations=4",
            "ap": f"{NBINS}-bin，正样本仅 small 组 GT，负样本所有 GT==0，pos/(pos+bg) 分母",
            "bootstrap": f"按图像有放回 {args.bootstrap} 次，seed={args.seed}，每次重算计数比",
        },
    }

    # ---------------------------------------------------------------- 模型（只读）
    model, minfo = build_model(args.variant, device=args.device, ckpt_path=ckpt_path,
                               strict=True)
    assert not minfo["missing_keys"] and not minfo["unexpected_keys"], minfo
    model_deploy = copy.deepcopy(model)
    model_deploy.eval()
    model_deploy.switch_to_deploy()
    print("[PREFLIGHT] train graph + deploy copy built; hooks not yet registered", flush=True)

    # ---------------------------------------------------------------- val 选择
    val_loader, val_ds = _make_loader(args.dataset_root, args.train_list, args.batch_size,
                                      args.num_workers,
                                      names_filter=lambda n: _hash_split(n, 0.10) == "val")
    val_names = list(val_ds.file_list)
    print(f"[PREFLIGHT] val split (D3 sha1-hash 10% of train.txt): {len(val_names)} images",
          flush=True)
    val = _run_pass(model, model_deploy, val_loader, val_names, args.device, args,
                    PAIR_NAMES, probes, [], "VAL", t_start)
    n_missed_val = int(sum(r["n_missed"] for r in val["per_image"]))
    n_fp_val = int(sum(r["n_fp"] for r in val["per_image"]))
    inputs["n_val_images"] = val["n_seen"]

    val_cand = {}
    vrows = [o for o in val["per_object"] if o["kind"] == "missed_small"]
    for pn in PAIR_NAMES:
        hk, tk = f"HF_{pn}", f"TAP_{pn}"
        n_hf = sum(1 for o in vrows if _is_hit(o, hk))
        n_tap = sum(1 for o in vrows if _is_hit(o, tk))
        n_only = sum(1 for o in vrows if _is_hit(o, hk) and not _is_hit(o, tk))
        rescue = (n_hf / n_missed_val) if n_missed_val else None
        fpr = (val["agg"][hk]["n_fp_hit"] / n_fp_val) if n_fp_val else None
        val_cand[pn] = {
            "n_missed": n_missed_val, "n_fp_like": n_fp_val,
            "n_hf_hit": n_hf, "n_tap_hit": n_tap, "n_hf_only": n_only,
            "rescue": rescue, "fp_like": fpr,
            "gap": (None if rescue is None or fpr is None else rescue - fpr),
            "novel_rescue": (n_only / n_missed_val) if n_missed_val else None,
            "tap_rescue": (n_tap / n_missed_val) if n_missed_val else None,
            "margins": val["agg"][hk]["margins"], "n_margin": val["agg"][hk]["n_margin"],
            "small_ap": val["agg"][hk]["small_ap"],
            "tap_margins": val["agg"][tk]["margins"], "tap_small_ap": val["agg"][tk]["small_ap"],
            "deploy_delta_params": PAIR_SPECS[pn]["deploy_delta_params"],
            "deploy_mac": PAIR_SPECS[pn]["deploy_mac"],
        }
        print(f"[VAL] {pn}: n_missed={n_missed_val} n_fp={n_fp_val} rescue={rescue} "
              f"fp_like={fpr} gap={val_cand[pn]['gap']} "
              f"novel={val_cand[pn]['novel_rescue']}", flush=True)

    eligible = [pn for pn in PAIR_NAMES
                if val_cand[pn]["gap"] is not None and val_cand[pn]["novel_rescue"] is not None
                and val_cand[pn]["gap"] >= VAL_SCREEN_GAP_MIN
                and val_cand[pn]["novel_rescue"] >= VAL_SCREEN_NOVEL_MIN]
    counts_ok = (n_missed_val >= VAL_MIN_MISSED and n_fp_val >= VAL_MIN_FP
                 and not val["timed_out"])
    val_ok = counts_ok and len(eligible) >= 1
    val_screen_status = classify_val_screen(counts_ok, val["timed_out"], eligible)

    selected, tie_breaker = (select_pair(val_cand, eligible) if val_ok else (None, None))

    selection = {
        "preflight_version": PREFLIGHT_VERSION,
        "selection_source": "train.txt D3 sha1-hash 10% probe-val only",
        "selection_from_train_val_only": True,
        "test_used_for_selection": False,
        "n_val_images": val["n_seen"],
        "val_rule": {"min_missed": VAL_MIN_MISSED, "min_fp_like": VAL_MIN_FP,
                     "screen_gap_min": VAL_SCREEN_GAP_MIN,
                     "screen_novel_min": VAL_SCREEN_NOVEL_MIN, "tie_eps": TIE_EPS},
        "n_missed_val": n_missed_val, "n_fp_like_val": n_fp_val,
        "candidates": val_cand, "eligible": eligible,
        "selected_pair": selected,
        "selected_stage": PAIR_SPECS[selected]["stage"] if selected else None,
        "selected_hf_channels": PAIR_SPECS[selected]["hf_channels"] if selected else None,
        "selected_target": (f"TAR.stage{PAIR_SPECS[selected]['stage']}.temporal"
                            if selected else None),
        "selected_hf_source": (f"encoder.{PAIR_SPECS[selected]['hf_attr']}"
                               if selected else None),
        "tie_breaker": tie_breaker, "locked": True,
        "val_timed_out": val["timed_out"],
        "val_screen_status": val_screen_status,
        "val_elapsed_s": round(time.time() - t_start, 1),
    }
    write_json(os.path.join(out_dir, "selection_val.json"), selection)

    val_cols = ["image", "id", "kind", "area", "model_recall"]
    for pn in PAIR_NAMES:
        val_cols += [f"hf_rank_margin_{pn}", f"tap_rank_margin_{pn}",
                     f"hf_hit_{pn}", f"tap_hit_{pn}"]
    with open(os.path.join(out_dir, "val_per_object.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=val_cols)
        w.writeheader()
        for o in val["per_object"]:
            row = {"image": o["image"], "id": o["obj_id"], "kind": o["kind"],
                   "area": o["area"], "model_recall": o["pixel_recall"]}
            for pn in PAIR_NAMES:
                for base, k in (("hf_rank_margin", f"HF_{pn}"), ("tap_rank_margin", f"TAP_{pn}")):
                    row[f"{base}_{pn}"] = o["margins"].get(k)
                for base, k in (("hf_hit", f"HF_{pn}"), ("tap_hit", f"TAP_{pn}")):
                    v = o["margins"].get(k)
                    row[f"{base}_{pn}"] = int(v is not None and v >= MARGIN_THRESHOLD)
            w.writerow(row)

    if not val_ok:
        gate = {
            "stage": "run5_hf_pair_preflight",
            "status": ("INCONCLUSIVE" if val_screen_status != "BOTH_SCREENED_OUT" else "FAIL"),
            "val_screen_status": val_screen_status,
            "preflight_version": PREFLIGHT_VERSION,
            "selected_pair": None,
            "test_pass_executed": False,
            "test_pass_not_run_reason": (
                "§5.3.2/§5.3.3：val 初筛后无合格候选，禁止对 test 做任何挑选"
                if val_screen_status == "BOTH_SCREENED_OUT" else
                "§5.3.2：val 有效样本数不足/超时 → INCONCLUSIVE"),
            "reason": ("val split insufficient or both candidates screened out "
                       "(§5.3.2 → NO_80K；不转备选位置、不改阈值重跑)"),
            "gates": {
                "G0": _gate("G0", True, "P0 inputs OK", "SHA256 + frozen probes + out-dir new"),
                "G1": _gate("G1", False, None, "n_test=4000 & elapsed<=600 & boot>=950",
                            "test 未运行：val 初筛未通过"),
                "G2": _gate("G2", False, None, f"test 未运行（val n_missed={n_missed_val}）"),
                "G3": _gate("G3", False, None, f"test 未运行（val n_fp_like={n_fp_val}）"),
                "G4": _gate("G4", False, None, ">=0.25", "test 未运行"),
                "G5": _gate("G5", False, None, "gap>=0.15 & CI_low>0.03", "test 未运行"),
                "G6": _gate("G6", False, None, "novel>=0.10 & CI_low>0.02", "test 未运行"),
                "G7": _gate("G7", False, None, "margin_delta>=0.02 & CI_low>0", "test 未运行"),
                "G8": _gate("G8", False, None, "selected_pair locked", "test 未运行"),
            },
            "val_summary": {"n_missed": n_missed_val, "n_fp_like": n_fp_val,
                            "counts_ok": counts_ok, "eligible": eligible,
                            "val_screen_status": val_screen_status,
                            "candidates": val_cand},
            "decision": "NO_80K",
            "elapsed_s": round(time.time() - t_start, 1),
            "note": ("§5.4：任一不通过即当日停止；不另选次优 Pair、不改阈值重跑。"
                     "test 未运行 ⇒ 未产生任何对象级 test 统计，不存在事后选层。"),
        }
        write_json(os.path.join(out_dir, "inputs.json"), inputs)
        write_json(os.path.join(out_dir, "gate.json"), gate)
        # §5.3 输出结构必备：test 未运行时写显式占位（表头 + not_run 标记），不留静默缺口
        write_json(os.path.join(out_dir, "bootstrap.json"), {
            "run_status": "NOT_RUN_VAL_SCREEN_FAILED",
            "bootstrap": args.bootstrap, "seed": args.seed, "unit": "image",
            "val_screen_status": val_screen_status, "eligible": eligible,
        })
        write_json(os.path.join(out_dir, "layer_diagnostics.json"), {
            "run_status": "NOT_RUN_VAL_SCREEN_FAILED",
            "test_pass_executed": False,
            "val_screen_status": val_screen_status,
            "val_agg_both_pairs": {pn: val["agg"][f"HF_{pn}"] for pn in PAIR_NAMES},
            "val_tap_agg_both_pairs": {pn: val["agg"][f"TAP_{pn}"] for pn in PAIR_NAMES},
            "val_hook_shapes": val["hook_shapes"],
            "val_deploy_crosscheck": val["deploy_crosscheck"],
            "note": ("val 初筛未通过 → §5.3.3 规定不得对 test 选层/评价，故无 test 层诊断。"
                     "候选级 val 指标见 gate.json/selection_val.json。"),
        })
        for fname, cols in (
                ("test_per_object.csv",
                 ["image", "id", "kind", "area", "model_recall", "hf_rank_margin",
                  "tap_rank_margin", "hf_hit", "tap_hit", "iou_max_any_pred", "fp_like_id"]),
                ("test_per_image.csv",
                 ["image", "n_missed", "n_fp", "n_hf_hit", "n_tap_hit", "n_hf_only",
                  "n_tap_only", "n_hf_fp_hit", "n_skipped_ring", "sum_margin_diff",
                  "n_margin_pairs"])):
            with open(os.path.join(out_dir, fname), "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=cols).writeheader()
        _write_sums(out_dir)
        print(f"[PREFLIGHT] STATUS={gate['status']} "
              f"val_screen_status={val_screen_status} -> NO_80K "
              f"(eligible={eligible})", flush=True)
        return 2

    # ---------------------------------------------------------------- test pass（锁定对）
    spec_sel = PAIR_SPECS[selected]
    test_loader, test_ds = _make_loader(args.dataset_root, args.test_list, args.batch_size,
                                        args.num_workers)
    names_test = list(test_ds.file_list)
    assert names_test == test_names, "test list order mismatch"
    print(f"[PREFLIGHT] locked {selected} stage={spec_sel['stage']} → full test "
          f"({len(test_names)} images)", flush=True)
    evidence_keys = ["L01b_norm0", "L03b_norm2", "L06b_norm4", "L07_network5",
                     "DOUT_decoder_refine", "P00_head_logits"]
    test = _run_pass(model, model_deploy, test_loader, names_test, args.device, args,
                     [selected], probes, evidence_keys, "TEST", t_start,
                     deploy_crosscheck_n=args.deploy_crosscheck_n, keep_maps=True)

    hf, tap = test["agg"]["HF"], test["agg"]["TAP"]
    n_missed = int(sum(r["n_missed"] for r in test["per_image"]))
    n_fp = int(sum(r["n_fp"] for r in test["per_image"]))
    n_skipped = int(sum(r["n_skipped_ring"] for r in test["per_image"]))
    small_ref = {
        "n_small_objects": int(sum(r["small_n"] for r in test["per_image"])),
        "small_pixels": int(sum(r["small_px"] for r in test["per_image"])),
        "small_tp_pixels": int(sum(r["small_tp"] for r in test["per_image"])),
        "small_hit25": int(sum(r["small_hit25"] for r in test["per_image"])),
        "n_missed_small_pre_ring": int(sum(r["n_missed_pre_ring"]
                                           for r in test["per_image"])),
        "n_skipped_ring_small": int(sum(r["n_skipped_ring_small"]
                                        for r in test["per_image"])),
        "n_skipped_ring_fp_like": n_skipped - int(sum(r["n_skipped_ring_small"]
                                                      for r in test["per_image"])),
        "frozen_reference_diag1": {"n_small_objects": 364,
                                   "small_pooled_recall": 0.19742165917906074,
                                   "small_hit25_rate": 0.18956043956043955,
                                   "n_missed_small": 295},
    }
    small_ref["small_pooled_recall"] = (
        small_ref["small_tp_pixels"] / small_ref["small_pixels"]
        if small_ref["small_pixels"] else None)
    small_ref["small_hit25_rate"] = (
        small_ref["small_hit25"] / small_ref["n_small_objects"]
        if small_ref["n_small_objects"] else None)
    small_ref["reconciles_with_diag1"] = bool(
        small_ref["n_small_objects"] == 364
        and abs((small_ref["small_pooled_recall"] or -1) - 0.19742165917906074) < 1e-9
        and abs((small_ref["small_hit25_rate"] or -1) - 0.18956043956043955) < 1e-9)
    n_hf_hit, n_tap_hit, n_hf_fp_hit = hf["n_hit"], tap["n_hit"], hf["n_fp_hit"]
    mrows = [o for o in test["per_object"] if o["kind"] == "missed_small"]
    n_hf_only = sum(1 for o in mrows if _is_hit(o, "HF") and not _is_hit(o, "TAP"))
    n_tap_only = sum(1 for o in mrows if _is_hit(o, "TAP") and not _is_hit(o, "HF"))
    rescue = (n_hf_hit / n_missed) if n_missed else None
    fpr = (n_hf_fp_hit / n_fp) if n_fp else None
    gap = None if (rescue is None or fpr is None) else rescue - fpr
    novel = (n_hf_only / n_missed) if n_missed else None

    d_missed = np.array([r["n_missed"] for r in test["per_image"]], dtype=np.float64)
    d_fp = np.array([r["n_fp"] for r in test["per_image"]], dtype=np.float64)
    a_hf = np.array([r.get("n_hit_HF", 0) for r in test["per_image"]], dtype=np.float64)
    a_fp = np.array([r.get("n_fp_hit_HF", 0) for r in test["per_image"]], dtype=np.float64)
    a_novel = np.array([r.get("n_hf_only", 0) for r in test["per_image"]], dtype=np.float64)
    ms = np.array([r.get("sum_margin_diff_pair", 0.0) for r in test["per_image"]],
                  dtype=np.float64)
    mn = np.array([r.get("n_margin_pairs", 0) for r in test["per_image"]], dtype=np.float64)

    gap_obs, gap_lo, gap_hi, gap_valid = _boot_gap(a_hf, a_fp, d_missed, d_fp,
                                                   args.bootstrap, args.seed)
    nov_obs, nov_lo, nov_hi, nov_valid = _boot_ratio(a_novel, d_missed, args.bootstrap,
                                                     args.seed)
    mdiff_obs, mdiff_lo, mdiff_hi, mdiff_valid = _boot_mean_pair(ms, mn, args.bootstrap,
                                                                 args.seed)

    elapsed = time.time() - t_start
    n_seen = test["n_seen"]
    dry = bool(args.limit)
    g = {}
    g["G0"] = _gate("G0",
                    (not dry) and test["deploy_crosscheck"]["n_flip"] == 0,
                    {"ckpt_sha256": m1_sha, "ckpt_sha256_expected": M1_SYSU_SHA,
                     "test_list_sha256": test_list_sha,
                     "test_list_sha256_kind": SYSU_TEST_LIST_SHA_KIND,
                     "test_list_sha256_expected": SYSU_TEST_LIST_SHA,
                     "test_list_raw_sha256": test_list_sha_raw,
                     "probes_frozen": True, "out_dir_new": True,
                     "selection_from_train_val_only": True, "selected_pair": selected,
                     "p0_deploy_binary_crosscheck": test["deploy_crosscheck"]},
                    "SHA256 严格匹配 + frozen + selection_from_train_val_only + "
                    "真实 batch train/deploy 二值一致",
                    "二值来源为 train graph（与 Diag1 同口径）；deploy 交叉核验为 P0 证据")
    g["G1"] = _gate("G1",
                    (n_seen == EXPECT_N_TEST) and (elapsed <= args.budget_sec)
                    and (min(gap_valid, nov_valid, mdiff_valid) >= 950)
                    and (not test["timed_out"]) and (not dry),
                    {"n_test": n_seen, "n_test_expected": EXPECT_N_TEST,
                     "elapsed_s": round(elapsed, 1), "budget_sec": args.budget_sec,
                     "valid_bootstraps": {"gap": gap_valid, "novel": nov_valid,
                                          "margin": mdiff_valid}},
                    "n_test=4000 且 elapsed<=600 且 valid_bootstraps>=950/1000")
    g["G2"] = _gate("G2", n_missed >= GATES["G2_n_missed_small_min"], n_missed,
                    GATES["G2_n_missed_small_min"])
    g["G3"] = _gate("G3", n_fp >= GATES["G3_n_fp_like_min"], n_fp, GATES["G3_n_fp_like_min"])
    g["G4"] = _gate("G4", rescue is not None and rescue >= GATES["G4_rescue_rate_min"],
                    rescue, GATES["G4_rescue_rate_min"])
    g["G5"] = _gate("G5",
                    gap is not None and gap >= GATES["G5_gap_min"] and gap_lo is not None
                    and gap_lo > GATES["G5_gap_ci_low_min"],
                    {"gap": gap, "ci95": [gap_lo, gap_hi], "rescue": rescue, "fp_like": fpr},
                    {"gap_min": GATES["G5_gap_min"],
                     "ci_low_min": GATES["G5_gap_ci_low_min"]})
    g["G6"] = _gate("G6",
                    novel is not None and novel >= GATES["G6_novel_rescue_min"]
                    and nov_lo is not None and nov_lo > GATES["G6_novel_ci_low_min"],
                    {"novel_rescue": novel, "ci95": [nov_lo, nov_hi], "n_hf_only": n_hf_only,
                     "n_tap_only": n_tap_only,
                     "tap_rescue": (n_tap_hit / n_missed) if n_missed else None,
                     "hf_minus_tap_rescue": (((n_hf_hit - n_tap_hit) / n_missed)
                                             if n_missed else None)},
                    {"novel_min": GATES["G6_novel_rescue_min"],
                     "ci_low_min": GATES["G6_novel_ci_low_min"]})
    g["G7"] = _gate("G7",
                    mdiff_obs is not None and mdiff_obs >= GATES["G7_margin_delta_min"]
                    and mdiff_lo is not None and mdiff_lo > GATES["G7_margin_ci_low_min"],
                    {"mean_margin_HF_minus_TAP": mdiff_obs, "ci95": [mdiff_lo, mdiff_hi],
                     "margin_HF_small": hf["margins"]["small"],
                     "margin_TAP_small": tap["margins"]["small"]},
                    {"delta_min": GATES["G7_margin_delta_min"],
                     "ci_low_min": GATES["G7_margin_ci_low_min"]})
    g["G8"] = _gate("G8",
                    (selected in PAIR_NAMES) and (not dry) and selection["locked"]
                    and (not selection["test_used_for_selection"]),
                    {"selected_pair": selected, "no_new_train": True, "test_switching": False,
                     "all_candidates_archived": True},
                    "selected_pair ∈ {Pair-2, Pair-3}，test 不切换，no_new_train=true")

    all_pass = all(v["pass"] for v in g.values())
    if dry:
        status = "DRY_ONLY"
    elif test["timed_out"] or elapsed > args.budget_sec:
        status = "TIMEOUT"
    elif not g["G0"]["pass"]:
        status = "P0_INVALID"
    elif not all_pass:
        status = "FAIL"
    else:
        status = "PASS"

    gate = {
        "stage": "run5_hf_pair_preflight", "status": status,
        "preflight_version": PREFLIGHT_VERSION, "protocol_tag": PROTOCOL_TAG,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "selected_pair": selected, "selected_stage": spec_sel["stage"],
        "selected_hf_channels": spec_sel["hf_channels"],
        "selected_target": f"TAR.stage{spec_sel['stage']}.temporal",
        "selected_hf_source": f"encoder.{spec_sel['hf_attr']}",
        "gates": g, "n_test": n_seen,
        "n_missed_small": n_missed, "n_fp_like": n_fp, "n_skipped_no_ring": n_skipped,
        "small_reference_reconciliation": small_ref,
        "rescue_rate": rescue, "fp_like_rate": fpr, "gap": gap, "gap_ci95": [gap_lo, gap_hi],
        "novel_rescue": novel, "novel_rescue_ci95": [nov_lo, nov_hi],
        "mean_margin_HF_minus_TAP": mdiff_obs, "mean_margin_diff_ci95": [mdiff_lo, mdiff_hi],
        "bootstrap": args.bootstrap, "bootstrap_unit": "image", "seed": args.seed,
        "margin_threshold": MARGIN_THRESHOLD, "elapsed_s": round(elapsed, 1),
        "budget_sec": args.budget_sec, "val_elapsed_s": selection["val_elapsed_s"],
        "decision": ("80K ALLOWED (G0–G8 all PASS)" if status == "PASS" else
                     "NO_80K (gate not PASS) — 不另选次优 Pair、不改阈值重跑"),
        "note": "test 只对 train-val 锁定的单对评价一次；test 结果不参与选层（§5.3.3/R6）",
    }

    write_json(os.path.join(out_dir, "inputs.json"), inputs)
    write_json(os.path.join(out_dir, "gate.json"), gate)
    write_json(os.path.join(out_dir, "bootstrap.json"), {
        "bootstrap": args.bootstrap, "seed": args.seed, "unit": "image",
        "valid_bootstraps": {"gap": gap_valid, "novel_rescue": nov_valid,
                             "margin_diff": mdiff_valid},
        "gap": {"obs": gap_obs, "ci95": [gap_lo, gap_hi]},
        "novel_rescue": {"obs": nov_obs, "ci95": [nov_lo, nov_hi]},
        "margin_diff": {"obs": mdiff_obs, "ci95": [mdiff_lo, mdiff_hi]},
    })
    write_json(os.path.join(out_dir, "layer_diagnostics.json"), {
        "selected_pair": selected, "hook_shapes": test["hook_shapes"],
        "hf_channels": spec_sel["hf_channels"], "hf_spatial": spec_sel["hf_spatial"],
        "target": f"TAR.stage{spec_sel['stage']}.temporal",
        "scores": {"HF": hf, "TAP": tap},
        "evidence_hooks": test["evidence"],
        "val_agg_both_pairs": {pn: val["agg"][f"HF_{pn}"] for pn in PAIR_NAMES},
        "cross_scale_note": ("跨尺度（1/4→1/32）small margin 递减已由 Diag1 F05 冻结"
                             "（0.0779→0.0400→0.0238→0.0022）；"
                             "本脚本给出同尺度 HF vs TAP 的配对检验"),
        "deploy_budget": {
            "mainline_deploy_params": 4880190,
            "delta_params": spec_sel["deploy_delta_params"],
            "new_deploy_total_params": 4880190 + spec_sel["deploy_delta_params"],
            "delta_mac": spec_sel["deploy_mac"], "hard_cap": 5000000,
        },
    })
    with open(os.path.join(out_dir, "test_per_object.csv"), "w", newline="",
              encoding="utf-8") as f:
        cols = ["image", "id", "kind", "area", "model_recall", "hf_rank_margin",
                "tap_rank_margin", "hf_hit", "tap_hit", "iou_max_any_pred", "fp_like_id"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for o in test["per_object"]:
            w.writerow({"image": o["image"], "id": o["obj_id"], "kind": o["kind"],
                        "area": o["area"], "model_recall": o["pixel_recall"],
                        "hf_rank_margin": o["margins"].get("HF"),
                        "tap_rank_margin": o["margins"].get("TAP"),
                        "hf_hit": int(_is_hit(o, "HF")), "tap_hit": int(_is_hit(o, "TAP")),
                        "iou_max_any_pred": o.get("iou_max_any_pred"),
                        "fp_like_id": o.get("fp_like_id")})
    with open(os.path.join(out_dir, "test_per_image.csv"), "w", newline="",
              encoding="utf-8") as f:
        cols = ["image", "n_missed", "n_fp", "n_hf_hit", "n_tap_hit", "n_hf_only",
                "n_tap_only", "n_hf_fp_hit", "n_skipped_ring", "sum_margin_diff",
                "n_margin_pairs"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in test["per_image"]:
            w.writerow({"image": r["image"], "n_missed": r["n_missed"], "n_fp": r["n_fp"],
                        "n_hf_hit": r.get("n_hit_HF", 0), "n_tap_hit": r.get("n_hit_TAP", 0),
                        "n_hf_only": r.get("n_hf_only", 0),
                        "n_tap_only": r.get("n_tap_only", 0),
                        "n_hf_fp_hit": r.get("n_fp_hit_HF", 0),
                        "n_skipped_ring": r.get("n_skipped_ring", 0),
                        "sum_margin_diff": r.get("sum_margin_diff_pair", 0.0),
                        "n_margin_pairs": r.get("n_margin_pairs", 0)})
    figs = _write_figures(out_dir, args, test["kept_maps"]) if args.save_figures else 0
    _write_sums(out_dir)

    print(f"[PREFLIGHT] test images={n_seen} elapsed={elapsed:.0f}s "
          f"(val {selection['val_elapsed_s']}s)", flush=True)
    print(f"[PREFLIGHT] n_missed_small={n_missed} n_fp_like={n_fp} "
          f"n_skipped_no_ring={n_skipped}", flush=True)
    print(f"[PREFLIGHT] small ref: n={small_ref['n_small_objects']} "
          f"pooled={small_ref['small_pooled_recall']} "
          f"hit25={small_ref['small_hit25_rate']} "
          f"pre_ring_missed={small_ref['n_missed_small_pre_ring']} "
          f"ring_skipped_small={small_ref['n_skipped_ring_small']} "
          f"reconciles_diag1={small_ref['reconciles_with_diag1']}", flush=True)
    print(f"[PREFLIGHT] rescue={rescue} fp_like={fpr} gap={gap} "
          f"ci95=[{gap_lo},{gap_hi}]", flush=True)
    print(f"[PREFLIGHT] novel_rescue={novel} ci95=[{nov_lo},{nov_hi}] "
          f"margin_delta={mdiff_obs} ci95=[{mdiff_lo},{mdiff_hi}]", flush=True)
    print(f"[PREFLIGHT] figures={figs} selected={selected}", flush=True)
    for k in sorted(g):
        print(f"[GATE] {k} pass={g[k]['pass']} value={g[k]['value']}", flush=True)
    print(f"[PREFLIGHT] STATUS={status} -> {os.path.join(out_dir, 'gate.json')}", flush=True)
    return 0 if status == "PASS" else 2


def main():
    ap = argparse.ArgumentParser(
        description="Run5 §5 只读前测：原生 local_conv 高频 → TAR 同尺度时相投影（G0–G8）")
    ap.add_argument("--dataset", type=str, default="SYSU-CD-256")
    ap.add_argument("--run", type=str, default="Run1")
    ap.add_argument("--variant", type=str, default="M1_FULL")
    ap.add_argument("--ckpt-root", type=str,
                    default="/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM")
    ap.add_argument("--dataset-root", type=str, default="/share_datasets/CD/SYSU-CD-256")
    ap.add_argument("--train-list", type=str,
                    default="/share_datasets/CD/SYSU-CD-256/list/train.txt")
    ap.add_argument("--test-list", type=str,
                    default="/share_datasets/CD/SYSU-CD-256/list/test.txt")
    ap.add_argument("--probe-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--split-hash", type=str, default="sha1_10pct")
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug only；非零时状态恒为 DRY_ONLY，永不 PASS")
    ap.add_argument("--budget-sec", type=float, default=BUDGET_S)
    ap.add_argument("--deploy-crosscheck-n", type=int, default=64,
                    help="P0 证据：前 N 张真实图 train/deploy 二值一致性（必须 0 flip）")
    ap.add_argument("--save-figures", type=int, default=1)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
