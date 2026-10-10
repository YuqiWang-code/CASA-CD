"""CASA-CD 小目标诊断 —— 正确的 GT 面积分组 / 预测连通域 / 边界带指标（P0）。

本模块**纯 numpy + scipy**（不 import torch），可离线跑单元测试。

修正 `analyse/run2_zero_cost_diag.py::component_pr` 的评价语义缺陷：
旧实现的 `FP = ((pred>0) & (gt==0) & m).sum()` 中 `m ⊆ (gt>0)`，故 FP 恒为 0，
得到的是「受限正样本组内的 pseudo-Precision/pseudo-F1」，不是对象级 P/R/F1。
本模块给出语义明确、可复现的替代指标：

  A) GT 面积分组 + 全局累计像素 Recall（主诊断）
     - 面积一律在**原生 256×256 GT** 上按 4 邻接（SciPy 默认）连通域计算；
     - micro-pixel Recall（像素加权）、object-macro pixel Recall（对象等权）、
       ObjectHit@1 / @25%（预注册主对象召回）/ @50%；
     - 空组返回 None（NA），不置 0，不参与 macro 分母。
  B) 真正匹配预测连通域的对象级指标（第二诊断）
     - 预测图按同样连通性标记，逐图构造 GT×Pred IoU 矩阵，
       用一次性最大权重二分匹配（`scipy.optimize.linear_sum_assignment`，确定性），
       仅 IoU>=0.10（宽松）/ >=0.50（严格）计为匹配；
     - 数据集级 ObjPrecision / ObjRecall / ObjF1；对象面积分层只报 ObjRecall；
     - 记录 N_gt / N_pred / N_matched / unmatched_pred / 预测连通域面积分布 / FP 像素。
  C) 像素级全图六指标（与 models/model/metric_tool.py 完全同口径）与边界带
     - 边界带 = dilate(GT, r) \\ erode(GT, r)，structuring element = 4 邻接，
       图像边界处理 border_value=0（带不能超出图像），并记录贴边 GT 像素数以说明截断；
     - 报告 band 内 TP/FP/FN、band 像素数、空 band 数，以及各面积组 band 内外 TP/FN。

协议常量集中在 `METRICS_PROTOCOL`，供 metrics_protocol.json 落盘。
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------------------
# 协议常量（写进 metrics_protocol.json）
# --------------------------------------------------------------------------------------
CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
CONN8 = np.ones((3, 3), dtype=bool)

AREA_EDGES = [1, 16, 64, 256, 1024]          # 分组边界（左闭右开）
AREA_GROUPS = ["tiny_1_15", "tiny_16_63", "small_64_255", "medium_256_1023", "large_1024_inf"]
# 对齐旧分组口径（1<=a<256 / 256<=a<1024 / a>=1024）
GROUP_ALIAS = {"small": ["tiny_1_15", "tiny_16_63", "small_64_255"],
               "medium": ["medium_256_1023"],
               "large": ["large_1024_inf"]}

IOU_LOOSE = 0.10
IOU_STRICT = 0.50
THRESHOLD_RULE = "pred_bool = (prob > 0.5)  # 与 models/train.py 一致（严格大于）"

METRICS_PROTOCOL = {
    "version": "tvim_object_metrics/v1 (Diag1)",
    "connectivity_primary": 4,
    "connectivity_sensitivity": 8,
    "st5": "4-邻接 structuring element [[0,1,0],[1,1,1],[0,1,0]]",
    "iou_loose": IOU_LOOSE,
    "iou_strict": IOU_STRICT,
    "matching": "scipy.optimize.linear_sum_assignment on -IoU (one-shot max-weight bipartite)",
    "threshold_rule": THRESHOLD_RULE,
    "area_edges": AREA_EDGES,
    "area_groups": AREA_GROUPS,
    "legacy_group_alias": GROUP_ALIAS,
    "empty_group_policy": "None (NA), macro 分母不含空组",
    "empty_image_policy": ("无 GT 且无 Pred → 不计入 N_gt/N_pred/N_matched；"
                           "有 Pred 无 GT → 计入 N_pred 且全部 unmatched(FP)"),
    "boundary_band": ("band_r = binary_dilation(gt, st5, iterations=r, border_value=0) "
                      "& ~binary_erosion(gt, st5, iterations=r, border_value=0)；"
                      "记录贴边 GT 像素数说明截断"),
    "note": "旧 run2_zero_cost_diag.py 的 comp_* 因 FP 恒为 0，仅为受限正样本伪指标，不得用作对象级 F1",
}


def bin_index(area):
    """面积 → 分组名（左闭右开）；area<1 视为无效（不应出现）。"""
    a = int(area)
    if a < 1:
        raise ValueError(f"invalid area {area}")
    if a < 16:
        return "tiny_1_15"
    if a < 64:
        return "tiny_16_63"
    if a < 256:
        return "small_64_255"
    if a < 1024:
        return "medium_256_1023"
    return "large_1024_inf"


def label_components(mask, connectivity=4):
    """连通域标记。connectivity=4 → 4 邻接；8 → 8 邻接。返回 (labels, n)。"""
    st = CONN4 if connectivity == 4 else CONN8
    return ndimage.label(np.asarray(mask).astype(bool), structure=st)


def _as_bool(a):
    return np.asarray(a).astype(bool)


# --------------------------------------------------------------------------------------
# A) GT 面积分组 + 全局累计像素 Recall
# --------------------------------------------------------------------------------------
def gt_object_stats(pred, gt, connectivity=4, image_name=None):
    """单图 GT 对象级统计（返回 (rows, group_agg)）。

    rows: 每个 GT 对象一行 dict（供 D1 明细使用）。
    group_agg: {group: {n_objects, n_pixels, tp_pixels, macro_recall_sum, hit1, hit25, hit50, n_missed}}
    """
    pred_b = _as_bool(pred)
    gt_b = _as_bool(gt)
    lab, n = label_components(gt_b, connectivity)
    rows = []
    agg = {g: {"n_objects": 0, "n_pixels": 0, "tp_pixels": 0, "macro_recall_sum": 0.0,
               "hit1": 0, "hit25": 0, "hit50": 0, "n_missed": 0} for g in AREA_GROUPS}
    if n == 0:
        return rows, agg
    objs = ndimage.find_objects(lab)
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    tp_map = pred_b & gt_b
    tp_counts = np.bincount(lab[tp_map].ravel(), minlength=n + 1) if tp_map.any() else np.zeros(n + 1, dtype=np.int64)
    for j in range(1, n + 1):
        sl = objs[j - 1]
        area = int(counts[j])
        tp = int(tp_counts[j])
        r = tp / area
        g = bin_index(area)
        a = agg[g]
        a["n_objects"] += 1
        a["n_pixels"] += area
        a["tp_pixels"] += tp
        a["macro_recall_sum"] += r
        if tp >= 1:
            a["hit1"] += 1
        else:
            a["n_missed"] += 1
        if r >= 0.25:
            a["hit25"] += 1
        if r >= 0.50:
            a["hit50"] += 1
        if image_name is not None:
            y0, y1 = sl[0].start, sl[0].stop
            x0, x1 = sl[1].start, sl[1].stop
            bbox_h, bbox_w = y1 - y0, x1 - x0
            bbox_area = bbox_h * bbox_w
            per = _perimeter_approx(lab, j, sl)
            rows.append({
                "sample_name": image_name, "gt_component_id": j, "area": area,
                "bbox_x1": int(x0), "bbox_y1": int(y0), "bbox_x2": int(x1), "bbox_y2": int(y1),
                "bbox_width": int(bbox_w), "bbox_height": int(bbox_h),
                "bbox_aspect_ratio": float(max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))),
                "perimeter_approx": int(per),
                "occupancy": float(area / max(1, bbox_area)),
                "touches_border": bool(y0 == 0 or x0 == 0 or y1 == gt_b.shape[0] or x1 == gt_b.shape[1]),
                "pixel_recall": float(r), "hit1": int(tp >= 1),
                "hit25": int(r >= 0.25), "hit50": int(r >= 0.50),
            })
    return rows, agg


def _perimeter_approx(lab, j, sl):
    """对象周长近似（4 邻接边界像素数），用于形态描述。"""
    sub = (lab[sl] == j)
    pad = np.zeros((sub.shape[0] + 2, sub.shape[1] + 2), dtype=bool)
    pad[1:-1, 1:-1] = sub
    # 4 邻接边界：与背景相邻的边
    h = np.count_nonzero(pad[1:-1, 1:-1] & ~pad[:-2, 1:-1]) + np.count_nonzero(pad[1:-1, 1:-1] & ~pad[2:, 1:-1])
    w = np.count_nonzero(pad[1:-1, 1:-1] & ~pad[1:-1, :-2]) + np.count_nonzero(pad[1:-1, 1:-1] & ~pad[1:-1, 2:])
    return h + w


# --------------------------------------------------------------------------------------
# B) GT×Pred 对象匹配（一次性最大权重二分匹配）
# --------------------------------------------------------------------------------------
def object_iou_matrix(pred, gt, connectivity=4):
    """单图 GT×Pred 连通域 IoU 矩阵及元数据（只算一次，供匹配与 per-object 统计复用）。

    返回 dict(iou (n_gt×n_pred), gt_areas, pred_areas, gt_objs, pr_objs, gt_lab, pr_lab)
    """
    pred_b = _as_bool(pred)
    gt_b = _as_bool(gt)
    gt_lab, n_gt = label_components(gt_b, connectivity)
    pr_lab, n_pred = label_components(pred_b, connectivity)
    gt_objs = ndimage.find_objects(gt_lab) if n_gt else []
    pr_objs = ndimage.find_objects(pr_lab) if n_pred else []
    gt_areas = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)[1:].astype(int).tolist() if n_gt else []
    pr_areas = np.bincount(pr_lab.ravel(), minlength=n_pred + 1)[1:].astype(int).tolist() if n_pred else []
    iou = np.zeros((n_gt, n_pred), dtype=np.float64)
    if n_gt and n_pred:
        for i in range(1, n_gt + 1):
            for j in range(1, n_pred + 1):
                y0 = max(gt_objs[i - 1][0].start, pr_objs[j - 1][0].start)
                y1 = min(gt_objs[i - 1][0].stop, pr_objs[j - 1][0].stop)
                x0 = max(gt_objs[i - 1][1].start, pr_objs[j - 1][1].start)
                x1 = min(gt_objs[i - 1][1].stop, pr_objs[j - 1][1].stop)
                if y0 >= y1 or x0 >= x1:
                    continue
                inter = int(np.count_nonzero((gt_lab[y0:y1, x0:x1] == i) & (pr_lab[y0:y1, x0:x1] == j)))
                if inter == 0:
                    continue
                union = gt_areas[i - 1] + pr_areas[j - 1] - inter
                iou[i - 1, j - 1] = inter / union
    return {"iou": iou, "gt_areas": gt_areas, "pred_areas": pr_areas, "gt_objs": gt_objs,
            "pr_objs": pr_objs, "gt_lab": gt_lab, "pr_lab": pr_lab, "n_gt": n_gt, "n_pred": n_pred}


def match_objects(pred, gt, connectivity=4, iou_loose=IOU_LOOSE, iou_strict=IOU_STRICT, cache=None):
    """单图对象匹配（一次性最大权重二分匹配）。

    返回 dict(含 n_gt, n_pred, matched_loose, matched_strict, pred_areas..., matched_gt_idx,
    matched_pred_idx)。`cache` 可传入 object_iou_matrix 的结果避免重复计算。
    """
    pred_b = _as_bool(pred)
    gt_b = _as_bool(gt)
    c = cache or object_iou_matrix(pred_b, gt_b, connectivity)
    iou, n_gt, n_pred = c["iou"], c["n_gt"], c["n_pred"]
    out = {"n_gt": int(n_gt), "n_pred": int(n_pred), "n_matched_loose": 0, "n_matched_strict": 0,
           "iou_pairs_loose": [], "pred_areas": list(c["pred_areas"]), "gt_areas": list(c["gt_areas"]),
           "unmatched_pred_areas": [], "matched_gt_idx": [], "matched_pred_idx": [],
           "fp_pixels": int(np.count_nonzero(pred_b & ~gt_b)),
           "pred_pixels": int(np.count_nonzero(pred_b)),
           "gt_pixels": int(np.count_nonzero(gt_b))}
    if n_gt == 0 or n_pred == 0:
        out["unmatched_pred_areas"] = list(c["pred_areas"])
        return out
    rows, cols = linear_sum_assignment(-iou)
    matched_pred = set()
    loose = strict = 0
    for r, cc in zip(rows, cols):
        v = iou[r, cc]
        if v >= iou_loose:
            loose += 1
            matched_pred.add(cc)
            out["iou_pairs_loose"].append(float(v))
            out["matched_gt_idx"].append(int(r))
            out["matched_pred_idx"].append(int(cc))
        if v >= iou_strict:
            strict += 1
    out["n_matched_loose"] = int(loose)
    out["n_matched_strict"] = int(strict)
    out["iou_max"] = float(iou.max()) if iou.size else 0.0
    out["unmatched_pred_areas"] = [c["pred_areas"][j] for j in range(n_pred) if j not in matched_pred]
    return out


# --------------------------------------------------------------------------------------
# C) 像素级 + 边界带
# --------------------------------------------------------------------------------------
def pixel_metrics(pred, gt):
    """全图像素级计数与六指标（与 metric_tool 同口径）。"""
    pred_b = _as_bool(pred)
    gt_b = _as_bool(gt)
    tp = int(np.count_nonzero(pred_b & gt_b))
    fp = int(np.count_nonzero(pred_b & ~gt_b))
    fn = int(np.count_nonzero(~pred_b & gt_b))
    tn = int(np.count_nonzero(~pred_b & ~gt_b))
    s = _counts_to_scores(tp, fp, fn, tn)
    s.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "gt_positive_pixels": int(np.count_nonzero(gt_b)),
              "pred_positive_pixels": int(np.count_nonzero(pred_b)),
              "n_pixels": int(gt_b.size)})
    return s


def _counts_to_scores(tp, fp, fn, tn):
    eps = float(np.finfo(np.float32).eps)
    total = tp + fn + fp + tn
    if total == 0:
        # 无任何像素（如空 boundary band）：指标无定义 → nan，不臆造 0
        return {"recall": float("nan"), "precision": float("nan"), "OA": float("nan"),
                "F1": float("nan"), "IoU": float("nan"), "Kappa": float("nan")}
    oa = (tp + tn) / (tp + fn + fp + tn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * recall * precision / (recall + precision + eps)
    iou = tp / (tp + fp + fn + eps)
    pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
    # 与 metric_tool 同式（无 eps）：单类退化时 0/0 → nan（numpy 语义）
    kappa = float("nan") if (1.0 - pre) == 0.0 else (oa - pre) / (1 - pre)
    return {"recall": float(recall), "precision": float(precision), "OA": float(oa),
            "F1": float(f1), "IoU": float(iou), "Kappa": float(kappa)}


def boundary_band(gt, r, structure=CONN4):
    """band = dilate(gt,r) \\ erode(gt,r)（border_value=0）。返回 (band_bool, meta)。"""
    gt_b = _as_bool(gt)
    dil = ndimage.binary_dilation(gt_b, structure=structure, iterations=r, border_value=0)
    ero = ndimage.binary_erosion(gt_b, structure=structure, iterations=r, border_value=0)
    band = dil & ~ero
    border_px = int(np.count_nonzero(gt_b[0, :]) + np.count_nonzero(gt_b[-1, :]) +
                    np.count_nonzero(gt_b[:, 0]) + np.count_nonzero(gt_b[:, 1]))
    return band, {"band_pixels": int(np.count_nonzero(band)), "gt_border_pixels": border_px,
                  "radius": int(r)}


def band_metrics(pred, gt, r):
    """band 内像素级 TP/FP/FN 与 F1。"""
    pred_b = _as_bool(pred)
    gt_b = _as_bool(gt)
    band, meta = boundary_band(gt_b, r)
    tp = int(np.count_nonzero(pred_b & gt_b & band))
    fp = int(np.count_nonzero(pred_b & ~gt_b & band))
    fn = int(np.count_nonzero(~pred_b & gt_b & band))
    s = _counts_to_scores(tp, fp, fn, int(band.size - band.sum()))
    s.update({"tp": tp, "fp": fp, "fn": fn, "empty_band": bool(meta["band_pixels"] == 0)})
    s.update(meta)
    return s


# --------------------------------------------------------------------------------------
# 数据集级累加器（流式；与图像顺序、batch size 无关）
# --------------------------------------------------------------------------------------
class ObjectMetricAccumulator:
    """流式累加：像素级 / GT 面积分组 / 对象匹配 / 边界带 / 预测连通域面积分布。

    所有 macro 统计均用 sum+count 累加，因此结果与图像顺序、batch 大小无关。
    """

    def __init__(self, connectivity=4, bands=(2, 4), iou_loose=IOU_LOOSE, iou_strict=IOU_STRICT):
        self.connectivity = connectivity
        self.bands = tuple(bands)
        self.iou_loose = iou_loose
        self.iou_strict = iou_strict
        self.n_images = 0
        self.tp = self.fp = self.fn = self.tn = 0
        self.gt_pos = self.pred_pos = 0
        self.n_pixels = 0
        self.groups = {g: {"n_objects": 0, "n_pixels": 0, "tp_pixels": 0, "macro_recall_sum": 0.0,
                           "hit1": 0, "hit25": 0, "hit50": 0, "n_missed": 0,
                           "n_images_with_group": 0} for g in AREA_GROUPS}
        self.obj = {"n_gt": 0, "n_pred": 0, "n_matched_loose": 0, "n_matched_strict": 0,
                    "fp_pixels": 0, "unmatched_pred_areas": [], "all_pred_areas": [],
                    "gt_areas": []}
        self.band = {r: {"tp": 0, "fp": 0, "fn": 0, "band_pixels": 0, "gt_border_pixels": 0,
                         "n_empty_band": 0} for r in self.bands}
        self.group_band = {g: {"tp_in": 0, "fn_in": 0, "tp_out": 0, "fn_out": 0} for g in AREA_GROUPS}
        self.empty_gt_images = 0
        self.empty_pred_images = 0
        self.empty_both_images = 0
        self.per_image = []            # (name, gt_pos, pred_pos, tp, fp, fn, gt_area_by_group, ...)
        self.object_rows = []          # GT 对象明细（仅当 want_rows=True）
        self._want_rows = False

    # ---------------------------------------------------------------- add one image
    def add(self, pred, gt, name=None, want_rows=False, prob=None):
        pred_b = _as_bool(pred)
        gt_b = _as_bool(gt)
        assert pred_b.shape == gt_b.shape, f"shape mismatch {pred_b.shape} vs {gt_b.shape}"
        self.n_images += 1
        px = pixel_metrics(pred_b, gt_b)
        self.tp += px["tp"]; self.fp += px["fp"]; self.fn += px["fn"]; self.tn += px["tn"]
        self.gt_pos += px["gt_positive_pixels"]; self.pred_pos += px["pred_positive_pixels"]
        self.n_pixels += px["n_pixels"]
        if px["gt_positive_pixels"] == 0:
            self.empty_gt_images += 1
        if px["pred_positive_pixels"] == 0:
            self.empty_pred_images += 1
        if px["gt_positive_pixels"] == 0 and px["pred_positive_pixels"] == 0:
            self.empty_both_images += 1

        cache = object_iou_matrix(pred_b, gt_b, self.connectivity)
        rows, agg = gt_object_stats(pred_b, gt_b, self.connectivity, image_name=name)
        if want_rows or self._want_rows:
            self._enrich_rows(rows, cache, prob, px)
            self.object_rows.extend(rows)
        for g in AREA_GROUPS:
            a = agg[g]
            self.groups[g]["n_objects"] += a["n_objects"]
            self.groups[g]["n_pixels"] += a["n_pixels"]
            self.groups[g]["tp_pixels"] += a["tp_pixels"]
            self.groups[g]["macro_recall_sum"] += a["macro_recall_sum"]
            self.groups[g]["hit1"] += a["hit1"]
            self.groups[g]["hit25"] += a["hit25"]
            self.groups[g]["hit50"] += a["hit50"]
            self.groups[g]["n_missed"] += a["n_missed"]
            if a["n_objects"] > 0:
                self.groups[g]["n_images_with_group"] += 1

        m = match_objects(pred_b, gt_b, self.connectivity, self.iou_loose, self.iou_strict,
                          cache=cache)
        self.obj["n_gt"] += m["n_gt"]
        self.obj["n_pred"] += m["n_pred"]
        self.obj["n_matched_loose"] += m["n_matched_loose"]
        self.obj["n_matched_strict"] += m["n_matched_strict"]
        self.obj["fp_pixels"] += m["fp_pixels"]
        self.obj["unmatched_pred_areas"].extend(m["unmatched_pred_areas"])
        self.obj["all_pred_areas"].extend(m["pred_areas"])
        self.obj["gt_areas"].extend(m["gt_areas"])

        for r in self.bands:
            band, meta = boundary_band(gt_b, r)
            btp = int(np.count_nonzero(pred_b & gt_b & band))
            bfp = int(np.count_nonzero(pred_b & ~gt_b & band))
            bfn = int(np.count_nonzero(~pred_b & gt_b & band))
            acc = self.band[r]
            acc["tp"] += btp; acc["fp"] += bfp; acc["fn"] += bfn
            acc["band_pixels"] += meta["band_pixels"]
            acc["gt_border_pixels"] += meta["gt_border_pixels"]
            if meta["band_pixels"] == 0:
                acc["n_empty_band"] += 1
            # 各面积组在 band 内/外的 TP/FN（复用同一次连通域标记）
            if cache["n_gt"]:
                gcounts = np.bincount(cache["gt_lab"].ravel(), minlength=cache["n_gt"] + 1)
                for j in range(1, cache["n_gt"] + 1):
                    g = bin_index(int(gcounts[j]))
                    obj_mask = cache["gt_lab"] == j
                    self.group_band[g]["tp_in"] += int(np.count_nonzero(pred_b & obj_mask & band))
                    self.group_band[g]["fn_in"] += int(np.count_nonzero(~pred_b & obj_mask & band))
                    self.group_band[g]["tp_out"] += int(np.count_nonzero(pred_b & obj_mask & ~band))
                    self.group_band[g]["fn_out"] += int(np.count_nonzero(~pred_b & obj_mask & ~band))

        self.per_image.append({
            "sample_name": name, "gt_positive_pixels": px["gt_positive_pixels"],
            "pred_positive_pixels": px["pred_positive_pixels"], "tp": px["tp"], "fp": px["fp"],
            "fn": px["fn"], "tn": px["tn"], "F1": px["F1"], "IoU": px["IoU"],
            "recall": px["recall"], "precision": px["precision"],
            "gt_area_by_group": {g: agg[g]["n_pixels"] for g in AREA_GROUPS},
            "n_gt_objects_by_group": {g: agg[g]["n_objects"] for g in AREA_GROUPS},
            "n_pred_objects": m["n_pred"], "n_matched_loose": m["n_matched_loose"],
            "n_matched_strict": m["n_matched_strict"],
        })
        return px

    def _group_band_counts(self, pred_b, gt_b, r, group):
        """（保留接口，内部已由 add() 内联优化替代。）"""
        lab, n = label_components(gt_b, self.connectivity)
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        if n == 0:
            return {"tp_in": 0, "fn_in": 0, "tp_out": 0, "fn_out": 0}
        band, _ = boundary_band(gt_b, r)
        sel = np.zeros(n + 1, dtype=bool)
        for j in range(1, n + 1):
            if bin_index(int(counts[j])) == group:
                sel[j] = True
        if not sel.any():
            return {"tp_in": 0, "fn_in": 0, "tp_out": 0, "fn_out": 0}
        gm = sel[lab]
        tp_map = pred_b & gt_b & gm
        fn_map = (~pred_b) & gt_b & gm
        return {"tp_in": int(np.count_nonzero(tp_map & band)),
                "fn_in": int(np.count_nonzero(fn_map & band)),
                "tp_out": int(np.count_nonzero(tp_map & ~band)),
                "fn_out": int(np.count_nonzero(fn_map & ~band))}

    def _enrich_rows(self, rows, cache, prob, px):
        """为 GT 对象明细补充：图像级上下文、模型概率统计、per-object 最大 IoU 与匹配标志。"""
        iou = cache["iou"]
        gt_lab = cache["gt_lab"]
        matched_gt = set()
        if iou.size:
            rows_l, cols_l = linear_sum_assignment(-iou)
            for r, c in zip(rows_l, cols_l):
                if iou[r, c] >= self.iou_loose:
                    matched_gt.add(int(r) + 1)
        ratio = float(px["gt_positive_pixels"] / px["n_pixels"])
        for r in rows:
            j = r["gt_component_id"]
            r["image_gt_area_ratio"] = ratio
            r["image_fp_pixels"] = px["fp"]
            r["image_fn_pixels"] = px["fn"]
            r["connectivity"] = self.connectivity
            r["matched_iou_010"] = int(j in matched_gt)
            r["matched_iou_050"] = int(iou[j - 1].max() >= self.iou_strict) if iou.size else 0
            r["iou_max_any_pred"] = float(iou[j - 1].max()) if iou.size else 0.0
            if prob is not None:
                m = (gt_lab == j)
                vals = prob[m]
                if vals.size:
                    r["model_p_mean_in_gt"] = float(vals.mean())
                    r["model_p_max_in_gt"] = float(vals.max())
                    r["model_p_p90_in_gt"] = float(np.quantile(vals, 0.90))
                else:
                    r["model_p_mean_in_gt"] = r["model_p_max_in_gt"] = r["model_p_p90_in_gt"] = None
            # 形状分层标签（预注册）
            w, h = r["bbox_width"], r["bbox_height"]
            r["bbox_min_width"] = int(min(w, h))
            r["width_stratum"] = "<4" if min(w, h) < 4 else ("4-7" if min(w, h) < 8 else ">=8")
            r["elongated"] = int(r["bbox_aspect_ratio"] >= 3.0)
            r["density_stratum"] = ("0-1%" if ratio <= 0.01 else
                                    ("1-5%" if ratio <= 0.05 else ">5%"))

    # ---------------------------------------------------------------- summary
    def summary(self, label=""):
        px = _counts_to_scores(self.tp, self.fp, self.fn, self.tn)
        px.update({"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                   "gt_positive_pixels": self.gt_pos, "pred_positive_pixels": self.pred_pos,
                   "n_pixels": self.n_pixels, "positive_prior": self.gt_pos / max(1, self.n_pixels)})
        groups = {}
        for g in AREA_GROUPS:
            a = self.groups[g]
            n = a["n_objects"]
            groups[g] = {
                "n_objects": n, "n_pixels": a["n_pixels"], "tp_pixels": a["tp_pixels"],
                "pixel_recall_micro": (a["tp_pixels"] / a["n_pixels"]) if n else None,
                "object_macro_pixel_recall": (a["macro_recall_sum"] / n) if n else None,
                "hit1": (a["hit1"] / n) if n else None,
                "hit25": (a["hit25"] / n) if n else None,
                "hit50": (a["hit50"] / n) if n else None,
                "n_missed": a["n_missed"], "n_images_with_group": a["n_images_with_group"],
            }
        # 对齐旧分组
        legacy = {}
        for lg, members in GROUP_ALIAS.items():
            nobj = sum(groups[m]["n_objects"] for m in members)
            npix = sum(groups[m]["n_pixels"] for m in members)
            tpp = sum(groups[m]["tp_pixels"] for m in members)
            macro_sum = sum(self.groups[m]["macro_recall_sum"] for m in members)
            hit1 = sum(self.groups[m]["hit1"] for m in members)
            hit25 = sum(self.groups[m]["hit25"] for m in members)
            hit50 = sum(self.groups[m]["hit50"] for m in members)
            legacy[lg] = {
                "n_objects": nobj, "n_pixels": npix, "tp_pixels": tpp,
                "pixel_recall_micro": (tpp / npix) if nobj else None,
                "object_macro_pixel_recall": (macro_sum / nobj) if nobj else None,
                "hit1": (hit1 / nobj) if nobj else None,
                "hit25": (hit25 / nobj) if nobj else None,
                "hit50": (hit50 / nobj) if nobj else None,
            }
        obj = dict(self.obj)
        n_gt, n_pred = obj["n_gt"], obj["n_pred"]
        matched_l, matched_s = obj["n_matched_loose"], obj["n_matched_strict"]
        op_l = matched_l / n_pred if n_pred else None
        or_l = matched_l / n_gt if n_gt else None
        op_s = matched_s / n_pred if n_pred else None
        or_s = matched_s / n_gt if n_gt else None
        object_metrics = {
            "n_gt": n_gt, "n_pred": n_pred,
            "n_matched_loose": matched_l, "n_matched_strict": matched_s,
            "unmatched_pred_loose": n_pred - matched_l,
            "ObjPrecision_loose": op_l, "ObjRecall_loose": or_l,
            "ObjF1_loose": (2 * op_l * or_l / (op_l + or_l)) if (op_l and or_l) else None,
            "ObjPrecision_strict": op_s, "ObjRecall_strict": or_s,
            "ObjF1_strict": (2 * op_s * or_s / (op_s + or_s)) if (op_s and or_s) else None,
            "fp_pixels": obj["fp_pixels"],
            "pred_cc_count_by_area": _area_hist(obj["all_pred_areas"]),
            "unmatched_pred_cc_count_by_area": _area_hist(obj["unmatched_pred_areas"]),
            "gt_cc_count_by_area": _area_hist(obj["gt_areas"]),
        }
        bands = {}
        for r, a in self.band.items():
            s = _counts_to_scores(a["tp"], a["fp"], a["fn"],
                                  a["band_pixels"] - a["tp"] - a["fp"] - a["fn"])
            s.update({"tp": a["tp"], "fp": a["fp"], "fn": a["fn"],
                      "band_pixels": a["band_pixels"],
                      "gt_border_pixels": a["gt_border_pixels"],
                      "n_empty_band": a["n_empty_band"]})
            s["group_tp_fn"] = {g: dict(self.group_band[g]) for g in AREA_GROUPS}
            bands[f"band{r}"] = s
        return {
            "label": label, "connectivity": self.connectivity,
            "n_images": self.n_images,
            "empty_gt_images": self.empty_gt_images,
            "empty_pred_images": self.empty_pred_images,
            "empty_both_images": self.empty_both_images,
            "pixel": px,
            "groups": groups,
            "legacy_groups": legacy,
            "object": object_metrics,
            "bands": bands,
        }

    # ---------------------------------------------------------------- per-image arrays (for bootstrap)
    def per_image_metric(self, key):
        return np.array([img[key] if img[key] is not None else np.nan for img in self.per_image],
                        dtype=np.float64)


def _area_hist(areas):
    h = {g: 0 for g in AREA_GROUPS}
    for a in areas:
        h[bin_index(int(a))] += 1
    return h


def aggregate_summaries(items, label=""):
    """对多个 accumulator.summary() 做像素级/分组计数合并（用于跨变体一致性校验）。"""
    raise NotImplementedError("use ObjectMetricAccumulator directly for aggregation")
