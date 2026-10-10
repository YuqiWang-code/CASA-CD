"""P0 单元测试：analyse/tvim_object_metrics.py（对应 Run-Diag 文档 §3.3 的 10 项）。

全部使用**人造小阵列 + 手算预期**，不依赖真实数据。
torch 相关用例（第 9 项）在 torch 不可用时自动 skip（服务器上会真实执行）。

运行：python -m unittest discover -s analyse/tests -p 'test_tvim_*.py' -v
"""
import os
import sys
import unittest

import numpy as np

_ANALYSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ANALYSE_DIR not in sys.path:
    sys.path.insert(0, _ANALYSE_DIR)
_MODELS_DIR = os.path.join(os.path.dirname(_ANALYSE_DIR), "models")
if os.path.isdir(_MODELS_DIR) and _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from tvim_object_metrics import (  # noqa: E402
    AREA_GROUPS, ObjectMetricAccumulator, band_metrics, bin_index, boundary_band,
    gt_object_stats, label_components, match_objects, pixel_metrics,
)

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def canvas(h=32, w=32):
    return np.zeros((h, w), dtype=bool)


def rect(a, y, x, h, w):
    a[y:y + h, x:x + w] = True
    return a


class TestObjectMetrics(unittest.TestCase):

    # ---------------------------------------------------------------- 1
    def test_01_area4_full_hit(self):
        gt = rect(canvas(), 10, 10, 2, 2)          # area 4
        pred = gt.copy()
        acc = ObjectMetricAccumulator()
        acc.add(pred, gt, name="img1")
        s = acc.summary()
        g = s["groups"]["tiny_1_15"]
        self.assertEqual(g["n_objects"], 1)
        self.assertEqual(g["n_pixels"], 4)
        self.assertEqual(g["pixel_recall_micro"], 1.0)
        self.assertEqual(g["object_macro_pixel_recall"], 1.0)
        self.assertEqual(g["hit25"], 1.0)
        self.assertEqual(g["hit1"], 1.0)
        self.assertEqual(s["object"]["n_matched_loose"], 1)
        self.assertEqual(s["object"]["n_matched_strict"], 1)   # IoU=1 >= 0.5
        self.assertAlmostEqual(s["object"]["ObjF1_strict"], 1.0, delta=1e-6)
        # metric_tool 口径含 float32 eps，命中全对时 precision/recall ≈ 1（非精确 1）
        self.assertAlmostEqual(s["pixel"]["precision"], 1.0, delta=1e-6)
        self.assertAlmostEqual(s["pixel"]["recall"], 1.0, delta=1e-6)

    # ---------------------------------------------------------------- 2
    def test_02_area4_total_miss(self):
        gt = rect(canvas(), 10, 10, 2, 2)
        pred = canvas()
        acc = ObjectMetricAccumulator()
        acc.add(pred, gt, name="img1")
        s = acc.summary()
        g = s["groups"]["tiny_1_15"]
        self.assertEqual(g["n_objects"], 1)
        self.assertEqual(g["pixel_recall_micro"], 0.0)
        self.assertEqual(g["object_macro_pixel_recall"], 0.0)
        self.assertEqual(g["hit1"], 0.0)
        self.assertEqual(g["hit25"], 0.0)
        self.assertEqual(g["n_missed"], 1)
        # 无预测 → 对象 Precision 分母为 0 → None（不是 0）
        self.assertEqual(s["object"]["n_pred"], 0)
        self.assertIsNone(s["object"]["ObjPrecision_loose"])
        self.assertIsNone(s["object"]["ObjPrecision_strict"])
        self.assertEqual(s["object"]["ObjRecall_loose"], 0.0)

    # ---------------------------------------------------------------- 3
    def test_03_fp_reduces_precision_regression_old_bug(self):
        """旧 component_pr 的 FP 恒为 0：本用例是其直接回归测试。"""
        gt = rect(canvas(64, 64), 10, 10, 2, 2)
        pred_hit = gt.copy()
        pred_hit_fp = gt.copy()
        for (y, x) in ((40, 40), (40, 50), (50, 40)):
            rect(pred_hit_fp, y, x, 2, 2)

        a = ObjectMetricAccumulator().add(pred_hit, gt)
        acc1 = ObjectMetricAccumulator(); acc1.add(pred_hit, gt, name="a")
        acc2 = ObjectMetricAccumulator(); acc2.add(pred_hit_fp, gt, name="b")
        s1, s2 = acc1.summary(), acc2.summary()

        # GT 组 Recall 不变
        self.assertEqual(s1["groups"]["tiny_1_15"]["pixel_recall_micro"],
                         s2["groups"]["tiny_1_15"]["pixel_recall_micro"])
        self.assertEqual(s1["groups"]["tiny_1_15"]["hit25"], s2["groups"]["tiny_1_15"]["hit25"])
        # 全图像素 Precision 必须下降
        self.assertLess(s2["pixel"]["precision"], s1["pixel"]["precision"])
        self.assertGreater(s2["pixel"]["fp"], s1["pixel"]["fp"])
        # 对象级 Precision 必须下降
        self.assertLess(s2["object"]["ObjPrecision_loose"], s1["object"]["ObjPrecision_loose"])
        self.assertGreater(s2["object"]["ObjPrecision_loose"], 0.0)   # 仍检出 1/4

    # ---------------------------------------------------------------- 4
    def test_04_two_gt_one_pred_cc(self):
        """一个预测连通域跨接两个 GT 小对象：一对一匹配至多命中一个。"""
        gt = canvas(64, 64)
        rect(gt, 10, 10, 4, 4)      # GT1 area16
        rect(gt, 20, 10, 4, 4)      # GT2 area16
        pred = canvas(64, 64)
        pred[10:24, 10:14] = True   # 单个连通域同时覆盖两者（含中间 6 行空隙）
        m = match_objects(pred, gt)
        self.assertEqual(m["n_gt"], 2)
        self.assertEqual(m["n_pred"], 1)
        self.assertEqual(m["n_matched_loose"], 1)
        acc = ObjectMetricAccumulator(); acc.add(pred, gt, name="x")
        s = acc.summary()
        self.assertEqual(s["object"]["n_matched_loose"], 1)
        self.assertAlmostEqual(s["object"]["ObjRecall_loose"], 0.5)

    # ---------------------------------------------------------------- 5
    def test_05_large_gt_one_pixel_hit(self):
        gt = rect(canvas(64, 64), 8, 8, 40, 40)     # area 1600 >= 1024 → large
        pred = canvas(64, 64); pred[20, 20] = True
        acc = ObjectMetricAccumulator(); acc.add(pred, gt, name="x")
        g = acc.summary()["groups"]["large_1024_inf"]
        self.assertEqual(g["n_objects"], 1)
        self.assertEqual(g["hit1"], 1.0)
        self.assertEqual(g["hit25"], 0.0)           # 证明 Hit@1 与 Hit@25 语义不同
        self.assertAlmostEqual(g["pixel_recall_micro"], 1 / 1600)

    # ---------------------------------------------------------------- 6
    def test_06_area_bins_and_empty_spec(self):
        self.assertEqual(bin_index(15), "tiny_1_15")
        self.assertEqual(bin_index(16), "tiny_16_63")
        self.assertEqual(bin_index(63), "tiny_16_63")
        self.assertEqual(bin_index(64), "small_64_255")
        self.assertEqual(bin_index(255), "small_64_255")
        self.assertEqual(bin_index(256), "medium_256_1023")
        self.assertEqual(bin_index(1023), "medium_256_1023")
        self.assertEqual(bin_index(1024), "large_1024_inf")
        # 端到端：面积 256 的块 → medium
        gt = rect(canvas(64, 64), 4, 4, 16, 16)
        acc = ObjectMetricAccumulator(); acc.add(gt.copy(), gt, name="m")
        s = acc.summary()
        self.assertEqual(s["groups"]["medium_256_1023"]["n_objects"], 1)
        self.assertEqual(s["groups"]["medium_256_1023"]["pixel_recall_micro"], 1.0)
        # 空组为 None，不参与 macro
        for g in ("tiny_1_15", "tiny_16_63", "small_64_255", "large_1024_inf"):
            self.assertIsNone(s["groups"][g]["pixel_recall_micro"])
            self.assertIsNone(s["groups"][g]["object_macro_pixel_recall"])
            self.assertEqual(s["groups"][g]["n_objects"], 0)
        # 全空图（GT 与 Pred 均空）
        acc2 = ObjectMetricAccumulator()
        empty = canvas(16, 16)
        acc2.add(empty.copy(), empty.copy(), name="e1")
        acc2.add(empty.copy(), empty.copy(), name="e2")
        s2 = acc2.summary()
        self.assertEqual(s2["n_images"], 2)
        self.assertEqual(s2["empty_both_images"], 2)
        self.assertEqual(s2["object"]["n_gt"], 0)
        self.assertEqual(s2["object"]["n_pred"], 0)
        self.assertIsNone(s2["object"]["ObjPrecision_loose"])
        self.assertIsNone(s2["object"]["ObjRecall_loose"])
        self.assertIsNone(s2["object"]["ObjF1_loose"])
        for g in AREA_GROUPS:
            self.assertIsNone(s2["groups"][g]["pixel_recall_micro"])
        self.assertAlmostEqual(s2["pixel"]["OA"], 1.0, delta=1e-6)   # metric_tool 含 eps 口径
        with np.errstate(invalid="ignore", divide="ignore"):
            self.assertTrue(np.isnan(s2["pixel"]["Kappa"]))   # 与 metric_tool 同式：0/0
        # 有 Pred 无 GT → 全部 FP，且不计入 N_gt
        acc3 = ObjectMetricAccumulator()
        p = canvas(16, 16); rect(p, 2, 2, 3, 3)
        acc3.add(p, empty.copy(), name="fp_only")
        s3 = acc3.summary()
        self.assertEqual(s3["object"]["n_gt"], 0)
        self.assertEqual(s3["object"]["n_pred"], 1)
        self.assertEqual(s3["object"]["n_matched_loose"], 0)
        self.assertEqual(s3["object"]["unmatched_pred_loose"], 1)
        self.assertEqual(s3["pixel"]["fp"], 9)

    # ---------------------------------------------------------------- 7
    def test_07_connectivity_diagonal(self):
        m = canvas(8, 8)
        m[2, 2] = True
        m[3, 3] = True
        _, n4 = label_components(m, 4)
        _, n8 = label_components(m, 8)
        self.assertEqual(n4, 2)
        self.assertEqual(n8, 1)
        # 累加器主口径 = 4 连通
        acc4 = ObjectMetricAccumulator(connectivity=4); acc4.add(m.copy(), m.copy(), name="d")
        acc8 = ObjectMetricAccumulator(connectivity=8); acc8.add(m.copy(), m.copy(), name="d")
        self.assertEqual(acc4.summary()["object"]["n_gt"], 2)
        self.assertEqual(acc8.summary()["object"]["n_gt"], 1)

    # ---------------------------------------------------------------- 8
    def test_08_background_fp_not_filtered_by_gt_mask(self):
        gt = rect(canvas(32, 32), 2, 2, 2, 2)
        pred = gt.copy()
        acc = ObjectMetricAccumulator(); acc.add(pred, gt, name="x")
        fp0 = acc.summary()["pixel"]["fp"]
        acc2 = ObjectMetricAccumulator(); acc2.add(pred.copy(), gt, name="x")
        # 在远离 GT 处再加 5 个背景假阳性
        for i in range(5):
            acc2.add(_with_extra_fp(pred, 20 + i, 20), gt, name=f"x{i+1}")
        s2 = acc2.summary()
        self.assertGreater(s2["pixel"]["fp"], fp0)
        # GT 分组统计不受背景 FP 影响（gt 组 pixel recall 保持 1.0）
        self.assertEqual(s2["groups"]["tiny_1_15"]["pixel_recall_micro"], 1.0)
        # band 内 FP 也独立计数
        self.assertGreaterEqual(s2["bands"]["band2"]["fp"], 0)

    # ---------------------------------------------------------------- 9
    @unittest.skipUnless(HAS_TORCH, "torch unavailable (local Windows env)")
    def test_09_torch_numpy_cpu_gpu_consistency(self):
        import torch
        rng = np.random.default_rng(0)
        pred = rng.random((2, 1, 64, 64)) > 0.8
        gt = rng.random((2, 1, 64, 64)) > 0.85
        base = ObjectMetricAccumulator()
        for i in range(pred.shape[0]):
            base.add(pred[i, 0], gt[i, 0], name=f"i{i}")
        ref = base.summary()
        variants = {
            "torch_bool_cpu": (torch.as_tensor(pred).bool(), torch.as_tensor(gt).bool()),
            "numpy_uint8": (pred.astype(np.uint8), gt.astype(np.uint8)),
        }
        if torch.cuda.is_available():
            variants["torch_bool_cuda"] = (torch.as_tensor(pred).cuda().bool(),
                                           torch.as_tensor(gt).cuda().bool())
        for name, (p, g) in variants.items():
            acc = ObjectMetricAccumulator()
            for i in range(pred.shape[0]):
                pi = p[i, 0].cpu().numpy() if hasattr(p[i, 0], "cpu") else p[i, 0]
                gi = g[i, 0].cpu().numpy() if hasattr(g[i, 0], "cpu") else g[i, 0]
                acc.add(pi, gi, name=f"i{i}")
            s = acc.summary()
            self.assertEqual(s["pixel"]["tp"], ref["pixel"]["tp"], name)
            self.assertEqual(s["pixel"]["fp"], ref["pixel"]["fp"], name)
            self.assertEqual(s["object"], ref["object"], name)
            self.assertEqual(s["groups"], ref["groups"], name)

    # ---------------------------------------------------------------- 10
    def test_10_batch_and_order_invariance(self):
        rng = np.random.default_rng(16)
        imgs = []
        for idx in range(16):
            gt = rng.random((48, 48)) > 0.93
            pred = gt.copy()
            if idx % 3 == 0:
                pred[:8, :8] = True                    # 假阳性
            if idx % 4 == 0:
                pred = rng.random((48, 48)) > 0.9      # 噪声预测
            imgs.append((f"img{idx:02d}.png", pred, gt))

        def run(order, batch_size):
            acc = ObjectMetricAccumulator()
            for start in range(0, len(order), batch_size):
                for name, pred, gt in order[start:start + batch_size]:
                    acc.add(pred, gt, name=name)
            return acc.summary()

        ref = run(imgs, 1)
        for bs in (4, 16):
            _close(run(imgs, bs)["pixel"], ref["pixel"])
            _close(run(imgs, bs)["groups"], ref["groups"])
            _close(run(imgs, bs)["object"], ref["object"])
        shuffled = imgs[::-1]
        s_shuf = run(shuffled, 7)
        _close(s_shuf["pixel"], ref["pixel"])
        _close(s_shuf["groups"], ref["groups"])
        _close(s_shuf["object"], ref["object"])
        # 计数必须严格相等（整数量）
        self.assertEqual(s_shuf["pixel"]["tp"], ref["pixel"]["tp"])
        self.assertEqual(s_shuf["pixel"]["fp"], ref["pixel"]["fp"])
        self.assertEqual(s_shuf["object"], ref["object"])
        self.assertEqual(s_shuf["n_images"], len(imgs))


def _with_extra_fp(mask, y, x):
    m = mask.copy()
    m[y:y + 2, x:x + 2] = True
    return m


def _close(a, b, tol=1e-9, path=""):
    """递归比较嵌套 dict/list/数值（浮点容差，nan==nan）。"""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"keys differ at {path}: {set(a) ^ set(b)}"
        for k in a:
            _close(a[k], b[k], tol, f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"len differ at {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, tol, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            assert a is b, f"None mismatch at {path}: {a} vs {b}"
            return
        if np.isnan(a) and np.isnan(b):
            return
        assert abs(a - b) <= tol * max(1.0, abs(a), abs(b)), f"value differ at {path}: {a} vs {b}"
    else:
        assert a == b, f"value differ at {path}: {a} vs {b}"


class TestBoundaryBand(unittest.TestCase):
    def test_band_definition_and_empty(self):
        # 大目标：腐蚀 2 次后仍有内部，内部像素不应在 band 内
        gt = rect(canvas(48, 48), 10, 10, 16, 16)
        band, meta = boundary_band(gt, 2)
        self.assertEqual(meta["band_pixels"], int(band.sum()))
        self.assertEqual(meta["gt_border_pixels"], 0)
        self.assertTrue(band[10, 8])       # x=8 距 GT(x=10..25) 2 px
        self.assertFalse(band[10, 7])      # 3 px
        self.assertFalse(band[17, 17])     # 内部深处不在带内（erosions=2 后剩余内部）
        # 小目标（4×4）：4 邻接腐蚀 2 次后为空 → 整块都在 band 内（形态学定义的自然结果）
        small = rect(canvas(32, 32), 10, 10, 4, 4)
        band_s, _ = boundary_band(small, 2)
        self.assertTrue(band_s[10, 10])
        empty = canvas(32, 32)
        b2, m2 = boundary_band(empty, 2)
        self.assertEqual(m2["band_pixels"], 0)
        out = band_metrics(empty, empty, 2)
        self.assertTrue(out["empty_band"])

    def test_border_truncation_recorded(self):
        gt = rect(canvas(32, 32), 0, 0, 4, 4)     # 贴角
        band, meta = boundary_band(gt, 4)
        self.assertGreater(meta["gt_border_pixels"], 0)
        self.assertGreater(meta["band_pixels"], 0)


class TestPixelMetricsMatchMetricTool(unittest.TestCase):
    def test_counts_to_scores_matches_metric_tool(self):
        from model.metric_tool import cm2score
        rng = np.random.default_rng(3)
        pred = (rng.random((2, 8, 8)) > 0.5).astype(np.int64)
        gt = (rng.random((2, 8, 8)) > 0.6).astype(np.int64)
        # 用 metric_tool 的官方实现算一次
        from model.metric_tool import get_confuse_matrix
        cm = get_confuse_matrix(2, list(gt), list(pred))
        ref = cm2score(cm)
        # 用本模块的逐图像素统计算一次
        tp = int(np.count_nonzero((pred == 1) & (gt == 1)))
        fp = int(np.count_nonzero((pred == 1) & (gt == 0)))
        fn = int(np.count_nonzero((pred == 0) & (gt == 1)))
        tn = int(np.count_nonzero((pred == 0) & (gt == 0)))
        mine = _scores(tp, fp, fn, tn)
        for k_src, k_dst, tol in (("recall", "recall", 1e-12), ("precision", "precision", 1e-12),
                                  ("OA", "OA", 1e-12), ("F1", "F1", 1e-12), ("IoU", "IoU", 1e-12),
                                  ("Kappa", "Kappa", 1e-12)):
            self.assertAlmostEqual(ref[k_src], mine[k_dst], delta=tol)


def _scores(tp, fp, fn, tn):
    eps = float(np.finfo(np.float32).eps)
    oa = (tp + tn) / (tp + fn + fp + tn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * recall * precision / (recall + precision + eps)
    iou = tp / (tp + fp + fn + eps)
    pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
    return {"recall": recall, "precision": precision, "OA": oa, "F1": f1, "IoU": iou,
            "Kappa": (oa - pre) / (1 - pre)}


if __name__ == "__main__":
    unittest.main(verbosity=2)
