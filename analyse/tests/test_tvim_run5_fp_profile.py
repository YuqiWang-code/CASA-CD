"""Run5 §10 FP 空间画像的纯逻辑单测（本地可跑，不需要 GPU / 数据集 / checkpoint）。"""
import os
import sys
import unittest

import numpy as np

_ANALYSE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ANALYSE not in sys.path:
    sys.path.insert(0, _ANALYSE)

import tvim_run5_fp_profile as P  # noqa: E402


class TestEdgeField(unittest.TestCase):
    def test_edge_sparse_field_degeneracy_guard(self):
        # 只有两列有梯度 ⇒ 90 分位为 0；必须退化为 mag>0，而不是把整幅图判成边缘
        img = np.zeros((3, 32, 32))
        img[:, :, 16:] = 5.0
        _mag, edge = P.edge_field(img)
        self.assertFalse(bool(edge[:, :14].any()))
        self.assertTrue(bool(edge[:, 15:17].any()))
        self.assertLess(float(edge.mean()), 0.2)

    def test_edge_on_step_boundary(self):
        img = np.zeros((3, 32, 32))
        img[:, :, 16:] = 5.0                      # 竖直阶跃
        mag, edge = P.edge_field(img)
        self.assertEqual(edge.shape, (32, 32))
        # 阶跃处应有边缘，远处平坦区不应有
        self.assertTrue(bool(edge[:, 15:17].any()))
        self.assertFalse(bool(edge[:, :8].any()))
        self.assertGreater(float(mag[:, 15:17].mean()), float(mag[:, :8].mean()))

    def test_edge_is_deterministic_and_scale_equivariant(self):
        rng = np.random.default_rng(0)
        img = rng.normal(size=(3, 48, 48))
        _m1, e1 = P.edge_field(img)
        _m2, e2 = P.edge_field(img.copy())
        np.testing.assert_array_equal(e1, e2)
        _m3, e3 = P.edge_field(img * 4.0)          # 正比缩放不改变 90 分位阈值下的边缘集
        np.testing.assert_array_equal(e1, e3)

    def test_uniform_image_yields_no_edges(self):
        # 常量场：梯度恒 0 ⇒ 退化保护给出空边缘集（而不是“全图都是边缘”）
        img = np.full((3, 16, 16), 2.0)
        _mag, edge = P.edge_field(img)
        self.assertFalse(bool(edge.any()))


class TestDistToGt(unittest.TestCase):
    def test_distance_zero_on_gt(self):
        gt = np.zeros((32, 32), bool)
        gt[20:22, 20:22] = True
        d = P.dist_to_gt(gt)
        self.assertEqual(float(d[20, 20]), 0.0)
        self.assertAlmostEqual(float(d[20, 18]), 2.0, places=6)
        self.assertAlmostEqual(float(d[20, 10]), 10.0, places=6)

    def test_no_gt_is_infinite(self):
        d = P.dist_to_gt(np.zeros((8, 8), bool))
        self.assertTrue(np.all(np.isinf(d)))


class TestBestShiftIou(unittest.TestCase):
    def _scene(self, dy=0, dx=0):
        gt_lab = np.zeros((64, 64), np.int32)
        gt_lab[20 + dy:24 + dy, 20 + dx:24 + dx] = 1
        gt_areas = [16]
        return gt_lab, gt_areas

    def test_zero_shift_when_aligned(self):
        gt_lab, areas = self._scene()
        comp = np.zeros((64, 64), bool)
        comp[20:24, 20:24] = True
        sl = (slice(20, 24), slice(20, 24))
        iou, sh, k = P.best_shift_iou(comp, gt_lab, 1, areas, sl)
        self.assertAlmostEqual(iou, 1.0, places=6)
        self.assertEqual(sh, (0, 0))
        self.assertEqual(k, 1)

    def test_recovers_two_pixel_shift(self):
        gt_lab, areas = self._scene(dy=0, dx=2)
        comp = np.zeros((64, 64), bool)
        comp[20:24, 20:24] = True
        sl = (slice(20, 24), slice(20, 24))
        iou, sh, k = P.best_shift_iou(comp, gt_lab, 1, areas, sl)
        self.assertAlmostEqual(iou, 1.0, places=6)
        self.assertEqual(sh, (0, -2))     # 把 GT 平移 (0,-2) 与 comp 对齐
        self.assertTrue(P.MAX_SHIFT >= 2)

    def test_no_gt_returns_zero(self):
        comp = np.zeros((16, 16), bool)
        comp[4:8, 4:8] = True
        iou, sh, k = P.best_shift_iou(comp, np.zeros((16, 16), np.int32), 0, [],
                                      (slice(4, 8), slice(4, 8)))
        self.assertEqual(iou, 0.0)
        self.assertEqual(sh, (0, 0))
        self.assertIsNone(k)


class TestObjectMetrics(unittest.TestCase):
    def _scene(self):
        gt = np.zeros((64, 64), bool)
        gt[30:34, 30:34] = True
        gt_lab, n_gt = __import__("scipy.ndimage", fromlist=["ndimage"]).label(
            gt, structure=P.CONN4)
        from scipy import ndimage as ndi
        gt_objs = ndi.find_objects(gt_lab)
        gt_areas = [int((gt_lab == 1).sum())]
        magA = np.zeros((64, 64)); magA[30:34, 30:34] = 3.0
        magB = np.zeros((64, 64))
        edge = np.zeros((64, 64), bool); edge[30:34, 30:34] = True
        edge_d = P.dilate1(edge)
        dgt = P.dist_to_gt(gt)
        chg = np.zeros((64, 64)); chg[30:34, 30:34] = 0.5
        return gt, gt_lab, n_gt, gt_areas, gt_objs, magA, magB, edge_d, dgt, chg

    def test_fp_component_next_to_gt(self):
        gt, gl, n, areas, objs, magA, magB, ed, dgt, chg = self._scene()
        comp = np.zeros((64, 64), bool)
        comp[30:34, 35:39] = True          # 紧邻 GT（间隔 1 列）
        from scipy import ndimage as ndi
        sl = ndi.find_objects(comp.astype(np.int64))[0]
        m = P.object_metrics(comp, gt, gl, n, areas, objs, sl, magA, magB, ed, ed,
                             dgt, chg, want_shift=False)
        self.assertEqual(m["area"], 16)
        self.assertAlmostEqual(m["dist_to_gt_min_px"], 2.0, places=6)
        # comp 列 35..38 到 GT（列 ..33）的距离为 2,3,4,5 → 只有 3/4 的像素在 4px 内
        self.assertAlmostEqual(m["frac_px_within_4px_of_gt"], 0.75, places=6)
        self.assertEqual(m["frac_px_within_1px_of_gt"], 0.0)
        self.assertEqual(m["frac_px_on_any_edge_1px"], 0.0)
        self.assertEqual(m["gradA_mean"], 0.0)
        self.assertEqual(m["change_abs_mean"], 0.0)
        self.assertIsNone(m["best_shift_iou"])

    def test_fp_component_touching_gt_band(self):
        gt, gl, n, areas, objs, magA, magB, ed, dgt, chg = self._scene()
        comp = np.zeros((64, 64), bool)
        comp[30:34, 33:37] = True          # 与 GT 对角/贴边相邻
        from scipy import ndimage as ndi
        sl = ndi.find_objects(comp.astype(np.int64))[0]
        m = P.object_metrics(comp, gt, gl, n, areas, objs, sl, magA, magB, ed, ed,
                             dgt, chg, want_shift=False)
        # comp 为 col 33..36：col 33 与 GT 重叠（d=0）、col 34 d=1、col 35 d=2、col 36 d=3
        self.assertAlmostEqual(m["frac_px_within_1px_of_gt"], 0.5, places=6)
        self.assertAlmostEqual(m["frac_px_within_2px_of_gt"], 0.75, places=6)
        self.assertAlmostEqual(m["dist_to_gt_min_px"], 0.0, places=6)

    def test_shift_search_flags_misalignment(self):
        gt, gl, n, areas, objs, magA, magB, ed, dgt, chg = self._scene()
        comp = np.zeros((64, 64), bool)
        comp[30:34, 28:32] = True          # 与 GT 相差 2px 平移
        from scipy import ndimage as ndi
        sl = ndi.find_objects(comp.astype(np.int64))[0]
        m = P.object_metrics(comp, gt, gl, n, areas, objs, sl, magA, magB, ed, ed,
                             dgt, chg, want_shift=True)
        self.assertAlmostEqual(m["best_shift_iou"], 1.0, places=6)
        self.assertEqual(m["shifted_matched"], 1)
        self.assertEqual(m["matched_at_zero_shift"], 0)

    def test_no_gt_gives_none_distance(self):
        gt = np.zeros((32, 32), bool)
        comp = np.zeros((32, 32), bool)
        comp[4:6, 4:6] = True
        z = np.zeros((32, 32))
        m = P.object_metrics(comp, gt, np.zeros((32, 32), np.int32), 0, [], [],
                             (slice(4, 6), slice(4, 6)), z, z, z.astype(bool),
                             z.astype(bool), P.dist_to_gt(gt), z, want_shift=True)
        self.assertIsNone(m["dist_to_gt_min_px"])
        self.assertEqual(m["frac_px_within_4px_of_gt"], 0.0)
        self.assertEqual(m["best_shift_iou"], 0.0)


class TestBackgroundControl(unittest.TestCase):
    def test_control_avoids_gt_and_pred(self):
        gt = np.zeros((64, 64), bool)
        gt[:32, :] = True
        pred = np.zeros((64, 64), bool)
        pred[:, :16] = True
        rng = np.random.default_rng(0)
        for area in (4, 16, 64, 200):
            win = P.background_control(area, pred, gt, rng)
            if win is None:
                continue
            self.assertFalse(bool((win & gt).any()))
            self.assertFalse(bool((win & pred).any()))
            self.assertEqual(int(win.sum()), int(round(np.sqrt(area))) ** 2)

    def test_returns_none_when_image_full(self):
        gt = np.ones((16, 16), bool)
        self.assertIsNone(P.background_control(4, np.zeros((16, 16), bool), gt,
                                              np.random.default_rng(0)))


class TestBootstrap(unittest.TestCase):
    def test_ratio_brackets_obs(self):
        num = np.array([2.0, 0.0, 3.0, 1.0])
        den = np.array([4.0, 2.0, 6.0, 2.0])
        obs, lo, hi, valid = P.bootstrap_ratio(num, den, n_boot=200, seed=16)
        self.assertAlmostEqual(obs, num.sum() / den.sum())
        self.assertEqual(valid, 200)
        self.assertLessEqual(lo, obs)
        self.assertGreaterEqual(hi, obs)

    def test_ratio_zero_denominator(self):
        obs, lo, hi, valid = P.bootstrap_ratio([1.0], [0.0], n_boot=50, seed=16)
        self.assertIsNone(obs)
        self.assertEqual(valid, 0)

    def test_paired_diff(self):
        a = np.array([0.5, 0.4, 0.6, 0.3])
        b = np.array([0.2, 0.1, 0.3, 0.2])
        obs, lo, hi, n = P.bootstrap_paired(a, b, n_boot=200, seed=16)
        self.assertAlmostEqual(obs, float((a - b).mean()))
        self.assertLessEqual(lo, obs)
        self.assertGreaterEqual(hi, obs)
        self.assertEqual(n, 4)

    def test_paired_ignores_nonfinite(self):
        obs, _lo, _hi, n = P.bootstrap_paired(np.array([1.0, np.nan]), np.array([0.0, 0.0]),
                                              n_boot=50, seed=16)
        self.assertAlmostEqual(obs, 1.0)
        self.assertEqual(n, 1)


class TestConstants(unittest.TestCase):
    def test_frozen_locks(self):
        self.assertEqual(len(P.M1_SYSU_SHA), 64)
        self.assertEqual(len(P.SYSU_TEST_LIST_SHA), 64)
        self.assertEqual(P.MAX_SHIFT, 3)
        self.assertEqual(P.SHIFT_MATCH_IOU, 0.50)
        self.assertEqual(P.EDGE_Q, 0.90)
        self.assertEqual(P.BOOTSTRAP, 1000)

    def test_small_alias_covers_1_to_255(self):
        for a in (1, 255):
            self.assertIn(P.bin_index(a), P.SMALL_ALIAS)
        self.assertNotIn(P.bin_index(256), P.SMALL_ALIAS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
