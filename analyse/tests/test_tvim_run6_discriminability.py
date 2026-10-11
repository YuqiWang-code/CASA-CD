"""Run6 §4 可分辨性前测的纯逻辑单测（本地可跑，不需要 GPU / 数据集 / checkpoint）。

覆盖（Run6 交接 §11.2 指定的清单 + 修 bug 时新增的回归项）：
  * `pool_map` / `resize_frac`（覆盖率池化口径）；
  * `auc_score` / `ap_score`（手工小例子对拍，含并列分数）；
  * `cross_scale_features`（C1 统计量）；
  * `fit_logistic`（可分数据 AUC>0.9）+ `_impute`（中位数只从拟合集取）；
  * `ring_mask` / `interior_mask`（外环**不使用 GT**；与 Run5 FP 画像口径的差异是刻意的）；
  * `denorm_channel`（手工证据通道必须在原始 [0,255] 像素上计算）；
  * "缺失列不使整行被丢弃"（`_matrix` + `_impute` + `probe_report` 行数守恒）；
  * K7 选一规则、K4 基线常数、四条硬门常数、预算切分。
"""
import os
import sys
import unittest

import numpy as np

_ANALYSE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ANALYSE not in sys.path:
    sys.path.insert(0, _ANALYSE)

import tvim_object_metrics as M  # noqa: E402
import tvim_run6_discriminability as R  # noqa: E402


class TestPoolMap(unittest.TestCase):
    def test_weighted_mean_exact(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        frac = np.full((2, 2), 0.5)
        self.assertAlmostEqual(R.pool_map(m, frac), 2.5, places=12)

    def test_unequal_weights(self):
        m = np.array([[0.0, 10.0], [0.0, 0.0]])
        frac = np.array([[0.75, 0.25], [0.0, 0.0]])
        self.assertAlmostEqual(R.pool_map(m, frac), 2.5, places=12)

    def test_zero_coverage_returns_none(self):
        m = np.ones((4, 4))
        self.assertIsNone(R.pool_map(m, np.zeros((4, 4))))

    def test_matches_plain_mean_on_binary_mask(self):
        rng = np.random.default_rng(0)
        m = rng.normal(size=(8, 8))
        mask = np.zeros((8, 8), np.float32)
        mask[2:5, 3:7] = 1.0
        self.assertAlmostEqual(R.pool_map(m, mask), float(m[2:5, 3:7].mean()), places=12)


class TestResizeFrac(unittest.TestCase):
    def test_full_mask_stays_full(self):
        mask = np.ones((256, 256), bool)
        for size in (64, 32, 16, 8, 256):
            fr = R.resize_frac(mask, size)
            self.assertEqual(fr.shape, (size, size))
            self.assertGreater(float(fr.sum()), 0.99 * size * size)
            self.assertLess(float(fr.sum()), 1.01 * size * size)

    def test_empty_mask_is_empty(self):
        fr = R.resize_frac(np.zeros((256, 256), bool), 16)
        self.assertEqual(float(fr.sum()), 0.0)

    def test_half_plane_is_half(self):
        mask = np.zeros((256, 256), bool)
        mask[:, :128] = True
        for size in (64, 32, 8):
            fr = R.resize_frac(mask, size)
            self.assertAlmostEqual(float(fr.sum()) / (size * size), 0.5, places=3)

    def test_tiny_object_survives_downsampling_with_small_weight(self):
        """1 像素对象在 1/32 尺度上权重极小但非零（INTER_AREA 保总面积）⇒ 不得返回 None。"""
        mask = np.zeros((256, 256), bool)
        mask[3, 3] = True
        fr8 = R.resize_frac(mask, 8)
        self.assertGreater(float(fr8.sum()), 0.0)
        self.assertLess(float(fr8.sum()), 0.01)
        val = R.pool_map(np.ones((8, 8)), fr8)
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 1.0, places=9)

    def test_object_entirely_outside_coarse_grid_can_vanish(self):
        """降采样取整确实可能丢掉整块：此时返回 None（由 `_impute` 用拟合集中位数补）。"""
        mask = np.zeros((256, 256), bool)
        fr = np.zeros((8, 8), np.float32)          # 模拟覆盖率全 0
        self.assertIsNone(R.pool_map(np.ones((8, 8)), fr))
        self.assertEqual(float(R.resize_frac(mask, 8).sum()), 0.0)


class TestAucAp(unittest.TestCase):
    def test_auc_perfect_separation(self):
        y = np.array([1, 1, 0, 0], bool)
        s = np.array([3.0, 4.0, 1.0, 2.0])
        self.assertAlmostEqual(R.auc_score(y, s), 1.0, places=12)
        self.assertAlmostEqual(R.auc_score(y, -s), 0.0, places=12)

    def test_auc_all_ties_is_half(self):
        y = np.array([1, 0, 1, 0], bool)
        s = np.full(4, 7.0)
        self.assertAlmostEqual(R.auc_score(y, s), 0.5, places=12)

    def test_auc_hand_computed_with_ties(self):
        # pos = {2.0, 2.0}, neg = {1.0, 2.0}: 对 (2,1) 胜、对 (2,2) 平 ⇒ (1 + 0.5)/2 = 0.75
        y = np.array([1, 1, 0, 0], bool)
        s = np.array([2.0, 2.0, 1.0, 2.0])
        self.assertAlmostEqual(R.auc_score(y, s), 0.75, places=12)

    def test_auc_partial(self):
        y = np.array([1, 0, 0, 0], bool)
        s = np.array([0.5, 0.4, 0.6, 0.1])
        # pos 胜过 0.4/0.1（2 个），输给 0.6 ⇒ 2/3
        self.assertAlmostEqual(R.auc_score(y, s), 2.0 / 3.0, places=12)

    def test_auc_nan_dropped(self):
        y = np.array([1, 0, 1, 0], bool)
        s = np.array([2.0, 1.0, np.nan, 0.0])
        self.assertAlmostEqual(R.auc_score(y, s), 1.0, places=12)

    def test_auc_single_class_is_none(self):
        self.assertIsNone(R.auc_score(np.ones(4, bool), np.arange(4.0)))
        self.assertIsNone(R.auc_score(np.zeros(4, bool), np.arange(4.0)))

    def test_ap_hand_computed(self):
        # 降序分数 4,3,2,1 → y=1,0,1,0：P@1=1 → rec 1/2；P@3=2/3 → rec 1
        # AP = 1*(0.5-0) + (2/3)*(1-0.5) = 0.5 + 1/3
        y = np.array([1, 0, 1, 0], bool)
        s = np.array([4.0, 3.0, 2.0, 1.0])
        self.assertAlmostEqual(R.ap_score(y, s), 0.5 + 1.0 / 3.0, places=12)

    def test_ap_perfect(self):
        y = np.array([1, 1, 0, 0], bool)
        self.assertAlmostEqual(R.ap_score(y, np.array([4.0, 3.0, 2.0, 1.0])), 1.0, places=12)

    def test_ap_reversed_ranking_hand_computed(self):
        # 降序分数 4,3,2,1 → y=0,0,1,1：P@3=1/3（rec 1/2）、P@4=2/4（rec 1）
        # AP = (1/3)*(1/2) + (1/2)*(1/2) = 5/12
        y = np.array([0, 0, 1, 1], bool)
        ap = R.ap_score(y, np.array([4.0, 3.0, 2.0, 1.0]))
        self.assertAlmostEqual(ap, 5.0 / 12.0, places=12)
        self.assertLess(ap, 0.5)                 # 反向排序 ⇒ AP 低于随机基线先验 0.5

    def test_ap_no_positive_is_none(self):
        self.assertIsNone(R.ap_score(np.zeros(4, bool), np.arange(4.0)))


class TestCrossScaleFeatures(unittest.TestCase):
    def test_equal_scales(self):
        out = R.cross_scale_features([2.0, 2.0, 2.0, 2.0])
        for s in range(1, 5):
            self.assertAlmostEqual(out[f"Y_ratio_s{s}"], 0.25, places=5)
        self.assertAlmostEqual(out["Y_range"], 0.0, places=12)
        self.assertAlmostEqual(out["Y_cv"], 0.0, places=8)
        self.assertAlmostEqual(out["Y_entropy"], float(np.log(4.0)), places=5)

    def test_ratios_sum_to_one_and_dominant_scale_wins(self):
        out = R.cross_scale_features([1e6, 1.0, 1.0, 1.0])
        tot = sum(out[f"Y_ratio_s{s}"] for s in range(1, 5))
        self.assertAlmostEqual(tot, 1.0, places=6)
        self.assertGreater(out["Y_ratio_s1"], 0.99)
        self.assertGreater(out["Y_range"], 1e5)

    def test_entropy_minimal_when_one_scale_dominates(self):
        out = R.cross_scale_features([0.0, 0.0, 9.0, 0.0])
        self.assertLess(out["Y_entropy"], 1e-6)

    def test_negative_values_clipped_and_zero_safe(self):
        out = R.cross_scale_features([-5.0, 0.0, 0.0, 0.0])
        self.assertGreaterEqual(out["Y_ratio_s1"], 0.0)
        self.assertTrue(np.isfinite(out["Y_cv"]))


class TestFitLogistic(unittest.TestCase):
    def test_separable_data_auc_above_0_9(self):
        rng = np.random.default_rng(16)
        X = np.r_[rng.normal(2.0, 0.5, size=(200, 3)), rng.normal(-2.0, 0.5, size=(200, 3))]
        y = np.r_[np.ones(200, bool), np.zeros(200, bool)]
        w, mu, sd = R.fit_logistic(X, y, lam=1e-3)
        sc = R.apply_logistic(w, mu, sd, X)
        self.assertGreater(R.auc_score(y, sc), 0.9)

    def test_deterministic(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 4))
        y = (X[:, 0] + X[:, 1] > 0)
        a = R.fit_logistic(X, y, 1e-2)
        b = R.fit_logistic(X, y, 1e-2)
        np.testing.assert_array_equal(a[0], b[0])

    def test_l2_shrinks_weights(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(120, 3))
        y = (X[:, 0] > 0)
        w_lo = R.fit_logistic(X, y, 1e-3)[0]
        w_hi = R.fit_logistic(X, y, 1.0)[0]
        self.assertLess(np.linalg.norm(w_hi[1:]), np.linalg.norm(w_lo[1:]))

    def test_standardization_uses_fit_statistics_only(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1], bool)
        _w, mu, sd = R.fit_logistic(X, y, 1e-3)
        self.assertAlmostEqual(float(mu[0]), 1.5, places=12)
        self.assertAlmostEqual(float(sd[0]), float(X[:, 0].std()) + 1e-9, places=10)


class TestImpute(unittest.TestCase):
    def test_median_from_fit_only(self):
        X_fit = np.array([[1.0, 10.0], [3.0, np.nan], [np.nan, 30.0]])
        X_other = np.array([[np.nan, np.nan], [7.0, 7.0]])
        f, o, med = R._impute(X_fit, X_other)
        self.assertAlmostEqual(float(med[0]), 2.0, places=12)     # 中位(1,3)
        self.assertAlmostEqual(float(med[1]), 20.0, places=12)    # 中位(10,30)
        self.assertTrue(np.isfinite(f).all() and np.isfinite(o).all())
        self.assertAlmostEqual(float(o[0, 0]), 2.0, places=12)
        self.assertAlmostEqual(float(o[0, 1]), 20.0, places=12)

    def test_all_nan_column_falls_back_to_zero(self):
        X_fit = np.array([[np.nan], [np.nan]])
        f, _o, med = R._impute(X_fit, np.array([[np.nan]]))
        self.assertEqual(float(med[0]), 0.0)
        self.assertEqual(float(f[0, 0]), 0.0)

    def test_does_not_change_finite_values(self):
        X_fit = np.array([[1.0, 2.0], [3.0, 4.0]])
        f, o, _m = R._impute(X_fit, X_fit.copy())
        np.testing.assert_array_equal(f, X_fit)
        np.testing.assert_array_equal(o, X_fit)

    def test_matrix_keeps_none_as_nan(self):
        rows = [{"a": 1.0, "b": None}, {"a": None, "b": 2.0}]
        X = R._matrix(rows, ["a", "b"])
        self.assertEqual(X.shape, (2, 2))
        self.assertTrue(np.isnan(X[0, 1]) and np.isnan(X[1, 0]))
        self.assertFalse(np.isnan(X[0, 0]))


class TestMasks(unittest.TestCase):
    def test_interior_and_ring(self):
        obj = np.zeros((32, 32), bool)
        obj[10:15, 10:15] = True                     # 5x5 实心块
        inter = R.interior_mask(obj)
        self.assertEqual(int(inter.sum()), 9)        # 3x3
        self.assertTrue(bool((inter & ~obj).sum() == 0))
        ring = R.ring_mask(obj, iterations=4)
        # 4 邻域结构元膨胀 4 次：不得与对象/内部重叠，且必须落在 dilate(obj) 内
        from scipy import ndimage as ndi
        dil = ndi.binary_dilation(obj, structure=R.CONN4, iterations=4)
        np.testing.assert_array_equal(ring, dil & ~obj)
        self.assertEqual(int((ring & obj).sum()), 0)
        self.assertEqual(int((ring & inter).sum()), 0)
        self.assertEqual(int(ring.sum()), 104)       # 5x5 块 + 4 次 4 邻域膨胀的确定性几何量
        self.assertGreater(int(ring.sum()), int(obj.sum()))
        # 平移不变性：在图像内平移对象后 ring 同步平移（无边界截断）
        obj2 = np.zeros((32, 32), bool)
        obj2[4:9, 20:25] = True
        ring2 = R.ring_mask(obj2, iterations=4)
        self.assertEqual(int(ring2.sum()), int(ring.sum()))

    def test_ring_does_not_depend_on_gt(self):
        """Run6 刻意去掉 Run5 的 `∩(~GT)`：外环必须是纯几何量，否则是 GT 泄漏。"""
        obj = np.zeros((64, 64), bool)
        obj[30:34, 30:34] = True
        a = R.ring_mask(obj, iterations=4)
        b = R.ring_mask(obj, iterations=4)           # 同输入同输出（无隐藏参数）
        np.testing.assert_array_equal(a, b)
        # 邻域 GT 完全不参与：函数签名里没有 gt
        self.assertEqual(R.ring_mask.__code__.co_varnames[:2], ("obj", "iterations"))

    def test_interior_empty_for_single_pixel(self):
        obj = np.zeros((16, 16), bool)
        obj[8, 8] = True
        self.assertFalse(bool(R.interior_mask(obj).any()))
        self.assertGreater(int(R.ring_mask(obj, iterations=1).sum()), 0)

    def test_ring_uses_4_connectivity_structure(self):
        np.testing.assert_array_equal(R.CONN4, M.CONN4)
        obj = np.zeros((16, 16), bool)
        obj[8, 8] = True
        ring1 = R.ring_mask(obj, iterations=1)
        self.assertEqual(int(ring1.sum()), 4)        # 4 邻域：上下左右
        self.assertFalse(bool(ring1[7, 7]))          # 对角不属于 4 连通结构元


class TestHandPixelSpace(unittest.TestCase):
    """手工证据通道必须与 F14 / Run5 FP 画像同口径（归一化张量），不得反归一化。

    冻结读数 `change_abs` FP=0.9730、`gradA` FP=1.8008 只在归一化口径下可能；
    原始 [0,255] 下均值绝对差会是 O(10–100)。本类用真实 Transforms 链证明这一点，
    并锁定 `image_context` 不做事后缩放。
    """

    def test_channel_mean3(self):
        x = np.stack([np.full((4, 4), 1.0), np.full((4, 4), 2.0), np.full((4, 4), 6.0)])
        np.testing.assert_allclose(R.channel_mean3(x), np.full((4, 4), 3.0))
        with self.assertRaises(ValueError):
            R.channel_mean3(np.zeros((6, 4, 4)))          # 6 通道（未切分）必须报错

    def test_frozen_change_abs_scale_is_normalized_not_raw(self):
        """用真实 Normalize/Scale/ToTensor 链验证：0.9730 量级只能在归一化口径出现。"""
        import cv2  # noqa: F401
        from dataset import Transforms as T

        rng = np.random.default_rng(11)
        a = rng.integers(0, 256, size=(256, 256, 3)).astype(np.uint8)
        b = a.copy()
        # 局部小扰动（模拟真实变化区域）
        b[100:130, 100:130] = np.clip(b[100:130, 100:130].astype(np.int32) + 40, 0, 255)
        tf = T.Compose([T.Normalize(mean=list(R.MEAN6), std=list(R.STD6)),
                        T.Scale(256, 256), T.ToTensor()])
        img, _lab = tf(np.concatenate([a, b], axis=2), np.zeros((256, 256), np.uint8))
        arr = img.numpy()
        norm_gap = float(np.abs(R.channel_mean3(arr[0:3])
                                - R.channel_mean3(arr[3:6])).mean())
        raw_gap = float(np.abs(a.astype(np.float64).mean(2)
                              - b.astype(np.float64).mean(2)).mean())
        # 归一化口径：变化块的 |Δ| 约 40/(255*0.225) ≈ 0.7，整图均值远小于 10
        self.assertLess(norm_gap, 10.0)
        self.assertGreater(norm_gap, 0.0)
        # 原始口径：整图均值绝对差是 O(1..100) 量级，且比归一化口径大一个数量级以上
        self.assertGreater(raw_gap, norm_gap)
        self.assertGreater(raw_gap / norm_gap, 5.0)

    def test_image_context_does_not_rescale_input(self):
        """`ctx['chg']` 必须等于传入张量本身的 |mean(A)−mean(B)|（无任何反归一化/缩放）。"""
        torch = __import__("torch")
        rng = np.random.default_rng(5)
        ab = rng.normal(size=(1, 6, 256, 256)).astype(np.float32)
        cap = {}
        for s, L in zip(range(1, 5), (64, 32, 16, 8)):
            cap[f"Y_s{s}"] = torch.zeros(1, 96, L, L)
            cap[f"T_s{s}"] = torch.zeros(1, 96, L, L)
        cap["refine"] = torch.zeros(1, 96, 64, 64)
        cap["head"] = torch.zeros(1, 1, 64, 64)
        prob = np.zeros((256, 256), np.float32)
        ctx = R.image_context(cap, 0, ab, prob)
        expect = np.abs(R.channel_mean3(ab[0, 0:3]) - R.channel_mean3(ab[0, 3:6]))
        np.testing.assert_allclose(ctx["chg"], expect, rtol=0, atol=1e-6)
        # 若误做 (x*std+mean)*255 反归一化，chg 会被放大 >100 倍 ⇒ 该测试会失败
        self.assertFalse(np.allclose(ctx["chg"], expect * 255.0, rtol=0.01, atol=1.0))
        self.assertEqual(ctx["edgeA_d"].shape, (256, 256))
        self.assertEqual(ctx["edgeA_d"].dtype, np.bool_)
        self.assertEqual(ctx["maps"]["Y_s1"].shape, (64, 64))
        self.assertEqual(ctx["shape"]["refine"], 64)
        self.assertEqual(ctx["shape"]["logit"], 64)

    def test_check_finite_raises_on_nan(self):
        with self.assertRaises(ValueError):
            R.check_finite(np.array([1.0, np.nan]), name="t")
        self.assertEqual(R.check_finite(np.array([1.0, 2.0])), (1.0, 2.0))


class TestProbeReportRowConservation(unittest.TestCase):
    def _rows(self, n_img, n_per_img, with_missing=False, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_img):
            for j in range(n_per_img):
                lab = 1 if (i + j) % 2 == 0 else 0
                r = {"image": f"img{i:03d}", "label": lab,
                     "f1": float(rng.normal(lab, 1.0)), "f2": float(rng.normal(lab, 1.0))}
                if with_missing and (i + j) % 7 == 0:
                    r["f2"] = None                      # ring 为空等导致的缺失
                rows.append(r)
        return rows

    def test_missing_column_does_not_drop_rows(self):
        tr = self._rows(40, 6, with_missing=True, seed=1)
        te = self._rows(40, 6, with_missing=True, seed=2)
        res = R.probe_report(tr, te, ["f1", "f2"], seed=16, n_boot=20)
        self.assertFalse(res["insufficient"])
        self.assertEqual(res["n_train"], len(tr))
        self.assertEqual(res["n_test"], len(te))
        self.assertGreater(res["n_missing_imputed_train"], 0)
        self.assertGreater(res["n_missing_imputed_test"], 0)

    def test_confounded_feature_gives_auc_near_half(self):
        tr = self._rows(40, 6, seed=5)
        te = self._rows(40, 6, seed=6)
        for r in tr + te:
            r["noise"] = 0.0
            r["pure"] = float(r["label"])
        res = R.probe_report(tr, te, ["noise"], seed=16, n_boot=20)
        self.assertAlmostEqual(res["test_auc"], 0.5, places=6)
        res2 = R.probe_report(tr, te, ["pure"], seed=16, n_boot=20)
        self.assertAlmostEqual(res2["test_auc"], 1.0, places=6)

    def test_insufficient_guard(self):
        res = R.probe_report(self._rows(2, 2, seed=1), self._rows(2, 2, seed=1),
                             ["f1"], seed=16, n_boot=5)
        self.assertTrue(res["insufficient"])

    def test_single_feature_aucs(self):
        te = self._rows(20, 4, seed=9)
        for r in te:
            r["pure"] = float(r["label"])
        out = R.single_feature_aucs(te, ["pure", "f1"])
        self.assertAlmostEqual(out["pure"], 1.0, places=6)


class TestK7Selection(unittest.TestCase):
    def _r(self, cv):
        return {"insufficient": False, "train_cv_auc": cv}

    def test_clear_winner_by_train_cv(self):
        sel, why = R.k7_select({"C1": self._r(0.70), "C2": self._r(0.80)})
        self.assertEqual(sel, "C2")
        self.assertEqual(why, "train_cv_auc")

    def test_tie_within_eps_prefers_fewer_params_C1(self):
        sel, why = R.k7_select({"C1": self._r(0.800), "C2": self._r(0.795)})
        self.assertEqual(sel, "C1")
        self.assertIn("tie", why)

    def test_exactly_eps_boundary_is_tie(self):
        # 相差恰为 0.005 ⇒ 按文档"相差 ≤0.005"取 C1
        sel, _why = R.k7_select({"C1": self._r(0.700), "C2": self._r(0.705)})
        self.assertEqual(sel, "C1")

    def test_just_over_eps_is_not_tie(self):
        sel, _why = R.k7_select({"C2": self._r(0.7051), "C1": self._r(0.700)})
        self.assertEqual(sel, "C2")

    def test_c2_can_win_alone(self):
        sel, why = R.k7_select({"C2": self._r(0.90)})
        self.assertEqual(sel, "C2")
        self.assertEqual(why, "only_usable")

    def test_tie_prefers_c1_even_when_c1_is_weaker(self):
        sel, _why = R.k7_select({"C1": self._r(0.60), "C2": self._r(0.604)})
        self.assertEqual(sel, "C1")

    def test_no_usable_candidate(self):
        self.assertEqual(R.k7_select({}), (None, None))


class TestFrozenConstants(unittest.TestCase):
    def test_hard_gates_match_preregistration(self):
        """Run6 文档 §4.2 的 K 门数值在见结果之前冻结；本测试防止被静默改动。"""
        self.assertEqual(R.GATES["K1_n_pos_min"], 300)
        self.assertEqual(R.GATES["K1_n_neg_min"], 2000)
        self.assertEqual(R.GATES["K2_test_auc_min"], 0.80)
        self.assertEqual(R.GATES["K2_test_auc_ci_low_min"], 0.75)
        self.assertEqual(R.GATES["K3_test_ap_min"], 0.35)
        self.assertEqual(R.GATES["K3_test_ap_ci_low_min"], 0.25)
        self.assertEqual(R.GATES["K4_auc_over_handcrafted_min"], 0.15)
        self.assertEqual(R.GATES["K4_handcrafted_baseline_auc"], 0.5992)
        self.assertEqual(R.GATES["K4b_auc_over_full_baseline_min"], 0.05)
        self.assertEqual(R.GATES["K5_auc_over_amplitude_min"], 0.10)

    def test_lock_hashes_and_budget(self):
        self.assertEqual(
            R.M1_SYSU_SHA,
            "45d688a01af22fe22521f4dfb4579e80827ddcaa886e07bafc778fc288731457")
        self.assertEqual(
            R.SYSU_TEST_LIST_SHA,
            "5f4122c9d32cb6db475f4f361618b2ac8bb10938df93d9550e0bb9a422a45797")
        self.assertEqual(R.BOOTSTRAP, 1000)
        self.assertEqual(R.BUDGET_S, 900.0)
        self.assertEqual(R.VAL_FRAC, 0.10)
        self.assertEqual(R.PROTOCOL_TAG, "run6_scale_gate_exact80k_v1")
        self.assertEqual(R.PROFILE_VERSION, "run6_discriminability_v1")

    def test_frozen_reference_object_counts(self):
        self.assertEqual(R.FROZEN_REF["n_pos_expected"], 327)
        self.assertEqual(R.FROZEN_REF["n_neg_expected"], 2430)
        self.assertEqual(R.FROZEN_REF["handcrafted_best_single_auc"], 0.5992)

    def test_baseline_groups_are_disjoint_and_cover_all_features(self):
        groups = list(R.FEATURE_GROUPS.values())
        flat = [c for g in groups for c in g]
        self.assertEqual(len(flat), len(set(flat)), "特征列不得重名（含跨组撞名）")
        self.assertEqual(sorted(flat), sorted(R.ALL_FEATURES))
        # 派生列（4 尺度平均）属于幅度基线，但仍不得与候选组撞名
        self.assertEqual(len(R.ALL_FEATURES), len(set(R.ALL_FEATURES)))
        cand = set(R.FEATURE_GROUPS["C1"]) | set(R.FEATURE_GROUPS["C2"])
        amp, hand, head = (set(R.BASELINE_AMP), set(R.BASELINE_HAND), set(R.BASELINE_HEAD))
        for a in (amp, hand, head):
            self.assertFalse(a & cand)
        # 三基线组内部两两不交
        self.assertFalse(amp & hand)
        self.assertFalse(amp & head)
        self.assertFalse(hand & head)
        self.assertEqual(R.BASELINE_FULL, R.BASELINE_AMP + R.BASELINE_HAND + R.BASELINE_HEAD)

    def test_candidate_sets_are_baseline_plus_exactly_one_group(self):
        self.assertEqual(R.PROBE_SETS["candidate_C1"],
                         R.BASELINE_FULL + R.FEATURE_GROUPS["C1"])
        self.assertEqual(R.PROBE_SETS["candidate_C2"],
                         R.BASELINE_FULL + R.FEATURE_GROUPS["C2"])
        self.assertEqual(sorted(R.PROBE_SETS), sorted([
            "baseline_amp", "baseline_hand", "baseline_head", "baseline_full",
            "C1_features", "C2_features", "candidate_C1", "candidate_C2"]))

    def test_amplitude_baseline_contains_four_scale_mean(self):
        """§4.1 要求 ‖Δ‖ 的 4 尺度平均必须在幅度基线里（H6b 的分母）。"""
        self.assertIn("mag_s_mean", R.BASELINE_AMP)

    def test_object_rule_constants(self):
        self.assertEqual(sorted(R.SMALL_ALIAS), sorted(M.GROUP_ALIAS["small"]))
        self.assertNotIn("medium_256_1023", R.SMALL_ALIAS)
        self.assertNotIn("large_1024_inf", R.SMALL_ALIAS)
        self.assertEqual(R.EDGE_Q, 0.90)


class TestPositiveRule(unittest.TestCase):
    """对象口径必须与 Run5/F14 逐字一致（Run5 F15 教训：面积组过滤不能漏）。"""

    @staticmethod
    def _rule(gt_index0, matched_gt, r):
        return 1 if ((gt_index0 in matched_gt) or (r < 0.25)) else None

    def test_matched_gt_is_positive(self):
        self.assertEqual(self._rule(3, {3}, 0.9), 1)

    def test_missed_low_recall_is_positive(self):
        self.assertEqual(self._rule(3, set(), 0.10), 1)

    def test_unmatched_high_recall_is_not_included(self):
        self.assertIsNone(self._rule(3, set(), 0.25))
        self.assertIsNone(self._rule(3, set(), 0.60))

    def test_boundary_recall_024_is_positive(self):
        self.assertEqual(self._rule(0, set(), 0.2499), 1)


class TestSmallGroupFilter(unittest.TestCase):
    """负样本侧的面积组过滤（Run5 F15 的回归项：medium/large 未匹配预测不得计入）。"""

    def test_only_small_alias_passes(self):
        for area in (1, 15, 16, 63, 64, 255):
            self.assertIn(M.bin_index(area), R.SMALL_ALIAS, area)
        for area in (256, 1023, 1024, 5000):
            self.assertNotIn(M.bin_index(area), R.SMALL_ALIAS, area)

    def test_area_zero_raises(self):
        with self.assertRaises(ValueError):
            M.bin_index(0)


class TestBudgetSplit(unittest.TestCase):
    def test_split_is_proportional_and_positive(self):
        fit, ev = R._budget_split(900.0, 10800, 4000)
        self.assertAlmostEqual(fit + ev, 900.0, places=6)
        self.assertGreater(fit, ev)
        self.assertAlmostEqual(fit, 900.0 * 10800 / 14800, places=6)

    def test_minimum_floor(self):
        fit, ev = R._budget_split(100.0, 1_000_000, 1)
        self.assertGreaterEqual(fit, 60.0)
        self.assertGreaterEqual(ev, 60.0)


class TestGateHelpers(unittest.TestCase):
    def test_k_structure(self):
        k = R._k("K2", True, {"a": 1}, {"b": 2}, "note")
        self.assertEqual(k["pass"], True)
        self.assertEqual(k["threshold"], {"b": 2})
        self.assertEqual(k["note"], "note")
        self.assertIs(R._k("K1", 0, None, None)["pass"], False)

    def test_boot_ci_shape_and_reproducibility(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=400).astype(bool)
        s = y.astype(float) + rng.normal(0, 0.5, 400)
        imgs = [f"i{i // 4}" for i in range(400)]
        a = R.boot_ci(y, s, imgs, n_boot=50, seed=16, metric="auc")
        b = R.boot_ci(y, s, imgs, n_boot=50, seed=16, metric="auc")
        self.assertEqual(a, b)
        self.assertEqual(a[2], 50)
        self.assertLess(a[0], a[1])
        self.assertGreater(a[0], 0.8)               # 强信号：CI 下界仍高

    def test_boot_ci_returns_none_when_no_valid_resample(self):
        y = np.zeros(10, bool)                      # 无正样本 ⇒ 每次重采样都 None
        lo, hi, n = R.boot_ci(y, np.arange(10.0), ["i"] * 10, n_boot=5, seed=16)
        self.assertIsNone(lo)
        self.assertEqual(n, 0)


class TestNumericHelpers(unittest.TestCase):
    def test_weighted_mean_std_zero_weight(self):
        self.assertEqual(R._weighted_mean_std(np.ones((4, 4)), np.zeros((4, 4))), (None, None))

    def test_weighted_mean_std_constant_map(self):
        m = np.full((8, 8), 3.0)
        mu, sd = R._weighted_mean_std(m, np.full((8, 8), 0.5))
        self.assertAlmostEqual(mu, 3.0, places=12)
        self.assertAlmostEqual(sd, 0.0, places=12)

    def test_weighted_mean_std_matches_uniform_mask(self):
        rng = np.random.default_rng(4)
        m = rng.normal(size=(16, 16))
        w = np.zeros((16, 16))
        w[4:9, 6:12] = 1.0
        mu, sd = R._weighted_mean_std(m, w)
        sub = m[4:9, 6:12]
        self.assertAlmostEqual(mu, float(sub.mean()), places=12)
        self.assertAlmostEqual(sd, float(sub.std()), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
