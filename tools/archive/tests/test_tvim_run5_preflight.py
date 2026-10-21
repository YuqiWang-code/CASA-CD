"""Run5 §5 只读前测的纯逻辑单测（本地可跑，不需要 GPU / 数据集 / checkpoint）。

覆盖：平均秩与逐图 rank、环带 margin、对象集合口径（missed_small / FP-like）、
逐图计数、图像级 bootstrap（比值 / gap / 配对均值）、§5.3.3 选层规则、gate 记录形状。
不触网、不读 .pth、不训练。
"""
import os
import sys
import unittest

import numpy as np

_ANALYSE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ANALYSE not in sys.path:
    sys.path.insert(0, _ANALYSE)

import tvim_run5_hf_pair_preflight as P  # noqa: E402


class TestRank(unittest.TestCase):
    def test_avg_rank_range_and_ties(self):
        x = np.array([10.0, 20.0, 20.0, 40.0])
        r = P._avg_rank01(x)
        self.assertEqual(r.shape, x.shape)
        self.assertAlmostEqual(float(r.min()), 0.0)
        self.assertAlmostEqual(float(r.max()), 1.0)
        # ties 取平均秩：值 20 的两个元素 rank 相同
        self.assertAlmostEqual(float(r[1]), float(r[2]))
        # 单调：高分位秩更大
        self.assertLess(float(r[0]), float(r[1]))
        self.assertLess(float(r[2]), float(r[3]))

    def test_avg_rank_label_independent_and_deterministic(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=4096)
        a = P._avg_rank01(x)
        b = P._avg_rank01(x.copy())
        np.testing.assert_array_equal(a, b)
        # 打乱顺序不改变“值→秩”的对应（按值索引验证）
        perm = rng.permutation(x.size)
        c = P._avg_rank01(x[perm])
        np.testing.assert_allclose(c, a[perm])

    def test_avg_rank_constant_is_half(self):
        # 完全并列 → 平均秩取中点 → 0.5（rank_tensor 的插值会引入浮点抖动，故直接测 rank 本体）
        r = P._avg_rank01(np.full(4096, 3.14))
        np.testing.assert_allclose(r, 0.5, atol=1e-12)

    def test_rank_tensor_is_monotone_in_upsampled_values(self):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(0)
        t = torch.rand(1, 16, 16)
        out = P._rank_tensor(t)[0]
        vals = F.interpolate(t[None].float(), size=(256, 256), mode="bilinear",
                             align_corners=False)[0, 0].numpy()
        order = np.argsort(vals.ravel(), kind="stable")
        r = out.ravel()[order]
        self.assertTrue(bool(np.all(np.diff(r) >= -1e-12)))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_rank_tensor_constant_stays_in_range(self):
        # 注意：双线性上采样的常量场存在 ULP 级抖动，再 rank 会把 65536 个近似并列值
        # 分成两组 → 秩跨度可达 0.5。这是预注册口径的固有性质（HF 与 TAP 同等处理），
        # 只要求结果落在 [0,1] 且有限。
        import torch
        t = torch.full((1, 8, 8), 3.14)
        out = P._rank_tensor(t)
        self.assertEqual(out.shape, (1, 256, 256))
        self.assertTrue(bool(np.isfinite(out).all()))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_rank_tensor_upsample_then_rank(self):
        import torch
        t = torch.zeros((1, 16, 16))
        t[0, :8, :] = 1.0
        out = P._rank_tensor(t)[0]
        self.assertEqual(out.shape, (256, 256))
        self.assertGreater(float(out[:128].mean()), float(out[128:].mean()))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


class TestRingMargin(unittest.TestCase):
    def test_margin_positive_for_inside_high(self):
        gt = np.zeros((64, 64), bool)
        gt[10:14, 10:14] = True          # 4x4 = 16 px
        score = np.zeros((64, 64))
        score[10:14, 10:14] = 1.0
        obj = gt.copy()
        sl = (slice(10, 14), slice(10, 14))
        m, ok = P._ring_margin(score, obj, gt, sl)
        self.assertTrue(ok)
        self.assertAlmostEqual(m, 1.0)

    def test_empty_ring_is_skipped(self):
        gt = np.ones((8, 8), bool)
        score = np.zeros((8, 8))
        m, ok = P._ring_margin(score, gt.copy(), gt, (slice(0, 8), slice(0, 8)))
        self.assertFalse(ok)
        self.assertIsNone(m)

    def test_ring_excludes_other_gt(self):
        gt = np.zeros((32, 32), bool)
        gt[5:9, 5:9] = True
        gt[10:14, 5:9] = True            # 紧邻的另一个 GT 对象
        score = np.zeros((32, 32))
        score[5:9, 5:9] = 1.0
        score[10:14, 5:9] = 0.0          # 若把它当环带会拉低 margin
        obj = np.zeros((32, 32), bool)
        obj[5:9, 5:9] = True             # 必须是全尺寸 mask（与生产调用一致）
        m, ok = P._ring_margin(score, obj, gt, (slice(5, 9), slice(5, 9)))
        self.assertTrue(ok)
        self.assertGreater(m, 0.5)       # 另一个 GT 对象被 ∩(~GT) 排除


def _mk_case():
    """合成 64² 场景：1 个漏检 small GT、1 个已命中 large GT、1 个未匹配 small 预测。"""
    gt = np.zeros((64, 64), bool)
    gt[40:44, 40:44] = True                      # small (16px)，完全漏检
    gt[5:25, 5:25] = True                        # large (400px)，被预测覆盖
    prob = np.zeros((64, 64))
    prob[5:25, 5:25] = 0.9                       # 命中 large
    prob[55:58, 55:58] = 0.9                     # 未匹配 small 预测 (9px)
    hf = np.zeros((64, 64))
    hf[40:44, 40:44] = 1.0                       # HF 在漏检对象上高响应
    hf[55:58, 55:58] = 0.0                       # HF 在 FP-like 上低响应
    tap = np.zeros((64, 64))                     # TAP 无响应 → HF-only
    scores = {"HF": hf, "TAP": tap}
    pred_b = prob > 0.5
    from tvim_object_metrics import match_objects, object_iou_matrix
    cache = object_iou_matrix(pred_b, gt, 4)
    mt = match_objects(pred_b, gt, 4, cache=cache)
    return prob, gt, scores, cache, mt


class TestImageRecords(unittest.TestCase):
    def test_counts_and_sets(self):
        prob, gt, scores, cache, mt = _mk_case()
        rows, gt_rows, rec, small_mask = P._image_records(prob, gt, scores, cache, mt)
        self.assertEqual(rec["n_missed"], 1)
        self.assertEqual(rec["n_fp"], 1)
        self.assertEqual(rec["n_hit_HF"], 1)
        self.assertEqual(rec["n_hit_TAP"], 0)
        self.assertEqual(rec["n_fp_hit_HF"], 0)
        self.assertEqual(rec["n_hf_only"], 1)
        self.assertEqual(rec["n_margin_pairs"], 1)
        kinds = sorted(r["kind"] for r in rows)
        self.assertEqual(kinds, ["fp_like", "missed_small"])
        ms = [r for r in rows if r["kind"] == "missed_small"][0]
        self.assertEqual(ms["area"], 16)
        self.assertAlmostEqual(ms["pixel_recall"], 0.0)
        self.assertTrue(P._is_hit(ms, "HF"))
        self.assertFalse(P._is_hit(ms, "TAP"))
        self.assertEqual(ms["iou_max_any_pred"], 0.0)
        fp = [r for r in rows if r["kind"] == "fp_like"][0]
        self.assertEqual(fp["area"], 9)
        self.assertIsNotNone(fp["fp_like_id"])
        # small_mask 只含 1–255px 的 GT
        self.assertEqual(int(small_mask.sum()), 16)
        # gt_rows 覆盖全部两个 GT 对象
        self.assertEqual(len(gt_rows), 2)

    def test_tap_hit_kills_novel(self):
        prob, gt, scores, cache, mt = _mk_case()
        scores = {"HF": scores["HF"], "TAP": scores["HF"].copy()}
        rows, _g, rec, _s = P._image_records(prob, gt, scores, cache, mt)
        self.assertEqual(rec["n_hit_HF"], 1)
        self.assertEqual(rec["n_hit_TAP"], 1)
        self.assertEqual(rec["n_hf_only"], 0)
        self.assertEqual(rec["n_tap_only"], 0)

    def test_medium_large_not_counted_as_missed_small(self):
        """回归：medium/large 对象即使 r<0.25 也不得进入 missed_small（§5.2.1）。"""
        gt = np.zeros((128, 128), bool)
        gt[10:14, 10:14] = True          # small 16px，漏检 → 应计入
        gt[40:60, 40:60] = True          # medium 400px，全漏检 → 不得计入
        gt[70:110, 70:110] = True        # large 1600px，全漏检 → 不得计入
        prob = np.zeros((128, 128))
        scores = {"HF": np.zeros((128, 128)), "TAP": np.zeros((128, 128))}
        from tvim_object_metrics import match_objects, object_iou_matrix
        cache = object_iou_matrix(prob > 0.5, gt, 4)
        mt = match_objects(prob > 0.5, gt, 4, cache=cache)
        rows, gt_rows, rec, _s = P._image_records(prob, gt, scores, cache, mt)
        ms = [r for r in rows if r["kind"] == "missed_small"]
        self.assertEqual(rec["n_missed"], 1)
        self.assertEqual(rec["n_missed_pre_ring"], 1)
        self.assertEqual([r["area"] for r in ms], [16])
        self.assertEqual(rec["small_n"], 1)
        self.assertEqual(rec["small_px"], 16)
        self.assertEqual(rec["small_tp"], 0)
        self.assertEqual(rec["small_hit25"], 0)
        self.assertTrue(all(r["galias"] == "small" for r in ms))

    def test_small_hit25_excluded_from_missed(self):
        gt = np.zeros((64, 64), bool)
        gt[10:14, 10:14] = True          # 16px
        prob = np.zeros((64, 64))
        prob[10:12, 10:12] = 0.9         # 命中 4/16 = 0.25 → hit25 边界，不算漏检
        scores = {"HF": np.zeros((64, 64)), "TAP": np.zeros((64, 64))}
        from tvim_object_metrics import match_objects, object_iou_matrix
        cache = object_iou_matrix(prob > 0.5, gt, 4)
        mt = match_objects(prob > 0.5, gt, 4, cache=cache)
        _rows, _g, rec, _s = P._image_records(prob, gt, scores, cache, mt)
        self.assertEqual(rec["small_hit25"], 1)
        self.assertEqual(rec["n_missed"], 0)

    def test_matched_pred_is_not_fp_like(self):
        prob, gt, scores, cache, mt = _mk_case()
        rows, _g, rec, _s = P._image_records(prob, gt, scores, cache, mt)
        fps = [r["fp_like_id"] for r in rows if r["kind"] == "fp_like"]
        self.assertNotIn(int(np.argmax(cache["pred_areas"])), fps)


class TestBootstrap(unittest.TestCase):
    def test_boot_ratio_brackets_obs(self):
        num = np.array([3.0, 0.0, 5.0, 1.0, 2.0])
        den = np.array([10.0, 4.0, 10.0, 5.0, 8.0])
        obs, lo, hi, valid = P._boot_ratio(num, den, n_boot=200, seed=16)
        self.assertAlmostEqual(obs, num.sum() / den.sum())
        self.assertEqual(valid, 200)
        self.assertLessEqual(lo, obs)
        self.assertGreaterEqual(hi, obs)

    def test_boot_gap_matches_obs(self):
        hit = np.array([2.0, 0.0, 1.0])
        fph = np.array([1.0, 1.0, 0.0])
        dm = np.array([4.0, 2.0, 4.0])
        df = np.array([6.0, 3.0, 3.0])
        obs, lo, hi, valid = P._boot_gap(hit, fph, dm, df, n_boot=200, seed=16)
        self.assertAlmostEqual(obs, hit.sum() / dm.sum() - fph.sum() / df.sum())
        self.assertEqual(valid, 200)
        self.assertLessEqual(lo, obs)
        self.assertGreaterEqual(hi, obs)

    def test_boot_gap_zero_denominator_is_invalid(self):
        obs, lo, hi, valid = P._boot_gap(np.array([1.0]), np.array([1.0]),
                                         np.array([1.0]), np.array([0.0]),
                                         n_boot=50, seed=16)
        self.assertIsNone(obs)
        self.assertEqual(valid, 0)

    def test_boot_mean_pair(self):
        s = np.array([0.4, 0.0, -0.2, 0.6])
        k = np.array([2.0, 1.0, 1.0, 2.0])
        obs, lo, hi, valid = P._boot_mean_pair(s, k, n_boot=200, seed=16)
        self.assertAlmostEqual(obs, s.sum() / k.sum())
        self.assertEqual(valid, 200)


class TestSelectionRule(unittest.TestCase):
    @staticmethod
    def _c(novel, gap, mac):
        return {"novel_rescue": novel, "gap": gap, "deploy_mac": mac}

    def test_higher_novel_wins(self):
        cand = {"Pair-2": self._c(0.20, 0.20, 6291456),
                "Pair-3": self._c(0.10, 0.30, 4128768)}
        self.assertEqual(P.select_pair(cand, ["Pair-2", "Pair-3"])[0], "Pair-2")

    def test_novel_tie_broken_by_gap(self):
        cand = {"Pair-2": self._c(0.200, 0.10, 6291456),
                "Pair-3": self._c(0.205, 0.30, 4128768)}
        sel, why = P.select_pair(cand, ["Pair-2", "Pair-3"])
        self.assertEqual(sel, "Pair-3")
        self.assertEqual(why, "novel_rescue>gap")

    def test_gap_tie_broken_by_lower_mac(self):
        cand = {"Pair-2": self._c(0.200, 0.300, 6291456),
                "Pair-3": self._c(0.205, 0.302, 4128768)}
        sel, why = P.select_pair(cand, ["Pair-2", "Pair-3"])
        self.assertEqual(sel, "Pair-3")
        self.assertIn("deploy_mac", why)

    def test_full_tie_lexicographic(self):
        cand = {"Pair-2": self._c(0.20, 0.30, 100),
                "Pair-3": self._c(0.20, 0.30, 100)}
        sel, why = P.select_pair(cand, ["Pair-2", "Pair-3"])
        self.assertEqual(sel, "Pair-2")
        self.assertIn("lexicographic", why)

    def test_single_candidate(self):
        cand = {"Pair-3": self._c(0.20, 0.30, 4128768)}
        self.assertEqual(P.select_pair(cand, ["Pair-3"]), ("Pair-3", "novel_rescue"))

    def test_val_screen_classification(self):
        # §5.3.2：样本数不足 / 超时 / 全部候选被初筛排除，必须区分（决定都是 NO_80K）
        self.assertEqual(P.classify_val_screen(True, False, []), "BOTH_SCREENED_OUT")
        self.assertEqual(P.classify_val_screen(False, False, []), "INSUFFICIENT_COUNTS")
        self.assertEqual(P.classify_val_screen(False, True, []), "VAL_TIMEOUT")
        self.assertEqual(P.classify_val_screen(True, False, ["Pair-2"]), "PASSED")

    def test_val_screen_thresholds_frozen(self):
        self.assertEqual(P.VAL_MIN_MISSED, 30)
        self.assertEqual(P.VAL_MIN_FP, 50)
        self.assertEqual(P.VAL_SCREEN_GAP_MIN, 0.05)
        self.assertEqual(P.VAL_SCREEN_NOVEL_MIN, 0.05)
        self.assertEqual(P.TIE_EPS, 0.01)

    def test_empty_eligible(self):
        self.assertEqual(P.select_pair({}, []), (None, None))


class TestConstants(unittest.TestCase):
    def test_gate_values_frozen(self):
        self.assertEqual(P.GATES["G4_rescue_rate_min"], 0.25)
        self.assertEqual(P.GATES["G5_gap_min"], 0.15)
        self.assertEqual(P.GATES["G5_gap_ci_low_min"], 0.03)
        self.assertEqual(P.GATES["G6_novel_rescue_min"], 0.10)
        self.assertEqual(P.GATES["G6_novel_ci_low_min"], 0.02)
        self.assertEqual(P.GATES["G7_margin_delta_min"], 0.02)
        self.assertEqual(P.MARGIN_THRESHOLD, 0.03)
        self.assertEqual(P.BUDGET_S, 600.0)
        self.assertEqual(P.EXPECT_N_TEST, 4000)

    def test_list_lock_uses_text_lines_hash(self):
        # §5.3 的 list 锁来自 Diag1 manifest 的 sha256_text_lines（非 raw bytes）；
        # 两者在同一文件上不同，写死 kind 以防实现被改成 raw sha256。
        self.assertEqual(P.SYSU_TEST_LIST_SHA_KIND, "sha256_text_lines")
        self.assertEqual(len(P.SYSU_TEST_LIST_SHA), 64)
        self.assertEqual(len(P.M1_SYSU_SHA), 64)

    def test_pair_specs_budget(self):
        for pn, spec in P.PAIR_SPECS.items():
            self.assertEqual(spec["hf_channels"], spec["contract_shape"][0])
            self.assertEqual(spec["hf_spatial"], spec["contract_shape"][1])
            self.assertEqual(spec["deploy_delta_params"],
                             96 * 2 * spec["hf_channels"])
            self.assertLess(4880190 + spec["deploy_delta_params"], 5000000)
        self.assertEqual(P.PAIR_SPECS["Pair-2"]["deploy_delta_params"], 6144)
        self.assertEqual(P.PAIR_SPECS["Pair-3"]["deploy_delta_params"], 16128)

    def test_small_alias_is_1_to_255(self):
        for a in (1, 15, 16, 63, 64, 255):
            self.assertIn(P.bin_index(a), P.SMALL_ALIAS)
        for a in (256, 1023, 1024):
            self.assertNotIn(P.bin_index(a), P.SMALL_ALIAS)

    def test_small_ap_hist_excludes_other_gt(self):
        gt = np.zeros((16, 16), bool)
        gt[0:2, 0:2] = True          # small 4px
        gt[8:16, 8:16] = True        # large 64px
        small = np.zeros((16, 16), bool)
        small[0:2, 0:2] = True
        score = np.zeros((16, 16))
        score[0:2, 0:2] = 1.0
        score[8:16, 8:16] = 1.0      # 其它 GT 对象像素不得进入样本
        hp, ha = P._small_ap_hist(score, small, gt)
        self.assertEqual(int(ha.sum()), int((small | ~gt).sum()))
        self.assertEqual(int(hp.sum()), 4)
        self.assertLessEqual(int(ha.sum()), int(score.size) - 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
