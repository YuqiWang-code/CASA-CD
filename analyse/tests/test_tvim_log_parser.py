"""诊断日志解析器一致性测试：本模块 `_extract_test_block`/`_parse_block` 必须与
`analyse/extract_metrics_to_excel.py` 的官方实现逐字一致（覆盖真实 train_log）。

说明：官方模块 import openpyxl（服务器 casacd 未安装），因此诊断侧复制了这两个纯正则函数；
本测试在 openpyxl 可用时直接对拍，不可用时自动 skip。

运行：python -m unittest discover -s analyse/tests -p 'test_tvim_*.py' -v
"""
import glob
import os
import sys
import unittest

_ANALYSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ANALYSE_DIR not in sys.path:
    sys.path.insert(0, _ANALYSE_DIR)
_ROOT = os.path.dirname(_ANALYSE_DIR)
_MODELS_DIR = os.path.join(_ROOT, "models")
if os.path.isdir(_MODELS_DIR) and _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from tvim_diag_common import _extract_test_block, _parse_block, parse_test_block  # noqa: E402

try:
    from extract_metrics_to_excel import extract_test_block as off_extract, parse_block as off_parse
    HAS_OFFICIAL = True
except Exception:
    HAS_OFFICIAL = False

DEFAULT_LOGS = [
    "outputs/CASA-TViM/Run1/M1_FULL/SYSU-CD-256/train_log.txt",
    "outputs/CASA-TViM/Run1/A2_STR/SYSU-CD-256/train_log.txt",
    "outputs/CASA-TViM/Run3/E5_FS_TAR/SYSU-CD-256/train_log.txt",
]


class TestLogParserEquivalence(unittest.TestCase):
    def _logs(self):
        found = [os.path.join(_ROOT, p) for p in DEFAULT_LOGS if os.path.isfile(os.path.join(_ROOT, p))]
        if not found:
            found = sorted(glob.glob(os.path.join(_ROOT, "outputs", "**", "train_log.txt"), recursive=True))[:3]
        return found

    @unittest.skipUnless(HAS_OFFICIAL, "official parser needs openpyxl")
    def test_matches_official_on_real_logs(self):
        logs = self._logs()
        self.assertTrue(logs, "no train_log.txt available to compare")
        n_checked = 0
        for log in logs:
            with open(log, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            b_mine = _extract_test_block(text)
            b_off = off_extract(text)
            self.assertEqual(b_mine, b_off, f"block differs for {log}")
            if b_mine is None:
                continue
            self.assertEqual(_parse_block(b_mine), off_parse(b_off), f"parsed dict differs for {log}")
            n_checked += 1
        self.assertGreater(n_checked, 0, "no log contained a TEST RESULTS block")

    def test_parse_test_block_extra_fields(self):
        logs = self._logs()
        if not logs:
            self.skipTest("no logs available")
        block, info, span = parse_test_block(logs[0])
        self.assertIsNotNone(block)
        self.assertIn("F1", info)
        self.assertIsNotNone(span)
        lo, hi = span
        self.assertLess(lo, hi)


if __name__ == "__main__":
    unittest.main(verbosity=2)
