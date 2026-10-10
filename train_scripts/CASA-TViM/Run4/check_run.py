#!/usr/bin/env python
"""Run4 单 run 收尾硬门（方案 §8.4）：只从日志**最后一个完整 TEST 区块**判定，不看 python rc。

用法（由 train_{dataset}.sh 自动调用）：

    python check_run.py --log <train_log.txt> --variant E6_FET1 --dataset SYSU-CD-256 \
        --ckpt-dir <ckpt_dir>

硬门（任一不满足 ⇒ 退出码 1，run 记 INVALID）：
  * 存在最后一个完整 `=== TEST RESULTS ===` … `=== END TEST RESULTS ===` 区块；
  * 区块内 `[ACTUAL-OPT-STEPS] 80000`（exact-80K 协议）；
  * `[REPARAM-ARGMAX-DISAGREE]` 与 `[REPARAM-REAL-ARGMAX-DISAGREE]` 均为 0（折叠二值分歧硬门）；
  * `[DEPLOY-PARAMS] total=…` ≤ 5M；
  * `E6_FET1`：区块含 `[FINE-TAP] 1`、`[FET-GAMMA]`、`[FET-FOLD] conv1x1 [96, 144, 1, 1]`；
    `M1_R4CTRL`：区块含 `[FINE-TAP] 0` 且不含任何 `[FET-` 行；
  * 日志中不存在 `[MANIFEST-MISMATCH]` / `[PROTOCOL-MISMATCH]` / `[EXACT-STEPS] budget already reached`
    （后者说明启动时预算已被耗尽 —— 除认证续跑外视为异常）；
  * `run_manifest.json` 与 `arch.json` 的 `fine_tap` / `protocol_version` 与 logs 一致。
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "..", "..", "..", "analyse")))
from tvim_diag_common import parse_test_block  # noqa: E402

EXPECT_STEPS = 80000
BUDGET = 5_000_000


def _get(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    args = ap.parse_args()

    checks = []
    log_text = ""
    if os.path.isfile(args.log):
        with open(args.log, "r", encoding="utf-8", errors="replace") as f:
            log_text = f.read()

    block, info, span = parse_test_block(args.log)
    checks.append(("last complete TEST block present", block is not None,
                   f"span={span}"))

    if block is None:
        _report(args, checks)
        return 1

    steps = _get(r"\[ACTUAL-OPT-STEPS\]\s*(\d+)", block, int)
    checks.append((f"[ACTUAL-OPT-STEPS] == {EXPECT_STEPS}", steps == EXPECT_STEPS, f"got {steps}"))

    d1 = _get(r"\[REPARAM-ARGMAX-DISAGREE\]\s*([0-9.eE+-]+)", block)
    d2 = _get(r"\[REPARAM-REAL-ARGMAX-DISAGREE\]\s*([0-9.eE+-]+)", block)
    checks.append(("fold binary disagreement == 0 (random batch)", d1 == 0.0, f"got {d1}"))
    checks.append(("fold binary disagreement == 0 (real batch)", d2 == 0.0, f"got {d2}"))

    depl = _get(r"\[DEPLOY-PARAMS\]\s*total=([0-9.]+)\s*M", block)
    checks.append((f"deploy params <= {BUDGET}", depl is not None and depl <= BUDGET / 1e6,
                   f"got {depl} M"))

    fine = _get(r"\[FINE-TAP\]\s*(\d+)", block, int)
    if args.variant == "E6_FET1":
        checks.append(("[FINE-TAP] == 1", fine == 1, f"got {fine}"))
        checks.append(("[FET-GAMMA] present", "[FET-GAMMA]" in block,
                       "gamma logged before folding"))
        checks.append(("[FET-FOLD] conv1x1 [96, 144, 1, 1]",
                       bool(re.search(r"\[FET-FOLD\]\s*conv1x1\s*\[\s*96\s*,\s*144\s*,\s*1\s*,\s*1\s*\]", block)),
                       "folded kernel shape"))
        checks.append(("[FET-FOLD] params == 13,920",
                       _get(r"\[FET-FOLD\][^\n]*params=(\d+)", block, int) == 13920,
                       "deploy increment"))
    else:
        checks.append(("[FINE-TAP] == 0", fine == 0, f"got {fine}"))
        checks.append(("no [FET- line in the CTRL TEST block", "[FET-" not in block, "ctr1 has no FET"))

    for tag in ("[MANIFEST-MISMATCH]", "[PROTOCOL-MISMATCH]"):
        checks.append((f"log has no {tag}", tag not in log_text, ""))
    checks.append(("[EXACT-STEPS] budget not pre-exhausted at launch",
                   "budget already reached" not in log_text, ""))

    # sidecar 一致性
    for name, keys in (("arch.json", ("fine_tap", "protocol_version")),
                       ("run_manifest.json", ("fine_tap", "protocol_version"))):
        p = os.path.join(args.ckpt_dir, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                obj = json.load(f)
            exp_fine = 1 if args.variant == "E6_FET1" else 0
            checks.append((f"{name} fine_tap == {exp_fine}", obj.get("fine_tap") == exp_fine,
                           f"got {obj.get('fine_tap')}"))
            checks.append((f"{name} protocol_version == run4_exact80k_v1",
                           obj.get("protocol_version") == "run4_exact80k_v1",
                           f"got {obj.get('protocol_version')}"))
        else:
            checks.append((f"{name} present", False, p))

    _report(args, checks)
    return 0 if all(c[1] for c in checks) else 1


def _report(args, checks):
    print(f"\n=== RUN4 CHECK_RUN {args.variant} / {args.dataset} ===")
    for name, ok, extra in checks:
        print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {extra}" if extra else ""))
    bad = [c[0] for c in checks if not c[1]]
    print(f"--- {'ALL PASS' if not bad else 'FAILED: ' + '; '.join(bad)} ---")


if __name__ == "__main__":
    sys.exit(main())
