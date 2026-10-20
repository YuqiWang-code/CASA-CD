#!/usr/bin/env python
"""Run9 早停判定：比较 S0_CTRL 与 S1_HSAM 在已封口数据集上的 ΔF1。

**预注册规则**（写死在 README §6，不由本脚本决定）：
前两个数据集（CDD、LEVIR）两边都封口后，
  * 若两个库上 `ΔF1 = HSAM - CTRL` **都 ≤ 0**  ⇒ 输出 `STOP`
  * 否则                                        ⇒ 输出 `CONTINUE`

输出一行，形如：
    CDD=+0.0012 LEVIR=-0.0003 => CONTINUE
    CDD=-0.0008 LEVIR=-0.0011 => STOP

只读日志，不训练。解析口径与 `_r9_state.sh` 一致：取**最后一个完整** TEST 区块。
"""
import os
import re
import sys

LOG_ROOT = "/home/yqwang/outputs/CASA-CD/CASA-TViM/Run9"
VARIANTS = ("S0_CTRL", "S1_HSAM")
GATE_DATASETS = ("CDD-CD-256", "LEVIR-CD-256")

F1_RE = re.compile(r"F1=([0-9]*\.?[0-9]+)")


def last_test_f1(path):
    """返回最后一个完整 TEST 区块里的 F1；没有就返回 None。"""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    if "=== END TEST RESULTS ===" not in txt:
        return None
    blocks, cur = [], None
    for line in txt.splitlines():
        if line.strip() == "=== TEST RESULTS ===":
            cur = []
        elif line.strip() == "=== END TEST RESULTS ===":
            if cur is not None:
                blocks.append(cur)
            cur = None
        elif cur is not None:
            cur.append(line)
    if not blocks:
        return None
    for line in reversed(blocks[-1]):
        if line.startswith("Recall=") or "F1=" in line:
            m = F1_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def main():
    vals = {}
    for ds in GATE_DATASETS:
        short = ds.split("-")[0]
        per = {}
        for v in VARIANTS:
            per[v] = last_test_f1(os.path.join(LOG_ROOT, v, ds, "train_log.txt"))
        if per["S0_CTRL"] is None or per["S1_HSAM"] is None:
            print("INCOMPLETE（前两库尚未两边封口）")
            return 0
        vals[short] = per["S1_HSAM"] - per["S0_CTRL"]
    parts = " ".join(f"{k}={v:+.4f}" for k, v in vals.items())
    stop = all(v <= 0.0 for v in vals.values())
    print(f"{parts} => {'STOP' if stop else 'CONTINUE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
