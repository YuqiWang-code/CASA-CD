#!/usr/bin/env python
"""Run7 一致性自检（本地可跑，不需 GPU/数据）。

校验 `variant → (hf_edge, hf_edge_form)` 与 `wave → variant` 在**四个来源**中完全一致：
`run_all.sh`（真正下发 CLI 的地方）、`models/_legacy/run7_dry_run.py`、`Run7/SELECTION_LOCK.json`、
`Run7/README.md`。

存在的理由：波二/三各要烧 ~4.5 h GPU 时间。若 runner 把 E9/E10 的 flag 传错、或把 CTRL 的
`--hf_edge` 传成 1，会**静默训出"错误标签的实验组"**，事后无法从目录名分辨，整轮消融作废。
**开每一波之前都应先跑本脚本**（`run_all.sh` 也在启动时校验 variant 名合法性）。

    python train_scripts/CASA-TViM/Run7/check_mapping.py
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
RUNNER = os.path.join(ROOT, "train_scripts", "CASA-TViM", "Run7", "run_all.sh")
DRYRUN = os.path.join(ROOT, "models", "run7_dry_run.py")
LOCK = os.path.join(ROOT, "train_scripts", "CASA-TViM", "Run7", "SELECTION_LOCK.json")
README = os.path.join(ROOT, "train_scripts", "CASA-TViM", "Run7", "README.md")

# 预注册冻结的映射（改这里等于改实验定义，必须同时改 SELECTION_LOCK 与预注册文档）
FROZEN = {"M1_R7CTRL": (0, "none"), "E9_HFAE_A": (1, "aux_only"),
          "E10_HFAE_B": (1, "aux_inject")}
FROZEN_WAVES = {1: "M1_R7CTRL", 2: "E9_HFAE_A", 3: "E10_HFAE_B"}


def parse_runner(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"variant_args\(\)\s*\{(.*?)\n\}", txt, re.S)
    out = {}
    for line in m.group(1).splitlines():
        mm = re.match(r'\s*([A-Za-z0-9_]+)\)\s*echo\s+"(\d+)\s+(\S+)"', line)
        if mm:
            out[mm.group(1)] = (int(mm.group(2)), mm.group(3))
    wv = {}
    m2 = re.search(r"wave_variant\(\)\s*\{(.*?)\n\}", txt, re.S)
    for line in m2.group(1).splitlines():
        mm = re.match(r"\s*(\d)\)\s*echo\s+(\S+)", line)
        if mm:
            wv[int(mm.group(1))] = mm.group(2)
    return out, wv


def parse_dryrun(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"VARIANTS\s*=\s*\{(.*?)\n\}", txt, re.S)
    return {mm.group(1): (int(mm.group(2)), mm.group(3))
            for mm in re.finditer(r'"([A-Za-z0-9_]+)":\s*\((\d+),\s*"(\w+)"\)', m.group(1))}


def main():
    runner_map, wave_map = parse_runner(RUNNER)
    dry_map = parse_dryrun(DRYRUN)
    lock = json.load(open(LOCK, encoding="utf-8"))
    lock_map = {v["name"]: (int(v["hf_edge"]), v["hf_edge_form"]) for v in lock["variants"]}
    readme = open(README, encoding="utf-8").read()
    runner_txt = open(RUNNER, encoding="utf-8").read()

    checks = {
        "runner_matches_frozen": runner_map == FROZEN,
        "dryrun_matches_frozen": dry_map == FROZEN,
        "lock_matches_frozen": lock_map == FROZEN,
        "runner_equals_dryrun": runner_map == dry_map,
        "runner_equals_lock": runner_map == lock_map,
        "wave_map_correct": wave_map == FROZEN_WAVES,
        # flags 必须由 variant_args 变量下发，不能写死（否则三波会跑成同一个形态）
        "runner_passes_flags_by_var": '--hf_edge "${HF}" --hf_edge_form "${FORM}"' in runner_txt,
        # 协议串与 λ 必须与冻结值一致
        "runner_protocol_frozen": "PROTOCOL=run7_hf_edge_exact80k_v1" in runner_txt,
        "runner_lambda_frozen": re.search(r"^LAMBDA=0\.2$", runner_txt, re.M) is not None,
        "runner_steps_frozen": re.search(r"^MAX_STEPS=80000$", runner_txt, re.M) is not None,
        "runner_batch_frozen": re.search(r"^BATCH=32$", runner_txt, re.M) is not None,
    }
    for v, (hf, form) in FROZEN.items():
        pat = re.compile(re.escape(v) + r".{0,80}?" + str(hf) + r".{0,80}?" + re.escape(form), re.S)
        checks[f"readme_row_{v}"] = bool(pat.search(readme))

    print("=== Run7 mapping / protocol consistency ===")
    print("runner :", runner_map)
    print("dryrun :", dry_map)
    print("lock   :", lock_map)
    print("waves  :", wave_map)
    bad = 0
    for k in sorted(checks):
        print(f"  [{'OK' if checks[k] else 'FAIL'}] {k}")
        bad += 0 if checks[k] else 1
    print(f"MAPPING CHECK = {'PASS' if bad == 0 else f'FAIL ({bad})'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
