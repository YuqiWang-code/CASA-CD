#!/usr/bin/env python
"""Run8 check_mapping.py — 变体定义在**三处**是否同步（照 Run7 的同类工具结构）。

必须一致的三处：
  1. `train_scripts/CASA-TViM/Run8/run_all.sh` 的 `variant_args()` / `wave_variant()`
  2. `models/run8_dry_run.py` 的 `VARIANTS`
  3. `train_scripts/CASA-TViM/Run8/SELECTION_LOCK.json` 的 `variants[].name`

另附两条 Run8 专属一致性检查：
  4. `run_all.sh` 里传给 `train.py` 的参数名必须存在于 `train.py` 的 argparse 中（防拼写漂移）
  5. `run_all.sh` 的 `PROTOCOL` 必须等于 `train.py` 的 `PROTOCOL_VERSION`
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RUN_DIR = os.path.join(ROOT, "train_scripts", "CASA-TViM", "Run8")


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def parse_run_all(txt):
    """返回 (variant -> tuple, wave -> variant, protocol)。"""
    body = txt.split("variant_args()", 1)[1]
    variants = {}
    for m in re.finditer(r"^\s*([A-Z0-9_]+)\)\s*echo\s+\"([^\"]+)\"\s*;;", body, re.M):
        variants[m.group(1)] = tuple(m.group(2).split())
    wbody = txt.split("wave_variant()", 1)[1]
    waves = {}
    for m in re.finditer(r"^\s*(\d+)\)\s*echo\s+([A-Z0-9_]+)\s*;;", wbody, re.M):
        waves[int(m.group(1))] = m.group(2)
    proto = re.search(r"^PROTOCOL=(\S+)", txt, re.M)
    return variants, waves, (proto.group(1) if proto else None)


def parse_dry_run(txt):
    body = txt.split("VARIANTS = {", 1)[1].split("\n}", 1)[0]
    out = {}
    for m in re.finditer(r'"([A-Z0-9_]+)":\s*dict\(([^)]*)\)', body):
        name, kw = m.group(1), m.group(2)
        d = {}
        for km in re.finditer(r"(\w+)\s*=\s*(\"[^\"]*\"|\d+|True|False)", kw):
            k, v = km.group(1), km.group(2)
            d[k] = v.strip('"')
        out[name] = d
    return out


def main():
    problems = []
    run_all = read(os.path.join(RUN_DIR, "run_all.sh"))
    dry = read(os.path.join(ROOT, "models", "run8_dry_run.py"))
    lock = json.loads(read(os.path.join(RUN_DIR, "SELECTION_LOCK.json")))
    tr = read(os.path.join(ROOT, "models", "train.py"))

    v_args, waves, proto = parse_run_all(run_all)
    v_dry = parse_dry_run(dry)
    v_lock = [v["name"] for v in lock["variants"]]

    print(f"[1] run_all.sh variant_args : {sorted(v_args)}")
    print(f"[2] run8_dry_run VARIANTS   : {sorted(v_dry)}")
    print(f"[3] SELECTION_LOCK variants : {sorted(v_lock)}")
    if set(v_args) != set(v_dry) or set(v_args) != set(v_lock):
        problems.append(f"变体名三处不一致: run_all={sorted(v_args)} dry={sorted(v_dry)} lock={sorted(v_lock)}")
    if set(waves.values()) != set(v_args):
        problems.append(f"wave_variant 覆盖不全: {waves} vs {sorted(v_args)}")
    if sorted(waves) != list(range(1, len(v_args) + 1)):
        problems.append(f"波次编号不连续: {sorted(waves)}")

    # 字段级对拍（hqkv / geometry / hgsl / hgsl_detach_score / inject_mode）
    keymap = {"HQ": "hqkv", "GEOM": "geometry", "GS": "hgsl",
              "GSD": "hgsl_detach_score", "INJ": "inject_mode"}
    order = ["HQ", "GEOM", "GS", "GSD", "INJ"]
    for v, tup in v_args.items():
        d = v_dry.get(v)
        if d is None:
            continue
        for i, k in enumerate(order):
            a, b = tup[i], d.get(keymap[k])
            if str(a) != str(b):
                problems.append(f"{v}.{keymap[k]}: run_all={a} dry={b}")

    # [4] train.py 参数名存在性（只扫真正的 train.py 调用块，排除注释与 run_all.sh 自己的 --wave）
    call_block = run_all.split("python train.py", 1)[1].split('>> "${logd}', 1)[0]
    train_args = set(re.findall(r"add_argument\('(--[a-zA-Z0-9_]+)'", tr))
    used = set(re.findall(r"(--[a-zA-Z0-9_]+)(?=[\s\"'$])", call_block))
    missing = sorted(x for x in used if x not in train_args)
    print(f"[4] run_all.sh 用到 {len(used)} 个 --flag；train.py 未定义: {missing}")
    if missing:
        problems.append(f"train.py 缺这些参数: {missing}")

    # [5] 协议版本
    tp = re.search(r'^PROTOCOL_VERSION = "([^"]+)"', tr, re.M)
    tp = tp.group(1) if tp else None
    print(f"[5] run_all PROTOCOL={proto}  train.py PROTOCOL_VERSION={tp}")
    if proto != tp:
        problems.append(f"协议版本不一致: {proto} vs {tp}")

    # [6] lite_dim 必须等于 str_dim
    sd = re.search(r"^STR_DIM=(\d+)", run_all, re.M)
    ld = re.search(r"--hqkv_lite_dim \"\$\{([A-Z_]+)\}\"", run_all)
    print(f"[6] STR_DIM={sd.group(1) if sd else '?'} lite_dim 引用={ld.group(1) if ld else '?'}")
    if sd and ld and ld.group(1) != "STR_DIM":
        problems.append("hqkv_lite_dim 未引用 STR_DIM（会导致 stage 输出与 str_dim 不匹配）")

    print()
    if problems:
        print("check_mapping: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print(f"check_mapping: PASS ({len(v_args)} variants, {len(waves)} waves, "
          f"{len(used)} flags, protocol={proto})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
