#!/usr/bin/env python
"""Run8b check_mapping.py — 变体定义三处同步 + 协议 + 抗自锁开关（照 Run8 同类工具改）。

必须一致的四处：
  1. `train_scripts/CASA-TViM/Run8b/_r8b_run_one_ds.sh::variant_args()`
  2. `train_scripts/CASA-TViM/Run8b/_r8b_pipeline_supervisor_single.sh::wave_variant()`
  3. `models/run8b_dry_run.py::VARIANTS`
  4. `train_scripts/CASA-TViM/Run8b/SELECTION_LOCK.json::variants[].name`

另附 Run8b 专属硬检查：
  5. `_r8b_run_one_ds.sh` 传给 train.py 的 `--flag` 必须都存在于 `train.py` 的 argparse
  6. 协议串必须等于 `train.py::PROTOCOL_VERSION`（即 `run8b_..._v2`）
  7. **`--hqkv_gamma_init` 必须出现在编排器里，且默认 > 0**（抗自锁的承重开关）
  8. `--arch` 必须是 `casa_vmamba_str_v2`
  9. Run8b **不得**写 Run8 的日志/ckpt 目录（防串味）
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RUN_DIR = os.path.join(ROOT, "train_scripts", "CASA-TViM", "Run8b")


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def parse_variant_args(txt):
    body = txt.split("variant_args()", 1)[1].split("read -r", 1)[0]
    out = {}
    for m in re.finditer(r"^\s*([A-Z0-9_]+)\)\s*echo\s+\"([^\"]+)\"\s*;;", body, re.M):
        out[m.group(1)] = tuple(m.group(2).split())
    return out


def parse_waves(txt):
    """解析督导里的 `wave_variant()` 分支。

    ⚠️ 两个坑（首版都踩了）：
    1. 变体名 `CTRL_R8B` 里含 `R8`，宽松正则回溯后会漏匹配 ⇒ 必须要求 `;;` 收尾；
    2. **多个分支写在同一行**（`1) ... ;; 2) ... ;; 3) ... ;; 4) ... ;;`），
       而 `^` 只匹配行首 ⇒ 只解析出第 1 个。改用逐行 `finditer`（不加 `^` 锚）。
    """
    body = txt.split("wave_variant()", 1)[1].split("esac", 1)[0]
    out = {}
    for line in body.splitlines():
        for m in re.finditer(r"(\d+)\)\s*echo\s+([A-Z0-9_]+)\s*;;", line):
            out[int(m.group(1))] = m.group(2)
    return out


def parse_dry(txt):
    body = txt.split("VARIANTS = {", 1)[1].split("\n}", 1)[0]
    out = {}
    for m in re.finditer(r'"([A-Z0-9_]+)":\s*dict\(([^)]*)\)', body):
        d = {}
        for km in re.finditer(r"(\w+)\s*=\s*(\"[^\"]*\"|\d+|True|False)", m.group(2)):
            d[km.group(1)] = km.group(2).strip('"')
        out[m.group(1)] = d
    return out


def main():
    problems = []
    run_one = read(os.path.join(RUN_DIR, "_r8b_run_one_ds.sh"))
    sup = read(os.path.join(RUN_DIR, "_r8b_pipeline_supervisor_single.sh"))
    dry = read(os.path.join(ROOT, "models", "run8b_dry_run.py"))
    lock = json.loads(read(os.path.join(RUN_DIR, "SELECTION_LOCK.json")))
    tr = read(os.path.join(ROOT, "models", "train.py"))

    v_args = parse_variant_args(run_one)
    v_waves = parse_waves(sup)
    v_dry = parse_dry(dry)
    v_lock = [x["name"] for x in lock["variants"]]

    print(f"[1] _r8b_run_one_ds variant_args : {sorted(v_args)}")
    print(f"[2] supervisor wave_variant      : {sorted(set(v_waves.values()))}")
    print(f"[3] run8b_dry_run VARIANTS       : {sorted(v_dry)}")
    print(f"[4] SELECTION_LOCK variants      : {sorted(v_lock)}")

    if not (set(v_args) == set(v_dry) == set(v_lock) == set(v_waves.values())):
        problems.append(f"变体名四处不一致: run_one={sorted(v_args)} waves={sorted(set(v_waves.values()))} "
                        f"dry={sorted(v_dry)} lock={sorted(v_lock)}")
    if sorted(v_waves) != list(range(1, len(v_args) + 1)):
        problems.append(f"波次编号不连续: {sorted(v_waves)}")

    order = ["HQ", "GEOM", "GS", "GSD", "INJ"]
    keymap = {"HQ": "hqkv", "GEOM": "geometry", "GS": "hgsl",
              "GSD": "hgsl_detach_score", "INJ": "inject_mode"}
    for v, tup in v_args.items():
        d = v_dry.get(v)
        if d is None:
            continue
        for i, k in enumerate(order):
            if str(tup[i]) != str(d.get(keymap[k])):
                problems.append(f"{v}.{keymap[k]}: run_one={tup[i]} dry={d.get(keymap[k])}")

    # [5] flags
    call = run_one.split("python train.py", 1)[1].split('>> "${LOGD}', 1)[0]
    train_args = set(re.findall(r"add_argument\('(--[a-zA-Z0-9_]+)'", tr))
    used = set(re.findall(r"(--[a-zA-Z0-9_]+)(?=[\s\"'$])", call))
    missing = sorted(x for x in used if x not in train_args)
    print(f"[5] 编排器用到 {len(used)} 个 --flag；train.py 未定义: {missing}")
    if missing:
        problems.append(f"train.py 缺这些参数: {missing}")

    # [6] protocol
    p_run = re.search(r"^PROTOCOL=(\S+)", run_one, re.M)
    p_tr = re.search(r'^PROTOCOL_VERSION = "([^"]+)"', tr, re.M)
    print(f"[6] run_one PROTOCOL={p_run.group(1) if p_run else None}  "
          f"train.py PROTOCOL_VERSION={p_tr.group(1) if p_tr else None}")
    if not p_run or not p_tr or p_run.group(1) != p_tr.group(1):
        problems.append("协议串不一致")
    if p_tr and not p_tr.group(1).startswith("run8b_"):
        problems.append(f"Run8b 的协议串应以 run8b_ 开头，实际 {p_tr.group(1)}")

    # [7] gamma_init 必须存在且默认 > 0
    g = re.search(r'GAMMA_INIT="\$\{GAMMA_INIT:-([0-9.]+)\}"', run_one)
    has_flag = "--hqkv_gamma_init" in call
    print(f"[7] GAMMA_INIT 默认={g.group(1) if g else None}  --hqkv_gamma_init 在调用里={has_flag}")
    if not g:
        problems.append("编排器缺 GAMMA_INIT 默认值")
    elif float(g.group(1)) <= 0:
        problems.append(f"GAMMA_INIT 默认必须 > 0（抗自锁），实际 {g.group(1)}")
    if not has_flag:
        problems.append("编排器没有把 --hqkv_gamma_init 传给 train.py ⇒ 抗自锁失效")

    # [8] arch
    a = re.search(r"--arch ([a-z0-9_]+)", call)
    print(f"[8] --arch {a.group(1) if a else None}")
    if not a or a.group(1) != "casa_vmamba_str_v2":
        problems.append(f"--arch 必须是 casa_vmamba_str_v2，实际 {a.group(1) if a else None}")

    # [9] 目录不得串味
    for pat in ("outputs/CASA-CD/CASA-TViM/Run8/", "checkpoints/CASA-CD/CASA-TViM/Run8/",
                "CASA-TViM/Run8/"):
        if pat in run_one or pat in sup:
            problems.append(f"Run8b 的脚本引用了 Run8 的路径 `{pat}` ⇒ 会串味")
    print("[9] 未引用 Run8 的日志/ckpt 路径" if not problems or
          not any("串味" in p for p in problems) else "[9] ⚠️ 有串味引用")

    print()
    if problems:
        print("check_mapping: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print(f"check_mapping: PASS ({len(v_args)} variants, {len(v_waves)} waves, "
          f"{len(used)} flags, protocol={p_run.group(1)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
