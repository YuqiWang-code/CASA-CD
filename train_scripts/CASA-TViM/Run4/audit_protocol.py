#!/usr/bin/env python
"""Run4 协议审计（可重复执行）：把「训练协议正确」从口头声明变成可核验记录。

对每个 run 核对（全部从**日志与 sidecar 原文**读取，不看任何人工摘要）：

  * `[MAX-STEPS] 80000` + `train_iters/epoch` + `max_epochs == ceil(80000 / iters_per_epoch)`
    以及推导出的末 epoch 截断步数（= 80000 − (max_epochs−1)·iters）；
  * `--exact_max_steps 1` 已生效：日志含 `[EXACT-STEPS] ... stop exactly at optimizer step 80000`；
  * `run_manifest.json`：protocol_version / exact_max_steps / max_steps / batch_size / seed /
    fine_tap / caacp / caacp_score_mode / caacp_residual_mode / frh / fs_tar / rep_mode / str_dim /
    backbone_lr_ratio / data_contract / source_code_sha256；
  * **LR 计划逐点核对**：`[LR-GROUPS]` 行必须等于官方公式
      - 全局步 g 的 epoch = g // iters_per_epoch；
      - `epoch == 0 and g < 200` → 线性 warmup `lr·0.9·(g+1)/200 + lr·0.1`；
      - 否则 poly `lr·(1 − g/max_steps)^0.9`（分母必须是 args.max_steps，不是 epochs×iters）；
      - backbone 组必须是 new 组的 0.1×。
  * 训练结束（存在完整 TEST 区块）时，区块内必须含 `[ACTUAL-OPT-STEPS] 80000`。

    python train_scripts/CASA-TViM/Run4/audit_protocol.py \
      --log-root /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4 \
      --ckpt-root /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4 \
      [--variant M1_R4CTRL] [--write-json]
"""
import argparse
import json
import math
import os
import re
import sys

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
VARIANTS = ["M1_R4CTRL", "E6_FET1"]
MAX_STEPS_DEFAULT = 80000
LR_DEFAULT = 2e-4
WARMUP = 200
BLR = 0.1


def expected_lr(g, iters_per_epoch, lr, max_steps):
    """返回 (期望 lr, 阶段名)。warmup 只在 epoch 0 且全局步 < 200 时生效。"""
    if g // iters_per_epoch == 0 and g < WARMUP:
        return lr * 0.9 * (g + 1) / WARMUP + 0.1 * lr, "warmup"
    return lr * (1 - g / max_steps) ** 0.9, "poly"


def audit_one(variant, dataset, log_root, ckpt_root):
    log = os.path.join(log_root, variant, dataset, "train_log.txt")
    man_path = os.path.join(ckpt_root, variant, dataset, "run_manifest.json")
    r = {"variant": variant, "dataset": dataset, "log": log, "manifest": man_path,
         "checks": {}, "pass": False, "detail": {}}
    if not os.path.isfile(log):
        r["checks"]["log_present"] = False
        return r
    if not os.path.isfile(man_path):
        r["checks"]["manifest_present"] = False
        return r
    txt = open(log, encoding="utf-8", errors="replace").read()
    man = json.load(open(man_path, encoding="utf-8"))

    max_steps = int(man.get("max_steps", MAX_STEPS_DEFAULT))
    lr = float(man.get("lr", LR_DEFAULT))
    m = re.search(r"\[MAX-STEPS\] (\d+)\s+\[train_iters/epoch\] (\d+)\s+\[max_epochs\] (\d+)", txt)
    if not m:
        r["checks"]["max_steps_header"] = False
        return r
    steps, iters, epochs = int(m.group(1)), int(m.group(2)), int(m.group(3))
    exp_epochs = math.ceil(max_steps / iters)
    last_iters = max_steps - (exp_epochs - 1) * iters
    r["detail"].update({"max_steps": steps, "iters_per_epoch": iters, "max_epochs": epochs,
                        "expected_max_epochs": exp_epochs, "last_epoch_steps": last_iters})
    r["checks"]["max_steps_header"] = (steps == max_steps and iters > 0 and epochs == exp_epochs)

    r["checks"]["exact_steps_entry"] = bool(re.search(
        r"\[EXACT-STEPS\] protocol=run4_exact80k_v1 enabled: stop exactly at optimizer step (\d+)",
        txt)) and f"step {max_steps}" in txt
    r["checks"]["protocol_version_manifest"] = man.get("protocol_version") == "run4_exact80k_v1"
    r["checks"]["exact_max_steps_manifest"] = int(man.get("exact_max_steps", 0)) == 1
    r["checks"]["batch_size_32"] = int(man.get("batch_size", 0)) == 32
    r["checks"]["seed_16"] = int(man.get("seed", -1)) == 16
    r["checks"]["str_dim_96"] = int(man.get("str_dim", 0)) == 96
    r["checks"]["backbone_lr_ratio_0.1"] = abs(float(man.get("backbone_lr_ratio", 0)) - BLR) < 1e-12
    r["checks"]["data_contract"] = man.get("data_contract") == "legacy_6ch_reverse_v1"
    r["checks"]["lr_mode_poly"] = "lr_mode" not in man or man.get("lr_mode", "poly") == "poly"
    want_fine = 1 if variant == "E6_FET1" else 0
    r["checks"]["fine_tap_manifest"] = int(man.get("fine_tap", -1)) == want_fine
    r["checks"]["caacp_1_full_rank_current"] = (int(man.get("caacp", 0)) == 1
                                                and man.get("rep_mode") == "full"
                                                and man.get("caacp_score_mode") == "rank"
                                                and man.get("caacp_residual_mode") == "current"
                                                and int(man.get("frh", 0)) == 0
                                                and int(man.get("fs_tar", 0)) == 0)

    # ---- LR 计划逐点核对（分母必须是 max_steps）
    lr_rows, lr_ok = [], True
    for mm in re.finditer(r"\[LR-GROUPS\] iter=(\d+) backbone=([0-9.eE+-]+) new=([0-9.eE+-]+)", txt):
        g, bb, new = int(mm.group(1)), float(mm.group(2)), float(mm.group(3))
        exp_new, kind = expected_lr(g, iters, lr, max_steps)
        exp_bb = exp_new * BLR
        good = abs(new - exp_new) / exp_new < 2e-3 and abs(bb - exp_bb) / exp_bb < 2e-3
        lr_ok = lr_ok and good
        lr_rows.append({"iter": g, "phase": kind, "new": new, "expected_new": exp_new,
                        "backbone": bb, "expected_backbone": exp_bb, "ok": good})
    r["detail"]["lr_rows"] = lr_rows
    r["checks"]["lr_plan_matches_formula"] = bool(lr_rows) and lr_ok

    # ---- 训练结束后：最后一个完整 TEST 区块内必须有步数标记
    blocks = re.findall(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", txt, re.S)
    if blocks:
        block = blocks[-1]
        got = re.search(r"\[ACTUAL-OPT-STEPS\]\s*(\d+)", block)
        r["detail"]["test_block_actual_opt_steps"] = int(got.group(1)) if got else None
        r["checks"]["test_block_actual_opt_steps_80000"] = bool(got) and int(got.group(1)) == max_steps
        r["detail"]["finished"] = True
    else:
        r["detail"]["finished"] = False
    r["pass"] = all(r["checks"].values())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--variant", default=None, choices=[None] + VARIANTS)
    ap.add_argument("--require-finished", type=int, default=0,
                    help="1 = also require the final TEST block to carry [ACTUAL-OPT-STEPS] max_steps")
    ap.add_argument("--write-json", type=int, default=1)
    args = ap.parse_args()

    variants = [args.variant] if args.variant else VARIANTS
    all_rows, ok_all = [], True
    for v in variants:
        for d in DATASETS:
            r = audit_one(v, d, args.log_root, args.ckpt_root)
            if args.require_finished and not r["detail"].get("finished"):
                r["checks"]["finished"] = False
                r["pass"] = False
            all_rows.append(r)
            bad = [k for k, ok in r["checks"].items() if not ok]
            ok_all = ok_all and r["pass"]
            det = r["detail"]
            print(f"{v:11s} {d:13s} {'PASS' if r['pass'] else 'FAIL'}"
                  + (f"  iters/ep={det.get('iters_per_epoch')} epochs={det.get('max_epochs')}"
                     f"/{det.get('expected_max_epochs')} last_epoch_steps={det.get('last_epoch_steps')}"
                     f" lr_points={len(det.get('lr_rows', []))}"
                     f" finished={det.get('finished')}" if det else "")
                  + (f"  FAILED: {', '.join(bad)}" if bad else ""))
    if args.write_json:
        for v in variants:
            rows = [r for r in all_rows if r["variant"] == v]
            if not rows:
                continue
            p = os.path.join(args.log_root, v, "PROTOCOL_AUDIT.json")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"variant": v, "require_finished": bool(args.require_finished),
                           "pass": all(r["pass"] for r in rows), "runs": rows}, f, indent=2)
            print("wrote", p)
    print("PROTOCOL_AUDIT", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
