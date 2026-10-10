"""CASA-TViM Run4 结果报告与三级裁决（方案 §6.3/§6.4/§8.5）。

两种模式：

1) 对象指标 pass（每个 run 一次，部署图 fold 之后、原生 256²、4 连通、p>0.5）：

       python analyse/run4_fet_report.py --mode object \
         --variant E6_FET1 --dataset SYSU-CD-256 --run Run4 \
         --ckpt-root /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM \
         --out-dir /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/objects

2) 裁决（读全部 8 个 train_log.txt 的**最后一个完整 TEST 区块** + 上一步对象 JSON）：

       python analyse/run4_fet_report.py --mode verdict \
         --log-root /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4 \
         --ckpt-root /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM \
         --object-dir /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/objects \
         --out /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/verdict.json

裁决层级（严格按预注册）：
  * 层 A P0 有效性：deploy ≤5M、train/deploy 双 batch 二值 disagreement=0、TEST 区块完整、
    `[ACTUAL-OPT-STEPS] 80000`、6 指标/FLOPs 齐全；任一失败 ⇒ INVALID；
  * 层 B 机制（SYSU 主判）：small(1–255px, 4 连通) pooled Recall 与 Hit@25 同时满足**绝对门槛**
    （≥0.2474 / ≥0.2396）**与同期增量门槛**（≥CTRL+0.05）；ObjRecall ≥CTRL+0.01；
    ObjPrecision 不低于 CTRL；<256px 未匹配预测不多于 CTRL；F1 ≥ CTRL；
  * 层 C 跨库守门 + 论文硬目标：CDD ≥97.18、LEVIR ≥91.07、SYSU ≥83.47、WHU ≥95.07 且不低于同期 CTRL；
    四库同时 ≥98.00/92.50/85.00/95.00 才是 PAPER-TARGET-PASS。

失败决策树（§6.4）直接编码为 `decision` 字段，禁止失败后继续堆模块。
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvim_diag_common import (  # noqa: E402
    DATA_ROOT, build_model, ensure_dir, make_loader, parse_test_block, read_list_names,
    set_eval_numerics, sha256_file, write_json,
)
from tvim_object_metrics import ObjectMetricAccumulator  # noqa: E402

VARIANTS = ["M1_R4CTRL", "E6_FET1"]
DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
SYSU = "SYSU-CD-256"

# ---------------- 预注册门槛（§6.3） ----------------
MECH_ABS = {"small_pooled_recall": 0.2474, "small_hit25": 0.2396}
MECH_DELTA = {"small_pooled_recall": 0.0500, "small_hit25": 0.0500,
              "ObjRecall_loose": 0.0100}
HIST_M1_SYSU = {"small_pooled_recall": 0.1974, "small_hit25": 0.1896,
                "ObjRecall_loose": 0.7674, "ObjPrecision_loose": 0.5320,
                "unmatched_pred_lt256": 2430, "F1": 83.47, "IoU": 71.63}
GUARDRAIL = {"CDD-CD-256": 97.18, "LEVIR-CD-256": 91.07, "SYSU-CD-256": 83.47, "WHU-CD-256": 95.07}
PAPER_TARGET = {"CDD-CD-256": 98.00, "LEVIR-CD-256": 92.50, "SYSU-CD-256": 85.00, "WHU-CD-256": 95.00}
BUDGET = 5_000_000


def _pct(x):
    """TEST 区块里的 F1/IoU 是比例值（0.8316 = 83.16%）；门槛表用百分数，统一换算并显式校验。"""
    if x is None:
        return None
    if x > 1.5:
        raise SystemExit(f"[REPORT] metric {x} looks like a percentage; expected a proportional "
                         "value in the TEST block (F1=0.8316 means 83.16%) -- refusing to compare")
    return x * 100.0


# ======================================================================================
# 模式 1：对象指标 pass（部署图）
# ======================================================================================
def build_run_model(variant, ckpt_path, device, ckpt_root):
    """按 sidecar 严格构造 Run4 变体并严格加载 train 图权重，然后折叠为部署图。

    返回 (model, deploy_info)：deploy_info 含**从 checkpoint 独立实测**的部署参数量
    （T10 的日志外证据：E6=4,894,110 / CTRL=4,880,190）与 FET 折叠后的卷积形状。
    """
    import torch
    from model.casa_tvim_str_net import CASATViMSTRNet

    expected = 4_894_110 if variant == "E6_FET1" else 4_880_190
    fine_tap = 1 if variant == "E6_FET1" else 0
    model = CASATViMSTRNet(
        os.path.join(os.environ.get("CASA_PROJECT", "/home/yqwang/projects/CASA-CD"),
                     "pretrained_weight", "tinyvim_s_1000e.pth"),
        caacp=True, rep_mode="full", str_dim=96, caacp_score_mode="rank", frh=False,
        caacp_residual_mode="current", fs_tar=False, fine_tap=bool(fine_tap)).float().to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    res = model.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    # 折叠前 = 训练图（§8.5 要求报告「训练图总/可训练参数」，此处从 checkpoint 独立实测）
    train_params = int(sum(p.numel() for p in model.parameters()))
    train_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.switch_to_deploy()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    deploy_info = {
        "train_params": train_params,
        "train_trainable": train_trainable,
        "expected_train": 5_097_938 if variant == "E6_FET1" else 5_084_017,
        "deploy_params": int(sum(p.numel() for p in model.parameters())),
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "expected": expected,
        "fet_shape": (model.fet_deploy_conv_shape() if fine_tap else None),
    }
    return model, deploy_info


def object_pass(args):
    import torch

    out_dir = ensure_dir(args.out_dir)
    device = args.device
    set_eval_numerics()
    ckpt_dir = os.path.join(args.ckpt_root, args.run, args.variant, args.dataset)
    import glob
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "best_F1=*.pth")))
    if not cands:
        raise SystemExit(f"[REPORT] no best_F1=*.pth in {ckpt_dir}")
    ckpt_path = cands[-1]
    model, deploy_info = build_run_model(args.variant, ckpt_path, device, args.ckpt_root)

    loader, list_path = make_loader(args.dataset, "test", batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    names = read_list_names(list_path)
    acc = ObjectMetricAccumulator(connectivity=4)
    n = 0
    with torch.no_grad():
        for img, label in loader:
            pre = img[:, 0:3].to(device).float()
            post = img[:, 3:6].to(device).float()
            out = model(pre, post)
            pb = out[:, 0].float().cpu().numpy()
            gts = label.numpy()[:, 0]
            for i in range(pb.shape[0]):
                nm = names[n] if n < len(names) else f"idx{n}"
                acc.add(pb[i] > 0.5, gts[i] > 0.5, name=nm)
                n += 1

    s = acc.summary(label=f"{args.variant}/{args.dataset}")
    small = s["legacy_groups"]["small"]
    payload = {
        "variant": args.variant,
        "dataset": args.dataset,
        "run": args.run,
        "ckpt": ckpt_path,
        "ckpt_sha256": sha256_file(ckpt_path),
        "deploy_graph": True,
        "train_params": deploy_info["train_params"],
        "train_trainable": deploy_info["train_trainable"],
        "train_params_expected": deploy_info["expected_train"],
        "train_params_ok": deploy_info["train_params"] == deploy_info["expected_train"],
        "deploy_params": deploy_info["deploy_params"],
        "deploy_params_expected": deploy_info["expected"],
        "deploy_params_ok": deploy_info["deploy_params"] == deploy_info["expected"],
        "deploy_trainable": deploy_info["trainable"],
        "deploy_params_le_5M": deploy_info["deploy_params"] <= BUDGET,
        "fet_deploy_conv_shape": deploy_info["fet_shape"],
        "connectivity": 4,
        "threshold": "p > 0.5",
        "n_images": s["n_images"],
        "pixel_scores": s["pixel"],
        "small": small,
        "medium": s["legacy_groups"]["medium"],
        "large": s["legacy_groups"]["large"],
        "object": s["object"],
        "unmatched_pred_lt256": int(sum(
            v for k, v in s["object"]["unmatched_pred_cc_count_by_area"].items()
            if k in ("tiny_1_15", "tiny_16_63", "small_64_255"))),
        "groups": s["groups"],
        "bands": {k: {kk: vv for kk, vv in v.items() if kk != "group_tp_fn"}
                  for k, v in s["bands"].items()},
    }
    path = os.path.join(out_dir, f"{args.variant}__{args.dataset}.json")
    write_json(path, payload)
    print(f"[REPORT] {args.variant}/{args.dataset}: n={n} "
          f"F1={s['pixel']['F1']:.4f} small_recall={small['pixel_recall_micro']} "
          f"small_hit25={small['hit25']} ObjP={s['object']['ObjPrecision_loose']} "
          f"ObjR={s['object']['ObjRecall_loose']} unmatched_lt256={payload['unmatched_pred_lt256']}")
    print(f"[REPORT] {args.variant}/{args.dataset}: deploy_params={deploy_info['deploy_params']} "
          f"(expected {deploy_info['expected']}, ok={payload['deploy_params_ok']}, "
          f"<=5M={payload['deploy_params_le_5M']}) fet_conv={deploy_info['fet_shape']}")
    print(f"[REPORT] wrote {path}")
    return 0


# ======================================================================================
# 模式 2：裁决
# ======================================================================================
def _deploy_params(block):
    m = re.search(r"\[DEPLOY-PARAMS\]\s*total=([0-9.]+)\s*M", block)
    return float(m.group(1)) if m else None


def _trainable_params(block):
    m = re.search(r"\[DEPLOY-PARAMS\][^\n]*trainable=([0-9.]+)\s*M", block)
    return float(m.group(1)) if m else None


def _deploy_flops(block):
    m = re.search(r"\[DEPLOY-FLOPS\]\s*([0-9.]+)\s*G", block)
    return float(m.group(1)) if m else None


def _unsupported(block):
    m = re.search(r"\[DEPLOY-FLOPS\][^\n]*unsupported_ops=(\d+)", block)
    return int(m.group(1)) if m else None


def _fet_gamma(block):
    m = re.search(r"\[FET-GAMMA\]\s*([0-9.eE+-]+)", block)
    return float(m.group(1)) if m else None


def read_last_test_block(log_root, variant, dataset):
    path = os.path.join(log_root, variant, dataset, "train_log.txt")
    block, info, span = parse_test_block(path)
    if block is None:
        return {"log": path, "valid": False, "reason": "no complete TEST RESULTS block"}
    steps = re.search(r"\[ACTUAL-OPT-STEPS\]\s*(\d+)", block)
    dis1 = re.search(r"\[REPARAM-ARGMAX-DISAGREE\]\s*([0-9.eE+-]+)", block)
    dis2 = re.search(r"\[REPARAM-REAL-ARGMAX-DISAGREE\]\s*([0-9.eE+-]+)", block)
    return {
        "log": path,
        "valid": True,
        "test_span_lines": list(span) if span else None,
        "metrics": {k: (float(v) if v is not None else None) for k, v in info.items()
                    if k in ("Recall", "Precision", "OA", "F1", "IoU", "Kappa")},
        "actual_opt_steps": int(steps.group(1)) if steps else None,
        "disagree_random": float(dis1.group(1)) if dis1 else None,
        "disagree_real": float(dis2.group(1)) if dis2 else None,
        "deploy_params_M": _deploy_params(block),
        "deploy_trainable_M": _trainable_params(block),
        "deploy_flops_G": _deploy_flops(block),
        "unsupported_ops": _unsupported(block),
        "fet_gamma": _fet_gamma(block),
    }


def verdict(args):
    log_root = args.log_root
    runs = {}
    for v in VARIANTS:
        for d in DATASETS:
            runs[f"{v}/{d}"] = read_last_test_block(log_root, v, d)

    obj = {}
    for v in VARIANTS:
        for d in DATASETS:
            p = os.path.join(args.object_dir, f"{v}__{d}.json")
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    obj[f"{v}/{d}"] = json.load(f)

    ctrl, e6 = "M1_R4CTRL", "E6_FET1"
    layers = {}

    # ---------------------------------------------------------------- 层 A：P0 有效性
    p0 = {}
    for v in VARIANTS:
        for d in DATASETS:
            r = runs[f"{v}/{d}"]
            o = obj.get(f"{v}/{d}", {})
            checks = {
                "test_block_complete": bool(r.get("valid")),
                "actual_opt_steps_80000": r.get("actual_opt_steps") == 80000,
                "deploy_params_le_5M": (r.get("deploy_params_M") is not None
                                        and r["deploy_params_M"] <= BUDGET / 1e6),
                "disagree_random_zero": r.get("disagree_random") == 0.0,
                "disagree_real_zero": r.get("disagree_real") == 0.0,
                "six_metrics_present": all(r.get("metrics", {}).get(k) is not None
                                          for k in ("Recall", "Precision", "OA", "F1", "IoU", "Kappa")),
                "deploy_flops_recorded": r.get("deploy_flops_G") is not None,
            }
            if v == e6:
                checks["fet_gamma_recorded"] = r.get("fet_gamma") is not None
            # checkpoint 级独立证据（对象 pass 从 best ckpt 重新折叠后实测；与日志互为交叉校验）
            if o:
                checks["ckpt_deploy_params_expected"] = bool(o.get("deploy_params_ok"))
                checks["ckpt_deploy_params_le_5M"] = bool(o.get("deploy_params_le_5M"))
                checks["ckpt_deploy_params_matches_log"] = (
                    o.get("deploy_params") is not None and r.get("deploy_params_M") is not None
                    and abs(o["deploy_params"] / 1e6 - r["deploy_params_M"]) < 1e-3)
            p0[f"{v}/{d}"] = {"checks": checks, "pass": all(checks.values())}
    layers["A_p0_validity"] = {"per_run": p0, "pass": all(x["pass"] for x in p0.values())}

    # ---------------------------------------------------------------- 层 B：机制（SYSU）
    key_c, key_e = f"{ctrl}/{SYSU}", f"{e6}/{SYSU}"
    mech = {"available": key_c in obj and key_e in obj}
    if mech["available"]:
        oc, oe = obj[key_c], obj[key_e]
        rows = {}
        for name, getter, abs_thr, delta_thr in (
            ("small_pooled_recall", lambda x: x["small"]["pixel_recall_micro"], MECH_ABS["small_pooled_recall"], MECH_DELTA["small_pooled_recall"]),
            ("small_hit25", lambda x: x["small"]["hit25"], MECH_ABS["small_hit25"], MECH_DELTA["small_hit25"]),
            ("ObjRecall_loose", lambda x: x["object"]["ObjRecall_loose"], None, MECH_DELTA["ObjRecall_loose"]),
        ):
            a, b = getter(oe), getter(oc)
            row = {"ctrl": b, "e6": a, "delta": (a - b) if (a is not None and b is not None) else None,
                   "abs_threshold": abs_thr, "delta_threshold": delta_thr}
            row["abs_pass"] = (abs_thr is None) or (a is not None and a >= abs_thr)
            row["delta_pass"] = (a is not None and b is not None and (a - b) >= delta_thr)
            row["pass"] = row["abs_pass"] and row["delta_pass"]
            rows[name] = row
        rows["ObjPrecision_loose"] = {
            "ctrl": oc["object"]["ObjPrecision_loose"], "e6": oe["object"]["ObjPrecision_loose"],
            "pass": (oe["object"]["ObjPrecision_loose"] is not None
                     and oc["object"]["ObjPrecision_loose"] is not None
                     and oe["object"]["ObjPrecision_loose"] >= oc["object"]["ObjPrecision_loose"]),
            "rule": "不得低于同期 CTRL"}
        rows["unmatched_pred_lt256"] = {
            "ctrl": oc.get("unmatched_pred_lt256"), "e6": oe.get("unmatched_pred_lt256"),
            "pass": (oe.get("unmatched_pred_lt256") is not None
                     and oc.get("unmatched_pred_lt256") is not None
                     and oe["unmatched_pred_lt256"] <= oc["unmatched_pred_lt256"]),
            "rule": "不得高于同期 CTRL"}
        fc, fe = _pct(oc["pixel_scores"].get("F1")), _pct(oe["pixel_scores"].get("F1"))
        lc = _pct(runs[key_c].get("metrics", {}).get("F1"))
        le = _pct(runs[key_e].get("metrics", {}).get("F1"))
        rows["SYSU_F1_official_log"] = {"ctrl": lc, "e6": le, "delta": (le - lc) if None not in (le, lc) else None,
                                        "pass": (lc is not None and le is not None and le >= lc),
                                        "rule": "F1 >= 同期 CTRL（取自最后一个完整 TEST 区块）"}
        rows["SYSU_F1"] = {"ctrl": fc, "e6": fe, "delta": (fe - fc) if None not in (fe, fc) else None,
                           "pass": (fc is not None and fe is not None and fe >= fc),
                           "rule": "F1 >= 同期 CTRL（对象 pass 在同一部署图上重算，交叉校验）"}
        mech["rows"] = rows
        mech["pass"] = all(r["pass"] for r in rows.values())
        mech["false_change_amplification"] = (
            not rows["ObjPrecision_loose"]["pass"] or not rows["unmatched_pred_lt256"]["pass"])
    layers["B_mechanism_sysu"] = mech

    # ---------------------------------------------------------------- 层 C：跨库守门 + 论文目标
    guard, paper = {}, {}
    for d in DATASETS:
        e = _pct(runs[f"{e6}/{d}"].get("metrics", {}).get("F1"))
        c = _pct(runs[f"{ctrl}/{d}"].get("metrics", {}).get("F1"))
        guard[d] = {"e6_F1_pct": e, "ctrl_F1_pct": c, "guardrail_pct": GUARDRAIL[d],
                    "pass": (e is not None and c is not None
                             and e >= GUARDRAIL[d] and e >= c)}
        paper[d] = {"e6_F1_pct": e, "paper_target_pct": PAPER_TARGET[d],
                    "pass": (e is not None and e >= PAPER_TARGET[d])}
    layers["C_guardrail"] = {"per_dataset": guard, "pass": all(v["pass"] for v in guard.values())}
    layers["C_paper_target"] = {"per_dataset": paper, "pass": all(v["pass"] for v in paper.values())}

    # IoU 与 F1 必须同方向，且同混淆矩阵上须满足 IoU = F1/(2−F1)（按比例值计算）
    iou_rows, iou_ok = {}, True
    for d in DATASETS:
        e = runs[f"{e6}/{d}"].get("metrics", {})
        c = runs[f"{ctrl}/{d}"].get("metrics", {})
        for tag, m in (("e6", e), ("ctrl", c)):
            f1, iou = m.get("F1"), m.get("IoU")
            if f1 is None or iou is None:
                continue
            ref = f1 / (2.0 - f1) if f1 < 2.0 else None
            cons = ref is not None and abs(iou - ref) < 1e-3
            iou_rows[f"{d}:{tag}"] = {"F1": f1, "IoU": iou, "IoU_from_F1": ref, "consistent": cons}
            iou_ok = iou_ok and cons
        f1e, f1c = e.get("F1"), c.get("F1")
        ioue, iouc = e.get("IoU"), c.get("IoU")
        if None not in (f1e, f1c, ioue, iouc):
            same_dir = (f1e >= f1c) == (ioue >= iouc)
            iou_rows[f"{d}:direction"] = {"dF1": f1e - f1c, "dIoU": ioue - iouc, "same_direction": same_dir}
            iou_ok = iou_ok and same_dir
    layers["C_iou_direction"] = {
        "note": "same confusion matrix ⇒ IoU = F1/(2−F1); E6 vs CTRL must move in the same direction",
        "rows": iou_rows, "pass": bool(iou_ok and iou_rows)}

    # ---------------------------------------------------------------- 最终判定（§6.4 决策树）
    if not layers["A_p0_validity"]["pass"]:
        status = "INVALID"
        decision = ("P0 INVALID -> fix code/fold/protocol and re-run smoke + dry-run; "
                    "a run that did not reach 80,000 updates cannot enter the results table")
    elif not layers["C_iou_direction"]["pass"]:
        status = "INVALID"
        decision = ("IoU/F1 inconsistency (same confusion matrix requires IoU = F1/(2-F1)) or "
                    "E6/CTRL moving in opposite directions -> the TEST block is not self-consistent; "
                    "do not read F1 until the block is fixed")
    elif not mech.get("available") or not mech.get("pass"):
        if mech.get("available") and mech.get("false_change_amplification"):
            status = "MECHANISM-FAIL"
            decision = ("false-change amplification rather than evidence recovery -> stop the 1/4 tap line; "
                        "return to Diag1 D1 FP spatial profiling (diagnosis only, no new head)")
        else:
            status = "MECHANISM-FAIL"
            decision = ("SYSU small Recall/Hit or same-period delta below the pre-registered gate -> "
                        "FET mechanism veto: stop the 1/4 tap line; do NOT try 1/8 or 3x3")
    elif not layers["C_guardrail"]["pass"]:
        status = "GUARDRAIL-FAIL"
        decision = ("cross-dataset guardrail failed -> stop the paper main line; "
                    "do not claim generality from SYSU alone")
    elif not layers["C_paper_target"]["pass"]:
        status = "MECHANISM-PASS/PAPER-TARGET-FAIL"
        decision = ("keep the negative result and mechanism analysis; do not squeeze scores via "
                    "threshold/loss/seed search, and do not add modules")
    else:
        status = "PAPER-TARGET-PASS"
        decision = ("freeze the Run4 structure, report 6 metrics + params + FLOPs + gamma + object stats; "
                    "proceed to writing, no further modules")

    payload = {"status": status, "decision": decision, "layers": layers, "runs": runs,
               "object_metrics_available": sorted(obj),
               "object_metrics": {
                   k: {"train_params": o.get("train_params"),
                       "train_trainable": o.get("train_trainable"),
                       "deploy_params": o.get("deploy_params"),
                       "fet_deploy_conv_shape": o.get("fet_deploy_conv_shape"),
                       "small": {kk: o["small"].get(kk) for kk in ("n_objects", "pixel_recall_micro", "hit25")},
                       "object": {kk: o["object"].get(kk) for kk in ("ObjPrecision_loose", "ObjRecall_loose")},
                       "unmatched_pred_lt256": o.get("unmatched_pred_lt256"),
                       "ckpt_sha256": o.get("ckpt_sha256")}
                   for k, o in obj.items()},
               "thresholds": {
                   "mech_abs": MECH_ABS, "mech_delta": MECH_DELTA, "guardrail": GUARDRAIL,
                   "paper_target": PAPER_TARGET, "budget": BUDGET},
               "historical_m1_sysu": HIST_M1_SYSU}
    write_json(args.out, payload)

    print("\n=== RUN4 VERDICT ===")
    for v in VARIANTS:
        for d in DATASETS:
            r = runs[f"{v}/{d}"]
            o = obj.get(f"{v}/{d}", {})
            if r.get("valid"):
                m = r["metrics"]
                a_ok = layers["A_p0_validity"]["per_run"][f"{v}/{d}"]["pass"]
                run_status = "OK" if a_ok else "INVALID(A)"
                print(f"{v:11s} {d:13s} F1={m['F1']:.4f} IoU={m['IoU']:.4f} "
                      f"R={m['Recall']:.4f} P={m['Precision']:.4f} OA={m['OA']:.4f} K={m['Kappa']:.4f} | "
                      f"steps={r['actual_opt_steps']} "
                      f"train_params={o.get('train_params', '-')}/{o.get('train_trainable', '-')} "
                      f"deploy_params={o.get('deploy_params', '-')} (log {r['deploy_params_M']}M) "
                      f"flops={r['deploy_flops_G']}G unsup={r['unsupported_ops']} "
                      f"gamma={r['fet_gamma']} dis={r['disagree_random']}/{r['disagree_real']} | "
                      f"small_recall={o.get('small', {}).get('pixel_recall_micro')} "
                      f"small_hit25={o.get('small', {}).get('hit25')} "
                      f"ObjP={o.get('object', {}).get('ObjPrecision_loose')} "
                      f"ObjR={o.get('object', {}).get('ObjRecall_loose')} "
                      f"unm<256={o.get('unmatched_pred_lt256')} | {run_status}")
            else:
                print(f"{v:11s} {d:13s} INVALID ({r.get('reason')})")

    # 层 A 失败原因必须直接可读（否则 INVALID 无法定位）
    for run_key, info in layers["A_p0_validity"]["per_run"].items():
        bad = [k for k, ok in info["checks"].items() if not ok]
        if bad:
            print(f"A | {run_key:26s} FAILED checks: {', '.join(bad)}")
    for run_key, row in layers["C_iou_direction"].get("rows", {}).items():
        if "consistent" in row and not row["consistent"]:
            print(f"C | {run_key:26s} IoU {row['IoU']} != F1/(2-F1) {row['IoU_from_F1']}")
        if "same_direction" in row and not row["same_direction"]:
            print(f"C | {run_key:26s} E6/CTRL opposite: dF1={row['dF1']:+.4f} dIoU={row['dIoU']:+.4f}")

    if mech.get("available"):
        for k, r in mech["rows"].items():
            print(f"B | {k:24s} ctrl={r['ctrl']} e6={r['e6']} "
                  f"delta={r['delta'] if 'delta' in r else '-'} pass={r['pass']}")
    for d in DATASETS:
        g, p = guard[d], paper[d]
        print(f"C | {d:13s} E6={g['e6_F1_pct']} CTRL={g['ctrl_F1_pct']} guard={g['guardrail_pct']} "
              f"paper={p['paper_target_pct']} guard_pass={g['pass']} paper_pass={p['pass']}")
    print(f"STATUS={status}")
    print(f"DECISION={decision}")
    print(f"wrote {args.out}")
    return 0


def render_markdown(verdict_path, out_path=None):
    """把 verdict.json 渲染成可直接粘进 README 的 markdown（数字全部来自 JSON，不经手工转抄）。"""
    with open(verdict_path, encoding="utf-8") as f:
        v = json.load(f)
    runs, layers = v["runs"], v["layers"]
    om = v.get("object_metrics", {})

    def f(x, nd=4):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    L = [f"**STATUS：`{v['status']}`**", "", f"> {v['decision']}", ""]

    L += ["### 1. 每 run 六指标与部署实测（来源：各 `train_log.txt` 最后一个完整 TEST 区块）", "",
          "| run | F1 | IoU | Recall | Precision | OA | Kappa | steps | deploy M | FLOPs G | unsup | γ | 分歧(随机/真实) | 状态 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for key in sorted(runs):
        r = runs[key]
        vv, dd = key.split("/")
        if not r.get("valid"):
            L.append(f"| {vv} / {dd} | " + " | ".join(["-"] * 12) + f" | INVALID（{r.get('reason')}） |")
            continue
        m = r["metrics"]
        a_ok = layers["A_p0_validity"]["per_run"][key]["pass"]
        L.append(f"| {vv} / {dd} | {f(m['F1'])} | {f(m['IoU'])} | {f(m['Recall'])} | {f(m['Precision'])} | "
                 f"{f(m['OA'])} | {f(m['Kappa'])} | {r['actual_opt_steps']} | {f(r['deploy_params_M'], 3)} | "
                 f"{f(r['deploy_flops_G'])} | {r['unsupported_ops']} | {f(r['fet_gamma'], 6)} | "
                 f"{f(r['disagree_random'], 3)}/{f(r['disagree_real'], 3)} | {'OK' if a_ok else 'INVALID(A)'} |")

    L += ["", "### 2. 对象指标与参数（来源：`--mode object` 在同一部署图上重算）", "",
          "| run | train 参数(可训练) | deploy 参数 | FET 折叠卷积 | small n | small pooled Recall | Hit@25 | ObjP | ObjR | 未匹配 <256px |",
          "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(om):
        o = om[key]
        vv, dd = key.split("/")
        L.append(f"| {vv} / {dd} | {o['train_params']} ({o['train_trainable']}) | {o['deploy_params']} | "
                 f"{o.get('fet_deploy_conv_shape') or '-'} | {o['small']['n_objects']} | "
                 f"{f(o['small']['pixel_recall_micro'])} | {f(o['small']['hit25'])} | "
                 f"{f(o['object']['ObjPrecision_loose'])} | {f(o['object']['ObjRecall_loose'])} | "
                 f"{o['unmatched_pred_lt256']} |")

    mech = layers.get("B_mechanism_sysu", {})
    L += ["", "### 3. 层 B：SYSU 机制（绝对门槛与同期增量门槛必须同时满足）", ""]
    if mech.get("available"):
        L += ["| 指标 | M1_R4CTRL | E6_FET1 | 增量 | 绝对门槛 | 增量门槛 | 判定 |",
              "|---|---:|---:|---:|---:|---:|---|"]
        for k, row in mech["rows"].items():
            L.append(f"| {k} | {f(row.get('ctrl'))} | {f(row.get('e6'))} | {f(row.get('delta'))} | "
                     f"{f(row.get('abs_threshold'))} | {f(row.get('delta_threshold'))} | "
                     f"{'PASS' if row['pass'] else 'FAIL'} |")
        L.append("")
        L.append(f"机制总判：**{'PASS' if mech.get('pass') else 'FAIL'}**"
                 + ("；判定为**伪变化放大而非证据恢复**" if mech.get("false_change_amplification") else ""))
    else:
        L.append("（对象指标缺失，未评估）")

    L += ["", "### 4. 层 C：跨库守门与论文硬目标", "",
          "| 数据集 | E6 F1 | 同期 CTRL F1 | 不退守门 | 守门判定 | 论文硬目标 | 硬目标判定 |",
          "|---|---:|---:|---:|---|---:|---|"]
    g, p = layers["C_guardrail"]["per_dataset"], layers["C_paper_target"]["per_dataset"]
    for d in DATASETS:
        L.append(f"| {d} | {f(g[d]['e6_F1_pct'], 2)} | {f(g[d]['ctrl_F1_pct'], 2)} | {g[d]['guardrail_pct']:.2f} | "
                 f"{'PASS' if g[d]['pass'] else 'FAIL'} | {p[d]['paper_target_pct']:.2f} | "
                 f"{'PASS' if p[d]['pass'] else 'FAIL'} |")
    L += ["", f"IoU/F1 自洽（同一混淆矩阵 `IoU = F1/(2−F1)`）且 E6/CTRL 同向："
              f"**{'PASS' if layers['C_iou_direction']['pass'] else 'FAIL'}**", ""]

    text = "\n".join(L) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"wrote {out_path}")
    else:
        print(text)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["object", "verdict", "markdown"], required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--run", type=str, default="Run4")
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--log-root", type=str, default=None)
    ap.add_argument("--ckpt-root", type=str, default=None)
    ap.add_argument("--object-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--verdict", type=str, default=None,
                    help="--mode markdown: path to verdict.json")
    args = ap.parse_args()

    if args.mode == "object":
        assert args.variant and args.dataset and args.out_dir and args.ckpt_root
        return object_pass(args)
    if args.mode == "markdown":
        assert args.verdict, "--mode markdown needs --verdict <path>"
        return render_markdown(args.verdict, args.out)
    assert args.log_root and args.object_dir and args.out and args.ckpt_root
    return verdict(args)


if __name__ == "__main__":
    sys.exit(main())
