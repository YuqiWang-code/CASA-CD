#!/usr/bin/env python
"""Run7 裁决引擎（purity-first）：把预注册的接受标准机械地作用到结果上。

设计原则
--------
1. **先定规则，再看数据**：所有阈值从 `SELECTION_LOCK.json` 的
   `acceptance_criteria_frozen_numeric` 读取（不硬编码），而该区块是在**任何 E9/E10 训练之前**
   写入的。这样"事后放宽门槛"在结构上不可能发生。
2. **决策逻辑是纯函数** `decide(...)`：不碰文件、不碰 GPU，可被 `--selftest` 用合成夹具
   逐分支验证（PASS / MECHANISM-FAIL / GUARDRAIL-FAIL / INVALID / INCOMPLETE）。
3. **负面结论照样产出**：MECHANISM-FAIL 时输出机制归因模板，供论文"负结果 + 机制归因"一节使用。

    python train_scripts/CASA-TViM/Run7/run7_verdict.py \
      --log-root /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7 \
      --object-dir /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/_OBJECT_METRICS \
      --out /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/_RESULTS/verdict.json
    python train_scripts/CASA-TViM/Run7/run7_verdict.py --selftest
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOCK = os.path.join(HERE, "SELECTION_LOCK.json")
PROTOCOL = "run7_hf_edge_exact80k_v1"
VARIANTS = ["M1_R7CTRL", "E9_HFAE_A", "E10_HFAE_B"]
DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
EXPERIMENTS = ["E9_HFAE_A", "E10_HFAE_B"]

sys.path.insert(0, os.path.join(ROOT, "analyse"))


# --------------------------------------------------------------------------- 纯函数
def load_thresholds(lock_path=LOCK):
    lock = json.load(open(lock_path, encoding="utf-8"))
    th = lock.get("acceptance_criteria_frozen_numeric")
    if not th:
        raise SystemExit("[VERDICT] SELECTION_LOCK.json lacks acceptance_criteria_frozen_numeric")
    return th, lock


def scan_log(log_path):
    """从 train_log.txt 抽训练期机制证据（这些行不在 TEST 区块里，只有日志有）。"""
    out = {"exists": os.path.isfile(log_path), "hfae_grad_norm": None,
           "aux_losses": [], "hfae_grad_lines": 0}
    if not out["exists"]:
        return out
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"\[HFAE-GRAD\][^\n]*hf_tap_grad_norm=([0-9.eE+-]+)", txt):
        out["hfae_grad_lines"] += 1
        out["hfae_grad_norm"] = float(m.group(1))
    out["aux_losses"] = [float(x) for x in
                         re.findall(r"\[HFAE-LOSS\][^\n]*aux_loss=([0-9.eE+-]+)", txt)]
    return out


def fold_ok_knife(knife_dir, variant, dataset, dis_real, log_maxabs=None):
    """真实 batch 折叠门：`disagreement == 0`，**或**有吻合的刀锋定位证据（pre-reg §7.6）。

    本函数是独立实现（不 import check_run / independent_audit），以保持三条代码路径相互独立。
    缺证据 / 数值不吻合 / verdict 不是 KNIFE_EDGE_ARTIFACT ⇒ 一律 False（fail-closed）。
    """
    if dis_real == 0.0:
        return True, None, "disagreement == 0"
    if not knife_dir:
        return False, None, "nonzero disagreement and no --knife-dir"
    p = os.path.join(knife_dir, f"KNIFE_EDGE_{dataset}_{variant}.json")
    if not os.path.isfile(p):
        return False, p, "no knife-edge evidence file"
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return False, p, f"unreadable: {e}"
    if d.get("variant") != variant or d.get("dataset") != dataset:
        return False, p, "variant/dataset mismatch"
    if d.get("verdict") != "KNIFE_EDGE_ARTIFACT":
        return False, p, f"verdict={d.get('verdict')}"
    if not (d.get("n_flipped") or 0) > 0:
        return False, p, f"n_flipped={d.get('n_flipped')}"

    def rel(a, b):
        try:
            return abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)
        except (TypeError, ValueError):
            return float("inf")
    if rel(d.get("disagree"), dis_real) >= 1e-3:
        return False, p, f"disagree {d.get('disagree')} vs log {dis_real}"
    if log_maxabs is not None and rel(d.get("max_abs_diff"), log_maxabs) >= 1e-3:
        return False, p, f"max_abs_diff {d.get('max_abs_diff')} vs log {log_maxabs}"
    return True, p, "knife-edge artifact, numbers match"


def decide(rows, obj, logs, th, knife_dir=None):
    """纯决策：返回 verdict dict。`rows`/`obj`/`logs` 为普通 dict，便于 selftest 造夹具。"""
    by = {(r["variant"], r["dataset"]): r for r in rows}
    reasons, detail = [], {}

    # ---- 0) 完备性与有效性
    # 状态语义（必须区分"还没跑完"与"跑了但不合规"）：
    #   MISSING_LOG / NO_COMPLETE_TEST_BLOCK → 尚未完成 ⇒ INCOMPLETE
    #   INVALID（有完整区块但步数/协议不符）→ 真正不合规 ⇒ INVALID
    by = {(r["variant"], r["dataset"]): r for r in rows}
    allkeys = [(v, d) for v in VARIANTS for d in DATASETS]
    not_ready = [f"{v}/{d}" for (v, d) in allkeys
                 if (v, d) not in by
                 or by[(v, d)].get("status") in ("MISSING_LOG", "NO_COMPLETE_TEST_BLOCK")]
    invalid = [f"{v}/{d}" for (v, d) in allkeys
               if (v, d) in by and by[(v, d)].get("status") not in
               ("VALID", "MISSING_LOG", "NO_COMPLETE_TEST_BLOCK")]
    if not_ready or invalid:
        verdict = "INCOMPLETE" if not_ready else "INVALID"
        reasons = []
        if not_ready:
            reasons.append(f"{len(not_ready)}/{len(allkeys)} 个 run 尚未产出完整 TEST 区块"
                           f"（未开始或仍在训练）: {not_ready}")
        if invalid:
            reasons.append(f"{len(invalid)} 个 run 有完整区块但未满足 80000 步/协议门: {invalid}")
        return {"verdict": verdict, "missing": not_ready, "invalid": invalid,
                "reasons": reasons, "attribution": None}

    f1 = {(v, d): by[(v, d)]["metrics"]["F1"] for v in VARIANTS for d in DATASETS}

    # ---- 1) 机制自检（每个实验组 × 每库都必须过）
    # 注意：`all_ok` **只能**由下面 checks_d 里的布尔项决定。诊断用的原始读数必须放在
    # `per[d]["raw"]` 子字典里——若混进同一层，`all()` 会把 0.0 / None 当成 False，
    # 从而把本来通过的机制自检判死（这类"把报告值混进判定"的错误正是本轮要防的）。
    mech = {}
    for v in EXPERIMENTS:
        per = {}
        for d in DATASETS:
            r = by[(v, d)]
            lg = logs.get((v, d), {})
            aux = lg.get("aux_losses") or []
            _fold_ok, _fold_ev, _fold_why = fold_ok_knife(
                knife_dir, v, d, r.get("fold_real_disagree"),
                r.get("fold_real_max_abs_error"))
            checks_d = {
                "hfae_form_ok": r.get("hfae_form") == ("aux_only" if v == "E9_HFAE_A"
                                                       else "aux_inject"),
                "fold_real_ok": _fold_ok,
                "deploy_within_limit": (r.get("hfae_deploy_params") is not None
                                        and r["hfae_deploy_params"] <= th["deploy_param_limit"]),
                "grad_reaches_hf_branch": (lg.get("hfae_grad_norm") or 0.0) > th["grad_norm_min"],
                "aux_loss_present_and_decreasing": (
                    len(aux) >= 2 and aux[-1] <= aux[0]),
            }
            per[d] = dict(checks_d)
            per[d]["all_ok"] = all(checks_d.values())
            # 原始读数：**只作诊断与论文引用**，不参与判定
            per[d]["raw"] = {
                "hfae_form": r.get("hfae_form"),
                "hfae_deploy_params": r.get("hfae_deploy_params"),
                "deploy_param_limit": th["deploy_param_limit"],
                "fold_real_disagree": r.get("fold_real_disagree"),
                "fold_real_max_abs_error": r.get("fold_real_max_abs_error"),
                "fold_knife_evidence": _fold_ev,
                "fold_note": _fold_why,
                "hf_tap_grad_norm": lg.get("hfae_grad_norm"),
                "hfae_grad_lines": lg.get("hfae_grad_lines"),
                "n_aux_loss_epochs": len(aux),
                "aux_loss_first": aux[0] if aux else None,
                "aux_loss_last": aux[-1] if aux else None,
            }
        mech[v] = per

    # ---- 2) 主判据：F1 delta
    delta = {(v, d): f1[(v, d)] - f1[("M1_R7CTRL", d)] for v in EXPERIMENTS for d in DATASETS}
    best = {}
    for v in EXPERIMENTS:
        n_ge = sum(1 for d in DATASETS if delta[(v, d)] >= -th["f1_tie_tolerance"])
        sysu_ok = delta[(v, "SYSU-CD-256")] > 0.0
        best[v] = {"n_datasets_ge_ctrl": n_ge, "sysu_improved": sysu_ok,
                   "mech_all_ok": all(mech[v][d]["all_ok"] for d in DATASETS),
                   "primary_ok": (n_ge >= th["min_datasets_ge_ctrl"] and sysu_ok
                                  and all(mech[v][d]["all_ok"] for d in DATASETS))}
    winners = [v for v in EXPERIMENTS if best[v]["primary_ok"]]
    detail["delta_F1_vs_ctrl"] = {f"{v}/{d}": delta[(v, d)] for v in EXPERIMENTS for d in DATASETS}
    detail["per_variant"] = best
    detail["mechanism"] = mech

    # ---- 3) 守门：中大对象 recall 与 CDD/WHU F1
    # 完整性守卫：守门依赖 `run7_object_metrics.py` 的对象级输出。若缺失（例如忘了跑那一步），
    # **不得**当作"守门通过"——否则会在从未评估对象级指标的情况下判出 MECHANISM-PASS。
    REQUIRED_HEADLINE = ("medium_pooled_recall", "large_pooled_recall", "ObjRecall_loose",
                         "small_pooled_recall", "small_hit25", "unmatched_small_pred",
                         "ObjPrecision_loose", "F1")

    def guardrail_ok(v):
        fails = []
        for d in DATASETS:
            o = obj.get((v, d))
            oc = obj.get(("M1_R7CTRL", d))
            if o and oc:
                for k in ("medium_pooled_recall", "large_pooled_recall"):
                    a, b = o.get(k), oc.get(k)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        if a < b - th["guardrail_recall_drop_max"]:
                            fails.append(f"{d}.{k}: {a:.4f} < ctrl {b:.4f} "
                                         f"(-{b - a:.4f} > {th['guardrail_recall_drop_max']})")
                a, b = o.get("ObjRecall_loose"), oc.get("ObjRecall_loose")
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if a < b - th["guardrail_objrecall_drop_max"]:
                        fails.append(f"{d}.ObjRecall_loose: {a:.4f} < ctrl {b:.4f}")
            if d in th["guardrail_f1_datasets"]:
                if f1[(v, d)] < f1[("M1_R7CTRL", d)] - th["guardrail_f1_drop_max"]:
                    fails.append(f"{d}.F1: {f1[(v, d)]:.4f} < ctrl "
                                 f"{f1[('M1_R7CTRL', d)]:.4f} - {th['guardrail_f1_drop_max']}")
        return fails

    def guardrail_evaluable(v):
        """胜出变体与 CTRL 的**全部**数据集都需有对象级输出，否则守门无法评估。"""
        miss = [d for d in DATASETS if (v, d) not in obj or ("M1_R7CTRL", d) not in obj]
        bad = [d for d in DATASETS
               if (v, d) in obj and any(k not in (obj[(v, d)] or {}) for k in REQUIRED_HEADLINE)]
        return miss, bad

    detail["guardrail_fails"] = {v: guardrail_ok(v) for v in EXPERIMENTS}
    detail["guardrail_unevaluable"] = {v: guardrail_evaluable(v) for v in EXPERIMENTS}

    # ---- 3b) 小目标机制读数：**只做归因报告，不参与接受/守门判定**
    # 目标第(5)条要求"按四库 F1 与 SYSU small recall/未匹配小预测数给出 PASS/FAIL **与机制归因**"。
    # 这些量此前只在 guardrail_evaluable() 里被要求"必须存在"，却**从未出现在任何输出里**
    # （实测：verdict 输出里搜不到 small_pooled_recall / unmatched_small_pred / small_hit25）。
    # 不变量：本区块只报告数字；接受标准与守门阈值是在任何 E9/E10 结果之前冻结的，
    # 不得因这次增补而改变 —— selftest 有一条断言专门守这个不变量（裁决标签必须不变）。
    SMALL_KEYS = ("small_pooled_recall", "small_hit25", "small_hit50",
                  "unmatched_small_pred", "n_gt_small_objects", "medium_pooled_recall",
                  "large_pooled_recall", "ObjRecall_loose", "F1")

    def _num(v, d, k):
        x = (obj.get((v, d)) or {}).get(k)
        return x if isinstance(x, (int, float)) else None

    small_report = {}
    for v in EXPERIMENTS:
        per = {}
        for d in DATASETS:
            e = {k: _num(v, d, k) for k in SMALL_KEYS}
            c = {k: _num("M1_R7CTRL", d, k) for k in SMALL_KEYS}
            per[d] = {
                "exp": e, "ctrl": c,
                "delta": {k: (None if e[k] is None or c[k] is None else e[k] - c[k])
                          for k in SMALL_KEYS},
                # 未匹配小预测的"倍数"比绝对差更直观（同一库上 CTRL 的基数不同）
                "unmatched_small_ratio_vs_ctrl": (
                    None if not c["unmatched_small_pred"] or e["unmatched_small_pred"] is None
                    else e["unmatched_small_pred"] / c["unmatched_small_pred"]),
            }
        small_report[v] = per
    detail["small_object_report"] = small_report

    def sy_line(v, d="SYSU-CD-256"):
        """一行归因：该变体在该库上的小目标读数 vs 同期 CTRL。"""
        s = small_report.get(v, {}).get(d)
        if not s:
            return f"{v}: {d} 无对象级小目标读数"
        e, c, dd = s["exp"], s["ctrl"], s["delta"]

        def f4(x):
            return "n/a" if x is None else f"{x:.4f}"

        def i0(x):
            return "n/a" if x is None else f"{x}"

        r = s["unmatched_small_ratio_vs_ctrl"]
        return (f"{v}: {d} small pooledR {f4(c['small_pooled_recall'])}→"
                f"{f4(e['small_pooled_recall'])} (Δ{'n/a' if dd['small_pooled_recall'] is None else format(dd['small_pooled_recall'], '+.4f')}), "
                f"Hit@25 {f4(c['small_hit25'])}→{f4(e['small_hit25'])}, "
                f"未匹配小预测 {i0(c['unmatched_small_pred'])}→{i0(e['unmatched_small_pred'])}"
                f"{'' if r is None else f' (×{r:.2f})'}, "
                f"中大 pooledR Δ{'n/a' if dd['medium_pooled_recall'] is None else format(dd['medium_pooled_recall'], '+.4f')}"
                f"/{'n/a' if dd['large_pooled_recall'] is None else format(dd['large_pooled_recall'], '+.4f')}")

    # ---- 4) 裁决
    if winners:
        v = winners[0]
        miss, bad = detail["guardrail_unevaluable"][v]
        if miss or bad:
            return {"verdict": "INCOMPLETE", "winner": v,
                    "reasons": [f"守门无法评估：{v} 缺对象级指标（datasets={miss}，"
                                f"缺字段={bad}）；请先跑 "
                                f"`tools/archive/run7_object_metrics.py --all-variants`。"],
                    "detail": detail,
                    "attribution": f"（对象级指标未齐，尚不能归因）{sy_line(v)}"}
        gf = detail["guardrail_fails"][v]
        if gf:
            return {"verdict": "GUARDRAIL-FAIL", "winner": v, "reasons": gf,
                    "detail": detail,
                    "attribution": (f"{v} 在 F1 上达标（≥{th['min_datasets_ge_ctrl']}/4 库且 "
                                    f"SYSU 提升），但守门被触发：提升伴随中大对象/目标级 recall "
                                    f"的实质损失 ⇒ 判定为「以小目标为代价」的不可接受增益。"
                                    f"归因读数：{sy_line(v)}")}
        return {"verdict": "MECHANISM-PASS", "winner": v, "reasons": [], "detail": detail,
                "attribution": (f"{v} 在 ≥{th['min_datasets_ge_ctrl']}/4 库 F1 ≥ 同期 CTRL 且 "
                                f"SYSU F1 提升，机制自检全过，守门未触发。"
                                f"归因读数：{sy_line(v)}"
                                + "；".join(f" {sy_line(v, d)}" for d in DATASETS
                                            if d != "SYSU-CD-256"))}
    # 未达接受标准 → 负结果 + 归因
    why = []
    for v in EXPERIMENTS:
        b = best[v]
        if not b["mech_all_ok"]:
            bad = [d for d in DATASETS if not mech[v][d]["all_ok"]]
            why.append(f"{v}: 机制自检未过 {bad}")
        elif b["n_datasets_ge_ctrl"] < th["min_datasets_ge_ctrl"]:
            why.append(f"{v}: 仅 {b['n_datasets_ge_ctrl']}/4 库 F1 ≥ CTRL")
        elif not b["sysu_improved"]:
            why.append(f"{v}: SYSU F1 未提升 (Δ={delta[(v, 'SYSU-CD-256')]:+.4f})")
    return {"verdict": "MECHANISM-FAIL", "winner": None, "reasons": why, "detail": detail,
            "attribution": ("边缘监督未能在受控消融下带来四库一致的 F1 提升；"
                            "按预注册记负结果，不回调 λ/阈值，不新增证据通道。"
                            "归因读数：" + "；".join(sy_line(v) for v in EXPERIMENTS))}


# --------------------------------------------------------------------------- IO
def gather(log_root, object_dir):
    from collect_results import collect_one  # 同目录
    rows = [collect_one(log_root, v, d) for v in VARIANTS for d in DATASETS]
    logs = {}
    for v in VARIANTS:
        for d in DATASETS:
            logs[(v, d)] = scan_log(os.path.join(log_root, v, d, "train_log.txt"))
    obj = {}
    for p in glob.glob(os.path.join(object_dir, "object_metrics_*.json")):
        j = json.load(open(p, encoding="utf-8"))
        obj[(j["variant"], j["dataset"])] = j.get("headline") or {}
    return rows, obj, logs


def render_markdown(rows, verdict):
    by = {(r["variant"], r["dataset"]): r for r in rows}
    lines = ["| dataset | CTRL F1 | E9 F1 | ΔE9 | E10 F1 | ΔE10 |", "|---|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        c = by.get(("M1_R7CTRL", d), {})
        e9 = by.get(("E9_HFAE_A", d), {})
        e10 = by.get(("E10_HFAE_B", d), {})

        def g(r, k="F1"):
            m = (r.get("metrics") or {}).get(k)
            return m if isinstance(m, (int, float)) else None
        cf, a, b = g(c), g(e9), g(e10)
        lines.append(f"| {d} | {cf if cf is None else f'{cf:.4f}'} | "
                     f"{a if a is None else f'{a:.4f}'} | "
                     f"{'n/a' if a is None or cf is None else f'{a - cf:+.4f}'} | "
                     f"{b if b is None else f'{b:.4f}'} | "
                     f"{'n/a' if b is None or cf is None else f'{b - cf:+.4f}'} |")
    lines.append("")
    lines.append(f"**VERDICT = {verdict['verdict']}**"
                 + (f" (winner={verdict.get('winner')})" if verdict.get("winner") else ""))
    for r in (verdict.get("reasons") or []):
        lines.append(f"- {r}")
    if verdict.get("attribution"):
        lines.append("")
        lines.append(f"**机制归因**：{verdict['attribution']}")

    # 小目标 / 中大对象读数表（**只作归因**，不参与接受或守门判定）
    sr = ((verdict.get("detail") or {}).get("small_object_report") or {})
    if sr:
        def f4(x):
            return "n/a" if x is None else f"{x:.4f}"

        def d4(x):
            return "n/a" if x is None else f"{x:+.4f}"
        lines += ["", "**小目标 / 中大对象机制读数（只作归因，不参与判定）**", "",
                  "| variant | dataset | small pooledR (CTRL→EXP) | Δ | Hit@25 Δ | "
                  "未匹配小预测 (CTRL→EXP) | ×CTRL | 中大 pooledR Δ (med/large) |",
                  "|---|---|---|---:|---:|---|---:|---|"]
        for v in EXPERIMENTS:
            for d in DATASETS:
                s = (sr.get(v) or {}).get(d)
                if not s:
                    continue
                e, c, dd = s["exp"], s["ctrl"], s["delta"]
                rt = s.get("unmatched_small_ratio_vs_ctrl")
                lines.append(
                    f"| {v} | {d} | {f4(c['small_pooled_recall'])}→"
                    f"{f4(e['small_pooled_recall'])} | {d4(dd['small_pooled_recall'])} | "
                    f"{d4(dd['small_hit25'])} | "
                    f"{c['unmatched_small_pred']}→{e['unmatched_small_pred']} | "
                    f"{'n/a' if rt is None else f'{rt:.2f}'} | "
                    f"{d4(dd['medium_pooled_recall'])}/{d4(dd['large_pooled_recall'])} |")

    # 机制自检的**原始读数**（判定仍是上面的 reasons / all_ok；这里只给可引用的数字）
    mech = ((verdict.get("detail") or {}).get("mechanism") or {})
    if mech:
        def g6(x):
            return "n/a" if x is None else f"{x:.6g}"
        lines += ["", "**机制自检原始读数（只作诊断/引用，不参与判定）**", "",
                  "| variant | dataset | form | deploy params | fold disagree | "
                  "HF tap grad norm | [HFAE-LOSS] epochs | aux first→last | grad lines | all_ok |",
                  "|---|---|---|---:|---:|---:|---:|---|---:|---|"]
        for v in EXPERIMENTS:
            for d in DATASETS:
                m = (mech.get(v) or {}).get(d)
                if not m:
                    continue
                rw = m.get("raw") or {}
                lines.append(
                    f"| {v} | {d} | {rw.get('hfae_form')} | {rw.get('hfae_deploy_params')} | "
                    f"{g6(rw.get('fold_real_disagree'))} | {g6(rw.get('hf_tap_grad_norm'))} | "
                    f"{rw.get('n_aux_loss_epochs')} | {g6(rw.get('aux_loss_first'))}→"
                    f"{g6(rw.get('aux_loss_last'))} | {rw.get('hfae_grad_lines')} | "
                    f"{'OK' if m.get('all_ok') else 'FAIL'} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- selftest
def _mk_rows(f1s, form_map=None, fold=0.0, deploys=None, status="VALID"):
    rows = []
    for v in VARIANTS:
        for d in DATASETS:
            rows.append({
                "variant": v, "dataset": d, "status": status,
                "metrics": {"F1": f1s[(v, d)]},
                "hfae_form": (form_map or {}).get(v, "none" if v == "M1_R7CTRL"
                                                  else ("aux_only" if v == "E9_HFAE_A"
                                                        else "aux_inject")),
                "fold_real_disagree": fold,
                "hfae_deploy_params": (deploys or {}).get(v, 4880190),
            })
    return rows


def _mk_logs(grad=5e-2, aux=(0.34, 0.33)):
    return {(v, d): {"exists": True, "hfae_grad_norm": grad if v != "M1_R7CTRL" else None,
                     "aux_losses": list(aux) if v != "M1_R7CTRL" else [],
                     "hfae_grad_lines": 1 if v != "M1_R7CTRL" else 0}
            for v in VARIANTS for d in DATASETS}


def _mk_obj(ctrl=0.5, exp=0.5):
    """夹具必须与 `tools/archive/run7_object_metrics.py` 的真实 headline 字段**逐键一致**
    （已用 Run1 M1_FULL 的真实输出核对过 schema），否则守卫会把夹具判成"字段缺失"。"""
    out = {}
    for v in VARIANTS:
        for d in DATASETS:
            c = v == "M1_R7CTRL"
            out[(v, d)] = {
                "small_pooled_recall": 0.1974 if c else 0.2100,
                "small_object_macro_recall": 0.1654 if c else 0.1750,
                "small_hit1": 0.2280, "small_hit25": 0.1896 if c else 0.2000,
                "small_hit50": 0.1648, "n_gt_small_objects": 364,
                "medium_pooled_recall": ctrl if c else exp,
                "large_pooled_recall": ctrl if c else exp,
                "unmatched_small_pred": 2430 if c else 2200,
                "ObjPrecision_loose": 0.5320 if c else 0.5450,
                "ObjRecall_loose": 0.7674 if c else 0.7674,
                "F1": 0.8347 if c else 0.8400,
            }
    return out


def selftest():
    th, _ = load_thresholds()
    base = {}
    for d in DATASETS:
        base[("M1_R7CTRL", d)] = 0.9000
    cases = []

    # 1) PASS：E9 ≥ CTRL 于 4/4 且 SYSU 提升，机制与守门全过
    f = dict(base)
    for d in DATASETS:
        f[("E9_HFAE_A", d)] = 0.9020
        f[("E10_HFAE_B", d)] = 0.8990
    f[("E9_HFAE_A", "SYSU-CD-256")] = 0.9050
    v = decide(_mk_rows(f), _mk_obj(exp=0.50), _mk_logs(), th)
    cases.append(("PASS case", v["verdict"] == "MECHANISM-PASS", v["verdict"]))

    # 2) MECHANISM-FAIL：只 2/4 库 ≥ CTRL
    f2 = dict(base)
    for d in ("CDD-CD-256", "LEVIR-CD-256"):
        f2[("E9_HFAE_A", d)] = 0.9010
    for d in ("SYSU-CD-256", "WHU-CD-256"):
        f2[("E9_HFAE_A", d)] = 0.8980
    for d in DATASETS:
        f2[("E10_HFAE_B", d)] = 0.8980
    v = decide(_mk_rows(f2), _mk_obj(), _mk_logs(), th)
    cases.append(("MECHANISM-FAIL (2/4)", v["verdict"] == "MECHANISM-FAIL", v["verdict"]))

    # 3) MECHANISM-FAIL：SYSU 未提升（尽管 4/4 库 ≥ CTRL）
    f3 = dict(base)
    for d in DATASETS:
        f3[("E9_HFAE_A", d)] = 0.9010
        f3[("E10_HFAE_B", d)] = 0.9000
    f3[("E9_HFAE_A", "SYSU-CD-256")] = 0.8990
    v = decide(_mk_rows(f3), _mk_obj(), _mk_logs(), th)
    cases.append(("MECHANISM-FAIL (SYSU flat)", v["verdict"] == "MECHANISM-FAIL", v["verdict"]))

    # 4) 机制自检不过（梯度没进高频分支）→ 必须判负
    v = decide(_mk_rows(f), _mk_obj(), _mk_logs(grad=0.0), th)
    cases.append(("MECHANISM-FAIL (no HF grad)", v["verdict"] == "MECHANISM-FAIL", v["verdict"]))

    # 5) GUARDRAIL-FAIL：接受标准过，但中大对象 recall 掉超 0.02
    v = decide(_mk_rows(f), _mk_obj(ctrl=0.80, exp=0.70), _mk_logs(), th)
    cases.append(("GUARDRAIL-FAIL (recall drop)", v["verdict"] == "GUARDRAIL-FAIL", v["verdict"]))

    # 6) INVALID：某个 run 有完整区块但未达 80000 步
    rows = _mk_rows(f)
    rows[0]["status"] = "INVALID"
    v = decide(rows, _mk_obj(), _mk_logs(), th)
    cases.append(("INVALID (bad run)", v["verdict"] == "INVALID", v["verdict"]))

    # 6b) INCOMPLETE：run 尚未产出完整 TEST 区块（对应训练进行中）
    rows = _mk_rows(f)
    rows[0]["status"] = "NO_COMPLETE_TEST_BLOCK"
    v = decide(rows, _mk_obj(), _mk_logs(), th)
    cases.append(("INCOMPLETE (still training)", v["verdict"] == "INCOMPLETE", v["verdict"]))

    # 6c) INCOMPLETE：run 日志都不存在（尚未启动）
    rows = _mk_rows(f)
    rows[0]["status"] = "MISSING_LOG"
    v = decide(rows, _mk_obj(), _mk_logs(), th)
    cases.append(("INCOMPLETE (not started)", v["verdict"] == "INCOMPLETE", v["verdict"]))

    # 7) INCOMPLETE：缺 run
    rows = [r for r in _mk_rows(f) if not (r["variant"] == "E10_HFAE_B")]
    v = decide(rows, _mk_obj(), _mk_logs(), th)
    cases.append(("INCOMPLETE (missing runs)", v["verdict"] == "INCOMPLETE", v["verdict"]))

    # 8) deploy 超 5M → 机制自检不过
    v = decide(_mk_rows(f, deploys={"E9_HFAE_A": 5_100_000}), _mk_obj(), _mk_logs(), th)
    cases.append(("MECHANISM-FAIL (deploy >5M)", v["verdict"] == "MECHANISM-FAIL", v["verdict"]))

    # 9) aux loss 未下降 → 机制自检不过
    v = decide(_mk_rows(f), _mk_obj(), _mk_logs(aux=(0.30, 0.40)), th)
    cases.append(("MECHANISM-FAIL (aux not decreasing)", v["verdict"] == "MECHANISM-FAIL",
                  v["verdict"]))

    # 10) 守门无法评估（忘了跑对象级复算）→ 不得判 PASS，必须 INCOMPLETE
    v = decide(_mk_rows(f), {}, _mk_logs(), th)
    cases.append(("INCOMPLETE (guardrail unevaluable)", v["verdict"] == "INCOMPLETE",
                  v["verdict"]))

    # 11) 部分数据集缺对象级指标 → 同样 INCOMPLETE
    obj_partial = _mk_obj()
    for d in DATASETS:
        obj_partial.pop(("E9_HFAE_A", d), None)
    v = decide(_mk_rows(f), obj_partial, _mk_logs(), th)
    cases.append(("INCOMPLETE (partial object metrics)", v["verdict"] == "INCOMPLETE",
                  v["verdict"]))

    # 12) 不变量：小目标读数**只作归因**，把它改到极端也不得改变裁决标签
    #     （接受标准与守门阈值在任何结果之前冻结；这次增补是"报告"，不是"改规则"）
    obj_wild = _mk_obj()
    for k in list(obj_wild):
        for fld in ("small_pooled_recall", "small_hit25", "small_hit50",
                    "small_object_macro_recall"):
            if fld in obj_wild[k]:
                obj_wild[k][fld] = 0.0
        if "unmatched_small_pred" in obj_wild[k]:
            obj_wild[k]["unmatched_small_pred"] = 10 ** 6
    va = decide(_mk_rows(f), _mk_obj(), _mk_logs(), th)
    vb = decide(_mk_rows(f), obj_wild, _mk_logs(), th)
    cases.append(("small-object readings do NOT change verdict",
                  va["verdict"] == vb["verdict"],
                  f"{va['verdict']} == {vb['verdict']}"))

    # 13) 目标第(5)条：SYSU small recall / 未匹配小预测数必须真的**出现在输出里**
    att = va.get("attribution") or ""
    att_ok = all(t in att for t in ("SYSU-CD-256", "small pooledR", "未匹配小预测"))
    rep = ((va.get("detail") or {}).get("small_object_report") or {}).get(
        "E9_HFAE_A", {}).get("SYSU-CD-256") or {}
    rep_ok = all(k in (rep.get("exp") or {}) for k in
                 ("small_pooled_recall", "small_hit25", "unmatched_small_pred",
                  "n_gt_small_objects")) and isinstance(rep.get("delta"), dict)
    cases.append(("SYSU small-object readings surfaced (report + attribution)",
                  att_ok and rep_ok, f"attribution={att_ok} report={rep_ok}"))

    # 14) 不变量：`all_ok` 只能由布尔检查项决定，**不能**被 raw 诊断值污染。
    #     夹具里 fold_real_disagree=0.0（falsy）、aux_loss_last=0.3（truthy）——
    #     若把 raw 混进同一层做 all()，这条会挂。
    vm = decide(_mk_rows(f), _mk_obj(), _mk_logs(aux=(0.34, 0.30)), th)
    m9 = (vm["detail"]["mechanism"]["E9_HFAE_A"])["SYSU-CD-256"]
    raw = m9.get("raw") or {}
    chk_bool = all(m9[k] for k in ("hfae_form_ok", "fold_real_ok",
                                   "deploy_within_limit", "grad_reaches_hf_branch",
                                   "aux_loss_present_and_decreasing"))
    cases.append(("all_ok depends only on boolean checks (raw cannot poison it)",
                  m9["all_ok"] is True and chk_bool and m9["all_ok"] == chk_bool,
                  f"all_ok={m9['all_ok']} bools={chk_bool} fold={m9['fold_real_ok']}"))
    cases.append(("mechanism raw readings present (aux first→last, grad, fold)",
                  raw.get("n_aux_loss_epochs") == 2
                  and raw.get("aux_loss_first") == 0.34 and raw.get("aux_loss_last") == 0.30
                  and raw.get("fold_real_disagree") == 0.0
                  and raw.get("hf_tap_grad_norm") is not None,
                  f"n={raw.get('n_aux_loss_epochs')} {raw.get('aux_loss_first')}→"
                  f"{raw.get('aux_loss_last')} grad={raw.get('hf_tap_grad_norm')}"))

    # 15) 负向：aux loss 反向（末 > 首）时机制自检必须 FAIL（预注册 ⑤）
    vaux = decide(_mk_rows(f), _mk_obj(), _mk_logs(aux=(0.30, 0.40)), th)
    mx = (vaux["detail"]["mechanism"]["E9_HFAE_A"])["SYSU-CD-256"]
    cases.append(("aux not decreasing => all_ok False with raw evidence",
                  mx["all_ok"] is False
                  and mx["raw"]["aux_loss_first"] == 0.30 and mx["raw"]["aux_loss_last"] == 0.40,
                  f"all_ok={mx['all_ok']} {mx['raw']['aux_loss_first']}→"
                  f"{mx['raw']['aux_loss_last']}"))

    # 16) 折叠门（pre-reg §7.6）：真实 batch disagreement>0 时
    #     (a) 无证据 => 机制自检必须 FAIL（fail-closed，不允许口头豁免）；
    #     (b) 有吻合的 KNIFE_EDGE_ARTIFACT 证据 => 通过。
    import tempfile as _tf
    kd = _tf.mkdtemp(prefix="r7_knife_")
    f_fold = dict(f)
    rows_fold = _mk_rows(f_fold)
    for r in rows_fold:                       # 把 E9 全库的真实折叠 disagreement 设成非零
        if r["variant"] == "E9_HFAE_A":
            r["fold_real_disagree"] = 9.5367431640625e-07
            r["fold_real_max_abs_error"] = 6.198883056640625e-06
    v_no = decide(rows_fold, _mk_obj(), _mk_logs(), th, knife_dir=kd)
    m_no = (v_no["detail"]["mechanism"]["E9_HFAE_A"])["SYSU-CD-256"]
    cases.append(("nonzero real-batch fold WITHOUT evidence => mech FAILs",
                  m_no["all_ok"] is False and m_no["fold_real_ok"] is False,
                  f"fold_real_ok={m_no['fold_real_ok']} note={m_no['raw'].get('fold_note')}"))

    with open(os.path.join(kd, "KNIFE_EDGE_SYSU-CD-256_E9_HFAE_A.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"variant": "E9_HFAE_A", "dataset": "SYSU-CD-256",
                   "verdict": "KNIFE_EDGE_ARTIFACT", "n_flipped": 1,
                   "disagree": 9.5367431640625e-07,
                   "max_abs_diff": 6.198883056640625e-06}, fh)
    v_yes = decide(rows_fold, _mk_obj(), _mk_logs(), th, knife_dir=kd)
    m_yes = (v_yes["detail"]["mechanism"]["E9_HFAE_A"])["SYSU-CD-256"]
    cases.append(("nonzero real-batch fold WITH matching knife-edge evidence => mech passes",
                  m_yes["fold_real_ok"] is True and m_yes["all_ok"] is True,
                  f"fold_real_ok={m_yes['fold_real_ok']} ev={os.path.basename(str(m_yes['raw'].get('fold_knife_evidence')))}"))

    # 17) 证据存在但数值与日志不吻合 => 仍须 FAIL（防止"拿别的 run 的证据蒙过"）
    with open(os.path.join(kd, "KNIFE_EDGE_SYSU-CD-256_E9_HFAE_A.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"variant": "E9_HFAE_A", "dataset": "SYSU-CD-256",
                   "verdict": "KNIFE_EDGE_ARTIFACT", "n_flipped": 1,
                   "disagree": 1.0, "max_abs_diff": 1.0}, fh)
    v_bad = decide(rows_fold, _mk_obj(), _mk_logs(), th, knife_dir=kd)
    m_bad = (v_bad["detail"]["mechanism"]["E9_HFAE_A"])["SYSU-CD-256"]
    cases.append(("knife-edge evidence with MISMATCHED numbers => mech FAILs",
                  m_bad["fold_real_ok"] is False,
                  f"note={m_bad['raw'].get('fold_note')}"))

    print("=== verdict engine selftest (pure decide(), synthetic fixtures) ===")
    print("thresholds:", json.dumps(th, ensure_ascii=False))
    bad = 0
    for name, ok, got in cases:
        print(f"  [{'OK' if ok else 'FAIL'}] {name:38s} -> {got}")
        bad += 0 if ok else 1
    print(f"SELFTEST = {'PASS' if bad == 0 else f'FAIL ({bad})'} ({len(cases)} branches)")
    # markdown 渲染烟雾
    md = render_markdown(_mk_rows(f), {"verdict": "MECHANISM-PASS", "winner": "E9_HFAE_A"})
    print("--- markdown rendering smoke ---")
    print(md.split("\n")[0]); print(md.split("\n")[1]); print(md.strip().split("\n")[-1])
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", default="/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7")
    ap.add_argument("--object-dir",
                    default="/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/_OBJECT_METRICS")
    ap.add_argument("--out", default=None)
    ap.add_argument("--knife-dir",
                    default="/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/_KNIFE_EDGE",
                    help="刀锋定位证据目录；真实 batch disagreement>0 时必须有吻合的证据，否则该 run 机制自检不过")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    th, lock = load_thresholds()
    rows, obj, logs = gather(a.log_root, a.object_dir)
    verdict = decide(rows, obj, logs, th, knife_dir=a.knife_dir)
    verdict["policy"] = lock.get("policy_note")
    verdict["thresholds_source"] = "SELECTION_LOCK.json:acceptance_criteria_frozen_numeric"
    print(render_markdown(rows, verdict))
    print("\n--- reasons ---")
    for r in verdict.get("reasons") or []:
        print("  -", r)
    if verdict.get("attribution"):
        print("--- attribution ---\n ", verdict["attribution"])
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(verdict, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.splitext(a.out)[0] + ".md", "w", encoding="utf-8",
                  newline="\n") as f:
            f.write(render_markdown(rows, verdict) + "\n")
        print(f"[VERDICT] wrote {a.out} (+ .md)")
    return 0 if verdict["verdict"] in ("MECHANISM-PASS", "MECHANISM-FAIL",
                                       "GUARDRAIL-FAIL") else 1


if __name__ == "__main__":
    sys.exit(main())
