"""汇总 Diag1 全部证据 → DIAGNOSIS_REPORT.md（含预注册判据的自动评估）。

用法：
    python analyse/tvim_diag_report.py --root /home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1 \
        --out "$ROOT/DIAGNOSIS_REPORT.md"

本脚本只读取各阶段产物（gate.json / summary.json / csv），不修改它们；
所有"结论性文字"分为两类：
  * AUTO：由预注册阈值（§10.2）与决策矩阵（§10.3）机械推导，可复现；
  * INTERPRETATION：留给研究者复核后填写（脚本给出 [M] 占位并列出所需证据）。
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tvim_object_metrics import AREA_GROUPS  # noqa: E402

# 预注册阈值（§10.2）
TH_D1_SIZE_GAP_PP = 10.0
TH_D2_STAGE_DROP_REL = 0.10
TH_D3_PROBE_AP_DROP = 0.02
MIN_GROUP_OBJECTS = 100

# D2 编码器节点顺序（沿计算流；同分辨率相邻对用于减少空间 alias 混杂）
NODE_ORDER = ["L00_patch_embed", "L01_network0", "L01b_norm0", "L02_network1",
              "L03_network2", "L03b_norm2", "L04_network3", "L05_stage3_last_prefix",
              "L06b_norm4", "L07_network5", "L08b_norm6"]
NODE_RES = {"L00_patch_embed": "1/4 64²", "L01_network0": "1/4 64²", "L01b_norm0": "1/4 64²",
            "L02_network1": "1/8 32²", "L03_network2": "1/8 32²", "L03b_norm2": "1/8 32²",
            "L04_network3": "1/16 16²", "L05_stage3_last_prefix": "1/16 16²",
            "L06b_norm4": "1/16 16²", "L07_network5": "1/32 8²", "L08b_norm6": "1/32 8²"}


def _load(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _g(root, *parts):
    return os.path.join(root, *parts)


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float) and (np.isnan(v)):
        return "NaN"
    return f"{v:.{nd}f}"


def build(root):
    out = {"root": root}
    out["p0"] = _load(_g(root, "P0", "metrics_protocol.json"))
    out["d0"] = {}
    for d in sorted(glob.glob(_g(root, "D0", "*"))):
        name = os.path.basename(d)
        out["d0"][name] = {"gate": _load(_g(d, "gate.json")),
                           "test": _load(_g(d, "test_result.json")),
                           "fold": _load(_g(d, "fold_equivalence.json"))}
    out["d1"] = {}
    for d in sorted(glob.glob(_g(root, "D1", "*"))):
        name = os.path.basename(d)
        out["d1"][name] = {"gate": _load(_g(d, "gate.json")),
                           "summary": _load(_g(d, "summary.json")),
                           "paired": [_load(p) for p in sorted(glob.glob(_g(d, "paired_summary_*.json")))]}
    out["d2"] = {}
    for d in sorted(glob.glob(_g(root, "D2", "*"))):
        name = os.path.basename(d)
        out["d2"][name] = {"gate": _load(_g(d, "gate.json")),
                           "stage": _load(_g(d, "stage_summary.json")),
                           "caacp": _load(_g(d, "caacp_internal.json"))}
    out["d3"] = {}
    for d in sorted(glob.glob(_g(root, "D3", "*"))):
        name = os.path.basename(d)
        out["d3"][name] = {"gate": _load(_g(d, "gate.json")),
                           "results": _load(_g(d, "probe_results.json"))}
    out["d4"] = {}
    for d in sorted(glob.glob(_g(root, "D4", "*"))):
        name = os.path.basename(d)
        out["d4"][name] = {"gate": _load(_g(d, "gate.json")),
                           "summary": _load(_g(d, "summary.json"))}
    return out


def d2_stage_drops(stage):
    """相邻节点（沿计算流）的 small margin / pooled AP 变化，判定 §10.2 的 stage drop。"""
    drops = []
    nodes = [n for n in NODE_ORDER if n in stage.get("encoder_nodes", {})]
    for a, b in zip(nodes[:-1], nodes[1:]):
        na, nb = stage["encoder_nodes"][a], stage["encoder_nodes"][b]
        ap_a, ap_b = na.get("pooled_AP_hist"), nb.get("pooled_AP_hist")
        m_a = (na.get("small_object_margin") or {}).get("mean")
        m_b = (nb.get("small_object_margin") or {}).get("mean")
        rec = {}
        rel_ap = None
        if ap_a and ap_b:
            rel_ap = (ap_b - ap_a) / ap_a if ap_a else None
        ci_a = (na.get("per_image_AP_bootstrap_CI") or {})
        ci_b = (nb.get("per_image_AP_bootstrap_CI") or {})
        # 配对方向：两个 CI 落在同侧（近似判据；严格配对需 per-image AP 逐图差，见 stage_raw.csv）
        same_sign = None
        if ci_a and ci_b:
            if ci_b.get("lo") is not None and ci_a.get("hi") is not None:
                if ci_b["lo"] > ci_a["hi"]:
                    same_sign = +1
                elif ci_b["hi"] < ci_a["lo"]:
                    same_sign = -1
                else:
                    same_sign = 0
        drops.append({
            "from": a, "to": b, "res_from": NODE_RES.get(a), "res_to": NODE_RES.get(b),
            "AP_from": ap_a, "AP_to": ap_b, "rel_AP_change": rel_ap,
            "margin_from": m_a, "margin_to": m_b, "margin_delta": (None if (m_a is None or m_b is None) else m_b - m_a),
            "bootstrap_direction": same_sign,
            "criterion_stage_drop_met": bool(rel_ap is not None and rel_ap <= -TH_D2_STAGE_DROP_REL
                                             and same_sign == -1),
        })
    return drops


def build_report(root):
    R = build(root)
    args = type("A", (), {"root": root})()          # 兼容下方 args.root 引用
    L = []
    A = L.append

    A("# CASA-TViM 小目标瓶颈分阶段诊断报告（Run-Diag / Diag1）\n")
    A(f"- 诊断根目录：`{args.root}`")
    A(f"- 预注册阈值：D1 size gap ≥{TH_D1_SIZE_GAP_PP}pp、D2 stage drop ≥{TH_D2_STAGE_DROP_REL:.0%}（相对）+ bootstrap 同号、"
      f"D3 probe AP 绝对下降 ≥{TH_D3_PROBE_AP_DROP}、每组最少 {MIN_GROUP_OBJECTS} 个 GT 对象\n")

    # ---------------------------------------------------------------- 14.1 identity
    A("## 1. 实验身份与完整性（§14.1）\n")
    p0 = R["p0"] or {}
    A(f"- 指标口径协议：`{p0.get('version', 'MISSING')}`（连通性 {p0.get('connectivity_primary')}，"
      f"IoU 匹配阈值 loose={p0.get('iou_loose')} strict={p0.get('iou_strict')}，"
      f"判定规则 `{p0.get('threshold_rule')}`）")
    for name, d in R["d0"].items():
        t = d.get("test") or {}
        f = d.get("fold") or {}
        A(f"\n### D0 {name}\n")
        A(f"- checkpoint：`{t.get('checkpoint', {}).get('path')}`")
        A(f"  sha256：`{t.get('checkpoint', {}).get('sha256')}`")
        A(f"- 原始日志：`{t.get('log_path')}`（TEST 区块行 {t.get('test_block_lines')}）")
        A(f"- 复现六指标：R={_fmt((t.get('reproduced') or {}).get('recall'),5)} "
          f"P={_fmt((t.get('reproduced') or {}).get('precision'),5)} "
          f"OA={_fmt((t.get('reproduced') or {}).get('OA'),5)} "
          f"F1={_fmt((t.get('reproduced') or {}).get('F1'),5)} "
          f"IoU={_fmt((t.get('reproduced') or {}).get('IoU'),5)} "
          f"Kappa={_fmt((t.get('reproduced') or {}).get('Kappa'),5)}")
        A(f"- 与日志最大偏差：{_fmt(t.get('max_abs_delta'), 2)}（容差 {t.get('tolerance')}）")
        A(f"- 训练图参数：{t.get('params', {}).get('train_graph_total')}；"
          f"部署图参数：{f.get('deploy_params_total')}；"
          f"部署 FLOPs：{_fmt(f.get('deploy_flops_G'), 4)} G "
          f"（unsupported ops {f.get('deploy_flops_unsupported_count')}）")
        A(f"- 折叠等价性：随机 batch max_abs={_fmt((f.get('random_batch') or {}).get('max_abs_error'), 2)}, "
          f"disagreement={_fmt((f.get('random_batch') or {}).get('binary_disagreement'), 2)}；"
          f"真实 batch max_abs={_fmt((f.get('real_batch') or {}).get('max_abs_error'), 2)}, "
          f"disagreement={_fmt((f.get('real_batch') or {}).get('binary_disagreement'), 2)}"
          f"（flipped={((f.get('real_batch') or {}).get('n_flipped'))}）")
        g = (d.get("gate") or {})
        A(f"- Gate D0-REPRO：**{g.get('status')}**")
        A(f"- 数据集：{t.get('dataset_manifest', {}).get('n_images')} 张，list sha256 "
          f"`{t.get('dataset_manifest', {}).get('test_list_sha256')}`；空 GT 图 {t.get('empty_mask_count')} 张")
        A(f"- GT 连通域按面积计数：{t.get('gt_cc_by_size')}")
    A("")

    # ---------------------------------------------------------------- P0
    A("## 2. P0 指标口径修正（§3）\n")
    A("旧 `analyse/run2_zero_cost_diag.py::component_pr` 的 `FP` 用 GT 局部掩码过滤，恒等于 0，"
      "故其 `small/medium/large F1` 是**受限正样本组内的伪指标**，不是对象级 F1。"
      "新实现（`analyse/tvim_object_metrics.py`）给出语义分离的指标：\n")
    A("- pooled / object-macro **像素 Recall**（GT 面积分组）")
    A("- ObjectHit@1 / **@25%**（预注册主对象召回）/ @50%")
    A("- 真正匹配预测连通域的 **ObjPrecision / ObjRecall / ObjF1**（IoU≥0.10 loose、≥0.50 strict，"
      "一次性最大权重二分匹配）\n")
    for name, d in R["d1"].items():
        s = d.get("summary") or {}
        lv = s.get("legacy_pseudo_vs_new")
        if not lv:
            continue
        A(f"### {name}：旧伪指标 vs 新正确口径\n")
        A(f"- 旧 `component_pr` pseudo small F1：{lv.get('old_component_pr_pseudo')}")
        A(f"- 新 **small pooled pixel Recall**：{_fmt(lv.get('new_small_recall_micro'))}；"
          f"object-macro Recall：{_fmt(lv.get('new_small_object_macro_recall'))}；"
          f"Hit@25%：{_fmt(lv.get('new_small_hit25'))}")
        A(f"- 新对象级指标（IoU≥0.10）：ObjPrecision={_fmt(lv.get('new_ObjPrecision_loose'))}，"
          f"ObjRecall={_fmt(lv.get('new_ObjRecall_loose'))}\n")
    A("")

    # ---------------------------------------------------------------- D1
    A("## 3. D1：小目标错误画像（§5 / §14.2）\n")
    for name, d in R["d1"].items():
        s = d.get("summary") or {}
        if not s:
            continue
        A(f"### {name}（{s.get('n_images_processed')} 张，limited={s.get('limited')}）\n")
        px = s.get("pixel", {})
        A(f"- 全图像素：R={_fmt(px.get('recall'),5)} P={_fmt(px.get('precision'),5)} "
          f"F1={_fmt(px.get('F1'),5)} IoU={_fmt(px.get('IoU'),5)}；"
          f"TP/FP/FN/TN={px.get('tp')}/{px.get('fp')}/{px.get('fn')}/{px.get('tn')}；"
          f"正类先验={_fmt(px.get('positive_prior'),5)}")
        A("")
        A("| GT 面积组 | 对象数 | 像素数 | pooled pixel Recall | object-macro Recall | Hit@25% | Hit@50% | 证据充足 |")
        A("|---|---:|---:|---:|---:|---:|---:|---|")
        for g in AREA_GROUPS:
            gg = s["groups"][g]
            A(f"| {g} | {gg['n_objects']} | {gg['n_pixels']} | {_fmt(gg['pixel_recall_micro'])} | "
              f"{_fmt(gg['object_macro_pixel_recall'])} | {_fmt(gg['hit25'])} | {_fmt(gg['hit50'])} | "
              f"{'是' if gg['n_objects'] >= MIN_GROUP_OBJECTS else '否(<%d)' % MIN_GROUP_OBJECTS} |")
        A("")
        o = s.get("object", {})
        A(f"- 对象级（真实匹配）：N_gt={o.get('n_gt')} N_pred={o.get('n_pred')} "
          f"matched(loose)={o.get('n_matched_loose')} matched(strict)={o.get('n_matched_strict')}；"
          f"ObjP(loose)={_fmt(o.get('ObjPrecision_loose'))} ObjR(loose)={_fmt(o.get('ObjRecall_loose'))} "
          f"ObjF1(loose)={_fmt(o.get('ObjF1_loose'))}；"
          f"严格：ObjP={_fmt(o.get('ObjPrecision_strict'))} ObjR={_fmt(o.get('ObjRecall_strict'))}")
        A(f"- FP 像素={o.get('fp_pixels')}；预测连通域按面积分布={o.get('pred_cc_count_by_area')}；"
          f"未匹配（FP）连通域分布={o.get('unmatched_pred_cc_count_by_area')}")
        b = s.get("bands", {})
        for k in ("band2", "band4"):
            if k in b:
                A(f"- {k}：F1={_fmt(b[k].get('F1'))} TP={b[k].get('tp')} FP={b[k].get('fp')} "
                  f"FN={b[k].get('fn')} band 像素={b[k].get('band_pixels')} 空 band 数={b[k].get('n_empty_band')}")
        A(f"- Gate D1-PHENOMENON：**{(d.get('gate') or {}).get('status')}**"
          f"（size gap={_fmt((d.get('gate') or {}).get('checks', {}).get('size_gap_pp_pooled_recall'),2)}pp，"
          f"阈值 {TH_D1_SIZE_GAP_PP}pp："
          f"{'满足' if (d.get('gate') or {}).get('checks', {}).get('size_gap_threshold_10pp_met') else '未满足'}）")
        for p in (d.get("paired") or []):
            if not p:
                continue
            A(f"- paired {p.get('base_variant')} vs {p.get('compare_variant')}："
              f"共同对象 {p.get('n_common_objects')} 个 / {p.get('n_images')} 图；"
              f"对象级 mean Δrecall={_fmt(p.get('mean_delta_recall_object_level'),6)}；"
              f"图像级 bootstrap CI={_fmt((p.get('image_level_bootstrap_CI') or {}).get('lo'),6)}"
              f"~{_fmt((p.get('image_level_bootstrap_CI') or {}).get('hi'),6)}；"
              f"分组={ {k: (None if v.get('mean_delta_recall') is None else round(v['mean_delta_recall'],5)) for k, v in (p.get('by_area_group') or {}).items()} }")
        A("")
    A("")

    # ---------------------------------------------------------------- D2
    A("## 4. D2：分阶段证据衰减（§6 / §14.2 核心表）\n")
    for name, d in R["d2"].items():
        s = d.get("stage") or {}
        c = d.get("caacp") or {}
        if not s:
            continue
        A(f"### {name}（{s.get('n_images_processed')} 张，limited={s.get('limited')}）\n")
        A(f"- 重建校验：head logits→interp→sigmoid vs 模型输出 max_abs="
          f"{(s.get('reconstruction_check') or {}).get('max_abs_recon_vs_model')}，"
          f"二值 disagreement={(s.get('reconstruction_check') or {}).get('binary_disagreement')}")
        A(f"- 分辨率协议：主口径={s.get('resolution_protocol', {}).get('main')}")
        A(f"  副口径={s.get('resolution_protocol', {}).get('secondary')}\n")
        A("| 节点 | 原生尺度 | pooled AP | per-image AP | bootstrap CI | small margin | recall@1%FP | recall@5%FP | 小/中/大 GT 分数均值 |")
        A("|---|---|---:|---:|---|---:|---:|---:|---|")
        for n in NODE_ORDER:
            v = (s.get("encoder_nodes") or {}).get(n)
            if not v:
                continue
            ci = v.get("per_image_AP_bootstrap_CI") or {}
            gs = v.get("group_score_mean") or {}
            A(f"| {n} | {NODE_RES.get(n,'')} | {_fmt(v.get('pooled_AP_hist'))} | "
              f"{_fmt(v.get('per_image_AP_mean'))} | "
              f"{_fmt(ci.get('lo'))}~{_fmt(ci.get('hi'))} | "
              f"{_fmt((v.get('small_object_margin') or {}).get('mean'),5)} | "
              f"{_fmt((v.get('budget_recall') or {}).get('fp_budget_1pct', {}).get('per_image_mean'))} | "
              f"{_fmt((v.get('budget_recall') or {}).get('fp_budget_5pct', {}).get('per_image_mean'))} | "
              f"{_fmt(gs.get('small'))}/{_fmt(gs.get('medium'))}/{_fmt(gs.get('large'))} |")
        A("")
        A("**相邻节点衰减判据（§10.2）**：\n")
        A("| 从 → 到 | 分辨率 | AP 变化(相对) | margin Δ | bootstrap 方向 | 达标(≥10% 且同号) |")
        A("|---|---|---:|---:|---:|---|")
        for row in d2_stage_drops(s):
            rel = row["rel_AP_change"]
            rel_s = "—" if rel is None else f"{rel:.2%}"
            A(f"| {row['from']} → {row['to']} | {row['res_from']} → {row['res_to']} | "
              f"{rel_s} | {_fmt(row['margin_delta'],5)} | {row['bootstrap_direction']} | "
              f"{'是' if row['criterion_stage_drop_met'] else '否'} |")
        A("")
        A("**CAACP 内部证据**：\n")
        for k in ("beta", "mean_abs_deltaC", "mean_abs_beta_deltaC", "beta_delta_rms_exact",
                  "ratio_beta_delta_over_cavg", "cell_weight_entropy_mean", "cell_weight_top1_mean",
                  "xlow_feature_norm_mean", "n_samples"):
            A(f"- {k} = {_fmt(c.get(k), 6)}")
        for k in ("group_mean_abs_deltaC", "group_mean_abs_residual_current",
                  "group_mean_abs_residual_avg_anchor"):
            A(f"- {k} = { {kk: (None if vv is None else round(vv, 6)) for kk, vv in (c.get(k) or {}).items()} }")
        A("")
        A("单路融合节点（禁止 A/B cosine，仅描述性）：\n")
        A("| 节点 | 形状 | GT 内范数均值 | 背景范数均值 | small GT/背景 比值 |")
        A("|---|---|---:|---:|---:|")
        for k, v in (s.get("single_path_nodes") or {}).items():
            A(f"| {k} | {v.get('shapes')} | {_fmt(v.get('gt_norm_mean'))} | {_fmt(v.get('bg_norm_mean'))} | "
              f"{_fmt(v.get('small_gt_over_bg_norm_ratio_mean'))} |")
        A("")
        A(f"- Gate D2-VALID：**{(d.get('gate') or {}).get('status')}**\n")

    # ---------------------------------------------------------------- D3
    A("## 5. D3：冻结线性 Probe（§7）\n")
    if not R["d3"]:
        A("- **NOT_RUN**：本轮未执行 probe（见 §REPRODUCE 说明与 [M] 缺口）。\n")
    for name, d in R["d3"].items():
        res = d.get("results") or {}
        A(f"### {name}\n")
        A(f"- Gate D3-VALID：**{(d.get('gate') or {}).get('status')}**")
        A(f"- 预算偏差：{(res.get('protocol') or {}).get('budget_deviation')}")
        A("")
        A("| 节点 | C_in | probe-train 样本 | probe-train 子集 AP | **test pooled AP** | test per-image AP |")
        A("|---|---:|---:|---:|---:|---:|")
        for k, v in (res.get("per_layer") or {}).items():
            t = v.get("test") or {}
            A(f"| {k} | {v.get('C_in')} | {v.get('n_train_samples')} | "
              f"{_fmt(v.get('probe_train_subset_AP'))} | **{_fmt(t.get('pooled_AP'))}** | "
              f"{_fmt(t.get('per_image_AP_mean'))} |")
        A("")
        A(f"- 冻结模型参数校验：unchanged={res.get('frozen_unchanged')}")
        A("")

    # ---------------------------------------------------------------- D4
    A("## 6. D4：CAACP β 受控反事实（§8）\n")
    for name, d in R["d4"].items():
        s = d.get("summary") or {}
        if not s:
            continue
        px = s.get("pixel", {})
        A(f"### {name}（β_on={_fmt(s.get('beta_on'), 6)} → β_off={s.get('beta_off')}）\n")
        A(f"- 全图像素 ON：R={_fmt(px.get('on', {}).get('recall'),6)} P={_fmt(px.get('on', {}).get('precision'),6)} "
          f"F1={_fmt(px.get('on', {}).get('F1'),6)} IoU={_fmt(px.get('on', {}).get('IoU'),6)}")
        A(f"- 全图像素 OFF：R={_fmt(px.get('off', {}).get('recall'),6)} P={_fmt(px.get('off', {}).get('precision'),6)} "
          f"F1={_fmt(px.get('off', {}).get('F1'),6)} IoU={_fmt(px.get('off', {}).get('IoU'),6)}")
        A(f"- Δ(ON−OFF)：{ {k: round(v, 7) for k, v in (px.get('delta') or {}).items()} }")
        lg = s.get("legacy_groups", {})
        A(f"- small 组：microR {_fmt(lg.get('small', {}).get('on_microR'),6)} → "
          f"{_fmt(lg.get('small', {}).get('off_microR'),6)}（Δ={_fmt(lg.get('small', {}).get('delta_microR'),7)}）；"
          f"Hit@25 {_fmt(lg.get('small', {}).get('on_hit25'),5)} → {_fmt(lg.get('small', {}).get('off_hit25'),5)}")
        op = s.get("object_paired_delta") or {}
        A(f"- per-object paired Δrecall：n={op.get('n_common_objects')}，均值={_fmt(op.get('mean_delta_recall_object_level'),7)}，"
          f"图像级 bootstrap CI={_fmt((op.get('image_level_bootstrap_CI') or {}).get('lo'),7)}"
          f"~{_fmt((op.get('image_level_bootstrap_CI') or {}).get('hi'),7)}")
        A(f"- 完整性：only_beta_changed={s.get('integrity', {}).get('only_beta_changed')}，"
          f"β 别名键={s.get('integrity', {}).get('beta_alias_keys')}，"
          f"BN 校验和不变={s.get('integrity', {}).get('bn_checksum_on_unchanged_by_eval')}，"
          f"ON 可复现 max_abs={s.get('integrity', {}).get('on_forward_reproducible_max_abs')}")
        A(f"- Gate D4-COUNTERFACT：**{(d.get('gate') or {}).get('status')}**")
        A(f"- 解释边界：{s.get('interpretation_boundary')}\n")

    # ---------------------------------------------------------------- D3c 分层 probe
    strat = []
    for d in sorted(glob.glob(_g(root, "D3_stratified", "*"))):
        r = _load(_g(d, "probe_stratified.json"))
        if r:
            strat.append((os.path.basename(d), r, _load(_g(d, "gate.json"))))
    if strat:
        A("## 6b. D3c：分层 probe 可读性（自动表）\n")
        for name, r, g in strat:
            A(f"### {name}（{r['protocol'].get('n_images_scored')} 张；gate={(g or {}).get('status')}）\n")
            A("| 节点 | pooled AP | small vs 背景 AP | medium | large | small margin | medium margin | large margin |")
            A("|---|---:|---:|---:|---:|---:|---:|---:|")
            for k, v in (r.get("per_layer") or {}).items():
                gg = v["groups"]
                A(f"| {k} | {_fmt(v.get('pooled_AP_all_gt_vs_bg'))} | "
                  f"{_fmt(gg['small'].get('AP_vs_background'), 5)} | {_fmt(gg['medium'].get('AP_vs_background'), 5)} | "
                  f"{_fmt(gg['large'].get('AP_vs_background'))} | {_fmt(gg['small'].get('margin_mean'), 5)} | "
                  f"{_fmt(gg['medium'].get('margin_mean'), 5)} | {_fmt(gg['large'].get('margin_mean'), 5)} |")
            A("")
            A(f"- 口径：{r['protocol'].get('positives_per_group')} / 负样本：{r['protocol'].get('negatives')}")
        A("")

    # ---------------------------------------------------------------- D2b margin 配对 CI
    mp = None
    for d in sorted(glob.glob(_g(root, "D2", "*"))):
        mp = _load(_g(d, "margin_paired_ci.json"))
        if mp:
            A("## 6c. D2b：cosine small-object margin 的配对 bootstrap CI（自动表）\n")
            A("| 相邻边界 | 同分辨率 | margin 从 → 到 | Δ | 配对 95% CI | 相对 | ≥10% 且同号 |")
            A("|---|---|---:|---:|---|---:|---|")
            for a in mp["adjacent"]:
                rel = a["relative_delta"]
                A(f"| {a['from']} → {a['to']} | {a['same_resolution']} | "
                  f"{_fmt(a['margin_from_mean'], 5)} → {_fmt(a['margin_to_mean'], 5)} | "
                  f"{_fmt(a['delta_margin_mean'], 5)} | "
                  f"[{_fmt(a['delta_margin_CI'][0], 5)}, {_fmt(a['delta_margin_CI'][1], 5)}] | "
                  f"{'—' if rel is None else f'{rel:.2%}'} | "
                  f"{'是' if a['criterion_relative_drop_10pct_same_sign'] else '否'} |")
            A(f"\n- 抽样单位：{mp['method']['paired_unit']}；bootstrap={mp['method']['bootstrap']}，seed=16\n")
            break

    # ---------------------------------------------------------------- D5 跨数据集
    d5 = sorted(glob.glob(_g(root, "D5", "*")))
    if d5:
        A("## 6d. D5：跨数据集一致性（自动表；SYSU 为主对象，其余为收窄复现）\n")
        A("| 数据集 | D0 | D1 | D2 | 正类先验 | small 对象数 | small pooled R | large pooled R | small Hit@25 | CCA@L05 | CCA@L06b | CCA@L07 | ObjP(loose) | ObjR(loose) |")
        A("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        rows = []
        sysd1 = _load(_g(root, "D1", "M1_FULL", "summary.json"))
        sysd2 = _load(_g(root, "D2", "M1_FULL", "stage_summary.json"))
        if sysd1 and sysd2:
            rows.append(("SYSU-CD-256", _load(_g(root, "D0", "M1_FULL", "gate.json")),
                         _load(_g(root, "D1", "M1_FULL", "gate.json")),
                         _load(_g(root, "D2", "M1_FULL", "gate.json")), sysd1, sysd2))
        for d in d5:
            ds = os.path.basename(d)
            s1 = _load(_g(d, "D1", "summary.json"))
            s2 = _load(_g(d, "D2", "stage_summary.json"))
            if s1 and s2:
                rows.append((ds, _load(_g(d, "D0", "gate.json")), _load(_g(d, "D1", "gate.json")),
                             _load(_g(d, "D2", "gate.json")), s1, s2))
        for ds, g0, g1, g2, s1, s2 in rows:
            lg = s1["legacy_groups"]
            enc = s2["encoder_nodes"]
            A(f"| {ds} | {(g0 or {}).get('status')} | {(g1 or {}).get('status')} | {(g2 or {}).get('status')} | "
              f"{_fmt(s1['pixel'].get('positive_prior'))} | {lg['small']['n_objects']} | "
              f"{_fmt(lg['small'].get('pixel_recall_micro'))} | {_fmt(lg['large'].get('pixel_recall_micro'))} | "
              f"{_fmt(lg['small'].get('hit25'))} | "
              f"{_fmt(enc.get('L05_stage3_last_prefix', {}).get('pooled_AP_hist'))} | "
              f"{_fmt(enc.get('L06b_norm4', {}).get('pooled_AP_hist'))} | "
              f"{_fmt(enc.get('L07_network5', {}).get('pooled_AP_hist'))} | "
              f"{_fmt(s1['object'].get('ObjPrecision_loose'))} | {_fmt(s1['object'].get('ObjRecall_loose'))} |")
        A("\n- **跨数据集绝对 AP 不可直接横比**（正类先验差异极大，见 prior 列）；CCA = cosine-proxy pooled AP。\n")

    # ---------------------------------------------------------------- D1 8 连通敏感性
    c8 = _load(_g(root, "D1_conn8", "M1_FULL", "summary.json"))
    c4 = _load(_g(root, "D1", "M1_FULL", "summary.json"))
    if c8 and c4:
        A("## 6e. 8 连通敏感性分析（自动表；主口径 = 4 连通）\n")
        A("| 口径 | small 对象数 | small pooled R | medium | large | ObjP(loose) | ObjR(loose) | N_gt | N_pred |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for tag, s in (("4 连通（主口径）", c4), ("8 连通（敏感性）", c8)):
            lg = s["legacy_groups"]
            A(f"| {tag} | {lg['small']['n_objects']} | {_fmt(lg['small'].get('pixel_recall_micro'))} | "
              f"{_fmt(lg['medium'].get('pixel_recall_micro'))} | {_fmt(lg['large'].get('pixel_recall_micro'))} | "
              f"{_fmt(s['object'].get('ObjPrecision_loose'))} | {_fmt(s['object'].get('ObjRecall_loose'))} | "
              f"{s['object'].get('n_gt')} | {s['object'].get('n_pred')} |")
        A("")

    # ---------------------------------------------------------------- gates + auto assessment
    A("## 7. Gate 汇总与自动评估（§10.2 / §10.3）\n")
    gates = []
    for stage in ("d0", "d1", "d2", "d3", "d4"):
        for name, d in R.get(stage, {}).items():
            g = d.get("gate") or {}
            gates.append((f"{stage.upper()} {name}", g.get("status"), g.get("checks")))
    A("| 阶段 | 对象 | 状态 |")
    A("|---|---|---|")
    for n, s_, _ in gates:
        A(f"| {n.split()[0]} | {' '.join(n.split()[1:])} | {s_} |")
    A("")
    return L, R


def auto_assessment(R):
    """按 §10.2/§10.3 机械评估，不臆造结论。"""
    L = []
    A = L.append
    A("### 自动评估（机械应用预注册阈值）\n")
    # D1
    for name, d in R["d1"].items():
        s = d.get("summary") or {}
        if not s:
            continue
        ck = (d.get("gate") or {}).get("checks", {})
        gap = ck.get("size_gap_pp_pooled_recall")
        A(f"- **D1 size gap**：{name} small pooled recall={_fmt(ck.get('small_pixel_recall_micro'))} vs "
          f"large={_fmt(ck.get('large_pixel_recall_micro'))}，gap={_fmt(gap,2)}pp；"
          f"small n={ck.get('small_objects_ge_100') and '≥100' or '<100'}，"
          f"large n={ck.get('large_objects_ge_100') and '≥100' or '<100'} ⇒ "
          f"{'满足 ≥10pp 且样本充足' if (gap is not None and gap >= TH_D1_SIZE_GAP_PP and ck.get('small_objects_ge_100') and ck.get('large_objects_ge_100')) else '未满足（仅描述性差异或样本不足）'}")
    # D2
    for name, d in R["d2"].items():
        s = d.get("stage") or {}
        if not s:
            continue
        rows = d2_stage_drops(s)
        hits = [r for r in rows if r["criterion_stage_drop_met"]]
        worst = min(rows, key=lambda r: (r["rel_AP_change"] if r["rel_AP_change"] is not None else 0)) if rows else None
        A(f"- **D2 stage drop**：{name} 满足判据的相邻边界数={len(hits)}"
          + (f"（{', '.join(r['from'] + '→' + r['to'] for r in hits)}）" if hits else "")
          + (f"；AP 相对变化最大负向边界={worst['from']}→{worst['to']}"
             f"（{_fmt(worst['rel_AP_change'],4)}，bootstrap 方向={worst['bootstrap_direction']}）" if worst else ""))
    # D3
    if not R["d3"]:
        A(f"- **D3 recoverability**：未执行 ⇒ 无法判定'信息存在但被度量隐藏'，"
          f"所有'某阶段丢失信息'的结论只能停留在 [I]（proxy 层面）")
    else:
        for name, d in R["d3"].items():
            res = d.get("results") or {}
            pl = res.get("per_layer") or {}
            if not pl:
                continue
            enc = {k: (v.get("test") or {}).get("pooled_AP") for k, v in pl.items()
                   if k.startswith("L")}
            late = {k: (v.get("test") or {}).get("pooled_AP") for k, v in pl.items()
                    if not k.startswith("L")}
            if enc:
                lo = min(v for v in enc.values() if v is not None)
                hi = max(v for v in enc.values() if v is not None)
                A(f"- **D3 recoverability**：{name} 编码器节点 probe test AP 区间 "
                  f"[{_fmt(lo)}, {_fmt(hi)}]；下游融合/解码/head 节点 "
                  f"{ {k: (None if v is None else round(v, 4)) for k, v in late.items()} }"
                  f" ⇒ 若下游显著更高，说明'改变判别性'主要由 TAR/DCR 学习到的时相代数提供，"
                  f"而非编码器 A/B 差分本身（注意编码器 probe 输入被固定为 abs(F_A−F_B)，"
                  f"不含可学习的时相组合，属口径差异）")
                l05 = pl.get("L05_stage3_last_prefix", {}).get("test", {}).get("pooled_AP")
                l06 = pl.get("L06b_norm4", {}).get("test", {}).get("pooled_AP")
                if l05 is not None and l06 is not None:
                    d_abs = l06 - l05
                    A(f"- **D3 与 D2 同向性（CAACP 边界 L05→L06b）**：probe AP 变化 "
                      f"{_fmt(d_abs, 4)}（{'下降' if d_abs < 0 else '上升'}）；"
                      f"判据阈值 |Δ|≥{TH_D3_PROBE_AP_DROP} ⇒ "
                      f"{'满足' if abs(d_abs) >= TH_D3_PROBE_AP_DROP else '未满足（不足以称该阶段损失信息）'}")
    # D4
    for name, d in R["d4"].items():
        s = d.get("summary") or {}
        if not s:
            continue
        px = s.get("pixel", {}).get("delta", {})
        lg = s.get("legacy_groups", {}).get("small", {})
        d_small = lg.get("delta_microR")
        d_f1 = px.get("F1")
        verdict = ("微小/非稳健 ⇒ 不支持 β 本身是主因"
                   if (d_small is not None and abs(d_small) < 0.01 and d_f1 is not None and abs(d_f1) < 0.001)
                   else "存在 non-trivial 差异 ⇒ 需按 §10.2 检查 tiny-vs-overall 权衡")
        A(f"- **D4 on/off**：{name} ΔF1={_fmt(d_f1,7)}，Δsmall microR={_fmt(d_small,7)} ⇒ {verdict}")
    A("")
    A("> 结论等级只能按 §10.3 决策矩阵给出；`D2` 与 `D3` 同向才可称『高度怀疑该阶段损失』。\n")
    return L


def write_run_manifest(root, out_path):
    """写 RUN_MANIFEST.json：统一记录代码身份/环境/数据/模型 SHA 与各阶段 Gate。"""
    from tvim_diag_common import code_identity, env_info, sha256_text_lines, DATA_ROOT
    ident = code_identity()
    env = env_info()
    data = {}
    for ds in ("SYSU-CD-256",):
        lp = os.path.join(DATA_ROOT, ds, "list", "test.txt")
        trp = os.path.join(DATA_ROOT, ds, "list", "train.txt")
        data[ds] = {
            "test_list": lp, "test_list_sha256": sha256_text_lines(lp) if os.path.isfile(lp) else None,
            "n_test": sum(1 for _ in open(lp)) if os.path.isfile(lp) else None,
            "train_list": trp, "train_list_sha256": sha256_text_lines(trp) if os.path.isfile(trp) else None,
        }
    models = {}
    for d in sorted(glob.glob(os.path.join(root, "D0", "*"))):
        t = _load(os.path.join(d, "test_result.json")) or {}
        models[os.path.basename(d)] = {
            "checkpoint": (t.get("checkpoint") or {}).get("path"),
            "checkpoint_sha256": (t.get("checkpoint") or {}).get("sha256"),
            "variant_cfg": (t.get("build_info") or {}).get("cfg"),
        }
    gates = {}
    # 扫描任意深度的 gate.json（覆盖 D0/D1/D2/D3/D3_concat/D3_stratified/D4/D1_conn8/D5/*/...）
    for g in sorted(glob.glob(os.path.join(root, "**", "gate.json"), recursive=True)):
        rel = os.path.relpath(g, root).replace(os.sep, "/")
        gates[rel[:-len("/gate.json")]] = (_load(g) or {}).get("status")
    payload = {
        "diagnosis": "TViM-TinyLoss-Diag1",
        "created": env.get("time"),
        "code_identity": ident,
        "env": env,
        "datasets": data,
        "models": models,
        "gates": gates,
        "protocols": {"metrics": _load(os.path.join(root, "P0", "metrics_protocol.json")),
                      "probe": _load(os.path.join(root, "D3", "M1_FULL", "probe_protocol.json"))},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return payload


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--interpretation", default=None,
                    help="追加的解释性章节文件（默认 <root>/INTERPRETATION.md，存在即追加）")
    args = ap.parse_args()
    lines, R = build_report(args.root)
    lines += auto_assessment(R)
    interp = args.interpretation or os.path.join(args.root, "INTERPRETATION.md")
    if os.path.isfile(interp):
        with open(interp, encoding="utf-8") as f:
            lines.append("\n---\n")
            lines.append(f"<!-- 以下为解释性章节，来源：{os.path.basename(interp)} -->\n")
            lines.append(f.read())
        print(f"appended interpretation: {interp}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out} ({len(lines)} lines)")
    if args.write_manifest:
        mp = os.path.join(args.root, "RUN_MANIFEST.json")
        write_run_manifest(args.root, mp)
        print(f"wrote {mp}")


if __name__ == "__main__":
    main_cli()
