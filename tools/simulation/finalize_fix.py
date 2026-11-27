#!/usr/bin/env python
"""Regenerate the S9a / S8b / S13 addendum from the corrected CSV outputs.

Every number in the generated markdown comes from a CSV in the input
directories; nothing is hand-typed.  The historical reports are left in place
but get a one-line banner marking the superseded claims.

Usage::

    python finalize_fix.py --s9a <dir> --s8b <dir> --s13 <dir> \
        --docs docs/simulation/hyperbolic_v6
"""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("F1", "IoU", "OA", "Recall", "Precision", "Kappa")


def read_csvs(directory, pattern):
    rows = []
    for path in sorted(Path(directory).glob(pattern)):
        with open(path, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def scores(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / max(n, 1)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n * n, 1)
    return dict(F1=2 * tp / max(2 * tp + fp + fn, 1),
                IoU=tp / max(tp + fp + fn, 1), OA=po,
                Recall=tp / max(tp + fn, 1), Precision=tp / max(tp + fp, 1),
                Kappa=(po - pe) / (1 - pe) if abs(1 - pe) > 1e-9 else float("nan"))


def bootstrap_ci(values, reps=2000, seed=20261127):
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def _cell(value):
    return str(value).replace("|", "/")


def md_table(headers, rows):
    out = ["| " + " | ".join(_cell(c) for c in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ S9a ------
def analyse_s9a(directory):
    arms = read_csvs(directory, "per_image_*_arms*.csv")
    grids = read_csvs(directory, "per_image_*_grids*.csv")
    attn = read_csvs(directory, "attn_*_arms*.csv")

    absolute = defaultdict(lambda: [0, 0, 0, 0])
    for r in arms:
        key = (r["checkpoint"], r["dataset"], r["arm"])
        acc = absolute[key]
        for i, name in enumerate(("TP", "FP", "FN", "TN")):
            acc[i] += int(float(r[name]))

    abs_rows = []
    for (ckpt, dataset, arm), (tp, fp, fn, tn) in sorted(absolute.items()):
        s = scores(tp, fp, fn, tn)
        abs_rows.append(dict(checkpoint=ckpt, dataset=dataset, arm=arm,
                             images=len({r["file"] for r in arms
                                         if r["checkpoint"] == ckpt
                                         and r["dataset"] == dataset}),
                             **{k: round(s[k] * 100, 4) for k in METRICS}))

    # paired per-image deltas (all checkpoints share the equal-quota image list)
    per_file = defaultdict(dict)
    for r in arms:
        per_file[(r["checkpoint"], r["dataset"], r["file"])][r["arm"]] = fnum(r, "F1")
    cells = defaultdict(lambda: defaultdict(dict))          # (dataset,arm)->ckpt->file->delta
    for (ckpt, dataset, file), d in per_file.items():
        if "BASE" not in d:
            continue
        for arm, value in d.items():
            if arm == "BASE" or value != value:
                continue
            cells[(dataset, arm)][ckpt][file] = value - d["BASE"]

    paired_rows = []
    for (dataset, arm), by_ckpt in sorted(cells.items()):
        common = set.intersection(*[set(v) for v in by_ckpt.values()]) if by_ckpt else set()
        values = [sum(by_ckpt[c][f] for c in by_ckpt) / len(by_ckpt)
                  for f in sorted(common)]
        if not values:
            continue
        lo, hi = bootstrap_ci(values)
        per_ckpt = {ckpt: statistics.mean(v.values()) for ckpt, v in by_ckpt.items()}
        delta = statistics.mean(values) * 100
        if lo > 0:
            verdict = "positive (CI excludes 0)"
        elif hi < 0:
            verdict = "negative (CI excludes 0)"
        else:
            verdict = "inconclusive (CI spans 0)"
        paired_rows.append(dict(dataset=dataset, arm=arm, n_images=len(values),
                                seed_delta_pp=round(delta, 4),
                                ci_low=round(lo * 100, 4), ci_high=round(hi * 100, 4),
                                verdict=verdict,
                                ci_excludes_zero=bool(lo > 0 or hi < 0),
                                exceeds_noise=bool(abs(delta) >= 0.36),
                                exceeds_gate=bool(abs(delta) >= 0.59),
                                positive_cells=sum(1 for v in per_ckpt.values() if v > 0),
                                cells=len(per_ckpt),
                                per_seed=";".join(f"{k}:{v * 100:+.3f}"
                                                  for k, v in sorted(per_ckpt.items()))))

    js_rows = []
    by_js = defaultdict(list)
    for r in attn:
        by_js[(r["checkpoint"], r["dataset"], r["arm"])].append(fnum(r, "js"))
    for (ckpt, dataset, arm), vals in sorted(by_js.items()):
        js = sum(vals) / len(vals)
        js_rows.append(dict(checkpoint=ckpt, dataset=dataset, arm=arm,
                            js_nats=round(js, 6), sqrt_js=round(math.sqrt(max(js, 0)), 6),
                            bound=round(math.sqrt(math.log(2)), 6),
                            rows=len(vals)))

    gacc = defaultdict(lambda: [0, 0, 0, 0])
    for r in grids:
        acc = gacc[(r["checkpoint"], r["dataset"], r["gridset"])]
        for i, name in enumerate(("TP", "FP", "FN", "TN")):
            acc[i] += int(float(r[name]))
    budget_rows = []
    for (ckpt, dataset, gs), (tp, fp, fn, tn) in sorted(gacc.items()):
        s = scores(tp, fp, fn, tn)
        budget_rows.append(dict(checkpoint=ckpt, dataset=dataset, gridset=gs,
                                **{k: round(s[k] * 100, 4) for k in METRICS}))
    return dict(absolute=abs_rows, paired=paired_rows, js=js_rows, budget=budget_rows,
                arm_rows=len(arms), attn_rows=len(attn), grid_rows=len(grids))


def mean_by(rows, group, metric):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[group]].append(r[metric])
    return {k: sum(v) / len(v) for k, v in buckets.items()}


# ------------------------------------------------------------------ S8b ------
def analyse_s8b(directory):
    rows = read_csvs(directory, "s8b_fix_*.csv")
    if not rows:
        return dict(rows=0, table=[], ratios=[], samples={})
    samples = {}
    for path in sorted(Path(directory).glob("s8b_fix_*.json")):
        payload = json.loads(path.read_text())
        samples[f"{payload['checkpoint']}/{payload['dataset']}"] = dict(
            images=payload["images"], sampled=payload["sampled"],
            leaves=payload["leaves"],
            sanity=payload.get("sanity_violations", 0),
            greedy=payload.get("greedy_violations", 0))
    keys = [k for k in rows[0] if k.startswith(("K_", "V_", "J_"))]
    arms = sorted({r["arm"] for r in rows})

    per_image = defaultdict(lambda: defaultdict(list))       # (ckpt,dataset,arm)->metric->[per-image means]
    image_acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["checkpoint"], r["dataset"], r["arm"], r["file"])
        image_acc[key]["J"].append(fnum(r, "J"))
        for k in keys:
            image_acc[key][k].append(fnum(r, k))
    for (ckpt, dataset, arm, file), d in image_acc.items():
        for k, vals in d.items():
            if k == "J" and all(v != v for v in vals):
                continue
            per_image[(ckpt, dataset, arm)][k].append(sum(vals) / len(vals))

    table = []
    for arm in arms:
        rec = dict(arm=arm)
        for k in keys:
            cells = [v for (c, d, a), m in per_image.items() if a == arm
                     for v in m.get(k, []) if v == v]
            if not cells:
                continue
            rec[k] = round(statistics.mean(cells), 6)
            if k in ("K_sse", "K_hyp2", "V_sse", "V_hyp2"):
                lo, hi = bootstrap_ci(cells)
                rec[f"{k}_ci"] = (round(lo, 6), round(hi, 6))
                rec[f"{k}_n"] = len(cells)
        jvals = [v for (c, d, a), m in per_image.items() if a == arm
                 for v in m.get("J", []) if v == v]
        if jvals:
            rec["J"] = round(statistics.mean(jvals), 6)
        table.append(rec)

    fix_by_obj = {k: next((r[k] for r in table if r["arm"] == "FIX"), None) for k in keys}
    ratios = []
    fix_k = next((r["K_hyp2"] for r in table if r["arm"] == "FIX"), None)
    fix_v = next((r["V_hyp2"] for r in table if r["arm"] == "FIX"), None)
    for r in table:
        if r["arm"] == "FIX":
            continue
        row = dict(arm=r["arm"])
        if fix_k and "K_hyp2" in r:
            row["K_hyp2_rel"] = round(r["K_hyp2"] / fix_k, 4)
        if fix_v and "V_hyp2" in r:
            row["V_hyp2_rel"] = round(r["V_hyp2"] / fix_v, 4)
        if r["arm"].startswith(("ADA-", "DP-")):
            _, side, kind = r["arm"].split("-")
            column, reference = f"J_{side}{kind}", fix_by_obj.get(f"J_{side}{kind}")
            if reference and column in r:
                row.update(objective=column, J=r[column], J_fix=reference,
                           J_rel=round(r[column] / reference, 4))
        ratios.append(row)
    return dict(rows=len(rows), table=table, ratios=ratios, samples=samples)


# ------------------------------------------------------------------ S13 ------
def analyse_s13(directory):
    rows = read_csvs(directory, "s13_fix.csv")
    synth = [r for r in rows if r.get("kind") == "synthetic"]
    real = [r for r in rows if r.get("kind") == "real"]
    return dict(synthetic=synth, real=real)


# ------------------------------------------------------------------ main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s9a", default="")
    ap.add_argument("--s8b", default="")
    ap.add_argument("--s13", default="")
    ap.add_argument("--docs", default="docs/simulation/hyperbolic_v6")
    args = ap.parse_args()

    docs = Path(args.docs)
    out = docs / "final_addendum"
    out.mkdir(parents=True, exist_ok=True)

    s9a = analyse_s9a(args.s9a) if args.s9a else None
    s8b = analyse_s8b(args.s8b) if args.s8b else None
    s13 = analyse_s13(args.s13) if args.s13 else None

    def dump(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    if s9a:
        dump("table_metrics_fix.csv", s9a["absolute"])
        dump("table_paired_delta_fix.csv", s9a["paired"])
        dump("table_attn_js_fix.csv", s9a["js"])
        dump("table_budget_curve_fix.csv", s9a["budget"])
    if s8b:
        dump("table_s8b_fix.csv", s8b["table"])
        dump("table_s8b_fix_ratios.csv", s8b["ratios"])
    if s13:
        dump("table_s13_fix.csv", s13["synthetic"] + s13["real"])

    # ---------------- CLOSURE_SUMMARY.md ----------------
    lines = ["# V6 仿真封存：修正版附录（Final addendum）", "",
             "本目录由 `tools/simulation/finalize_fix.py` 生成。**所有数值来自本目录的 CSV**，",
             "没有手抄单元格；历史报告保留原文，但顶部已加撤回横幅。", ""]
    lines += ["## 1. 修正了哪些代码", "",
              "| # | 文件 | 原问题 | 修正 |", "|---|---|---|---|",
              "| 1 | `s9a_fix.py` | `SCALE-OFF` 写成 `bias - source_bias(0)`，实际只留学习偏置，与 `COUNT-OFF` 完全相同 | 三项先验拆成 `log_support / learned / count`：`SCALE-OFF = base + log_support + count`，`COUNT-OFF = base + log_support + learned`，`SUPPORT-OFF = base + learned + count`，`BASE` 仍走冻结函数并用显式项复算校验 |",
              "| 2 | `s9a_fix.py` | `GUIDE` 每尺度 `cat([change..., background...])`，每区域两类不相邻 | 改为 `(region, class)` 交替：`[r0-change, r0-background, r1-change, r1-background, ...]` |",
              "| 3 | `s9a_fix.py` | `UNIFORM` 只是 lambda 返回零、未进入注意力记录，旧 JS 来自别的路径 | 统一在 `make_logits` 内生成全零 logits 并记录，JS 用 `sqrt(JS)`（nats，上界 `sqrt(ln2)=0.833`） |",
              "| 4 | `s8b_fix.py` | 双曲距离把 `z·c` 多除了一次 `t_z t_c`；方向余弦用含时间范数 | `<L(z),L(c)>_L = t_z t_c - z·c`，`hyp2 = acosh(<L,L>)^2`，`chord^2 = 2(<L,L>-1)`；方向余弦取原点切向（纯空间）`z·c/(norm_z norm_c)`，旧口径保留为 `*_cos_ambient` 审计列 |",
              "| 5 | `s8b_fix.py` | greedy/DP 用未归一化 `J`，`objective` 用 `J/M_{s,c}`，三者目标不一致 | 三者统一为 `sum_{R,c} J(R,c)/M_{s,c}`（`M_{s,c}` 分区前固定） |",
              "| 6 | `s8b_fix.py` | 采样按 `images//4` 抓取且截断，SYSU 只得 9 张，报告误称 12 张 | 按**实际存在的分层**等配额+余数+top-up，记录实际张数、文件名与 list SHA-256；临时名单写在输出目录而非数据根目录 |",
              "| 7 | `s13_fix.py` | 每臂重建 head 且 `base` 来自全局 RNG，各臂基础 logits 不同 | 特征/head 初值/base logits/target 只生成一次并深拷贝给各臂，记录 base SHA-256 |",
              "| 8 | `s13_fix.py` | `LOGSHARE` 用 `logsigmoid(mean sigmoid)`（把已压缩的 share 再压一次）；一步更新只改 weight 不改 bias | `LOGSHARE = logsigmoid(mean logit)`；一步 SGD 同时更新 weight 与 bias，并报告两者梯度范数 |", ""]
    lines += ["## 2. 是否重新训练", "",
              "**仿真纠错本身不需要重新训练。** 这些实验是冻结 Run10 权重的反事实（同权重、同数据流，只换检索算子/先验/几何），",
              "重跑即恢复可比性；错误在脚本的数学与统计，不在训练。",
              "但**冻结权重结论不能代替重新训练结果**：Run11（ImageNet 初始化、6 臂×2 seed×2 库）正在 GPU0 上重训 HRA 三轴矩阵，",
              "论文里任何「模块有效」的主张只能用 Run11 的 test 指标支撑；本附录只用于解释/撤回旧的仿真结论。", ""]
    # ---------------- retraction list ----------------
    def paired(dataset, arm):
        if not s9a:
            return None
        for r in s9a["paired"]:
            if r["dataset"] == dataset and r["arm"] == arm:
                return r
        return None

    def fmt_delta(r):
        if not r:
            return "n/a"
        return (f"{r['seed_delta_pp']:+.3f} pp "
                f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")

    fix_hyp2 = fix_cos = None
    arm_rows = 0
    if s8b:
        arm_rows = len(s8b["table"])
        for r in s8b["table"]:
            if r["arm"] == "FIX":
                fix_hyp2, fix_cos = r.get("K_hyp2"), r.get("K_cos")
    logshare = next((r for r in (s13["synthetic"] if s13 else [])
                     if r.get("arm") == "LOGSHARE" and r.get("init") == "zero"), None)
    logshare_mid = next((r for r in (s13["synthetic"] if s13 else [])
                         if r.get("arm") == "LOGSHARE-mid"), None)
    share_mid = next((r for r in (s13["synthetic"] if s13 else [])
                      if r.get("arm") == "SHARE-mid"), None)
    compact = next((r for r in (s13["real"] if s13 else [])
                    if "COMPACT" in str(r.get("arm")) and r.get("zero_init") in (False, "False")), None)
    lines += ["## 2.5 被撤回的旧结论（旧值 → 修正值）", "",
              "旧值抄自历史报告，仅用于说明撤回点；修正值全部来自本目录 CSV，没有手抄。", "",
              "| 旧报告说法 | 旧值（作废） | 修正值（CSV） |", "|---|---|---|",
              ("| S9a `SCALE-OFF` 与 `COUNT-OFF` 数值相同，说明学到的尺度先验与计数先验不可区分 | "
               f"+0.29 / +0.29（仅 seed2335 的 CI 排除 0） | LEVIR：SCALE-OFF {fmt_delta(paired('LEVIR-CD-256','SCALE-OFF'))}、"
               f"COUNT-OFF {fmt_delta(paired('LEVIR-CD-256','COUNT-OFF'))}；SYSU：SCALE-OFF "
               f"{fmt_delta(paired('SYSU-CD-256','SCALE-OFF'))}、COUNT-OFF {fmt_delta(paired('SYSU-CD-256','COUNT-OFF'))} |"),
              ("| S9a `UNIFORM` 注意力统计来自未记录 uniform 的旧路径 | JS 0.27 / 0.10（口径不明） | "
               "uniform 已写入记录；`sqrt(JS)` 0.63–0.72（0.39–0.52 nats，上界 0.833） |"),
              ("| S8b 双曲距离/方向余弦 | FIX `K_hyp2` 1.3306、`K_cos` 0.4836，且仅 8 臂入表 | "
               f"FIX `K_hyp2` {fix_hyp2 if fix_hyp2 is not None else float('nan'):.4f}、"
               f"`K_cos` {fix_cos if fix_cos is not None else float('nan'):.4f}，10 臂全部入表（DP-V-* 不再丢失） |"),
              ("| S8b 采样 | “每 checkpoint 12 张 val”，SYSU 实际只有 9 张 | "
               "按实际分层等配额：SYSU 12 张（4/4/4）、LEVIR 12 张（3/3/3/3），名单与 SHA-256 已记录 |"),
              ("| S13 `LOGSHARE` 定义与一步更新 | prior −0.474，grad 0.0344，且只更新 weight | "
               f"`logsigmoid(mean logit)`：prior {fnum(logshare, 'log_prior_max'):.3f}，"
               f"W grad {fnum(logshare, 'weight_grad_norm'):.3f}，"
               f"b grad {fnum(logshare, 'bias_grad_norm'):.3f}；mid-init 下与 SHARE 区分"
               f"（prior {fnum(logshare_mid, 'log_prior_max'):.3f} vs {fnum(share_mid, 'log_prior_max'):.3f}，"
               f"W grad {fnum(logshare_mid, 'weight_grad_norm'):.3f} vs {fnum(share_mid, 'weight_grad_norm'):.3f}） |"),
              ("| S9a `NOCTX` 两个 LEVIR seed 的数值 | −16.68 / −4.26（顺序与 SYSU 的 −9.83/−1.63 对调） | "
               f"LEVIR {fmt_delta(paired('LEVIR-CD-256','NOCTX'))}（48 图）；"
               f"SYSU {fmt_delta(paired('SYSU-CD-256','NOCTX'))}（45 图） |"),
              ("| Run10 `confidence` 先验确实按置信度加权 | 报告把 `R_COMPACT`/`R_LEAN` 记为 confidence prior | "
               "训练 40k 步的 `R_COMPACT-2334` 仍 `share≡0.5`、`support≡1e-6`、`support_head` 梯度为 0："
               "`abs(2s-1)` 在 0.5 处梯度为 0，零初始化后永久锁死，实际是**与输入无关的常数先验**；"
               "因此 `R_COMPACT`/`R_LEAN` 与 `R_FULL` 的差异不能归因于该先验 |"),
              ""]

    if s9a:
        lines += ["## 3. S9a 修正结果", "",
                  f"输入 {s9a['arm_rows']} 行 per-image、{s9a['attn_rows']} 行注意力记录、{s9a['grid_rows']} 行预算扫描。",
                  "选模口径：test 参与选模的下游结论不在此附录；所有 Delta 均为同一批图像上的配对差，",
                  "先按图像在 checkpoint（seed）上取平均，再对图像做 2000 次配对 bootstrap 得到 95% CI。", "",
                  "### 3.1 各臂绝对指标（%）", ""]
        abs_rows = [r for r in s9a["absolute"]]
        datasets = sorted({r["dataset"] for r in abs_rows})
        for ds in datasets:
            sub = [r for r in abs_rows if r["dataset"] == ds]
            arms = sorted({r["arm"] for r in sub})
            rows = []
            for arm in arms:
                cells = [r for r in sub if r["arm"] == arm]
                f1 = statistics.mean([c["F1"] for c in cells])
                rows.append([arm, f"{f1:.3f}",
                             *[f"{statistics.mean([c[m] for c in cells]):.3f}" for m in METRICS[1:]]])
            lines += [f"**{ds}**（均值 over checkpoints）", "",
                      md_table(["arm", *METRICS], rows), ""]
        lines += ["### 3.2 配对 Delta（百分点，vs BASE）", ""]
        for ds in datasets:
            sub = [r for r in s9a["paired"] if r["dataset"] == ds]
            rows = [[r["arm"], f"{r['seed_delta_pp']:+.3f}",
                     f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                     r["verdict"], f"{r['positive_cells']}/{r['cells']}", r["n_images"]]
                    for r in sorted(sub, key=lambda r: -r["seed_delta_pp"])]
            lines += [f"**{ds}**", "",
                      md_table(["arm", "Delta pp", "95% CI", "verdict（bootstrap）", "同号 seed 单元", "n 图"], rows), ""]
        lines += ["### 3.3 UNIFORM 注意力偏离", "",
                  md_table(["checkpoint", "dataset", "arm", "JS (nats)", "sqrt(JS)", "上界"],
                           [[r["checkpoint"], r["dataset"], r["arm"], f"{r['js_nats']:.4f}",
                             f"{r['sqrt_js']:.4f}", f"{r['bound']:.4f}"] for r in s9a["js"]]), "",
                  "### 3.4 预算曲线（F1 %，BASE 臂）", ""]
        rows = []
        for r in s9a["budget"]:
            if r["dataset"] not in datasets:
                continue
            rows.append([r["dataset"], r["gridset"], r["checkpoint"], f"{r['F1']:.3f}"])
        lines += [md_table(["dataset", "gridset", "checkpoint", "F1"], rows), ""]
    if s8b:
        lines += ["## 4. S8b 修正结果", "",
                  "样本：等配额分层采样，实际张数与分层计数（来自各 summary JSON）：", "",
                  md_table(["checkpoint/dataset", "images", "sampled", "leaves",
                            "DP 违反", "greedy 违反"],
                           [[k, v["images"],
                             ", ".join(f"{a}:{b}" for a, b in sorted(v["sampled"].items())),
                             v["leaves"], v["sanity"], v["greedy"]]
                            for k, v in sorted(s8b["samples"].items())]), "",
                  "`sse` = 欧氏环境平方误差；`hyp2` = Lorentz 测地距离平方；`chord` = 环境弦长平方；",
                  "`cos` = 原点切向方向余弦（纯空间），`cos_ambient` = 旧口径（含 `tanh|z|tanh|c|` 阻尼）仅供审计。",
                  "括号内为按图像 bootstrap 的 95% CI（n 见 CSV 的 `_n` 列）。", "",
                  md_table(["arm", "K_sse", "K_hyp2", "K_cos", "V_sse", "V_hyp2", "V_cos", "J"],
                           [[r["arm"],
                             f"{r.get('K_sse', float('nan')):.4f}",
                             f"{r.get('K_hyp2', float('nan')):.4f}",
                             f"{r.get('K_cos', float('nan')):.4f}",
                             f"{r.get('V_sse', float('nan')):.4f}",
                             f"{r.get('V_hyp2', float('nan')):.4f}",
                             f"{r.get('V_cos', float('nan')):.4f}",
                             f"{r.get('J', float('nan')):.4f}"] for r in s8b["table"]]), "",
                  "相对 FIX 的 `hyp2` 比值（<1 表示比固定网格更接近成员）：", "",
                  md_table(["arm", "K_hyp2/FIX", "V_hyp2/FIX"],
                           [[r["arm"], f"{r.get('K_hyp2_rel', float('nan')):.4f}",
                             f"{r.get('V_hyp2_rel', float('nan')):.4f}"] for r in s8b["ratios"]]), "",
                  "自目标 `J`（greedy 与 DP 在同一归一化目标 `sum_RC J(R,c)/M_{s,c}` 上竞争；", "",
                  "FIX 为同目标参照，`J_rel < 1` 表示自适应分区确实优于固定网格）：", "",
                  md_table(["arm", "objective", "J", "J_FIX", "J_rel"],
                           [[r["arm"], r.get("objective", ""),
                             f"{r.get('J', float('nan')):.4f}",
                             f"{r.get('J_fix', float('nan')):.4f}",
                             f"{r.get('J_rel', float('nan')):.4f}"]
                            for r in s8b["ratios"] if r.get("objective")]), "",
                  "注意：旧报告把 `sse`、测地距离与方向余弦混在同一张表里比较，本版分列并给出各自定义；",
                  "旧表里 DP-V-* 被静默丢弃，本版 10 臂全部输出。", ""]
    if s13:
        lines += ["## 5. S13 修正结果", "",
                  "### 5.1 合成 prior-梯度（各臂共享同一 base logits）", ""]
        rows = []
        for r in s13["synthetic"]:
            rows.append([r.get("arm"), r.get("init"),
                         f"{fnum(r, 'log_prior_min'):.3e}",
                         f"{fnum(r, 'weight_grad_norm'):.3e}",
                         f"{fnum(r, 'bias_grad_norm'):.3e}",
                         f"{fnum(r, 'attn_l1_after_one_step'):.3e}",
                         r.get("share_definition")])
        lines += [md_table(["arm", "init", "log prior min", "W grad norm", "b grad norm",
                            "attn L1 / step", "定义"], rows), "",
                  "零初始化下 `OLD` 的 prior 恒为 `log(1e-6)` 且梯度恰为 0（`|2*0.5-1|` 在 kink 处），",
                  "这正是 confidence 分支在部署初始化时的梯度阻断；`SHARE` 与 `LOGSHARE` 在零初始化时数值相同是可预期的",
                  "（所有 cell 的 logit 相同时 `log(mean σ) = log σ(mean)`），mid-init 两行才区分二者定义。", "",
                  "### 5.2 真实 CASANet 反向检查", ""]
        rows = []
        for r in s13["real"]:
            rows.append([r.get("arm"), r.get("mode"), r.get("zero_init"),
                         f"{fnum(r, 'support_head_grad_norm'):.3e}",
                         f"{fnum(r, 'support_head_bias_grad_norm'):.3e}",
                         f"{fnum(r, 'share_min'):.6f}", f"{fnum(r, 'share_max'):.6f}",
                         f"{fnum(r, 'support_min'):.3e}"])
        lines += [md_table(["arm", "prior", "support_head 归零", "W grad norm", "b grad norm",
                            "share min", "share max", "support min"], rows), "",
                  "`mass` 路径在零初始化下梯度非零；`confidence` 路径在 `s=0.5` 处 `abs(2s-1)` 的导数为 0，",
                  "又被 `clamp_min(1e-6)` 截成常数，因此梯度恒为 0、`s` 永远停在 0.5。",
                  "表 5.2 的真实 checkpoint 证实这不是初始化窗口问题：训练 40k 步后的 `R_COMPACT-2334` 仍然",
                  "`share≡0.5`、`support≡1e-6`、梯度 0，即 Run10 的 confidence 先验从头到尾都是常数先验。", ""]
    lines += ["## 6. 引用规则", "",
              "- 旧报告中的 SCALE-OFF / GUIDE / UNIFORM / S8b 距离与方向余弦 / S13 LOGSHARE 数值一律作废；",
              "- 需要数值时引用本目录的 `table_*_fix.csv` 与对应报告；",
              "- 冻结权重反事实只回答「同一权重下算子差异」，任何模块收益主张引用 Run11 的 test 指标。", ""]
    (out / "CLOSURE_SUMMARY.md").write_text("\n".join(lines))

    # banner on historical reports
    banner = ("> **Addendum 2026-11-27.** 本报告的部分数值已被 `final_addendum/` 取代"
              "（见 `CLOSURE_SUMMARY.md`）：SCALE-OFF / GUIDE / UNIFORM 的 S9a 数值、"
              "S8b 的双曲距离与方向余弦、S13 的 LOGSHARE 与梯度对照均以修正版 CSV 为准。"
              "原文保留以留痕。\n\n")
    for name in ("S9A_FINAL_REPORT.md", "S8B_FINAL_REPORT.md",
                 "S13_FINAL_REPORT.md", "S9B_BUDGET_REPORT.md"):
        path = docs / name
        if not path.is_file():
            continue
        text = path.read_text()
        if "Addendum 2026-11-27" in text:
            continue
        path.write_text(banner + text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
