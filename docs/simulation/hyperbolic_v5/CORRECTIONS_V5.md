# CORRECTIONS_V5 —— 对本轮任务书（V4 复核）的逐条状态

依据：`docs/temporary/CASA-CD_V4仿真复核与下一轮实验执行方案.md`（665 行，基准 main@`38c7342`）。
本文件只记录**已经跑过、有产物**的事；未跑的一律写"未做"，不把计划当结论。

## 0 状态总表

| 条款 | 内容 | 状态 | 产物 |
|---|---|---|---|
| A0 | 冻结来源核查（protocol/选模/哈希/可用臂） | ✅ 完成 | `results/runtime_inventory.md`（本文件 §1） |
| A1 | S7b 半径重复缩放修正 + 交叉验证 | ✅ 完成 | `tools/simulation/s7c_capture.py`、`S7C_REPORT.md` §1 |
| A2 | parity 补齐（aux/kv/support/二值/各级 logit） | ✅ 完成（smoke 逐位 0.0） | 同上，`results/s7c_parity_note.md` |
| A3 | C2c 最小修复（K 档/行中心化 a/geo_norm/corr/JS/count） | ❌ 未做 | — |
| A4 | C3b 3×3 交叉、R1b/R2b 口径核对 | ❌ 未做 | — |
| A5 | 本文件 | ✅ 完成 | 本文件 |
| S7c | 真实特征诊断（4 checkpoint × 2 库） | ✅ 完成（60 val + 32 train/库） | `S7C_REPORT.md`、`results/s7c_*.csv` |
| S9a | 真实反事实（BASE/NOCTX/池化/距离/聚合 12 臂） | ✅ 完成 | `S9A_REPORT.md`、`results/s9a_*` |
| S8b | 同预算自适应分配 | 🟡 第一层（表示保真）完成；下游层未做 | `S8B_REPORT.md` |
| S9b | 预算/全量参考反事实 | ❌ 未做 | — |
| S13a-0 | support 零点失活复现 | ✅ 完成（V4 轮） | `docs/simulation/hyperbolic_v4/results/log_s13a.txt` |
| S13a-0b | 用真实 `log(support)` 路径重跑 | ❌ 未做 | — |
| S13a-1 | 冻结先验反事实 | ❌ 未做 | — |

## 1 A0 冻结来源核查（已闭合）

| 项 | 结果 |
|---|---|
| manifest `protocol` | `casa_pyramid_40k_testselect_heads_detail_v12r3`（4 套 checkpoint 全同） |
| 选模 | `selection_split=val`、`selection=best val F1@0.5`、`independent_test=True` |
| 完成步数 | SYSU 40125 / LEVIR 40182 |
| runtime `config.py` 的 `PROTOCOL` | `casa_minor_run10_lean_v1` —— **与 manifest 的 protocol 是不同字段**（Run10 精简协议 vs pipeline 协议），不是矛盾 |
| 源码哈希 | `hyperbolic_context.py 3bf6ee3a…`、`manifold_context.py 9d041b32…`、`lorentz_ops.py b1ac4ce6…`、`train_CASA.py 2aec1d86…`（与 `runtime_manifest.json['files']` 一致）；`train_scripts/.../config.py` 不在该 manifest |
| 可用臂 | R_FULL 四套齐全；R_COMPACT/R_LEAN 只有 SYSU；R_RES-2335 只有 SYSU；D_CG/D_CTRL 为 `kv_tokens=0/module=none`（无 HRA） |
| 结论 | 四个 R_FULL checkpoint 全部 val-selected，可用于机制选择（不涉及 test） |

## 2 A1 半径口径修正

- 旧口径：`asinh(‖z‖/√D)` 作用在**已经除过 √d 的 z** 上（重复缩放）；且 S7b 的 Q 半径取自
  与本次不同的张量路径，两套数值不可直接比较。旧 S7b 的半径数字（0.75–1.34）**作废**。
- 新口径：`rho = asinh(‖z‖)`，z 是真正送入 `lift` 的坐标；Q/K 已除 √d 一次，V 不除。
- 独立交叉验证：同一批原型用 `acosh(t)`（`t = x_0`）重算，与 `asinh(‖z‖)`
  的 max 偏差 **7.0e-07**（FP32 级），说明口径自洽。
- `memlegacy_*` 列保留了同一张量在旧口径下的值，用于量化口径差异（例如 K 成员在 s0：
  新 1.248 vs 旧 0.279），但不作为结论。

## 3 四条需要撤下或改写的历史表述

| 旧表述 | 现状 | 依据 |
|---|---|---|
| "K 被池化压到 Q 的 0.6–0.7，说明压缩损伤大" | **失效**：Q 与 K 来自不同流，比值无物理意义；应看"原型/成员保留率"：K 0.79–0.90、V 0.67–0.96 | `S7C_REPORT.md` §3 |
| "半径对变化没有超出 guide 的增量信息（已确定）" | **改写为**：全网格 ΔAUC = −0.003…+0.0024；SYSU 三尺度为正（CI 不含 0 但幅度 0.002），LEVIR 为 0 或负 → **无跨库一致的可利用增量** | §2 |
| "C2c 的 17–39% V 差是几何效应" | **仅受控条件成立**：那是同温度下的合成构造，98–99% 可被单一温度+行偏置吸收 | V4 `CORRECTIONS_V4.md` |
| "S8 的 B=32 等于 32 KV" | **失效**：B 是区域数，实际原型数 = 2B；S8b 必须按 32 区域→(8,8,8,8)=64 原型重新定义 | V4 §4（仍未修，属 S8b 工作项） |

新增的两条正面结论（本轮真实证据）：

1. **查询在近侧、键在远侧**：K 成员半径中位 0.9–1.2，Q 中位 0.78–1.17（LEVIR 更高）。
   由 `C = 2·min(ρ_Q, ρ_K)`，角度可分辨上限由 **Q** 决定，而不是由池化后的键决定。
2. **压缩损失主要在 V 侧且与数据集相关**：SYSU 的 V 保留 0.67–0.80（K 0.83–0.90），
   LEVIR 的 V 0.74–0.96。方向一致性 A=0.88–0.95、ESS/p=0.61–0.83，收缩来自方向抵消
   而不是权重弥散。

## 4 为什么本轮先做 S7c 而不是 S9a

S9a 的每个臂都要在同一 checkpoint 上做**在线算子替换 + 逐级 parity**，并且必须先把
"基准臂可复现"钉死（任务书 §5 S9a-0）。A1/A2 的采集修正直接决定了 S9a 里比较的半径/温度
口径是否正确；如果带着旧口径直接做 S9a，得到的是"口径差"而不是"几何差"。
本轮把 A0–A2 与 S7c 一次做透，S9a 留到下一轮按 §5 的 BASE→NOCTX→池化 2×2→距离/聚合 2×2
顺序做，避免又一轮"数字作废"。

## 5 复现命令

```bash
# 服务器，env casacd
cd /home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/runtime/tools_simulation
S7B_THREADS=6 python -u s7c_capture.py --arm R_FULL-2334 --dataset SYSU-CD-256 \
  --val-limit 60 --train-limit 32 --parity 0 --probe-cap 40000
# 本地汇总（把服务器 analysis/simulation/S7c/<arm>_<dataset>/ 的 CSV 拉回 .tmp/s7c/ 后）
python tools/simulation/s7c_aggregate.py
```

单次运行约 5 分钟（60 val + 32 train，CPU 6 线程，与同机其它任务共享负载）。

## 6 本轮新增结果（S9a / S8b-A）

| 实验 | 结论 | 产物 |
|---|---|---|
| S9a 算子反事实（12 臂 × 4 checkpoint × 8 图） | 几何替换对 F1 影响 ≤0.4 pp（SYSU）/≤0.2 pp（LEVIR）；NOCTX 掉 10.1/5.9 pp、UNIFORM 掉 2.3/12.1 pp ⇒ **机制有无比几何形式重要一个量级** | `S9A_REPORT.md` |
| S9a 先验/温度 | DISTONLY 在 LEVIR 与 BASE 持平、SYSU −1.05 pp；PRIORONLY 在 LEVIR −12.8 pp ⇒ 先验与距离的相对重要性跨库不同 | 同上 |
| S9a 标定 | `a = 0.674`、行中心化残差 0.738 ⇒ 欧氏/测地平方距离不是常数倍关系，"一个温度吸收几何差"只对注意力分布近似成立 | 同上 |
| S8b-A 同预算分区 | 自适应 dyadic 在 E 与 H 上**同时更差**（E 1.23–2.78×、H 1.16–2.11×），均匀网格已接近最优 ⇒ 换分区拿不到保真收益 | `S8B_REPORT.md` |

## 7 本轮对任务书 5 个问题的回答

1. **修正后是否改变旧结论、哪些数值失效**：半径"没有增量信息"改为"无跨库一致的可利用
   增量"（SYSU +0.002 AUC、LEVIR 0）；K/Q 比、"池化压到 0.6–0.7"失效；S7b 半径数值作废。
2. **真实半径是否提供 guide 之外的判别信息**：有但极小（P3−P1 ≤ +0.0024 AUC），且跨库不
   一致，不足以支撑专门机制。
3. **同预算自适应是否优于固定网格、E/H 谁更好**：不优。S8b-A 上四种自适应分区全部更差；
   S9a 上 E/H 替换的预测差 ≤0.4 pp。没有证据支持"自适应分配"作为下一步结构。
4. **几何与压缩差异能否传到 V 聚合/回写/预测**：注意力可以差很多（JS 0.03–0.34）而预测
   几乎不变；V 聚合换欧氏会使 FP 明显上升（SYSU 1850→2463）但 F1 基本不变 ⇒ 差异被回写
   与门控吸收，最终预测对几何不敏感。
5. **是否触发 S11/S12，或应精简 HRA**：不触发。按任务书 §9 的停止规则，本轮证据支持
   "保留 HRA 机制、冻结其几何形式"，把预算留给 support 修复（S13a）与预算效应（S9b）。
