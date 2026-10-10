# CASA-TViM Run-Diag / Diag1 — SYSU 微小变化瓶颈分阶段诊断

> 任务来源：`docs/temporary/CASA-CD 小目标瓶颈分阶段诊断｜DSH 完整执行文档.md`
> 性质：**只读诊断**（不训练新 80K、不改模型、不覆盖历史 checkpoint/日志/JSON）。
> 结论报告：服务器 `$DIAG/DIAGNOSIS_REPORT.md`；本地副本 `docs/temporary/CASA-TViM_Diag1/`。

## 一键运行

```bash
# 服务器
cd /home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Diag1
bash run_diag.sh                 # 默认：p0 + d0(M1) + d1 + d2 + d4 + report
bash run_diag.sh p0 d0 d1 d2 d3 d4 report   # 含 D3 probe
LIMIT=16 bash run_diag.sh p0 d0 d1 d2       # smoke（结果标 dry_run，禁止用于结论）
```

`run_diag.sh` 按 Gate 纪律执行：`P0` 与 `D0(M1)` 必须 PASS 才继续；`D3` 只在 D1/D2 均 PASS 时运行。
所有产物写入 `/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1`（独立目录，不复用历史输出）。

## 新增诊断文件（全部只增不改）

| 文件 | 作用 |
|---|---|
| `analyse/tvim_diag_common.py` | 路径/变体表、SHA256、严格 TEST 解析、checkpoint 唯一性校验、模型构造与严格键核对、混淆矩阵六指标（与 `metric_tool` 同式）、bootstrap、`audit` 子命令（D0） |
| `analyse/tvim_object_metrics.py` | **P0 核心**：正确的 GT 面积分组（tiny_1_15 / tiny_16_63 / small_64_255 / medium / large）、pooled & object-macro 像素 Recall、ObjectHit@1/25/50、真正的对象级 ObjP/ObjR/ObjF1（IoU≥0.10/0.50，一次性最大权重二分匹配）、边界带 ±2/±4 |
| `analyse/tvim_small_error_audit.py` | D1：逐 GT 对象的错误画像 + M1/A2 same-object 配对 Δrecall + 预注册样例图 |
| `analyse/tvim_stage_recoverability.py` | D2：L00–L08 / T / D / head 节点 hook；编码器 A/B cosine proxy + 原生 256² 评价、native occupancy 副口径；CAACP 内部（β、ΔC、βΔC、entropy、分组统计）；重建校验 |
| `analyse/tvim_linear_probe.py` | D3：冻结线性 probe（train 拟合 / val 固定 / test 一次性评估） |
| `analyse/tvim_caacp_counterfactual.py` | D4：β ON/OFF 受控反事实 + 完整性核对 |
| `analyse/tvim_diag_report.py` | 汇总报告 + 预注册判据自动评估 + `RUN_MANIFEST.json` |
| `analyse/tests/test_tvim_object_metrics.py` | P0 的 10 项人造阵列单测（含旧 FP≡0 bug 的回归用例） |
| `analyse/tests/test_tvim_log_parser.py` | 诊断侧日志解析器与官方实现逐字对拍 |

## 关键结论（详见 DIAGNOSIS_REPORT.md）

1. **[F] 口径修复**：旧 `component_pr` 的 `FP` 恒为 0，`small 0.1796` 是伪指标；新正确口径下 M1 的
   small(1–255px, 364 obj) pooled pixel Recall **0.1974**、Hit@25 **0.1896**，与 large(0.8477) 的
   **gap = 65.0pp** —— 现象真实存在，但旧叙事必须改写。
2. **[F]+[I] 叙事更正**：正确指标下的 same-object 配对显示 CAACP **提升**小目标 per-object recall
   （tiny_16_63 +1.24pp、small_64_255 +1.39pp），代价是中/大目标（−4.2/−4.7pp）；
   "CAACP 伤害 small" 的旧结论**撤回**。
3. **[I] 衰减位置**：编码器 proxy 中唯一满足预注册判据的相邻边界是 `L06b(1/16) → L07(1/32)`（AP −41%），
   但跨分辨率；同分辨率的 CAACP 边界 `L05→L06b` 呈 **AP ↑ / small margin ↓15.8%**，且 D3 probe 同向（−0.036）⇒ 该阶段存在**稳健但有限**的证据弱化。
4. **[F]+[I] CAACP β 修正项不是主因**：`||βΔC||/||c_avg|| = 0.19%`；β 置零后 ΔF1 = −2.2e-6、Hit@25 不变。
5. **[I] 输出端不是瓶颈**：decoder refine 与 head logits 的 probe AP 相同（0.9053 / 0.9049）。
6. **[M] 缺口**：编码器 probe 只跑了 `abs(F_A−F_B)` 紧凑口径，需要 `concat(F_A,F_B,|Δ|)` 才能区分
   "早期信息不足"与"proxy 口径不足"——这是下一轮的首要补证项（零 80K）。

## 注意事项

- 服务器工作副本不是 git 仓库 → 代码身份用**逐文件 SHA256** 记录在 `RUN_MANIFEST.json`。
- 诊断输出目录若已存在，脚本会**写入同名文件**；若需保留旧一轮结果，请先改 `DIAG` 目录名（例如加时间戳）。
- 样例图（`D1/M1_FULL/examples/`）仅保留在服务器，未导出到本地/公开仓库。
- `pkill -f` 匹配脚本名时会误杀自己的 SSH shell —— 请用 `pkill -f "[t]vim_..."` 形式或按 PID 处理。
