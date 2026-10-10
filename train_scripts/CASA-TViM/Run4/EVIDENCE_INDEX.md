# Run4 证据索引（截至 2026-10-10 训练进行中）

本文件只索引**训练前/训练中**已产出的可核验证据与其结论。**任何 Run4 TEST 成绩都不在此列**——
正式结果只在两波 8×80,000 步跑完后由 `post_train.sh` 产出。

服务器路径前缀：
`$OUT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_preflight_checks`、
`$DRY=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_DRY`、
`$PF=/home/yqwang/outputs/CASA-CD/diagnostics/TViM-Run4-Preflight-20261010`、
`$LOG=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4`、
`$CK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4`。

## 一、代码身份与部署

| 证据 | 位置 | 结论 |
|---|---|---|
| 代码身份（19 文件 SHA256 + 与 Diag1 12 文件基线逐条差异） | 仓库 `train_scripts/CASA-TViM/Run4/SOURCE_IDENTITY.json`；服务器同路径 | 服务器与本地 **IDENTITY_MATCH**；Diag1 基线中仅 `casa_tvim_str_net.py` / `train.py` / `eval.py` 改动，其余 9 个逐字节相同 |
| 训练期身份未漂移 | `$CK/M1_R4CTRL/SYSU-CD-256/run_manifest.json` 的 `source_code_sha256` | `TRAINING_CODE_IDENTITY_UNCHANGED`（`models/train.py` = `16d13c33e594451d`，为 P0 修复后的版本） |

## 二、训练前门（⑶⑷⑸）

| 证据 | 位置 | 结论 |
|---|---|---|
| 既有 `casa_tvim_str` smoke（旧行为不变） | `$OUT/smoke_casa_tvim_str.log` | **ALL OK**（T-CA-2 `retained=744/805 worst=0.00e+00`） |
| T0–T9 验收（最终代码身份） | `$OUT/test_run4_fine_tap_FINAL2.log` | **PASS=53 / FAIL=0 / SKIP=0**（含 epoch-0 前向 `torch.equal`、整模型 train↔deploy `max_abs=0.000e+00` 且 disagreement=0、FET 折叠 FP64 1.2e-15、deploy 4,894,110 / 2.7313 G unsupported=20） |
| §10.2 互补性 preflight | `$PF/gate.json`、`$PF/preflight.json`、`$PF/per_object.csv` | **PASS**（4000 张 / 12.8 s；`n_missed_small=295`、`rescue_rate=0.5085`、`gap=0.1303`、按图 bootstrap 95%CI **[0.0669, 0.1934]**；复现 Diag1 small recall 0.19742 / Hit@25 0.18956 / n=364） |
| 真实数据 3-step dry-run（两变体） | `$DRY/E6_FET1/dry_run.json`、`$DRY/M1_R4CTRL/dry_run.json` | **RESULT=PASS**（几何增广同步 20/20 探针、label `gray>=128` 逐位一致、batch32 峰值 ≈9.8 GB/进程） |

## 三、门禁与工具链预校验（任何结果出现之前）

| 证据 | 位置 | 结论 |
|---|---|---|
| 日志硬门 `check_run.py` 正反例 | `$OUT/GATE_PREVALIDATION.json` → `check_run_cases` | 2 正例 rc=0；5 负例 rc≠0 且命中正确检查项；历史 Run3 日志被正确拒绝 |
| 真实格式区块的**接受**演练 | 同上 → `accept_rehearsal_on_real_blocks` | 真实 E6/CTRL 区块（仅步数改写为 80000）→ **ALL PASS / rc=0** |
| 对象指标口径四库标定 | `$OUT/objcheck4/CALIBRATION.json`（+4 个 per-dataset JSON） | **`OBJECT_PASS_MATCHES_DIAG1_D5`**：SYSU 0.19742/0.18956/0.53200/0.76744、LEVIR 0.70459/0.59924/0.91284/0.85262、CDD 0.73691/0.72735/0.85918/0.79399、WHU 0.64197/0.45570/0.78235/0.81762；F1 同时复现 Run1 best 文件名 |
| checkpoint 级 T10 证据 | 对象 JSON 的 `train_params` / `deploy_params` / `fet_deploy_conv_shape` | CTRL 5,084,017 → 4,880,190（四库一致）；E6 5,097,938 → 4,894,110，FET 折叠卷积 `[96,144,1,1]` |
| T11 断点恢复 + 对照 | `$OUT/T11_RESUME_EVIDENCE.json` | **协议精确、非逐位**：步数/LR/optimizer/四项 RNG 逐项恢复；对照（同 seed 两次连续从头）差异数与"连续 vs 恢复"完全相同（1098/1302、1288）⇒ 不声称"精确恢复" |
| 崩溃恢复 runbook | `$OUT/GATE_PREVALIDATION.json`（T11 段与本索引同目录） | 截断 `last.pth` 确实加载失败；`.bak` 可加载；`cp last.pth.bak last.pth` 恢复成功 → `RUNBOOK_OK` |
| 包装脚本崩溃重试 | 同上 → `wrapper_crash_retry_rehearsal` | 第一次尝试 SIGKILL(137) → `[RETRY 1/3]` → 第二次自动 resume → 跑满预算 → 收尾调用硬门；退出码由硬门决定 |
| 协议审计工具 | `$LOG/M1_R4CTRL/PROTOCOL_AUDIT.json`；工具 `train_scripts/CASA-TViM/Run4/audit_protocol.py` | 运行中波次四库 **PASS**（步数预算/epoch 算术/exact_max_steps/manifest 全字段/LR 计划逐点）；负例（区块步数 79875、区块 80000 但 manifest 30K）均被拒 |
| 收口脚本 fail-closed 守卫 | 工具 `train_scripts/CASA-TViM/Run4/post_train.sh`（实测于训练中途） | 8 个 run 未全部产出完整 TEST 区块时 **exit 3** 且不写 `verdict.json`/`objects/`；协议审计不过则 **exit 4**；仅 `FORCE=1` 可产出已记录的部分裁决 |
| README 报告机器生成 | `analyse/run4_fet_report.py --mode markdown`（`post_train.sh` 第 4 段 → `$LOG/README_SNIPPET.md`） | 四张表（每 run 六指标+部署实测 / 对象指标+参数 / 层 B / 层 C）全部取自 `verdict.json`，杜绝人工转抄；已在真实短跑 ckpt 上渲染验证 |
| 运行时防护 | `$LOG/ckpt_backup_watchdog.log` | `last.pth` → `last.pth.bak` 原子备份持续运行（只备份能成功 `torch.load` 的文件） |
| 同期 CTRL 轨迹对照（SYSU） | 工具 `analyse/run4_progress_eta.py` 同目录的轨迹核对；数据源 `$LOG/M1_R4CTRL/SYSU-CD-256/train_log.txt` vs Run1 `M1_FULL` 同名日志 | 31 个可比 epoch 的 ΔF1 均值 **−0.0008**（区间 [−0.0158, +0.0168]），CTRL best 0.8326 vs Run1 官方 0.8347 ⇒ **exact-80K 协议修正未实质改变轨迹**，文档要求的"≥ 同期 CTRL / 不退守门"比较才有意义 |
| 重启一致性抽查 | 同上（重启前波次已判 INVALID，仅作对照） | 四库同 epoch ΔF1 ≤ |0.003|、正负号混合 ⇒ 重启无系统性偏差 |
| 运行状态快照 | `$LOG/STATUS.md`（`analyse/run4_progress_eta.py --write-status` 生成） | 逐 run 步数/速率/ETA 与收口命令，可随时刷新 |
| 训练日志健康 | `$LOG/M1_R4CTRL/*/train_log.txt` 计数 | 四库 **0 error / 0 NaN**，仅各 8 条启动期 xFormers `not available` 告警；`run4_run_all.log` 无 ABORT/FAIL 标记 |
| 自动收口守候 | `train_scripts/CASA-TViM/Run4/auto_post_train.sh`；日志 `$LOG/auto_post_train.log`；完成标记 `$LOG/AUTO_POST_TRAIN_DONE` | 等 8 个 run 全部产出完整 TEST 区块后**自动**执行 `post_train.sh`（只读、自带 fail-closed 守卫）⇒ 即使监控会话中断，第 ⑧ 步结果也一定被产出 |

## 四、失效产物隔离

| 证据 | 位置 | 结论 |
|---|---|---|
| 第 1 波（缺区块内步数标记） | `$CK/_INVALID_pre_gatefix_20261010/README.json` + 对应 log 目录同名子目录 | **不删除**、移出正式路径、标注 `not_used_for_any_conclusion=true`；后处理工具在结构上够不到该树（已实测） |

## 五、尚未产出（依赖两波训练结束）

1. 8 个 run 的 `[ACTUAL-OPT-STEPS] 80000` 与正式 TEST 区块；
2. 部署图对象指标 ×8；
3. `verdict.json` 三层裁决（§6.3 / §6.4）与 `README_SNIPPET.md`。

产出命令：`GPU=0 bash train_scripts/CASA-TViM/Run4/post_train.sh`（五段：协议审计 → 日志硬门 ×8 →
部署图对象指标 ×8 → 三层裁决 → README markdown；前两段失败即 fail-closed 中止）。
