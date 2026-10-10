# Run4 交接说明（无人值守恢复用）

> 生成时间：2026-10-10 14:53 CST（训练进行中）。本文件用于**会话中断 / 轮次耗尽**时接手：
> 照此文件即可拿到 Run4 的全部结果，不需要任何额外上下文。

## 1. 现在正在跑什么

| 进程 | 位置 | 作用 |
|---|---|---|
| `run_all.sh` + 4 × `train.py` | 服务器 `bash train_scripts/CASA-TViM/Run4/run_all.sh` | 两波 × 4 库，每库 80,000 optimizer updates |
| `ckpt_backup_watchdog.py` | 服务器 | `last.pth` → `last.pth.bak` 原子备份（写盘崩溃缓解） |
| `auto_post_train.sh` | 服务器 | 当**8 个 run 全部产出完整 TEST 区块、且没有任何 `train.py` 在运行**时，**自动**执行收口（协议审计 → 日志硬门 ×8 → 部署图对象指标 ×8 → 三层裁决 → README markdown） |

映射：GPU0 = CDD + LEVIR，GPU1 = SYSU + WHU；先 `M1_R4CTRL`（fine_tap=0，同期对照），
后 `E6_FET1`（fine_tap=1，唯一正式主实验）。

## 2. 结果会自动出现在哪里

| 产物 | 路径 |
|---|---|
| 完成标记（含收口退出码） | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/AUTO_POST_TRAIN_DONE` |
| 收口守候日志 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/auto_post_train.log` |
| 协议审计 | `.../Run4/{M1_R4CTRL,E6_FET1}/PROTOCOL_AUDIT.json` |
| 三层裁决 | `.../Run4/verdict.json` |
| README 可用 markdown | `.../Run4/README_SNIPPET.md` |
| 对象指标（8 份） | `.../Run4/objects/{variant}__{dataset}.json` |
| 训练日志（正式结果唯一来源） | `.../Run4/{variant}/{dataset}/train_log.txt` |
| checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4/{variant}/{dataset}/` |

## 3. 若 `AUTO_POST_TRAIN_DONE` 尚未出现

按顺序排查：

1. 训练是否还在跑：`pgrep -af "[t]rain.py --dataset" | wc -l`（正常 20 = 4 主 + 16 worker）；
2. 进度：`bash /tmp/r4_poll.sh`（若脚本已丢失，直接看各 `train_log.txt` 的 `[ACTUAL-OPT-STEPS]` 行）；
3. 收口是否卡住：`cat .../Run4/auto_post_train.log`；守候进程是否存活：
   `pgrep -af "[a]uto_post_train"`；
4. 若 8 个 run 都已完成但没自动收口，手工执行：
   ```bash
   source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate casacd
   cd /home/yqwang/projects/CASA-CD
   GPU=0 bash train_scripts/CASA-TViM/Run4/post_train.sh
   ```
   （自带 fail-closed：8 个 run 未完成 → exit 3；协议审计不过 → exit 4）

## 4. 拿到结果后怎么读

1. 先看 `.../Run4/README_SNIPPET.md`（四张表，数字全部来自 `verdict.json`，无人工转抄）：
   每 run 六指标与部署实测 / 对象指标与参数 / 层 B 机制 / 层 C 守门与论文目标；
2. `verdict.json` 顶层 `status` ∈ {INVALID, MECHANISM-FAIL, GUARDRAIL-FAIL,
   MECHANISM-PASS/PAPER-TARGET-FAIL, PAPER-TARGET-PASS}，`decision` 给出 §6.4 决策树的对应处置；
3. 预注册门槛（**不得事后放宽**）写在
   `train_scripts/CASA-TViM/Run4/README.md` §8 与设计文档 §6.3。

## 5. 已固定的纪律（不要在接手时改动）

- 正式结果**只**取自各 `train_log.txt` 最后一个完整 `=== TEST RESULTS ===` 区块；
  `best_F1=*.pth` 的文件名、验证行、`last.pth` 字段都不是正式结果来源；
- 断点恢复是**协议精确、非逐位**（T11 有对照证据），不得声称"精确恢复"；
- 失效产物保存在 `.../Run4/_INVALID_pre_gatefix_20261010/`，标注
  `not_used_for_any_conclusion=true`，**不要删除、也不要用于任何结论**；
- 不得用 Run1/M1 best 微调 E6、不得改 loss/增广/阈值/seed/EMA/超参做搜索。

## 6. 完整证据索引

`train_scripts/CASA-TViM/Run4/EVIDENCE_INDEX.md`（服务器副本
`/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_preflight_checks/EVIDENCE_INDEX.md`）
列出 ①–⑥ 全部已完成的验证与其结论、路径。
