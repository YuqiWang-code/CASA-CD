# Run7 HANDOFF — HFAE 消融正在服务器执行（12 run / 3 波）

> 生成时间：2026-10-11 10:29（首版）；**最后更新 10:52**。本文件是**状态与恢复索引**，不是结果报告。
> 结果只能由 `finish_run7.sh` 在 12 个 run 全部跑完后产出。
>
> ⚠️ **本文件里的"进度/剩余时长"都是带时间戳的快照，必然随时间过期。**
> 请**以命令实测为准**，不要把下面任何进度数字当作当前值：
>
> ```bash
> # 当前进度（四库累计 optimizer.step / 80000）
> for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
>   printf '%-14s %s\n' "$d" "$(grep -E '^.ACTUAL-OPT-STEPS' \
>     /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/M1_R7CTRL/$d/train_log.txt | tail -1)"
> done
> # 督导是否还活着（日志每 120 s 一行）
> tail -3 /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7_supervisor.log
> ```
> `…/Run7/E9_HFAE_A/` 与 `…/E10_HFAE_B/` **出现** = 对应波次已启动（波二/波三就绪的信号）。

## 0. 新接手的 agent：按这个顺序做（约 5 分钟就能接手）

1. **读本文件 §1 / §1.1 / §1.2 / §1.3** —— 现状、ETA、健康读数、解读锚。
2. **跑一遍交付状态核对**：`train_scripts/CASA-TViM/Run7/README.md` **§9**（三条命令）。
   全 PASS 才继续；有 FAIL 先查那一条。
3. **看训练到哪了**：`tail -3 …/Run7_supervisor.log`；
   `…/Run7/E9_HFAE_A/` 是否出现 = 波二是否已开始。
4. **若 12 个 run 都齐了**（每变体 4/4 VALID）：
   ```bash
   cd /home/yqwang/projects/CASA-CD
   bash train_scripts/CASA-TViM/Run7/finish_run7.sh; echo "FINISH_RC=$?"
   ```
   `rc` 含义见 `README.md` §8.2（四个码都已端到端实测）。**不要**为了"更快"去改任何参数。
5. **写结论**：按 `README.md` **§8.4**（五种 verdict 写法）+ **§8.5b**（根 README 结果节骨架）
   + **§8.5**（三张表，`verdict.md` 里已直接产出）。**负结果照写**；硬目标未达≠失败，
   但**绝不可**回调 λ/阈值/seed。
6. **若中途出事**：督导死了 ≠ 训练死了。按 `README.md` **§8.8** 恢复
   （注意 `RESUME_SAME_RUN` 陷阱）；产物只改名不删除。

> **不要重验** §3 表里已验过的东西——重验只会花时间并引入 churn（每次改收口工具都要重跑
> 12-run 全量排练）。


## 1. 一句话现状

`M1_R7CTRL` / `E9_HFAE_A` / `E10_HFAE_B` × 四库（CDD/LEVIR/SYSU/WHU）各**严格 80,000** `optimizer.step`，
共 12 个 run，正在 2×RTX 5090 上按**三波串行**跑；监督进程自动接续下一波。**无错误**。

| 项 | 值 |
|---|---|
| 监督进程 | `/tmp/run7_supervisor.pid`（本轮 pid `2320921`），日志 `…/CASA-TViM/Run7_supervisor.log` |
| 波次 → 变体 | 1 → `M1_R7CTRL`，2 → `E9_HFAE_A`，3 → `E10_HFAE_B` |
| GPU 分配 | GPU0 = CDD + LEVIR；GPU1 = SYSU + WHU（每卡 2 个 run） |
| 并发 | 4 个 run 同时跑（每卡 2 个）；batch 32；`--exact_max_steps 1`；~5.8 step/s/run |
| 进度 | **随时实测**（见文首命令）。快照：10:29 波一 19.1–21.2k；10:52 约 26.3–29.4k / 80k（0 个 TEST 区块属正常：只在 80k 收尾写出） |
| ETA（**实测修正**，10:38） | 波一 ≈ **13:43**；三波全部 ≈ **22:15**（详见 §1.1） |
| 资源 | ckpt 目录稳定 **635 MB**；日志 ≈0.1 MB @80K；磁盘 2.6 T 空闲 |

### 1.1 ETA 的实测修正（此前用"总量速率"估算，偏乐观约 1.5 h）

波次门要求**四个 run 全部 VALID**，所以波一完成时间 = **最慢**那个 run 的完成时间，
不能用总量速率估算。10:38 逐 run 采样（相隔 120 s）：

| dataset | 速率 (步/s) | 剩余步 | 剩余时长 |
|---|---|---|---|
| CDD | **5.217** ← 最慢，决定波一 | 57,777 | **3.08 h** |
| LEVIR | 5.575 | 57,477 | 2.86 h |
| SYSU | 6.250 | 57,500 | 2.56 h |
| WHU | 6.200 | 55,634 | 2.49 h |

⇒ 波一 ETA **13:43**；波一时长 ≈ **4.27 h**（09:27 启动）⇒ 三波全部（12 run ÷ 4 并发 = 3 批）
≈ **22:15**（此前写 20:40，是用总量速率算的，偏乐观约 1.5 h）。

**安全性**：督导 20 小时上限自 09:31 起算 ⇒ 到 **次日 05:31**，22:15 远在门内 ✅
**波二/波三速率**：边缘头仅 7–10k 参数、aux 损失在 64² 上，额外算力可忽略（预计 <1% 差异），
故沿用 4.27 h/波；若明显更慢，以 `_r7_eta.sh` 复测为准。


### 1.2 波一中期健康读数（2026-10-11 10:32，**仅供健康核对，不是结果**）

| dataset | best F1 @~26% | 当前 epoch | 该库总 epoch | epoch×iters/epoch | ≈80,000? |
|---|---|---|---|---|---|
| CDD | **0.9538** | 64/256 | 256 | 256×313 = 80,128 | ✓ |
| LEVIR | **0.9089** | 92/359 | 359 | 359×223 = 80,057 | ✓ |
| SYSU | **0.8319** | 54/214 | 214 | 214×375 = 80,250 | ✓ |
| WHU | **0.9475** | 120/431 | 431 | 431×186 = 80,166 | ✓ |

**这些数字的用途**：① 训练健康（F1 仍在涨、Loss 在降、无 `[RESUME]`/异常行）；
② **独立佐证步数预算**：四库 `总 epoch × iters/epoch` 全部 ≈80,000；
③ 给后来者一个**解读锚**：CTRL 已接近 Run1 `M1_FULL` 的**终值**
（Run1：CDD 0.9718 / LEVIR 0.9107 / SYSU 0.8347 / WHU 0.9507），
符合"同构对照"的预期 ⇒ 之后看 E9/E10 就是与这条线比。
④ 每个 ckpt 目录**恰好 1 个** `best_F1=*.pth`（收口 `pick_best_ckpt` 依赖此不变量）。

### 1.3 解读锚（10:51 更新）：CTRL 会落在哪，以及"硬目标"实际要求多少提升

| dataset | CTRL bestF1 @~1/3 训练 | vs Run1 终值 | **距硬目标** |
|---|---|---|---|
| CDD | 0.9578 | −0.0140 | **−0.0222** |
| LEVIR | 0.9092 | −0.0015 | −0.0158 |
| SYSU | 0.8319 | −0.0028 | −0.0181 |
| WHU | 0.9487 | −0.0020 | −0.0013 |

**怎么用这张表**（写结论时必须分清两件事）：

1. **预注册的接受标准**只看**相对 CTRL**：`≥3/4 库 F1 ≥ CTRL` 且 `SYSU F1 > CTRL`。
2. **硬目标（98.00/92.50/85.00/95.00）是努力方向，不是必达门**（新研究政策）。
   由上表可见，若 CTRL 最终落在 Run1 终值附近，则"达到硬目标"意味着 E9/E10 相对 CTRL 需要
   **CDD +2.2pp / LEVIR +1.6pp / SYSU +1.8pp / WHU +0.1pp** —— 对一个**辅助监督**模块而言这是
   相当高的要求。因此：
   - 若 verdict 为 `MECHANISM-PASS` 但未达硬目标 ⇒ 照实写"机制有效、未达硬目标"，
     **不得**因此回调任何参数；
   - 若 verdict 为 `GUARDRAIL-FAIL` / `MECHANISM-FAIL` ⇒ 记为负结果 + 归因（同样有价值）。
3. 上表是**训练中期读数，不是结果**；CTRL 的终值以 `finish_run7.sh` 产出的四库 F1 表为准。



## 2. 收口只需一条命令（已 12-run 全量排练，`FINISH_RC=0`，耗时 207 s）

> ⚠️ **别依赖那个 bounded watcher**：`.claude/_r7_watch.ps1` 是 **11 小时**上限
> （2026-10-11 09:28:37 启动 ⇒ 约 **20:28–20:31 自行退出**），而按 §1.1 的实测修正
> 三波全部结束约 **22:15** ⇒ **watcher 会在完成前约 1 小时 45 分钟就到期**。
> 也就是说**最后约 1.75 小时完全没有 watcher 覆盖**。请直接轮询
> `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7_supervisor.log`（每 120 s 一行），
> 不要等 watcher 的通知。


```bash
cd /home/yqwang/projects/CASA-CD
bash train_scripts/CASA-TViM/Run7/finish_run7.sh; echo "FINISH_RC=$?"
```

**rc 含义、五种 verdict 的写法、必写的三张表、禁做清单** ⇒ 见 `train_scripts/CASA-TViM/Run7/README.md`
**§8 收口 runbook**（务必按它写结论，尤其是 `GUARDRAIL-FAIL`/`MECHANISM-FAIL` 也要记负结果）。

## 3. 已经验证过的东西（**不要重验**，重验只会浪费时间并引入 churn）

| 项 | 证据 | 出处 |
|---|---|---|
| 收口链 6 步整条 | 12-run 全量 DRY 排练 `FINISH_RC=0`（207 s） | README §7.9 |
| 逐 run 硬门 `check_run.py` | 排练 12/12 PASS；真实 CTRL 清单 9 条协议断言全过 | README §7.5 / §7.8 |
| 波次衔接（唯一自动交接点） | 逐字提取督导逻辑实测 → `4 4 320000` GATE PASS；3/4 正确拦截 | README §7.12 |
| 对象级复算 | 真实 E9/E10 权重三变体 `missing=0 unexpected=0` | README §7.6 |
| `object_metrics` → `verdict` 接口 | 真实 JSON 驱动，8 个必需字段全命中 | README §7.7 |
| deploy 参数门（三形态） | eval.py 实测 4.880/4.880/**4.890** M，FLOPs 2.6747/2.6747/**2.7170** G | README §7.13 |
| 注入点 | 读码核验：注到 **DCR 的 1/4 输入**（非 refine 之后） | README §7.14 |
| 预注册 ↔ 代码 | 逐条对拍一致；§6「不改既有算子」经 git 取证 | 预注册文档 §7 + README §7.10 |
| 崩溃恢复 | `RESUME_SAME_RUN` 陷阱实测确认 | README §8.8 |
| 单元/集成测试 | T0–T11 12 checks 0 FAIL；analyse 142 tests OK | 本会话 round 18 |

## 4. 如果会话在 run 跑完前结束

1. **先判断督导是否还活着**（pidfile + `kill -0`；日志 >5 min 无新行即为死）——
   注意**督导死了不等于训练死了**，4 个 `train.py` 是独立进程。
2. **按 `Run7/README.md` §8.8 恢复**。关键陷阱：督导调 `run_all.sh --wave N` 时**不带**
   `RESUME_SAME_RUN`，而**非空 ckpt 目录会被拒绝启动** ⇒
   - 波一未完成：直接重启督导即可（安全）；
   - 波二/波三之后：**必须先 `export RESUME_SAME_RUN=1`** 再启动（否则会在波二处 ABORT）。
3. 恢复后逐条核对：每变体 4/4 VALID、`check_run` 无 FAIL、日志无 `budget already reached`、
   `run_manifest.json` 的 `resume_source` 不得指向别的变体/库。
4. **绝不**删除任何 ckpt/日志（只 `_INVALID_*` / `_SUPERSEDED_*` 改名）。

## 5. 冻结项（见结果之前已固定，**任何情况下不得事后修改**）

- `λ = 0.2` **禁止搜索**；两部署形态（`aux_only` / `aux_inject`）固定；
- 判据与阈值以 `train_scripts/CASA-TViM/Run7/SELECTION_LOCK.json` 的
  `acceptance_criteria_frozen_numeric` 为**唯一规范形式**（冻结于 2026-10-11T09:35:00+08:00，
  当时 **12 个 run 都还没有任何 TEST 区块**，故无从观察结果）；
- **硬目标（98.00/92.50/85.00/95.00）是努力方向而非必达门** —— 未达不等于失败，
  **但绝不可因此回调 λ/阈值/seed**；
- 既有算子（`ss2d.py` / `caacp_ss2d.py` / `str_dcr.py` / `str_reparam.py` 等）Run7 期间未改动（git 已取证）。

## 6. 关键路径

| 用途 | 路径 |
|---|---|
| 训练日志 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/<variant>/<dataset>/train_log.txt` |
| 权重 | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run7/<variant>/<dataset>/` |
| 收口产物 | `…/Run7/_RESULTS/{summary,verdict,INDEPENDENT_AUDIT}.json` + `check_run_*.json` |
| 对象级指标 | `…/Run7/_OBJECT_METRICS/object_metrics_*.json` |
| 收口工具 | `train_scripts/CASA-TViM/Run7/{finish_run7.sh,check_run.py,collect_results.py,run7_verdict.py,independent_audit.py}` |
| 对象级工具 | `analyse/run7_object_metrics.py` |
| 预注册文档 | `docs/temporary/CASA-CD_Run7_高频边缘辅助监督HFAE_预注册_2026-10-11.md` |
| **收口 runbook** | `train_scripts/CASA-TViM/Run7/README.md` §8（rc→动作、五种 verdict 写法、三张表、禁做清单） |
| 督导 | `train_scripts/CASA-TViM/Run7/supervise_waves.sh` + `…/Run7_supervisor.log` |

## 7. 关于 `Run7/` 目录里的 `_r7_*.sh`（一次性诊断探针，**不是流水线的一部分**）

服务器上 `train_scripts/CASA-TViM/Run7/` 除了被跟踪的正式文件外，还有约 **34 个** `_r7_*.sh`
（我本会话做验证/监测时上传的一次性探针，如 `_r7_survey.sh`、`_r7_eta.sh`、`_r7_rerun.sh`、
`_r7_rc4.sh` …）。它们**不影响任何流程**，已验证：

| 断言 | 实测 |
|---|---|
| 收口链是否依赖目录列表 | `finish_run7.sh` 里 `glob`/`listdir` 出现 **0** 次 ⇒ 全部显式路径，探针无法影响 |
| `check_mapping.py` 是否被探针干扰 | 在 34 个探针在场时实跑 = **PASS（14/14）** |
| `check_imports.py` | 实跑 = **PASS** |
| 审计的 `glob` | 只用于 `object_metrics_*.json` 与 `SOURCE_FILES.txt` 清单，与探针无关 |

⇒ 看到这些文件**不必清理**（清理反而有改错路径的风险）；正式流水线只认
`SOURCE_FILES.txt` 里登记的 **26 个条目**（其中包含 `HANDOFF.md` 自身）。

