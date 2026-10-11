# Run7 HANDOFF — HFAE 消融正在服务器执行（12 run / 3 波）

> 生成时间：2026-10-11 10:29（会话仍在进行）。本文件是**状态与恢复索引**，不是结果报告。
> 结果只能由 `finish_run7.sh` 在 12 个 run 全部跑完后产出。

## 1. 一句话现状

`M1_R7CTRL` / `E9_HFAE_A` / `E10_HFAE_B` × 四库（CDD/LEVIR/SYSU/WHU）各**严格 80,000** `optimizer.step`，
共 12 个 run，正在 2×RTX 5090 上按**三波串行**跑；监督进程自动接续下一波。**无错误**。

| 项 | 值 |
|---|---|
| 监督进程 | `/tmp/run7_supervisor.pid`（本轮 pid `2320921`），日志 `…/CASA-TViM/Run7_supervisor.log` |
| 波次 → 变体 | 1 → `M1_R7CTRL`，2 → `E9_HFAE_A`，3 → `E10_HFAE_B` |
| GPU 分配 | GPU0 = CDD + LEVIR；GPU1 = SYSU + WHU（每卡 2 个 run） |
| 并发 | 4 个 run 同时跑（每卡 2 个）；batch 32；`--exact_max_steps 1`；~5.8 step/s/run |
| 进度（10:29） | 波一 4 库 **19.1k–21.2k / 80k**（0 个 TEST 区块，属正常：只在 80k 收尾写出） |
| ETA | 波一 ≈ **13:15**；三波全部 ≈ **20:40** |
| 资源 | ckpt 目录稳定 **635 MB**；日志 ≈0.1 MB @80K；磁盘 2.6 T 空闲 |

### 1.1 波一中期健康读数（2026-10-11 10:32，**仅供健康核对，不是结果**）

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


## 2. 收口只需一条命令（已 12-run 全量排练，`FINISH_RC=0`，耗时 207 s）

> ⚠️ **别依赖那个 bounded watcher**：`.claude/_r7_watch.ps1` 是 **11 小时**上限
> （2026-10-11 09:28:37 启动 ⇒ 约 **20:28–20:31 自行退出**），而三波全部结束约 **20:40** ——
> **watcher 会在完成前约 10 分钟到期**。最后一段请直接轮询
> `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7_supervisor.log`（每 120 s 一行）
> 或跑 `_r7_survey.sh`，不要等 watcher 的通知。


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
| 督导 | `train_scripts/CASA-TViM/Run7/supervise_waves.sh` + `…/Run7_supervisor.log` |
