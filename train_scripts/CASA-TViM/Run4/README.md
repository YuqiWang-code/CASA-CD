# CASA-TViM Run4 — R4-FET1（1/4 细尺度证据 1×1 可折叠旁路）

设计文档：`docs/temporary/CASA-CD_Run4_小目标瓶颈结构改进与实验设计_2026-10-10.md`（§2/§3 结构、§5 改动清单、
§6 预注册门槛、§7 T0–T11、§8 执行顺序、§10.2 前测门）。

> **状态：可执行设计 + 代码已实现，尚未产生任何 Run4 正式训练结果。**
> 本 README 不预填任何 Run4 TEST 成绩，也不声称已验证成功。

## 1. 唯一结构变量

| 变体 | `fine_tap` | 其它结构 | 用途 |
|---|---:|---|---|
| `M1_R4CTRL` | 0 | CAACP rank/current=1、TAR/DCR full、D=96、FRH=0、FS-TAR=0 | 同期唯一变量对照（排除 exact-80K 实施校正差异） |
| `E6_FET1` | 1 | 与 CTRL 完全相同，**仅新增** 1/4 尺度 1×1 FET | 唯一正式主实验 |

FET 定义（`models/model/str_fine_tap.py`，`P=f1a`、`Q=f1b` 取自共享编码器 `norm0`，1/4、48C、64²）：

```
D  = |Q − P|                       # abs 是卷积之前的输入特征构造
U  = W_pq * [P, Q] + b_pq          # 96×96×1×1 (bias)
V  = W_diff * D                    # 96×48×1×1 (no bias)，零初始化
T  = γ (U + V),  R' = R + T        # γ 零初始化；加在 DCR refine **之后**（不进 SiLU/RepLocalBlock）
```

* 训练图新增 **13,921** 参数；部署图折叠为单条 `Conv1x1(144→96, bias)`，新增 **13,920**；
  部署总量 **4,894,110 ≤ 5,000,000**（CTRL 4,880,190）。
* 融合 FP64 拼核、最后一次 cast 到 FP32；`γ=0` + `W_diff=0` ⇒ epoch-0 与关掉 FET **逐位一致**。
* 新模块**最后构造**且 `torch.random.fork_rng` 局部初始化 ⇒ 不消耗全局 RNG，公共权重逐位不变。

## 2. 精确 80,000 optimizer updates（P0 协议修正）

`--exact_max_steps 1`：`train_epoch(..., remaining_steps)` 在第 N 次 `optimizer.step()` 后 break，
`cur_iter += executed_steps`，`cur_iter >= 80000` 立即退出；poly 归一化分母固定 `args.max_steps=80000`，
warmup200 / BCE+Dice / Adam(2e-4, 0.9/0.99, wd=1e-4) / backbone_lr_ratio 0.1 全部不变。
日志出现 `[ACTUAL-OPT-STEPS] 80000`。`last.pth` 额外保存 `actual_steps`、`rng_state`、
`protocol_version="run4_exact80k_v1"`；`arch.json` / `run_manifest.json` 记录 `fine_tap*`、
`protocol_version`、`source_code_sha256`，eval 侧逐字段严格核对。
**默认 `--exact_max_steps 0`（历史整 epoch 口径）**，只有 Run4 脚本显式打开 1 —— 历史 Run1–3 的可复现性不受影响。

## 3. 目录与路径

```
train_scripts/CASA-TViM/Run4/
  _gen_scripts.py            生成下面 8 个 train_*.sh 与 run_all.sh
  M1_R4CTRL/train_<DS>.sh    fine_tap=0（先跑）
  E6_FET1/train_<DS>.sh      fine_tap=1（后跑）
  run_all.sh                 两波；每波 4 库并发（GPU0=CDD+LEVIR，GPU1=SYSU+WHU）
  check_run.py               收尾硬门：只认最后一个完整 TEST 区块 + [ACTUAL-OPT-STEPS] 80000
  audit_protocol.py          协议审计：步数预算/epoch 算术/exact_max_steps/manifest 全字段/
                             LR 计划逐点/完成后区块内步数标记；写 PROTOCOL_AUDIT.json
  post_train.sh              收口四段：协议审计 → check_run ×8 → 部署图对象指标 ×8 → verdict.json
  gpu_concurrency_probe.py   每 GPU 并发 2 个 batch32 进程的显存 probe（R8）
  ckpt_backup_watchdog.py    把可加载的 last.pth 原子备份为 last.pth.bak（写盘崩溃缓解）
  SOURCE_IDENTITY.json       Run4 代码身份（与 Diag1 manifest 的逐文件差异记录）

/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4/{M1_R4CTRL,E6_FET1}/<DS>/
/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/{M1_R4CTRL,E6_FET1}/<DS>/train_log.txt
```

## 4. 执行顺序（照做）

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate casacd
cd /home/yqwang/projects/CASA-CD
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}

# ① 现有 M1 原生 smoke 保持通过（不改旧行为）
cd models && python smoke_test.py --mode casa_tvim_str \
  --pretrained_weight_path ../pretrained_weight/tinyvim_s_1000e.pth \
  --tinyvim_pretrained_weight_path ../pretrained_weight/tinyvim_s_1000e.pth --gpu_id 0

# ② T0–T9（退出码必须 0；需 CUDA selective-scan 内核）
python test_run4_fine_tap.py --tinyvim-pretrained-weight-path ../pretrained_weight/tinyvim_s_1000e.pth \
  --device cuda:0 --real-dataset-root /share_datasets/CD/SYSU-CD-256 \
  --real-test-list /share_datasets/CD/SYSU-CD-256/list/test.txt --limit 16

# ③ §10.2 互补性 preflight（10 min 上限；gate 非 PASS ⇒ 到此为止，不消耗 8×80K）
cd /home/yqwang/projects/CASA-CD
DIAG=/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1
OUT=/home/yqwang/outputs/CASA-CD/diagnostics/TViM-Run4-Preflight-20261010
python analyse/tvim_run4_tap_preflight.py --device cuda:0 --dataset SYSU-CD-256 \
  --run Run1 --variant M1_FULL --probe-dir "$DIAG/D3_concat/M1_FULL" --out-dir "$OUT"
python -m json.tool "$OUT/gate.json" | head -80

# ④ 真实数据 3-step dry-run（T7/T8）
CUDA_VISIBLE_DEVICES=0 python models/run4_dry_run.py --variant E6_FET1 --device cuda:0 --steps 3 \
  --dataset-root /share_datasets/CD/SYSU-CD-256 --train-list /share_datasets/CD/SYSU-CD-256/list/train.txt \
  --output-dir /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_DRY/E6_FET1
python models/test_run4_fine_tap.py --device cuda:0 \
  --tinyvim-pretrained-weight-path pretrained_weight/tinyvim_s_1000e.pth \
  --dry-run-json /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_DRY/E6_FET1/dry_run.json  # T8 复核

# ⑤ 正式训练：两波 × 4 库（先 CTRL，后 E6）
bash train_scripts/CASA-TViM/Run4/run_all.sh

# ⑥ 收口：协议审计 → 8 个日志硬门 → 部署图对象指标 ×8 → 三层裁决
GPU=0 bash train_scripts/CASA-TViM/Run4/post_train.sh
# post_train.sh 内部四段（任一失败都会体现在汇总行与退出码）：
#   0) audit_protocol.py --require-finished 1   （LR 计划 / 步数预算 / manifest / 区块内步数标记）
#   1) check_run.py × 8                          （只认最后一个完整 TEST 区块 + 硬门）
#   2) run4_fet_report.py --mode object × 8      （部署图对象指标，含 train/deploy 参数实测）
#   3) run4_fet_report.py --mode verdict         （§6.3 三层裁决 → verdict.json）
```

**单独运行协议审计**（训练中也可跑，用于随时核对协议；训练未结束时会因缺少 `finished` 而 FAIL，
这是预期行为，去掉 `--require-finished` 即只查协议本身）：

```bash
python train_scripts/CASA-TViM/Run4/audit_protocol.py \
  --log-root /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4 \
  --ckpt-root /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM --run Run4
```

## 5. 训练后测量口径预校验（在任何 Run4 结果出现之前完成）

`analyse/run4_fet_report.py --mode object` 是第 ⑦ 步的**唯一**对象指标口径。用 Run1 `M1_FULL` 四库历史
best 的**只读软链**（临时目录，不写任何历史目录）跑一遍该口径，逐项对齐 Diag1 §14 D5：

| 数据集 | small 对象数 | small pooled Recall | Hit@25 | ObjPrecision | ObjRecall | F1 / IoU |
|---|---:|---:|---:|---:|---:|---|
| SYSU | 364 (ref 364) | **0.19742** (0.1974) | **0.18956** (0.1896) | **0.53200** (0.5320) | **0.76744** (0.7674) | 0.8347 / 0.7163 |
| LEVIR | 1572 (1572) | **0.70459** (0.7046) | **0.59924** (0.5992) | **0.91284** (0.9128) | **0.85262** (0.8526) | 0.9107 / 0.8360 |
| CDD | 6767 (6767) | **0.73691** (0.7369) | **0.72735** (0.7274) | **0.85918** (0.8592) | **0.79399** (0.7940) | 0.9718 / 0.9451 |
| WHU | 79 (79) | **0.64197** (0.6420) | **0.45570** (0.4557) | **0.78235** (0.7824) | **0.81762** (0.8176) | 0.9507 / 0.9060 |

结论 **`OBJECT_PASS_MATCHES_DIAG1_D5`**（证据：服务器
`$OUTObj/objcheck4/CALIBRATION.json`，`$OUTObj=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_preflight_checks`）。
四库 F1 同时复现 Run1 best 文件名（0.9718 / 0.9107 / 0.8347 / 0.9506），说明部署图 eval 也复现了官方成绩。

裁决脚本自身的决策树已用合成 TEST 区块双向演练：畸形区块（IoU 与 `F1/(2−F1)` 不符）→ `INVALID`
且打印具体失败检查项；自洽区块 → `PAPER-TARGET-PASS`。

## 6. 断点恢复（T11 实测）与崩溃缓解

T11 用小规模可控实验验收（SYSU 前 320 张 train / 128 张 test，batch32 → 10 iter/epoch，
`--max_steps 40 --exact_max_steps 1`，`fine_tap=1`）：

* **A**：一次连续跑完 40 步；
* **B**：在 `last.pth` 完整写入 `epoch=2 / actual_steps=20` 的瞬间用 SIGSTOP 冻结确认、
  再 SIGKILL，然后自动 resume 跑完剩余 20 步；
* **C**（对照）：**同 seed 16 的第二次连续从头 run**，新进程、新目录。

| 比较 | 键集合 | 不同张量 | max\|Δw\| | optimizer 状态不同项 | 步数 / 末步 LR |
|---|---|---:|---:|---:|---|
| A vs B（连续 vs 恢复） | 相同 | 1098/1302 | 1.97e+01 | 1288 | 40 / 40，7.23062774795963e-07（相同） |
| **A vs C（对照：两次连续从头）** | 相同 | **1098/1302** | 1.09e+00 | **1288** | 40 / 40，同上 |

**结论（如实记录，不得含糊）**

1. 恢复路径的**结构性事实全部正确**：`[MANIFEST] validated`（逐字段）、
   `[RESUME] actual_steps=20 (epoch=2, iters/epoch=10)`、`rng_state`（torch CPU / CUDA / numpy / python
   四项）恢复、per-group LR（0.1× / 1.0×）与 optimizer 状态恢复、`[EXACT-STEPS]` 预算续算、最终恰好 40 步。
2. **逐位等价不可达，且原因不在恢复路径**：对照组 A vs C 是**两次完全独立、无中断、同 seed**
   的从头训练，其不同张量数与 optimizer 状态差异数与 A vs B **完全相同**（1098/1302、1288），
   只有数值幅度不同。⇒ 本项目训练管线（`cudnn.benchmark=True` + 非确定性卷积核）
   **跨进程即不可逐位复现**，A vs B 的差异不能归因于 resume。
3. 因此按设计文档纪律**不声称"精确恢复"**；正确表述是
   **"协议精确恢复（protocol-exact），非逐位恢复"**。证据：
   `$OUTObj/T11_RESUME_EVIDENCE.json`（`$OUTObj=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_preflight_checks`）。
4. **崩溃缓解（零代码改动）**：`ckpt_backup_watchdog.py` 周期性把**能被成功 `torch.load`** 的
   `last.pth` 原子复制为 `last.pth.bak`（不触碰 `last.pth`、不介入训练进程、不改训练代码身份）。
   若 `last.pth` 因写盘中崩溃而半截化（实测会抛 `EOFError` 使自动恢复失败），可
   `cp last.pth.bak last.pth` 继续，最多损失一个 epoch；若备份也不可用，则按设计文档用
   **独立新目录**从头训练，不覆盖历史文件。

## 7. 事故与修复记录（2026-10-10，波次 1 重启）

**现象**：第 1 波 `M1_R4CTRL` 跑到 ~6.8k/80k 步时，用**真实**短跑（320 样本、`max_steps=30`）
取出真实 TEST 区块做门禁演练，发现区块里**没有 `[ACTUAL-OPT-STEPS]`**。

**根因**：`models/train.py` 里这两行被加到了 `test_best()`（changevit 路径）而不是
`test_best_strfusion()`（casa/STR 路径）。Run4 走的是后者，因此正式日志的最后一个完整 TEST
区块缺少设计文档 §6.2 / §7-T10 要求的 `[ACTUAL-OPT-STEPS] 80000`。

**为什么必须重启而不是放宽门禁**：§6.4 第 1 条明确「P0 INVALID → 修复后重跑 smoke + dry-run；
未达 80K 的训练不能入结果表」；`check_run.py` 的步数门是**预注册**的，事后放宽属禁止行为。
且两个变体必须跑在**同一份实现**上，若只修 E6 波次则 CTRL/E6 不同版本。

**处置**（全部留痕）：

1. 停机（SIGKILL 4 个 `train.py`；用 `[p]ython` 括号模式避免 `pkill -f` 误杀自己的 SSH shell）；
2. 修 `train.py`：把 `[ACTUAL-OPT-STEPS]` / `[PROTOCOL-VERSION]` 移入 `test_best_strfusion()`
   的 TEST 区块（紧接 `[BEST-F1]`），并撤回 `test_best()` 里的误加；
   `source_code_sha256` 随之更新，`SOURCE_IDENTITY.json` 已重算并上传；
3. **真实短跑复核**（非合成日志）：E6 与 CTRL 的 TEST 区块都出现 `[ACTUAL-OPT-STEPS] 30` 与
   `[PROTOCOL-VERSION] run4_exact80k_v1`，`check_run.py` 除 `steps==80000`（短跑预期）外**全 PASS**；
4. **重新验收**：T0–T9 `PASS=53 / FAIL=0 / SKIP=0`（含 SHA 身份核对），两个 dry-run `RESULT=PASS`；
5. 旧产物**不删除**，整体移到 `Run4/_INVALID_pre_gatefix_20261010/`（ckpt 与 log 各一份），
   附 `README.json` 标注 `not_used_for_any_conclusion=true`；
6. 在新代码上**同时重启两个波次**。

**代价**：损失约 35 min GPU 时间（~6.8k 步），换取两变体同版本 + 预注册门禁完整。

## 8. 预注册门槛（不得事后放宽）

* **层 A（P0）**：deploy ≤5M；train/deploy 双 batch 二值 disagreement=0；TEST 区块完整；`[ACTUAL-OPT-STEPS] 80000`。
* **层 B（SYSU 机制，绝对 + 同期增量同时满足）**：small pooled Recall ≥0.2474 且 ≥CTRL+0.05；
  Hit@25 ≥0.2396 且 ≥CTRL+0.05；ObjRecall ≥CTRL+0.01；ObjPrecision ≥CTRL；
  <256px 未匹配预测 ≤CTRL；F1 ≥CTRL。
* **层 C（守门 + 论文目标）**：CDD ≥97.18、LEVIR ≥91.07、SYSU ≥83.47、WHU ≥95.07 且不低于同期 CTRL；
  四库同时 ≥98.00 / 92.50 / 85.00 / 95.00 才是 `PAPER-TARGET-PASS`。

`gate.json` / `verdict.json` 非 PASS ⇒ 按 §6.4 决策树**停止该线**，不去试 FET 1/8、3×3、FRH+FET，
也不以阈值/loss/seed 搜索挤分；失败时保留日志与权重做机制归因。

## 9. 禁止的捷径

用 Run1/M1 best 微调 E6；沿用旧伪 small F1；用 1/32 cosine 代理推断信息总量丢失；提高 3×3 参数越过 5M；
`disagreement>0` 仍发 PASS；`strict=False` 载错 checkpoint；覆盖 Diag1/Run1–3 产物；
改变 loss / 增广 / 0.5 严格阈值 / seed / EMA / 超参做搜索。
