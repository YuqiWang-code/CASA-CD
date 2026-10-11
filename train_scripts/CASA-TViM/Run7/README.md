# Run7 — HFAE（前两层频率分解高频 → 边缘检测辅助监督）执行索引

> **当前状态：预注册完成，代码已实现并通过本地可跑的 T0/T2/T8/T9/T10/T11；待服务器跑全量 T0–T11
> + SYSU dry-run，通过后按波次开训。** `training_runs_executed = 0`。
> 依据 README「研究目标与实验纪律（2026-10-11 修订）」：硬目标是**努力方向**；**每轮改进都上服务器
> 跑完整 80K**；训练前诊断从宽，**有效性由服务器 80K 消融判定**。

## 0. 冻结输入

| 项 | 值 |
|---|---|
| 预注册文档 | `docs/temporary/CASA-CD_Run7_高频边缘辅助监督HFAE_预注册_2026-10-11.md` |
| 决策锁 | `train_scripts/CASA-TViM/Run7/SELECTION_LOCK.json` |
| 新模块 | `models/model/str_hf_edge.py`（唯一的架构新增） |
| 接线文件 | `models/model/casa_tvim_str_net.py`、`models/train.py`、`models/eval.py` |
| 验收测试 | `models/test_run7_hf_edge.py`（T0–T11） |
| 真实数据 dry-run | `models/run7_dry_run.py` |
| 编排 | `train_scripts/CASA-TViM/Run7/run_all.sh --wave {1,2,3,all}` |
| `protocol_version` | `run7_hf_edge_exact80k_v1`（**由 run4 升级**，因训练目标改变） |
| 预训练 | `pretrained_weight/tinyvim_s_1000e.pth`（sha 短 `f63e58da8109dbb1`） |

## 1. 机制一句话

从 **stage0（1/4，HF 36ch，HF 占比 75%）** 与 **stage1（1/8，HF 32ch）** 的频率分解高频分支
（`SS2D.forward_core` 里的 `x_high = self.local_conv(x_high)`，**只读 hook，不改 `ss2d.py`**）
接出证据 → 中高频增强 → **变化边缘检测辅助监督** → 梯度回流高频分支以**重塑表征**。

**为什么是新机制**：Run5 否证的是**无监督读出**、F14 否证的是**手工证据通道**、Run2/Run4 否证的是
**末端旁路**、Run6 否证的是**读出层门控**。本机制改的是**主干算出来的东西**，且 tap 在 Run5 从未
接过的 **1/4**。去掉边缘监督即退化为已被否决的路线 ⇒ **监督是承重部件**（已获批准，λ=0.2 固定）。

## 2. 三个变体（deploy 参数实测）

| variant | `--hf_edge` | `--hf_edge_form` | train 图参数 | **deploy 参数** | 增量 |
|---|---|---:|---:|---:|---:|
| `M1_R7CTRL` | 0 | `none` | 5,084,017 | **4,880,190** | — |
| `E9_HFAE_A` | 1 | `aux_only` | 5,091,058 | **4,880,190** | **+0（零推理代价）** |
| `E10_HFAE_B` | 1 | `aux_inject` | 5,094,226 | **4,890,366** | +10,176 |

（硬门 ≤5,000,000 全部满足；`switch_to_deploy()` 对 `aux_only` 删除整个边缘头，
对 `aux_inject` 只丢监督专用 `edge_out`。）

## 3. 执行顺序

| 步 | 内容 | 阻断条件 |
|---|---|---|
| S0 | 上传 + 远端 `sha256sum -c` 对拍（`_r7_stage.py`） | 任一不符 ⇒ STOP |
| S1 | **服务器 T0–T11 全量**（`python models/test_run7_hf_edge.py --device cuda:0`） | 任一 FAIL ⇒ 不训练 |
| S2 | **SYSU dry-run**（`models/run7_dry_run.py`，3 步，三个变体） | 任一 gate FAIL ⇒ 不训练 |
| S3 | **波一 `M1_R7CTRL` × 4 库** 80K | 波内有 run 失败 ⇒ 不启动下一波 |
| S4 | **波二 `E9_HFAE_A` × 4 库** 80K | 同上 |
| S5 | **波三 `E10_HFAE_B` × 4 库** 80K | 同上 |
| S6 | 收口：**一条命令** `bash train_scripts/CASA-TViM/Run7/finish_run7.sh`（12-run 门 → 对象级只读复算 → 裁决+归因 → 对照表；退出码 0=已裁决 / 2=尚有 run 未完成） | — |

GPU 布局：GPU0 = CDD + LEVIR，GPU1 = SYSU + WHU，每卡 2 并发、batch 32。
每个 run 自带 `last.pth` 崩溃恢复（仅限**本 run**，上限 3 次），**绝不复用 Run1–Run6 目录**。

## 4. 判据（预注册，见决策锁 `acceptance_criteria`）

- **主**：四库 F1 vs 同期 `M1_R7CTRL`；
- **次**：SYSU small(1–255px) pooled Recall / Hit@25（冻结参考 **0.1974 / 0.1896**）、
  未匹配小预测 CC（冻结 **2430**）、ObjP/ObjR（冻结 **0.5320 / 0.7674**）；
- **守门**：中/大对象 pooled Recall 不回退；CDD/WHU 不明显恶化；
- **机制自检**：`[HFAE-GRAD]` 首步 `hf_tap_grad_norm > 0`、epoch-0 `torch.equal(CTRL,EXP)`、
  真实 batch 折叠 disagreement = 0、deploy ≤5M、`[HFAE-LOSS]` 逐 epoch 下降；
- **接受**：`E9`/`E10` 至少一个在 **≥3/4 库 F1 ≥ CTRL** 且 **SYSU F1 > CTRL** ⇒ 记为机制有效；
  否则记**负结果 + 归因**（照 Run6 惯例，不回调阈值）。

## 5. 禁止项

1. **不得搜索 λ**（=0.2 预注册固定）；不得改动机制/两形态/预算/判据后重跑当"同一实验"；
2. 不得用 Run1–Run6 权重 warm start；不得多 seed；不得改 batch/步数/阈值/增强；
3. **不得改 `ss2d.py` / `caacp_ss2d.py` / `str_dcr.py` / `str_reparam.py` 任何现有算子**；
   高频 tap 只允许通过**只读 forward hook**；
4. 产物只做 `_INVALID_*` / `_SUPERSEDED_*` **改名隔离，绝不删除**；
5. `edge_out` 是**监督专用**，不得在推理期参与主输出（T7 折叠门守此线）；
6. 论文定位：按 **STR 内部增强/消融** 写；R8（EdgeRefNet, TGRS 2026）已占"边缘引导细化+边缘监督"，
   **不得**把边缘监督抬成独立第三贡献；投稿前须重新核验 R8 方法细节；
7. **不得在脚本仍在运行时上传对同一脚本的修改**：bash 对脚本文件按**字节偏移惰性读取**，
   插入/删除行会让运行中的实例重新读到错位区域，可能中途损坏编排。
   （本轮就险些把"开波护栏"加进**正在跑**的 `run_all.sh` —— `_r7_stage.py` 会上传清单里的
   **每一个**文件，所以一次常规 stage 就会覆盖活文件；已在**上传前**发现并回滚，并用
   `local == server` 的 SHA256 证明运行实例未受影响。要改脚本：要么等它退出，要么写成新文件。）

### 5.1 开波护栏的执行方式（因上述限制而固定为手动）
`check_mapping.py`（变体/波次映射 + 协议常量）与 `check_imports.py`（常量 import 完整性）
本应"每波前自动跑"，但把它们加进 `run_all.sh` 必须改**正在运行**的脚本（见 §5 第 7 条），
故改为**在每波收口门核验时手动执行并留痕**。这一取舍是可接受的，因为：
- 波二/三的代码路径**已由真实的 40 步 `train.py` 冒烟逐条验证**（两种形态 rc=0，
  含 `[HFAE-GRAD] hf_tap_grad_norm=4.90e-02`、sidecar 三处齐全）——比静态护栏更强的证据；
- 代码一次性上传后由 `SOURCE_SHA256.txt` 锁定，波间不会漂移。

## 6. 产物索引
```text
/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run7/<variant>/<dataset>/best_F1=*.pth
/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/<variant>/<dataset>/train_log.txt
/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7_DRY/<variant>/<dataset>/{dry_run_log.txt,dry_run_summary.json}
```

（运行完成后在此登记 sha256；当前为空。）

## 7. 收口链在真实已完成日志上的**事前**校验（2026-10-11，训练进行中）

为免"跑了 13 h 才发现收口读不出来"，在波一训练期间用**真实已完成 run** 验证了
`collect_results.py` 的解析与门结构：

| 验证对象 | 结果 |
|---|---|
| `Run1/M1_FULL` 四库（**pre-R4** 日志） | 六指标解析与 README 记录**逐位吻合**：CDD `0.9718` / LEVIR `0.9107` / SYSU `0.8347` / WHU `0.9507`；`params=4.88`、`flops=2.6747`；因缺 `[ACTUAL-OPT-STEPS]`/`[PROTOCOL-VERSION]` 被门正确判 `INVALID` |
| `Run4/M1_R4CTRL/SYSU-CD-256`（**post-R4** 日志） | 最后一个完整 TEST 区块内**确含** `[ACTUAL-OPT-STEPS] 80000` 与 `[PROTOCOL-VERSION] run4_exact80k_v1` ⇒ Run7 门所需的**结构假设在真实数据上成立**，仅协议串不同（run7 vs run4） |
| `Run2` 各变体（pre-R4） | 区块内无这两行 ⇒ 门能正确区分 R4 前后两代协议机制 |

⇒ Run7 的 `run_all.sh` 内联门与 `collect_results.py` 的 INVALID 判定不会因"读不到行"而误杀全部 run。

### 7.1 波次门本身与 `eval.py` fail-closed 门的实测（用真实日志/权重）

上面两张表是**解析层**的验证；下面两处是**执行层**的（这两段代码平时只在 4.4 h 训练跑完后才被执行，
若带 bug 会白烧数小时）：

| 被测代码 | 用例 | 结果 |
|---|---|---|
| `run_all.sh::check_test_block`（**从文件中逐字抽出**后运行，避免复制漂移） | 真实 Run4 日志 + 其自身协议 `run4_exact80k_v1` | `rc=0`，`[GATE] opt_steps=80000 protocol=run4_exact80k_v1 F1=0.8326 -> PASS` ✅ |
| 同上 | Run4 日志 + 换成 Run7 协议（模拟串错） | `rc=1` FAIL ✅ |
| 同上 | pre-R4 的 Run2 日志（区块内无标记行） | `rc=1` FAIL ✅ |
| `eval.py` arch 门 | Run1 `M1_FULL/WHU` 权重 + 匹配实参 | `rc=0`；`[ARCH] eval arch matches ckpt sidecar`；`[DEPLOY-PARAMS] total=4.880 M`、`[DEPLOY-FLOPS] 2.6747 G (unsupported_ops=19)` —— 与项目记录逐项吻合 ✅ |
| 同上 | 同一个 Run1 权重 + `--hf_edge 1` | `rc=1`，`[ARCH-MISMATCH] ckpt has no hf_edge field but --hf_edge=1; refusing to eval` ✅ |
| 同上 | `--hf_edge 0` 但 `--hf_edge_form aux_only` | `rc=1`，`AssertionError: [ARCH-MISMATCH] hf_edge=0 requires hf_edge_form=none` ✅ |

⇒ 波次门与 eval 门在**接受/拒绝两种方向**上都已被真实数据证明可用（另注：`eval.py` 没有
`--test_batch_size`，用 `--batch_size`；这是本项目既有 CLI 约定）。

### 7.2 收口工具自身的缺陷（训练期间发现并修复）

`analyse/run7_object_metrics.py` 原先复用 `tvim_diag_common.build_model()`，但该函数的
`VARIANT_CFG` **没有 Run7 变体**（`M1_R7CTRL`/`E9_HFAE_A`/`E10_HFAE_B`），且它**不传
`hf_edge*`** —— 即使把变体名补进去，E9/E10 的 train 图 state_dict（含 `hf_edge_head.*`）
也会 strict 失败。**这会在最终裁决那一步才炸、导致 12 个 run 的对象级指标全部算不出来。**

已改为显式构造：新增 `build_run7_model()`（变体 → `(hf_edge, form)` 映射后直接构造
`CASATViMSTRNet` 并严格加载），并验证：

| 验证 | 结果 |
|---|---|
| 三变体 CPU 严格加载（自建 state_dict 回灌） | `missing=0 unexpected=0`，参数量 5,084,017 / 5,091,058 / 5,094,226 ✅ |
| 未知变体名 | 明确 `KeyError`（不带默认值蒙混）✅ |
| **真实 Run7 ckpt 端到端**（波一 CTRL/SYSU 中途权重） | `built Run7 variant: hf_edge=0 form=none (missing=0 unexpected=0)`，正常输出对象级指标 ✅ |

同轮核实：每个 ckpt 目录**恰好 1 个** `best_F1=*.pth`（`train.py:792` 会先删旧 best），
故 `pick_best_ckpt` 的"唯一 best"假设对 Run7 成立 ✅。

**早期 sanity 读数**（波一中途 CTRL，约 3,000/80,000 步）：`n_small=364`（GT 固定不变）、
`small pooledR=0.2213`、**`unmatched_small=5,358`**、`ObjP=0.3822`、`F1=0.8238`
—— 对比 Run1 终值 `2430 / 0.5320 / 0.8347`：**训练早期模型过度开火**（FP 数量翻倍、ObjP 明显低），
故对象级指标确实随训练变化，且**只有终态 ckpt 才可与冻结参考对比**（收口只取 best ckpt）。

### 7.3 独立审计的参数化与**真实数据阳性对照**（2026-10-11 09:48）

收口第 3 步 `independent_audit.py` 原本把 `PROTOCOL/VARIANTS/DATASETS` 写死，只能审计 Run7
自己，**无法在训练结束前证明它的解析与自洽性检查在真实日志上真的成立**。已改为可传
`--variants/--datasets/--protocol`，并新增 `_fixture()` 端到端自测（`--selftest` 9 → **14/14 OK**）：

| 自测 | 断言 |
|---|---|
| e2e 合规夹具 | `n_failed == 0`（10 项全过） |
| e2e A9 重推导 | `picked == ["E9_HFAE_A"]`，与 `verdict.json` 的 winner 一致 |
| e2e A8 行数 | `n_code_identity_rows == len(variants) × len(datasets)`（证明 `--variants` 真被使用） |
| e2e 故意破坏一处 F1 | 被 **A2** 抓到，且 `f1_iou_worst_key` 精确指向该 run |
| e2e 故意写错一处协议串 | 被 **A4** 抓到 |

**真实数据阳性对照**：把审计指向**已完成**的 Run4（`--variants M1_R4CTRL --protocol run4_exact80k_v1`，
协议串是从真实日志里 `grep` 出来的、不是我猜的）：

| 检查 | 结果 | 说明 |
|---|---|---|
| A1/A2/A3/A4/A5/A6/A8/A10 | **8 项 OK** | 解析、`F1 == 2IoU/(1+IoU)`、量程、80k 步数、协议串、deploy 4.880 M、代码身份全部在真实日志上成立 |
| `f1_iou_max_delta` | **6.95e-05**（Run4/CDD） | 真实日志的六指标自洽性残差；阈值 1e-3 既有余量又能抓住真错误 |
| A7 sidecar 契约 | FAIL（**预期**） | Run4 `arch.json` 无 `hf_edge` 字段（`hf_edge=None ≠ 0`）——证明该检查确实按 Run7 契约判定 |
| A9 接受规则对拍 | FAIL（**预期**） | Run4 无 Run7 的 `verdict.json` |

对照 2：指向**正在训练中**的 Run7 树 → A1 正确报 `not_ready`（未完成路径不崩、也不假 PASS）。

**纪律副作用（已更正并留痕）**：对照跑初版会把审计结果哈希**追加**到 `--log-root` 的
`SHA256SUMS.txt`，于是那次跑在 `Run4/` 里**新建**了一个只含我自己那行的
`SHA256SUMS.txt`（未覆盖任何历史内容：`wc -l = 1` 且该行就是我写的）。
已按纪律**不删除、改名标记**为 `Run4/_INVALID_control_side_effect_20261011.SHA256SUMS.txt`，
并把代码改为：**只有规范跑（无覆盖参数）才写回 SHA256SUMS**，对照跑明确打印"不改动目标 Run"。
（目标 Run 的 `SOURCE_IDENTITY.json`/`EVIDENCE_INDEX.md` 均不引用该文件，故未破坏任何已登记身份。）

`independent_audit.py` 原先还**漏登记**在 `SOURCE_FILES.txt` 里，已补入（现 24 条）；
`SOURCE_SHA256.txt` **24/24 与服务器逐字节一致**（`_r7_stage.py --check` 退出 0）。

### 7.4 身份链能否**从 git 复现**——发现并修好一个此前无人察觉的基础设施缺陷

身份登记只有在"别人 clone 下来也能复核"时才有意义。实测**全新 clone** 再核验
`Run7/SOURCE_SHA256.txt`：

| 阶段 | clone 侧核验结果 |
|---|---|
| 修复前 | **FAIL**，`.py` 逐条 MISMATCH（`train.py`、`str_hf_edge.py`、`ss2d.py`、`str_tar/dcr/reparam.py`、`check_mapping.py`… 前 400 字节内就有 5~10 个 CRLF） |
| 只加 `* text=auto eol=lf` | 23/24，仅剩 `models/model/utils.py` |
| 再钉 `utils.py text eol=crlf` | **24/24 PASS** ✅ |

根因：`.gitattributes` 原先只钉了 `*.sh`，而 `core.autocrlf=true` 来自**系统级** gitconfig
（`C:/Program Files/Git/etc/gitconfig`，即 Windows 装完 Git 的默认值）⇒ 任何人在 Windows 上
clone 都会把 `.py/.json/.md` 检成 CRLF，**身份链只能在本机工作副本上闭合**。
（服务器工作副本是 SFTP 逐文件同步、不是 clone，故**已跑完/正在跑的训练不受影响**。）

`models/model/utils.py` 为什么单独钉 CRLF 而不是规范成 LF：它在 `models/train.py` 的
`SOURCE_IDENTITY_FILES` 里，sha256 会写进**每个 run** 的 `arch.json`/`last.pth`；而波一正在训，
此时改它的字节会让波一与波二/波三身份不同源 → A8「全部 run 同源代码身份」FAIL →
**错误地否定一个本来干净的实验**，且属于训练中途改代码。故改为把**检出内容**钉成 CRLF，
与运行时被哈希的字节逐字节一致；本地工作副本未被改写（`git status` 只有 `.gitattributes`）。
留 TODO：下个 Run 统一改 LF 并同时重建清单。

**附带核查（研究诚信）**：体检发现 Run5 有 2 条、Run6 有 3 条清单条目与当前工作副本不一致。
逐条查证：这 5 个文件都由 `1160eed`（Research policy revision + archive design docs）最后修改，
`git diff` 显示改动**仅为**设计文档路径改为 `docs/temporary/过去的想法/…` 并加一行
"内容与 SHA256 未变"说明；**没有任何阈值、判据、结论、数字被改动**，两份设计文档 sha256 复算未变。
⇒ Run5/Run6 预注册完整性不受影响；其清单记录的是"运行时身份"，其后有一次已记录在案的归档改名。

验证工具：`.claude/_r7_verify_clone.py`（对任意工作副本逐条核验清单）、
`.claude/_identity_health.py`（体检本地各 Run 清单 + CRLF 污染）。


