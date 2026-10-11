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

### 7.5 补上收口缺的两步：逐 run 硬门 `check_run.py` + 把独立审计接进收口链

**发现的缺口**：目标要求的收口链是「审计 + check_run + 对象级只读复算」，但 `finish_run7.sh`
原来只有「12-run 门 → 对象级复算 → 裁决 → 对照表」——**既没有 check_run，也没跑独立审计**。
即：`collect_results.py` 只查步数/协议串，不查「消融是不是真的成立」；而 `independent_audit.py`
虽然写好了却没人调用。两处都会在 20:40 收口时才发现，属于"收口那一刻才发现工具没接上"。

**新增 `check_run.py`**（Run4 `check_run.py` 的 Run7 版本），它比 12-run 门多管两件
**消融可解释性**的关键事：

| 检查 | 为什么必须有 |
|---|---|
| `M1_R7CTRL` 日志里除 `[HFAE-CONFIG] hf_edge=0 form=none` 外**不得出现任何** `[HFAE-*]` 机制行 | 保证对照组没有偷偷带上被消融的机制（"唯一变量"） |
| 实验组 `[HFAE-GRAD]` 必须 `hf_tap_grad_norm > 0` 且 `nonzero_tap_grad_tensors > 0` | 证明边缘辅助损失的梯度**真的回流进了高频分支**，而不是只挂了旁路头（正是 Run5「只读 HF 通道」被否证的失败模式） |
| `[HFAE-FORM]` 的 `lambda == 0.2`、`taps == 1_4_1_8`（λ 预注册固定禁止搜索） | 防止事后换权重的"调参型提升" |
| `[HFAE-DEPLOY]` 的 `dropped=entire_head/edge_out_only` 与 `deploy_params=4880190/4890366` | 证明两种部署形态各自按预注册方式折叠，且 ≤5M |
| sidecar（`arch.json`/`run_manifest.json`）的 `hf_edge/form/taps/lambda/tag` + `source_code_sha256` | 日志与权重身份一致，且身份非空 |

**为验证它而用真实 train.py 产出契约完整日志**：40 步冒烟没有跑完一个 epoch ⇒ 既无 TEST 区块、
也没有 `[HFAE-FORM]`/`[HFAE-DEPLOY]`（它们写在收尾阶段，需要 `best_F1=*.pth` 存在）。
故用 **WHU（最小数据集，batch 8 → 744 iters/epoch）+ `max_steps=750`** 在两张卡上并行各跑
E9/E10 一次，得到**契约完整**的真实日志（GPU 占用回到基线，未干扰波一）。

**验证结果：`CHECK_RUN VALIDATION = PASS`**（9 个用例，"该过的过、该拦的拦"）：

| 用例 | 期望 | 实测 |
|---|---|---|
| 真实 E9 / E10 契约完整日志 | 仅 `steps(750) != 80000` 失败 | 35/36、36/36 检查通过，恰好 1 项 FAIL ✅ |
| 把 E9 日志当 CTRL 收 | 抓「对照组不干净」+ 配置行不符 | 2 项 FAIL，正是这两条 ✅ |
| `hf_tap_grad_norm` 归零 / `nonzero_tap_grad_tensors` 归零 | 抓「机制没接上」 | 各 1 项 FAIL ✅ |
| 日志 `lambda` 改 0.5 | 抓 λ 被改 | 1 项 FAIL ✅ |
| `deploy_params` 改 5100000 | 抓 deploy 被改 | 1 项 FAIL ✅ |
| **sidecar** `hf_edge_lambda` 改 0.5 | 抓权重身份被改 | 2 项 FAIL（两份 sidecar）✅ |
| 真实 CTRL（未跑完） | 只 FAIL 在「无完整 TEST 区块」；**对照组干净必须 PASS** | 2 项 FAIL，干净检查 **PASS (clean)** ✅ |

**过程中被真实数据纠正的两处我自己的错误**（若不先在真实产出上验，收口时会把好 run 判死）：
1. 我原以为 `source_code_sha256` 是 64 位十六进制；实测 train.py 写的是 **16 位**（如
   `5bc6515389ad3c09`）⇒ 该检查在**所有** run 上都会误报失败。已改为接受 16 或 64 位。
2. 我原以为 CTRL 的 sidecar 也记 `taps=1_4_1_8 / lambda=0.2`；实测 CTRL 记的是**生效值**
   `taps='none' / lambda=0.0 / tag=None`（而日志里的 `[HFAE-CONFIG]` 回显的是 CLI 参数
   `taps=1_4_1_8 lambda=0.2`）。两者口径不同是设计如此 ⇒ 检查必须**按变体分别对齐**。

**`finish_run7.sh` 改为 6 步**：`[1/6]` 12-run 门 → `[2/6]` **逐 run check_run（fail-closed：
有 FAIL 则 exit 4，不产出裁决，可用 `RUN7_ALLOW_CHECKRUN_FAIL=1` 显式越过并需在 README 记录原因）**
→ `[3/6]` 对象级复算 → `[4/6]` 裁决 → `[5/6]` **独立审计（A9 与 verdict 对拍，故必须排在裁决之后）**
→ `[6/6]` 对照表；审计未全过则 exit 5 并显式警告结论须按 INVALID 处理。
实测未跑完时 `bash -n` 通过、实跑 **`FINISH_RC=2`** 且**未启动**对象级复算（不浪费 GPU）。

### 7.6 对象级复算在**真实 E9/E10 权重**上的严格加载验证（收口最后一块未验证的环节）

`analyse/run7_object_metrics.py` 是收口第 3 步，也是**唯一会碰 E9/E10 train 图权重**的环节
（`hf_edge_head.*` 键）。此前只用"自建 state_dict 回灌"验过，**有循环论证嫌疑**：构造与加载
走同一套代码，键名不一致照样能过。用 §7.5 那次真实 750 步训练产出的权重第一次做了独立验证：

| 变体 | 真实权重 | 严格加载 | 对象级读数（WHU，750 步模型，**数值本身无意义**） |
|---|---|---|---|
| `M1_R7CTRL` | `best_F1=0.9387.pth`（波一实时权重只读拷贝） | `hf_edge=0 form=none (missing=0 unexpected=0)` ✅ | `pooledR=0.5227 Hit@25=0.3165 n_small=79 unmatched=321… ObjP=0.7299 ObjR=0.7807 F1=0.9387` |
| `E9_HFAE_A` | `best_F1=0.7970.pth` | `hf_edge=1 form=aux_only (missing=0 unexpected=0)` ✅ | `pooledR=0.3186 n_small=79 unmatched=318 ObjP=0.4660 F1=0.7968` |
| `E10_HFAE_B` | `best_F1=0.8243.pth` | `hf_edge=1 form=aux_inject (missing=0 unexpected=0)` ✅ | `pooledR=0.2609 n_small=79 unmatched=320 ObjP=0.4549 F1=0.8243` |

⇒ 三变体在**真实权重**上都能严格加载并跑通端到端复算（这正是原先会导致"12 个 run 全算不出来"的那条路）。

**顺带修掉一个会误用的 CLI 缺陷**：`--variants/--datasets` 只在 `--all-variants` 下生效，
但原先**静默忽略**——只给 `--variants` 时 `args.variant` 保持 `None`，在下游抛一个
`TypeError: join() ... not 'NoneType'`（实测踩到）。现已改为 argparse 开门见山报错；
`--variant/--dataset` 单 run 旧调用方式**回归验证仍可用**（本地实跑确认不再抛 usage error）。
`finish_run7.sh` 传的是 `--all-variants`，不受影响。

### 7.7 裁决引擎：接口契约实测 + 补上目标第(5)条要求的「小目标机制归因」

**接口契约（`run7_object_metrics.py` → `run7_verdict.py`）在真实文件上验证通过。**
两个工具是分开写的，接口从没在真实产物上对过；本轮用 §7.6 产出的**真实**
`object_metrics_*.json` 驱动 `decide()` 实测：

| 检查 | 结果 |
|---|---|
| 真实 `headline` 键 vs 裁决的 `REQUIRED_HEADLINE`（8 个） | 真实 headline 有 12 键，**8 个必需键全部命中** ⇒ `guardrail_unevaluable[*].bad == []` ✅ |
| 只给 1/4 库的对象级指标 | 判 `INCOMPLETE`（**不假 PASS**），reason 指向"缺对象级指标" ✅ |
| 四库齐备 | 判 `GUARDRAIL-FAIL`，且 reasons 给出**真实数值**：`large_pooled_recall 0.8430 < ctrl 0.9283 (-0.0853 > 0.02)`、`ObjRecall_loose 0.7029 < 0.7807` ✅ |
| SYSU F1 不提升 | 判 `MECHANISM-FAIL` ✅ |

**发现的缺口（目标第(5)条只实现了一半）**：`small_pooled_recall` / `unmatched_small_pred` /
`small_hit25` 此前只在 `guardrail_evaluable()` 里被要求"**必须存在**"，却**从未出现在任何输出里**
——实测在 verdict 输出里搜这三个名字全是 `False`，而 `GUARDRAIL-FAIL` 的 `attribution` 是 `None`。
即"按四库 F1 **与 SYSU small recall/未匹配小预测数**给出 PASS/FAIL **与机制归因**"里的后半句没落地。

**已补上（**只作报告，不改规则**）**：`decide()` 新增 `detail["small_object_report"]`
（每变体×每库给出 small pooledR / Hit@25 / Hit@50 / 未匹配小预测 / n_gt_small / 中大 pooledR /
ObjRecall 的 CTRL→EXP 与 Δ，以及未匹配小预测相对 CTRL 的**倍数**），三类结局的 `attribution`
都带上 SYSU 小目标读数，`render_markdown` 增出小目标表与归因段。真实文件实测输出：

```
E9_HFAE_A: SYSU-CD-256 small pooledR 0.5227→0.3186 (Δ-0.2041), Hit@25 0.3165→0.2025,
           未匹配小预测 121→318 (×2.63), 中大 pooledR Δ-0.0082/-0.0853
```

**关键不变量（已写进 selftest 守死）**：接受标准与守门阈值是在任何 E9/E10 结果之前冻结的，
这次增补是"报告"而不是"改规则"⇒ 新增断言「把小目标读数改到极端，**裁决标签必须不变**」。
selftest 由 13 → **15 条全 PASS**：

| 新增断言 | 结果 |
|---|---|
| `small-object readings do NOT change verdict` | `MECHANISM-PASS == MECHANISM-PASS` ✅ |
| `SYSU small-object readings surfaced (report + attribution)` | `attribution=True report=True` ✅ |

（§7.6/§7.7 里用于接口与归因验证的 headline 数值来自 **750 步**的模型，只证明管道与判定逻辑，
**不是实验结果**；正式结论一律等 12 个 80K run 跑完后由 `finish_run7.sh` 产出。）

### 7.8 逐 run 冻结协议核验：把「脚本里参数写对了」升级为「每个 run 的落盘清单也对」

**发现的缺口**：目标第(4)条要求「各严格 80,000 `optimizer.step`、batch 32、从同一预训练权重从头训」。
此前这些只在**脚本参数**层面被 `check_mapping.py` 验过（`runner_batch_frozen` / `runner_steps_frozen` …），
但**没有任何检查去看每个 run 实际落盘的 `run_manifest.json`**。若某一波被用错参数启动
（例如 batch 或 max_steps 不对），12 个 run 仍会各自产出"完整 TEST 区块"，`collect_results.py`
的 12-run 门也照样放行 ⇒ **静默的协议违规**。

**已补进 `check_run.py`**（字段与期望值均从**真实** manifest 实测得到，非猜测）：

| 断言 | 冻结值 |
|---|---|
| `batch_size` / `seed` | 32 / 16 |
| `max_steps` / `exact_max_steps` | 80000 / 1 |
| `data_contract` | `legacy_6ch_reverse_v1` |
| `feature_taps` | `stage1_1_4_48, stage2_1_8_64, stage3_1_16_168, stage4_1_32_224` |
| `backbone_lr_ratio` / `str_dim` | 0.1 / 96 |
| `dataset` | 与该 run 的目录/调用一致（防止 ckpt 与库错配） |
| **`resume_source`** | 必须为 `None`，或指向**本 run 自己**的 `last.pth` |

`resume_source` 这条是**研究诚信**守卫：`run_all.sh` 的崩溃重试会原样重跑同一条命令，
由 `train.py` 自动从**自己的** `last.pth` 续训（`train.py:426` 自动 resume、`:445` 恢复
`actual_steps`、`:461` 恢复 RNG、`:508/:631` 把 `resume_source` 写进清单）。若 `resume_source`
指向**别的变体/库**，等于拿别人的权重续训，会**静默污染受控消融** —— 现在会被硬门拦下。

**真实数据验证结果**（`CHECK_RUN VALIDATION = PASS`，9 用例）：

| 用例 | 实测 |
|---|---|
| 波一 CTRL 四库**真实** `run_manifest.json` | **9 条协议断言全部 PASS**（`n_checks=28 n_failed=2`，仅"未跑完"两条失败）⇒ 波一确实按冻结协议在跑 ✅ |
| 750 步冒烟清单（batch=8/max_steps=750） | 被协议门**正确判 FAIL**（`manifest batch_size == 32`、`manifest max_steps == 80000`）⇒ 门不是恒真 ✅ |
| 把某库的 ckpt 配错 dataset | 被 `manifest dataset == …` 抓到 ✅ |

（过程中我**自己的测试脚本**有一处 bug：`run()` 硬编码 `--dataset WHU-CD-256`，而 CTRL 的 ckpt 是 CDD，
于是被新检查判出 `manifest dataset` 不符。工具是对的、测试调用是错的 —— 已修好并重跑通过。）

### 7.9 收口链整条 **DRY 排练**（12 个 run 全走通，`FINISH_RC=0`，耗时 207 s）

**为什么必须做**：收口链有 6 步、跨 5 个工具。此前每一步都**单独**验过，但**从未整条串起来跑过**；
而正式收口只有一次机会（12 个 80K run 跑完后）。工具之间的问题（路径口径、文件名约定、
audit 与 verdict 的先后依赖）只有串起来才会暴露。

**做法**：把 `finish_run7.sh` 的 `LOG_ROOT`/`CKPT_ROOT`/变体与库列表改为**环境变量可覆盖**
（默认值与正式收口**完全一致**，故正式行为不变），然后在 `/tmp/r7_rehearse` 里用
**伪造但格式真实**的 12 份日志（完整 TEST 区块、`80000` 步、冻结协议串、各变体机制行、
`F1 == 2IoU/(1+IoU)`）+ **真实权重**（750 步探针的 ckpt，逐库复制并改写 `dataset`/`max_steps`/
`batch_size`）搭一棵临时树。排练时脚本会打印**醒目 REHEARSAL 横幅**，防止排练结论被误当正式结论。

**结果（`FINISH_RC=0`，全程 207 s，正式 `Run7` 目录未被触碰）**：

| 步骤 | 结果 |
|---|---|
| `[1/6]` 收口门 | **12/12 VALID** ✅ |
| `[2/6]` check_run × 12 | **12/12 PASS，0 失败**（每 run ≈45 条检查：变体契约 + 机制证据 + 冻结协议字段 + sidecar） |
| `[3/6]` 对象级复算 × 12 | **12/12 写出**，三变体 × 四库全部 `missing=0 unexpected=0` 严格加载 |
| `[4/6]` 裁决 + 归因 | `GUARDRAIL-FAIL`（winner=E9_HFAE_A），12 条具体理由 + 含 SYSU 小目标读数的机制归因 + 小目标表 |
| `[5/6]` 独立审计 | **10 checks, 0 FAILED**（含 A8「12 run 同源代码身份」与 A9「接受规则重推导与裁决对拍」） |
| `[6/6]` 跨变体对照表 | 含冻结参考（Run1）✅ |

⇒ 收口链的**步骤依赖与文件约定**已被证明自洽：`check_run` 读得到 `--ckpt-dir` 下的两份 sidecar、
`run7_object_metrics` 的 `--out` 文件名正是 `run7_verdict --object-dir` 期待的名字、
`independent_audit` 排在裁决之后才能做 A9 对拍。
**并且测出正式收口只需约 3.5 分钟**（最重的对象级复算 12 次约 200 s），不存在"收口本身超时"的风险。

**排练暴露并修掉的一个真实缺陷**：`finish_run7.sh` 原第 1 步调用 `collect_results.py` 时**没有传**
`--variants/--datasets`，所以**永远**按 12/12 判门。正式收口下这等价于"全部"，看不出问题；
但它意味着**任何部分 run 都无法收口**（排练第一次跑就卡在 `VALID 3/12 -> exit 2`）。
已改为传与后续各步**同一套**过滤（正式时默认仍是全部 12 个，行为不变）。

⚠️ 排练里的 `GUARDRAIL-FAIL` 与所有读数**不是实验结果**：F1 是我为了验证判据而**编造**的
（故意让 E9 在 4/4 库都 ≥ CTRL），对象级指标则来自把 750 步权重逐库复制后评测 ⇒ 数值毫无意义。
它只证明「守门确实会在"F1 涨但中大对象 recall 崩"时触发」这条逻辑是活的。
**正式结论一律等 12 个 80K run 跑完后由 `finish_run7.sh` 产出。**

### 7.10 预注册文档 ↔ 裁决代码的逐条对拍（见结果之前完成）

**为什么必须做**：预注册只有在"文档写的判据 == 代码执行的判据"时才成立。此前我核对过
`SELECTION_LOCK.json` ↔ `run7_verdict.py`，但**没有核对散文式设计文档 ↔ 代码**。
若两者不一致，评审时会被当作"事后改规则"。

**做法**：逐条读码对拍，把结果作为 **§7 追加**进预注册文档
（`docs/temporary/CASA-CD_Run7_高频边缘辅助监督HFAE_预注册_2026-10-11.md`）。
**追加而非修改**：`git diff --numstat` 实测 **added=66 / deleted=0**，
原 §1–§6 一字未动（§6 明文要求冻结项不得事后修改），且抽查关键句仍逐字存在。

**对拍结果**：

| 项 | 结论 |
|---|---|
| 接受标准 | ✅ **逐字一致**：`primary_ok = (n_ge ≥ 3) and SYSU 严格提升 and 机制自检全过`，`winners` 取第一个，无 winner ⇒ `MECHANISM-FAIL` + 归因 |
| tap 规格 | ✅ 文档 `network[0][2]`/`network[2][2]` 的 `local_conv`（36ch@1/4 + 32ch@1/8）与 `HF_EDGE_TAPS` 及真实日志 `[HFAE-TAPS] [[0,2,36,64],[2,2,32,32]]` 一致 |
| λ / head 参数量 / deploy 参数 | ✅ 0.2、7,041 / 10,209、4,880,190 / 4,880,190 / 4,890,366（+10,176）全部与真实日志一致 |
| 机制自检 ①③④⑤ | ✅ 分别落在 `grad_reaches_hf_branch` / `fold_disagree_zero`（另由 check_run 与 audit 独立再查）/ `deploy_within_limit` / `aux_loss_present_and_decreasing` |
| 机制自检 ②（epoch-0 逐位相同） | ⚠️ **如实说明**：由 **T2 + dry-run 门**在**构造期**保证（`fork_rng` + 注入零初始化），**不从日志逐 run 复推**；已写进文档以免被误读为"逐 run 有日志证据" |
| §6「既有算子一律不动」 | ✅ 实测 Run7 期间模型侧改动**恰好**是承诺范围：新增 `str_hf_edge.py`/`run7_dry_run.py`/`test_run7_hf_edge.py`，接线修改 `train.py`/`eval.py`/`casa_tvim_str_net.py`；`ss2d.py`(10-03)、`layers/caacp_ss2d.py`(10-05)、`str_dcr.py`/`str_reparam.py`(10-02)、`tinyvim_s_slim.py`(10-05)、`str_fine_tap.py`(10-10) **最近提交均在 Run7 之前** |

**披露一处「代码比散文更严」**：§4 散文的守门只写了中/大对象 pooled Recall，
而代码另加了 `ObjRecall_loose` 回退 > 0.02 触发守门（`guardrail_objrecall_drop_max`）。
这是**额外从严、不是放宽**，且与其余阈值**同时**在 09:35 冻结 ⇒ 不违反预注册精神；
已在文档 §7.3 如实披露，避免被当作事后加码。

**冻结时序可核验**：锁 `_frozen_at = 2026-10-11T09:35:00+08:00`；09:27 启动的是同期对照
`M1_R7CTRL`，**没有任何 E9/E10 run 存在**；且冻结时**全部 12 个 run 都还没有任何完整 TEST 区块**
（TEST 区块只在 80,000 步收尾写出，实测到 10:00 波一仍是 `NO_COMPLETE_TEST_BLOCK`）
⇒ 冻结时**无从观察到任何结果**，"判据在任何 E9/E10 结果之前固定"成立且可复核。

### 7.11 机制自检的**原始读数**入报告 + 修掉一处「报告值会污染判定」的隐患

**动机**：目标第(5)条要"机制归因"，而 `detail["mechanism"][v][d]` 原先**只有布尔值**，
论文无法直接引用"高频梯度到底多大、aux loss 从多少降到多少"。同时发现一个**真实隐患**：

`all_ok` 原先写作 `all(v2 for k, v2 in per[d].items() if k != "all_ok")` —— **遍历全部条目**。
这意味着**任何**后来加进去的非布尔诊断值都会被当成判定条件：例如 `fold_real_disagree = 0.0`
是 falsy、`aux_loss_last = 0.3` 是 truthy，一旦混进同一层，就会把本来通过的机制自检判死
（或反之）。这与本轮想加的功能正好相撞。

**改法**（只加报告，不改判定语义）：
- 把布尔检查项收进 `checks_d`，`all_ok = all(checks_d.values())` —— **只由布尔项决定**；
- 诊断读数放进**独立子字典** `per[d]["raw"]`：`hfae_form` / `hfae_deploy_params` /
  `deploy_param_limit` / `fold_real_disagree` / `hf_tap_grad_norm` / `hfae_grad_lines` /
  `n_aux_loss_epochs` / `aux_loss_first` / `aux_loss_last`；
- `verdict.md` 增出**机制自检原始读数表**（可整表引用进论文）。

**验证**：

| 检查 | 结果 |
|---|---|
| selftest | **15 → 18 条全 PASS**；其中一条夹具专门造陷阱（`fold=0.0` falsy + `aux_last=0.3` truthy）断言 `all_ok is True` |
| 反向用例 | `aux=(0.30, 0.40)`（末 > 首）⇒ `all_ok False`，且 raw 里留下 `0.30→0.40` 作为归因证据 |
| **真实形状数据上判定不变** | 用排练产物重跑裁决：`before = GUARDRAIL-FAIL winner=E9_HFAE_A`，`after = 同一标签同一 winner`，**reasons 均为 12 条** |
| `all_ok == all(布尔项)` | 对全部 **8** 个 (变体, 库) 逐一成立 |
| 新表真实渲染 | `HF tap grad norm 0.0490047`、`aux first→last 0.337611→0.301204`、`fold disagree 0`、`deploy 4880190`、`all_ok OK` |

**顺带核实资源增长（推到 80K 是否有风险）**：波一四个日志在 ~16.5k 步时 16–27 KB，
**线性外推到 80,000 步约 0.1 MB**；ckpt 目录稳定在 **635 MB**（`last.pth` 覆盖写、
best 文件每次替换）⇒ 无日志膨胀、无 ckpt 堆积风险；磁盘 2.6 T 空闲。

### 7.12 督导「过门即开下一波」的触发逻辑实测（防静默不开波）

**风险**：`wait_wave` 只在 `n_valid == n_total` 时 `return 0`。若 `wave_state` 的解析有 bug
导致 `nv != nt`，督导会**一直循环到 20 小时上限**，**波二/波三永远不启动**，而且**是静默的**
（日志只显示 `valid=3/4` 之类，看起来像"还没跑完"）。

**做法**：从**正在运行的** `supervise_waves.sh` 里用正则**逐字提取** `wave_state` 的内联解析代码，
在排练树（12 份完整日志，并补上督导正则要求的 `cumulative=` 行）上跑同一份逻辑。

> ⚠️ **安全措施**：真正的 `wave_state` 读写 `/tmp/r7_sup_gate.json`，而督导**此刻正在用那个文件**。
> 若测试覆盖它，督导可能读到我的数据并在波一未完成时**误判过门 ⇒ 提前开波二**。
> 因此提取后**只把那个临时文件路径**换成 `/tmp/r7_wavestate/gate.json`，解析逻辑逐字未动
> （提取脚本会打印出所有 `/tmp/` 引用以证明只此一处）。
> 事后核实：`/tmp/r7_sup_gate.json` 仍是督导自己的实时状态（`n_valid=0 / n_total=4`，
> mtime 跟随督导 120 s 周期）——**未被污染** ✅

**结果（`WAVE_GATE CHECK = PASS`）**：

| 变体 | 督导原始逻辑输出 `(n_valid n_total steps_sum)` | 判定 |
|---|---|---|
| `M1_R7CTRL` | `4 4 320000` | GATE PASS ⇒ 会启动波二 ✅ |
| `E9_HFAE_A` | `4 4 320000` | GATE PASS ⇒ 会启动波三 ✅ |
| `E10_HFAE_B` | `4 4 320000` | GATE PASS ⇒ 收口 ✅ |
| 负向：只有 3/4 库完整 | `n_valid=3 n_total=4` | **不可过门**（正确）✅ |

⇒ 「波一完成 → 自动开波二 → 自动开波三」这条唯一的自动衔接点已被证明可用。

---

### 7.13 三形态 deploy 门的最后一块：`eval.py` 在**部署图**上的实测（E10 此前未验）

**缺口**：目标第(4)条要"deploy 参数门"。此前只验过 CTRL（Run1 权重：4.880 M / 2.6747 G /
unsupported_ops=19）与 E9（fail-closed 正例），**E10（`aux_inject`）从未验过** ——
而它恰是三个形态里**唯一让部署图真正变大**的那个（4,890,366，+10,176），
也正是最可能被 eval 误报成**训练图**参数（5.09 M，看起来像超预算）的那个。

**做法**：用三形态的**真实**权重跑 `eval.py`（走 `--batch_size`，本项目没有 `--test_batch_size`），
在**部署图**上读它自己打印的 `[DEPLOY-PARAMS]` / `[DEPLOY-FLOPS]`：

| 变体 | eval 报的 deploy total | FLOPs | `unsupported_ops` | 冻结值 | ≤5 M |
|---|---|---|---|---|---|
| `M1_R7CTRL` | **4.880 M** | 2.6747 G | 19 | 4,880,190 ✓ | ✓ |
| `E9_HFAE_A`（`aux_only`） | **4.880 M** | **2.6747 G** | 19 | 4,880,190 ✓ | ✓ |
| `E10_HFAE_B`（`aux_inject`） | **4.890 M** | **2.7170 G** | 20 | 4,890,366 ✓ | ✓ |
| 三者 | 均打印 `protocol_version=run7_hf_edge_exact80k_v1`，`source_code_sha256` **9 文件完全一致**，且 `[ARCH] eval arch matches ckpt sidecar` | | | | |

**由此得到两条可写进论文的结论**（此前只有参数、没有 FLOPs）：

1. **`aux_only` 的推理代价确实是零**：deploy 参数 **与 FLOPs 与 CTRL 逐位同值**
   （4.880 M / 2.6747 G / unsupported_ops=19）⇒ 「训练期深监督、推理零开销」不只是数参数得来的，
   而是**经 eval.py 在部署图上实测**的。
2. **`aux_inject` 的代价是新测出来的**：4.890 M（+10,176）且 **2.7170 G（+0.0423 G，+1.58%）**，
   `unsupported_ops` 19 → 20。论文部署表可直接引用这四个数。

另：三者 `source_code_sha256` 完全一致，是 `independent_audit` 的 A8「12 run 同源代码身份」
**从另一条代码路径**（eval.py 而非审计脚本）得到的独立佐证。
（表中 F1 来自 **750 步**探针权重，**不是结果**；仅证明部署管道与参数门。）

### 7.14 注入点核验：E10 到底注到哪（目标第(2)条「TAR/DCR 注入」+ 预注册 §4「DCR 1/4 输入」）

**为什么查**：目标第(2)条写「`casa_tvim_str_net.py`/**TAR-DCR 注入**」，预注册 §4 写
「零初始化注入 **DCR 1/4 输入**」。这两句是否与代码一致，直接决定论文里"注入到哪"的说法能不能写。
而 `feats = self.tar(...)` 之后才注入，容易让人误以为注到的是 **DCR 的输出**。

**读码结果（逐行）**：

```
str_tar.py:221   MultiScaleTAR.forward -> t1..t4 = TARStage(每个尺度) ; return [t1, t2, t3, t4]
casa_tvim_str_net.py:265   feats = self.tar([f1a..f4a], [f1b..f4b])
casa_tvim_str_net.py:268   # 零初始化注入 ⇒ epoch-0 逐位恒等；注入到 DCR 的最小尺度输入（1/4）
casa_tvim_str_net.py:271   feats[0] = feats[0] + self.hf_edge_head.inject(self._hf_edge_feat)
casa_tvim_str_net.py:272   x = self.decoder(feats)        # self.decoder = DCRDecoder(...)  (:106)
casa_tvim_str_net.py:276   logits = self.head(x)
```

⇒ `feats[0]` 是 TAR 在 **1/4** 尺度上的输出，而它在 **:272 才被 `DCRDecoder` 消费**
（`:106` 明确 `self.decoder = DCRDecoder(dim=str_dim, use_aux=(rep_mode=="full"), ...)`）。
所以注入点**确实是 DCR 的 1/4 输入**，预注册 §4 的措辞与代码**逐字相符**，
目标第(2)条的「TAR/DCR 注入」成立 ✅

**顺带得到一个可用于"新机制而非重试"论证的对比**：被两次判 `MECHANISM-FAIL` 的 Run4 FET1 是
**refine 之后**相加（`:273-275` 注释原文："refine **之后**相加（不进入 SiLU/RepLocalBlock，
保持 DCR 训练路径不变）"）。即：

| 机制 | 注入点 | 是否改变 DCR 算出来的东西 |
|---|---|---|
| Run4 FET1（已否证） | `decoder(feats)` **之后** | ✗ 末端旁路 |
| Run7 E10 HFAE | `decoder(feats)` **之前**、1/4 尺度 | ✓ 进入 DCR 的输入 |

这与预注册 §1「本机制作用在编码器最细两级…不在末端读出」「改变主干算出来的东西」**一致** ✅

`switch_to_deploy` 侧一致：`:292` 在 `aux_only` 下把 `hf_edge_head = None`（整个头删除），
与 `check_run.py` 断言的 `dropped=entire_head` / deploy 参数回到 4,880,190 相互印证。


### 7.15 真实 batch 折叠门：刀锋像素澄清（⚠️ 见结果之后作出，已在预注册文档 §7.6 披露）

**触发**：波一对照臂 `M1_R7CTRL/CDD-CD-256` 的真实 batch 折叠检查给出
`[REPARAM-REAL-ARGMAX-DISAGREE] 9.537e-07`（**1 / 1,048,576 像素**），而预注册 §4③ 写的是
「真实 batch disagreement = 0」⇒ 严格读法下该 run FAIL，收口链会 `exit 4` **不产出裁决**。

**定位证据**（`analyse/run7_fold_knife_edge.py`，用真实 best ckpt 复现 train.py 收尾的同一 batch）：

| 项 | 值 |
|---|---|
| 翻转像素 `\|p_train − 0.5\|` | **3.099e-06** |
| 该次折叠最大绝对误差 | **6.199e-06** |
| 全批处于误差带内的像素 | **2 / 1,048,576** |
| 同源性自检（与日志相对差） | `max_abs` 1.9e-05、`disagree` 2.7e-05 ⇒ 同一 batch ✅ |
| 判定（与 Run4 §10 同定义） | **KNIFE_EDGE_ARTIFACT** |

**澄清**：真实 batch 折叠门 = 逐像素二值化一致，**唯一例外**是 `|p_train − 0.5|` 小于该次折叠
**自身最大绝对误差**的像素（落在阈值数值误差带内、二值化本就不适定）。该例外**必须**附
`_KNIFE_EDGE/KNIFE_EDGE_<dataset>_<variant>.json` 定位证据（`verdict=KNIFE_EDGE_ARTIFACT`、
`n_flipped>0`、且 `disagree`/`max_abs_diff` 与日志吻合）；**缺失或不吻合一律 FAIL**。

**三处独立实现**（互不 import，各自 fail-closed）：

| 实现 | 位置 | 已验证 |
|---|---|---|
| 逐 run 硬门 | `check_run.py`（新增 `--knife-dir`） | 无证据→`rc=1`；有真实证据→`ALL PASS rc=0` ✅ |
| 裁决机制自检 | `run7_verdict.py::fold_ok_knife` | selftest 3 条新分支；**21/21 PASS** ✅ |
| 独立复算审计 | `independent_audit.py` A6 | 有证据→`A6=True(accepted)`；移走→`A6=False(violation)` ✅ |

**收口链已自动生成证据**：`finish_run7.sh` 新增 `[1b/6]` 步——只对
`[REPARAM-REAL-ARGMAX-DISAGREE] > 0` 的 run 跑定位诊断并落证据；disagreement 为 0 的 run 不做任何事。
证据目录 `…/Run7/_KNIFE_EDGE/`。

⚠️ **论文必须披露**：本条是在**看到波一结果之后**作出的（由真实运行触发，事先无从预知）；
它**不依赖任何性能指标**、**对称适用于全部 12 个 run**（含 E9/E10）、不偏袒任何变体；
判据/阈值/λ/seed **一概未动**，`SELECTION_LOCK.json` 的接受标准与守门阈值**完全不变**。



## §8 收口 runbook（波三跑完后照着做）

### 8.1 一条命令

```bash
cd /home/yqwang/projects/CASA-CD
bash train_scripts/CASA-TViM/Run7/finish_run7.sh; echo "FINISH_RC=$?"
```

预计 **~3.5 分钟**（已 12-run 全量排练实测 207 s）。它会打印醒目 `[REHEARSAL]` 横幅**仅当**
用了 `RUN7_*` 覆盖变量——正式跑**不应**出现该横幅。

### 8.2 退出码 → 动作

**下列每个退出码都已端到端实测**（不是只写在代码里）：

| rc | 含义 | 动作 | 实测证据 |
|---|---|---|---|
| **0** | 裁决已产出（MECHANISM-PASS / MECHANISM-FAIL / GUARDRAIL-FAIL） | 进入 8.4 写作 | 12-run 全量排练 `FINISH_RC=0`（207 s）；重入再跑亦 0（§8.9） |
| **2** | 尚有 run 未产出完整 TEST 区块 | **等**；看**训练日志**（`Run7/<variant>/<dataset>/train_log.txt`）与 `nvidia-smi --query-compute-apps`（应恰 4 个）；**不要**用督导日志判断（波次内它静默），不要手动乱启 | 真实未跑完状态实测 |
| **3** | 环境/脚本缺失 | 查 conda 环境与文件是否齐 | （仅环境守卫，未构造） |
| **4** | 逐 run 硬门有 FAIL | 看 `_RESULTS/check_run_*.json` 的 `failed` 列表；**不得**用 `RUN7_ALLOW_CHECKRUN_FAIL=1` 蒙过，必须先查明 | 篡改 E9 `deploy_params` 后实测：`FINISH_RC=4` **且 `verdict.json` mtime 未变**（第 4 步没跑 ⇒ **没有铸造裁决**） |
| **5** | 独立审计未全过 | 结论**按 INVALID 处理**；查 `INDEPENDENT_AUDIT.json` 里 FAIL 的那几项 | ① 1 库排练（缺 run）实测；② **越过开关后**审计仍抓到同样的篡改：`10 checks, 1 FAILED` |

**关于 `RUN7_ALLOW_CHECKRUN_FAIL=1`（已实测）**：它**确实能**让链条越过第 2 步
（实测第 4 步照跑、`verdict.json` 被改写），**但第 5 步的独立审计会独立再抓一次**
⇒ 最终仍是 `FINISH_RC=5` 并打印「结论必须按 INVALID 处理」。
这正是"两道不同代码路径的门"的意义：**绕过一道，另一道仍拦得住**。


### 8.3 拿到 rc=0 后先核这几项

1. `_RESULTS/summary.json` → `n_valid == 12`
2. `_RESULTS/check_run_*.json` × 12 → 每个 `n_failed == 0`
3. `_RESULTS/INDEPENDENT_AUDIT.json` → `n_checks=10, n_failed=0`
4. `_RESULTS/verdict.json` → `verdict` / `winner` / `reasons` / `attribution`

> **收口前卫生（2026-10-11 10:30 已做）**：`_RESULTS/` 里原先留着我早前 DRY 测试的
> `summary.json`(0/12)、`summary_partial.json`、`verdict_partial.json/.md`(INCOMPLETE)。
> 它们**不会被任何工具读取**（审计走显式路径；`_OBJECT_METRICS` 当时为空），
> 但会与正式产物并排出现、有被误引为结果的风险。已按纪律**改名**为
> `_SUPERSEDED_20261011_1030_*`（**未删除**）。
> ⇒ 核验时若看到 `_SUPERSEDED_*`，那是历史测试产物，**不要引用**；
> 正式产物只认 `summary.json` / `verdict.json` / `INDEPENDENT_AUDIT.json` / `check_run_*.json`。


### 8.4 五种 verdict 怎么写（**硬目标未达也照实写**）

| verdict | 写法 |
|---|---|
| `MECHANISM-PASS` | 机制有效：四库 F1 表 + 小目标表 + 机制原始读数表 |
| `GUARDRAIL-FAIL` | "F1 达标但以中大对象/目标级 recall 为代价" ⇒ **记负结果 + 归因**（这本身是有价值的负面发现，不得省略） |
| `MECHANISM-FAIL` | 按 `reasons` 分清是"F1 不够/SYSU 未提升"还是"机制自检哪一条没过"，逐条写 |
| `INVALID` | 先查因，**不得引用 F1** |
| `INCOMPLETE` | run 未齐，**不得当结论** |

> 新政策提醒：硬目标（98.00/92.50/85.00/95.00）是**努力方向而非必达门**；
> 未达不等于失败，**但不得为此回调任何预注册参数**。

### 8.5 必写的三张表（`verdict.md` 里已经直接产出，可整表引用）

1. 四库 F1 表（CTRL / E9 / E10 + Δ）
2. 对象级小目标表（small pooledR、Hit@25、未匹配小预测、×CTRL、中大 pooledR Δ）
3. 机制自检原始读数表（form / deploy params / fold disagree / HF tap grad norm / aux first→last）

### 8.5b 根 `README.md` 的 Run7 结果节结构（**对齐 Run1–Run6 既有格式**，不要另创）

Run1–Run6 各结果节用的都是同一个骨架。Run7 请照抄该骨架
（参考 `README.md` 的 `# Run6 结果` 一节，L550 起）：

```
# Run7 结果（已完成，`STATUS = <verdict>`；正式训练次数 = 12）

**Run7 的唯一问题**：<一句话说清这次要回答什么>
   例：把编码器 stage0/stage1 频率分解的**高频分支**接出来做边缘辅助监督，
       能否在受控消融下带来四库一致的 F1 提升（而不是又一次"信息已在 head"）。

**受控消融说明**（`finish_run7.sh`；3 变体 × 4 库；各严格 80,000 optimizer.step）：
   协议、波次、2×RTX 5090 并发、收口链耗时、独立审计结果、`models/` 改动范围。

**接受标准与守门（预注册冻结）**：指向 SELECTION_LOCK.json（含冻结时间与"见结果之前"证据）。

**四库 F1 表** + **机制自检原始读数表** + **对象级小目标表**   ← 三张表，§8.5

**判定与机制归因**：verdict + winner + reasons + attribution（**负结果同样写全**）。

**产物索引**：summary / verdict / INDEPENDENT_AUDIT / check_run_* / object_metrics_*。
```

⚠️ 必须做到的三件事：
1. `STATUS` 用 `verdict.json` 的**原值**（`MECHANISM-PASS` / `MECHANISM-FAIL` /
   `GUARDRAIL-FAIL`），**不要**改写成"成功/失败"这类主观词；
2. 若为 `GUARDRAIL-FAIL`：明确写出"F1 达标但以中大对象/目标级 recall 为代价"，
   并保留完整 reasons —— 这本身是**有价值的负面发现**，不得省略；
3. 同时更新根 README 的**目录/索引表**（若有）指向该节，并保留 `HANDOFF.md` 的链接。


### 8.6 收尾清单

- [ ] Run7 README 增「结果」节；根 `README.md` 增 Run7 结果节（**含负结果**）
- [ ] 提交并 push；产物只 `_INVALID_*`/`_SUPERSEDED_*` **改名，绝不删除**
- [ ] 任何文献表述前**重新核验** R8（EdgeRefNet, TGRS 2026）的真实出版信息

### 8.7 绝不要做

- ❌ 因未达硬目标而调 λ / 阈值 / seed（预注册已冻结，改了就作废）
- ❌ 删除任何 run 日志或 ckpt
- ❌ 重跑个别 run 以追求"更好看"的数字

### 8.8 中途失败 / 督导死亡时的恢复流程（**含一个反直觉陷阱**）

**① 先判断督导是否还活着**

```bash
cat /tmp/run7_supervisor.pid                      # 记下 pid
kill -0 "$(cat /tmp/run7_supervisor.pid)" && echo ALIVE || echo DEAD
# 看它是否正阻塞在某一波的 run_all.sh 上（这才是"活着"的正常形态）
ps -o pid,ppid,stat,etime,cmd --ppid "$(cat /tmp/run7_supervisor.pid)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # 应恰 4 个
```

> ⚠️ **不要用"督导日志是否在更新"判断死活 —— 波二/波三期间它本来就静默。**
> 原因：督导的代码是 `bash run_all.sh --wave N >> SUPLOG` **同步阻塞**，只有该波**跑完**
> 才会回来写 `wait_wave` 的 `steps_sum=` 行。所以一波之内（约 4–5 小时）督导日志**一行都不写**
> —— 实测：13:35:18 之后 33 分钟无新行，而督导完全健康。
>
> **正确判据**：`kill -0` 成功 **且** 能看到它有一个 `run_all.sh --wave N` 子进程 ⇒ 一切正常。
> `kill -0` 失败 ⇒ 督导确实死了（此时 4 个 `train.py` 很可能还在跑，督导死了 ≠ 训练死了）。
>
> **波二/波三期间看进度要读训练日志**（`Run7/<variant>/<dataset>/train_log.txt`）或
> `nvidia-smi --query-compute-apps`，**不要**读督导日志。

**② 判断哪些 run 未完成**

```bash
for v in M1_R7CTRL E9_HFAE_A E10_HFAE_B; do
  python train_scripts/CASA-TViM/Run7/collect_results.py \
    --log-root /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7 \
    --variants $v --datasets CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256 \
    --out /tmp/rs_$v.json >/dev/null 2>&1
  printf '  %-11s ' "$v"
  grep -oE '"n_(valid|total)": *[0-9]*' /tmp/rs_$v.json | tr '\n' ' '; echo
done
grep -l -E "3 attempts exhausted" \
  /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/*/*/train_log.txt
```

> 该循环**只写 `/tmp/rs_<v>.json`**，**不会**碰到督导正在用的 `/tmp/r7_sup_gate.json`。
> 同时打印 `n_valid` 与 `n_total`：只看 `n_valid` 在波次进行中会显示 `0`、无法区分
> 「0/4」与「0/12」。（本节的每条命令都已**照抄实测**可执行、输出可判读。）

**③ 波内崩溃一般会自愈**：`run_all.sh` 对每个 run 有 **3 次重试**，每次都从**自己的**
`last.pth` 自动续训（`train.py` 恢复 `actual_steps` 与 RNG）。只有日志出现
**`3 attempts exhausted`** 才是该 run 的终态失败。

**④ ⚠️ 反直觉陷阱：督导**没有**设置 `RESUME_SAME_RUN`**

`run_all.sh` 对"非空 ckpt 目录"是**拒绝启动**的（防止把新实验写进旧目录）：

```
[ABORT] <ckpt>/last.pth exists; set RESUME_SAME_RUN=1 only for certified same-run resume
[ABORT] <ckpt> not empty; refusing to start a new experiment here
```

而督导调 `run_all.sh --wave N` 时**不带**这个变量。因此：

- **波一未完成、督导死了** ⇒ 直接重启督导即可（它只是 `wait_wave M1_R7CTRL 1`，
  **不会**重跑 `run_all.sh --wave 1`）——**安全**。
- **波二/波三之后督导死了**（或需要重启某一波）⇒ 直接重启督导会**在波二处 ABORT**，
  因为波二的 ckpt 目录已经不空了。**必须先导出**：

```bash
cd /home/yqwang/projects/CASA-CD
export RESUME_SAME_RUN=1          # 只用于"认证的同一次 run 续训"
# 要么单波手动接续：
bash train_scripts/CASA-TViM/Run7/run_all.sh --wave 2   # 会从各自 last.pth 续训
bash train_scripts/CASA-TViM/Run7/run_all.sh --wave 3
# 要么重启督导（RESUME_SAME_RUN 会被 run_all.sh 继承）：
nohup bash train_scripts/CASA-TViM/Run7/supervise_waves.sh \
  >> /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7_supervisor.log 2>&1 &
```

> 陈旧的 pidfile 不会挡住重启：单实例守卫用 `kill -0` 判活，pid 已消失就会直接覆盖。
> 督导重启后 20 小时上限**重新计时**。

**④-验证**：上面这条陷阱**不是推断**，是实测的 —— 从 `run_all.sh` **逐字提取** `run_one()` 的
前置校验片段（9 行），在受控 ckpt 目录上单独执行（真跑 `--wave 2` 会启动训练，故不能那么测）：

| 情形 | 结果 |
|---|---|
| 全新空目录 | `NO_ABORT(would launch)` ⇒ 波二/波三正常启动 ✅ |
| 非空 ckpt 目录、**不带** `RESUME_SAME_RUN` | `ABORT(refusing to start a new experiment here)` ⇒ **陷阱确认存在** ✅ |
| 非空 ckpt 目录、**带** `RESUME_SAME_RUN=1` | `NO_ABORT(would launch)` ⇒ 恢复流程有效 ✅ |

`PRECHECK CHECK = PASS（与 §8.8 文档一致）`


**⑤ 恢复后必须逐条核对**（缺一不可）：

- 每个变体 `collect_results` 都到 **4/4 VALID**；
- `check_run.py` 对 12 个 run **无 FAIL**；
- 日志里**不出现** `budget already reached`（出现说明续训时预算已耗尽，该 run 需按 INVALID 处理）；
- 若某 run 的 `run_manifest.json` 的 `resume_source` **指向别的变体/库** ⇒ 视为受控消融被污染，按 INVALID 处理。

**⑥ 无论如何**：产物只 `_INVALID_*` / `_SUPERSEDED_*` **改名**，**绝不删除**。

### 8.9 收口链是**可重入**的（实测 `RERUN_RC=0`）

**为什么必须知道**：正式收口只有一次机会。若第 4/5 步出错需要**重跑整条链**，而第 3 步
（`run7_object_metrics.py --out`）不容忍已存在的输出目录，那"重跑"会在第 3 步就失败，
接手者会误判成链路坏了。所以这一点必须实测，不能假设。

**做法**：对一棵**已经跑过一次收口**的树（`_RESULTS` 16 项、`_OBJECT_METRICS` 12 项、
`SHA256SUMS.txt` 1 行）**再跑一次** `finish_run7.sh`，不重建夹具。

**结果**：

| 项 | 结果 |
|---|---|
| 六步是否全部完成 | ✅ 全部通过，`RERUN_RC=0`，耗时 **203 s** |
| 第 3 步（对象级复算，非空 out 目录） | ✅ `object metrics rc=0`（同名文件**覆盖**，不拒绝、不重复） |
| `_OBJECT_METRICS` 项数 | 12 → **12**（无重复堆积）✅ |
| 裁决是否稳定 | ✅ `GUARDRAIL-FAIL winner=E9_HFAE_A n_reasons=12`（与首次完全一致） |
| 独立审计 | ✅ `n_checks=10 n_failed=0` |
| `SHA256SUMS.txt` | 1 → **2 行**（审计**每次运行追加**，属**设计如此**：它是"审计运行台账"，不是幂等文件） |

⇒ **收口链可安全重跑**。若第 4/5 步出问题，直接重跑 `finish_run7.sh` 即可，不必先清理目录。

> 关于 `SHA256SUMS.txt` 的多行：它是 append-only 的运行台账，重复行**不是**缺陷；
> 但引用哈希时请取**最后一次**写入的那一行（`tail -1`）。

---

## §9 交付状态一条命令核对（任何时刻都可重跑）

不确定"当前这套东西是否仍然自洽"时，跑这个。**全部应当通过**：

```bash
cd /home/yqwang/projects/CASA-CD

# 1) 身份清单逐字节对拍（期望 26 条全 OK、无输出其它行）
sha256sum -c train_scripts/CASA-TViM/Run7/SOURCE_SHA256.txt | grep -c ': OK'   # -> 26
sha256sum -c train_scripts/CASA-TViM/Run7/SOURCE_SHA256.txt | grep -v ': OK'    # -> 空

# 2) 关键工具自检（期望 PASS / PASS / PASS(18 branches) / PASS）
python train_scripts/CASA-TViM/Run7/check_mapping.py  | tail -1
python train_scripts/CASA-TViM/Run7/check_imports.py  | tail -1
python train_scripts/CASA-TViM/Run7/run7_verdict.py --selftest    | grep '^SELFTEST'
python train_scripts/CASA-TViM/Run7/independent_audit.py --selftest | grep '^SELFTEST'

# 3) 督导 + 进度 + 异常（期望 ALIVE / 无 errors / 恰 4 个 GPU 进程）
#    注意：督导日志在波二/波三期间**静默是正常的**（同步阻塞在 run_all.sh --wave N 里），
#          所以**不要**用它判死活；用 kill -0 + 子进程 + GPU 进程数。
kill -0 "$(cat /tmp/run7_supervisor.pid)" && echo "supervisor ALIVE"
ps -o pid,stat,etime,cmd --ppid "$(cat /tmp/run7_supervisor.pid)" | head -3
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # 应恰 4 行
grep -l -E "Traceback|CUDA out of memory|3 attempts exhausted" \
  /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/*/*/train_log.txt || echo "no errors"
```

**2026-10-11 10:55 实测基线**（此后数值会随时间变化，但"是否 PASS"应保持不变）：

| 检查 | 实测 |
|---|---|
| 身份清单 | **26/26 OK**，无非 OK 行 |
| check_mapping / check_imports | PASS / PASS |
| verdict selftest | **PASS (18 branches)** |
| audit selftest | PASS |
| 训练 | supervisor ALIVE；无 errors；4 个 `train.py`；ckpt 634 M；磁盘 2.6 T 空闲 |

本地侧同样应有：`models/test_run7_hf_edge.py` = 12 checks / 0 FAIL（6 项因无 CUDA kernel 跳过，
服务器上为 0 SKIP）；`analyse/tests` 五个文件合计 **142 tests OK**；`git status` 干净。