# Run6 — 变化判别瓶颈的只读可分辨性前测 执行索引

> **当前状态：S1 前测脚本已审阅、修正、补测、本地端到端冒烟 PASS；尚未在服务器运行。**
> `decision = PREFLIGHT_PENDING`、`training_runs_executed = 0`、
> `models/` **未被修改**（与 Run5 结束后逐字节相同）；本目录**不含也不会补写训练脚本**。
> 这与 Run6 设计文档 §4.2 的硬要求一致：`gate.json` 非 PASS ⇒ 不写训练脚本、不改 `models/`、不开 80K。

## 0. 冻结输入

| 项 | 值 |
|---|---|
| 设计文档 | `docs/temporary/CASA-CD_Run6_变化判别瓶颈可分辨性前测与读出结构设计_2026-10-11.md` |
| 设计文档 SHA256 | `d77cacbd7b52363598487b348b2593ce5f989a6ae66d55dc688a0cd80969eb5b`（461 行 / 45282 bytes） |
| 冻结仓库版本 | `YuqiWang-code/CASA-CD@408099dbf65bfb351d23fa5d295f822645120ce1` |
| 前测脚本 | `analyse/tvim_run6_discriminability.py`（`run6_discriminability_v1`） |
| 本地单测 | `analyse/tests/test_tvim_run6_discriminability.py`（71 项） |
| 锁定 checkpoint | SYSU `Run1/M1_FULL/best_F1=0.8347.pth`，sha256 `45d688a01af22fe22521f4dfb4579e80827ddcaa886e07bafc778fc288731457` |
| 锁定 test list | sha256**\_text\_lines**（非 raw bytes）`5f4122c9d32cb6db475f4f361618b2ac8bb10938df93d9550e0bb9a422a45797` |
| 前测输出根 | `/home/yqwang/outputs/CASA-CD/diagnostics/TViM-Run6-DISCRIMINABILITY-20261011` |
| 决策锁 | `train_scripts/CASA-TViM/Run6/SELECTION_LOCK.json`（`decision=PREFLIGHT_PENDING`） |

## 1. S1 前测：唯一问题与唯一入口

**问题（H6a）**：冻结的 Run1 `M1_FULL` 既有特征里，能否把「FP-like 未匹配小预测」与
「真实 small 变化」分开？

- **拟合** = `train.txt` 的 D3 SHA1-hash **90% train** 部分（`_hash_split(name, 0.10) == "train"`，非新 seed）；
- **评估** = `test.txt` 全量 4000 张；**test 不参与拟合、不选特征、不选阈值**；
- probe = 标准化线性 logistic（无隐藏层），L2 由**按图像分组**的 5 折 CV 在拟合集上选（`λ ∈ {1e-3, 1e-2, 1e-1}`）；
- test AUC/AP 的 95%CI = **图像级**有放回 bootstrap 1000 次、seed 16（同图对象同进同出）。

**对象口径（必须与 Run5 前测 / F14 逐字一致）**

| 类别 | 定义 | test 期望数 |
|---|---|---|
| 正（真实 small 变化） | GT small 对象（面积 [1,255]、4 连通）中 `IoU>=0.10 已匹配` **或** `像素 recall r<0.25` 者 | **327**（matched 35 + missed 292） |
| 负（FP-like） | 预测 4 连通域（面积 [1,255]）中一次性最大权重匹配后**未匹配**者 | **2430** |

**两个候选（K7 二选一，互斥；test 不参与选择）**

| 候选 | 位置 | 机制 | 新 deploy 参数 | deploy 总数 |
|---|---|---|---|---|
| `C1` 跨尺度一致性门控 | TAR 四级输出 → DCR 之间 | `raw_s=f_s.abs().mean(1)` → 统一到 1/4 比较得 `d_s` → `f_s *= (1+γ_s·d_s_up)`，γ 零初始化 | 8（+可选共享 1×1 = 105） | 4,880,198 |
| `C2` 对象内部一致性门控 | `decoder.refine` → head 之间 | 3×3 DW + 1×1 产生"碎片 logit"，γ 零初始化 | 961 | 4,881,151 |

**K7 选一规则（预注册，禁止按 test 选）**：按 `train-CV AUC` 降序 → 相差 **≤0.005** 取 deploy 参数更少者
（C1）→ 仍平局按字典序。实现上用 `> 0.005 + 1e-12` 判定"领先"，`1e-12` 仅抵消 IEEE754 表示误差。

### 1.1 前测硬门（数值在见结果之前冻结，禁止事后回调）

| ID | 硬门 | 数值 | FAIL 后动作 |
|---|---|---|---|
| K0 | 源码/ckpt/list SHA 严格匹配、输出目录全新、test 未参与拟合 | SHA 相同 + `fitted_on=train_hash90` + `test_used_for_fit=false` | `P0_INVALID` → 修实现后重跑，无 80K |
| K1 | 对象有效数（test） | `n_pos>=300` 且 `n_neg>=2000` | `INCONCLUSIVE` → NO_80K |
| K2 | 胜出候选存在可分离信息 | test `AUC>=0.80` 且 CI 下界 `>0.75` | NO_80K |
| K3 | 不只是"更亮就更大" | test `AP>=0.35` 且 CI 下界 `>0.25` | NO_80K |
| K4 | 必须显著超过手工证据通道 | 胜出候选 `AUC − **0.5992** >= 0.15`（**冻结常数**） | NO_80K |
| K4b | 必须在完整基线之上再增量 | 胜出候选 `AUC − baseline_full >= 0.05` | NO_80K |
| K5 | 一致性统计量必须优于纯幅度基线（H6b） | 胜出候选 `AUC − AUC_amp >= 0.10` | **降级** `gate_stat=amplitude`（仍可训练，但论文不得称跨尺度一致性贡献） |
| K6 | 双候选可比性 | C1/C2 同一对象集合、同一 probe、同一 bootstrap；两者都存档 | `P0_INVALID` |
| K7 | 选一（见上） | `selected ∈ {C1,C2}` | `P0_INVALID` |

**K2/K3/K4/K4b 任一 FAIL ⇒ `NO_80K`，Run6 训练次数 = 0**，按设计文档 §7.6 收口（不再搜索新模块）。

四个基线 probe（同框架、同对象集合）：`baseline_amp`（幅度）、`baseline_hand`（手工证据通道）、
`baseline_head`（模型自身置信度）、`baseline_full`（三者拼接 = K4b 分母）。

### 1.2 输出结构（`--out-dir` 必须全新/为空）

```text
TViM-Run6-DISCRIMINABILITY-20261011/
  inputs.json          # 源码/ckpt/list SHA、特征定义、split 规则、口径说明、冻结常数、环境
  features_train.csv   # 拟合集逐对象特征（§4.3 命名）
  features_test.csv    # 评估集逐对象特征
  probe_results.json   # 8 个 probe 组：train-CV、test AUC/AP、CI、逐特征 test AUC、选一结果、冻结对账
  bootstrap.json       # 图像级 1000 次、seed16、valid_bootstraps
  gate.json            # K0..K7、status ∈ {PASS, FAIL, INCONCLUSIVE, DRY_ONLY, INCOMPLETE}
  SHA256SUMS.txt
```

`--limit>0` 只做链路自检，状态恒为 `DRY_ONLY`，**永不 PASS**。

### 1.3 时间预算（A06：≤900 s）

脚本按图像数把总预算切成两个 pass：`fit_sec = budget·n_fit/(n_fit+n_eval)`、`eval_sec = budget − fit_sec`
（`--budget-sec 900`，当 `n_fit=10800 / n_eval=4000` ⇒ fit ≈ 657 s、eval ≈ 243 s）。
**任一 pass 超时 ⇒ `status=INCOMPLETE`**（不是 FAIL：FAIL 意味着"机制不存在"，超时不是机制结论），
且**不删样本、不事后补跑**。

## 2. 本地实现口径与对草稿的修正（审计要点）

### 2.1 必须在 `inputs.json` 中留痕的三条口径

1. **手工证据通道在归一化张量口径计算**（与 F14 / Run5 FP 画像**逐字同口径**）：
   A = `img[:, 0:3]`、B = `img[:, 3:6]`（DataLoader 直接给出的 `Normalize→Scale→ToTensor` 结果），
   `edge_field` 直接复用 `tvim_run5_fp_profile.edge_field`；**不做反归一化**。
   依据：冻结读数 `change_abs` FP = **0.9730**、`gradA` FP = **1.8008** 只可能在归一化口径下出现
   （原始 [0,255] 下"均值绝对差 <2"不可能），⇒ 只有同口径重算才能与冻结的手工基线 0.5992 直接对账。
   该结论有单测固定（`TestHandPixelSpace::test_frozen_change_abs_scale_is_normalized_not_raw`）。
2. **C2 的外环不使用 GT**：`ring = dilate_4px(O) \ O`（4 邻域结构元，纯几何量）；
   内部 `interior = erode_1px(O)`。Run5 FP 画像的 `R=(dilate(O)\O)∩(~GT)` **只用于空间画像**，
   作为特征会引入 GT 依赖（正样本的外环被系统性削去 GT 邻域）⇒ 本前测**有意不同口径**。
3. **C1 的"对象内平均变化证据"**按设计文档 §5.1：`f_s` 是 TAR 已融合的单路特征（无 A/B 之分），
   故用 `f_s.abs().mean(dim=1)` 的固定统计量；**不新增任何张量入口、不碰频率/边缘通道**。

### 2.2 修正清单（对 `408099d` 入库草稿，逐条核对 HEAD）

| # | 级别 | 草稿缺陷 | 后果 | 本次修正 |
|---|---|---|---|---|
| 1 | **P0** | `compute_row` 9 参签名，却以 **11 个实参**调用（且把 batch size `B` 传进 `obj` 位） | 首个对象即 `TypeError`，链路**根本跑不通** | 重构为 `image_context(ctx) + compute_row(ctx, obj)` |
| 2 | P1（门语义） | K4 用**重算的多特征 hand probe AUC**，而冻结常数 `K4_handcrafted_baseline_auc=0.5992` 虽定义却**未使用** | 与预注册 §4.2「`AUC − 0.5992 ≥ 0.15`」不一致 | K4 改用**冻结常数**；重算值仅并列报告（`delta_vs_recomputed`），不进门 |
| 3 | P1（GT 泄漏） | C2 外环 `=(dilate(O)\O)∩(~GT)`，GT 进入特征；且无「内部腐蚀 1px」定义 | 正负样本环带几何被 GT 系统性区分 | 改为无 GT 的 `dilate_4px(O)\O` + `erode_1px(O)` |
| 4 | P1（特征不完整） | `Y_spatstd_mean` 与 `Y_spatstd_s1` 取同一个值（占位），s2/s3/s4 空间离散度缺失 | C1 特征组名不符实 | 逐尺度 `Y_spatstd_s1..s4` + 其均值 |
| 5 | P1（特征不完整） | 幅度基线缺 §4.1 明确要求的「`‖Δ‖` 的 4 尺度平均」（H6b / K5 的分母） | K5 分母不是预注册定义 | 增 `mag_s_mean`（由 4 个尺度均值派生，**不是新证据通道**） |
| 6 | P1（停机语义） | 超时被判为 `FAIL`（=NO_80K）；且两个 pass 共用同一个 `t0` 预算时钟 | 超时会被误读为"机制不存在" | 按图像数切分预算；超时 ⇒ `INCOMPLETE` |
| 7 | P2 | `gate.json` 残留 Run5 的 `val_screen_status` 字段 | 与 Run5 产物混淆 | 移除 |
| 8 | P2 | 输出名 `features_fit.csv`，与设计文档 §4.3 的 `features_train.csv` 不符 | 产物名与规格不符 | 改名 `features_train.csv` |
| 9 | P2 | 无与冻结 `n_pos=327 / n_neg=2430` 的对账块（§9 R7 的要求） | 对象口径写错不会被发现 | 增 `frozen_reference_reconciliation`（写进 `gate.json` 与 CSV 报告） |
| 10 | P2（预算） | 每个对象重复 `.cpu().numpy()` 全部 10 个 hook 张量、每对象每尺度重复 `resize` | 900 s 预算下 CPU 开销过大 | 每图一次预计算 + 每对象分辨率缓存 + **无合格对象的图直接跳过特征提取** |

**两处自我更正（记录在案，避免后续重复误判）**

- 初审时曾把 `_resolve(model, key)` 判为"实参个数不匹配"；核对 HEAD 后确认该处**本来就正确**
  （2 参对 2 参，映射 `Y_s*→tar.stage*.temporal`、`T_s*→tar.stage*`），**未做修改**。
- 初审时曾把"手工通道在归一化张量上计算"当成缺陷并改为原始 [0,255]；核对
  `tvim_run5_fp_profile.py` 源码与冻结读数（`change_abs` FP=0.9730）后确认**草稿是对的**，
  已**回退**并与 F14 逐字对齐（见 §2.1 第 1 条）。

### 2.3 性能实测（本地，预算依据）

| 量 | 实测 |
|---|---|
| `image_context`（每图固定开销：2×Sobel + 2×dilate1 + 10 张特征图搬运） | ≈ **13.4 ms/图** |
| 每对象特征行（C1+C2+confound+hand） | ≈ **3.1 ms/对象** |
| test 中「有合格对象的图」占比（按 Run5 `per_image.csv` 真值） | **38.8%**（1550/4000；对象总数 2794） |
| 优化后 CPU 投影 | fit ≈ 93 s + eval ≈ 34 s ≈ **127 s** |
| GPU 前向投影（2×RTX 5090，batch 16） | ≈ 400–450 s |
| 合计投影 | **≈ 550–600 s < 900 s**（含 margin） |

## 3. 本地可跑的正确性护栏（门 PASS 前不会上传训练脚本）

```bash
python -m compileall -q analyse models train_scripts
python -m unittest discover -s analyse/tests -p 'test_tvim_*.py' -v      # 138 项（Run6 新增 71 项）
python analyse/tvim_run6_discriminability.py --help
```

`analyse/tests/test_tvim_run6_discriminability.py` 覆盖：`pool_map` / `resize_frac`（覆盖率池化）、
`auc_score` / `ap_score`（手工小例子对拍，含并列分数与全并列）、`cross_scale_features`、
`fit_logistic`（可分数据 AUC>0.9、确定性、L2 单调收缩、标准化只用拟合集统计）、`_impute`
（中位数只从拟合集取、全 NaN 列兜底 0）、`ring_mask` / `interior_mask`（**无 GT 依赖**、
4 连通结构元、平移不变）、**缺失列不使整行被丢弃**（`_matrix`+`_impute`+`probe_report` 行数守恒）、
`k7_select`（含恰为 0.005 的边界）、冻结常数与 K 门数值、对象口径（`positive_rule` + small 面积组过滤，
Run5 F15 回归项）、预算切分、以及手工通道像素空间（§2.1 第 1 条）。

本地端到端冒烟（`.claude/_r6_smoke.py`，gitignored）：合成 8 图 + 桩模型跑通完整链路，
31 项检查全 PASS，含「CSV 表头 == 元数据 + `ALL_FEATURES`」「`--limit` ⇒ `DRY_ONLY` 且 rc=2」
「`gate.json` 无 Run5 残留」「`test_used_for_fit=false`」。

## 4. 执行顺序（S0→S9，任何一步 FAIL 即停止）

| 步 | 内容 | 阻断条件 |
|---|---|---|
| S0 | 环境（torch/CUDA/`selective_scan_cuda_oflex`）+ 逐文件 SHA 身份 | import 或 SHA 不符 ⇒ STOP |
| S1 | 单测 138 项 + P0 素材核验（ckpt SHA / test list `sha256_text_lines` / 输出目录全新） | 任一不符 ⇒ `P0_INVALID` |
| S2 | **DRY 排练** `--limit 400`（独立 out-dir） | 链路不健康 ⇒ STOP |
| S3 | **全量前测**（fit ~10800 + test 4000，预算 900 s） | `INCOMPLETE` ⇒ 只在修性能后重跑，不改口径 |
| S4 | **独立最小复算审计**（`.claude/_r6_audit.py`，与主脚本不同代码路径） | 任一不一致 ⇒ 禁止进入 §5 |
| S5 | 按 `gate.json` 裁决 | 非 PASS ⇒ `NO_80K`，归档证据 |

S0–S4 全部只读：不训练、不改 `models/`、不覆盖任何历史产物。
`--out-dir` 已存在且非空时脚本直接 `[P0_INVALID]` 退出；**失败/作废产物改名
`_INVALID_<stamp>` / `_SUPERSEDED_<stamp>` 保留，绝不删除**。

## 5. 尚未执行（门 PASS 后才允许）

S6 按 §5 实装**唯一一个**门控（C1 或 C2，由前测 K7 锁定）→ S7 `models/test_run6_gate.py` T0–T11
→ S8 `smoke_test.py` + 真实数据 3 步 dry-run + 两卡显存探针 → S9 两波 8×80K
（波一 `M1_R6CTRL`、波二 `E8_SCG`，四库各严格 80,000 step）。

**本目录在门 PASS 之前不会出现 `_gen_scripts.py` / `run_all.sh` / `post_train.sh`。**

## 6. 禁止项（红线）

1. 前测未 PASS 前**不得修改 `models/`**、不得写训练脚本、不得开 80K；
2. **不得新增证据通道类模块**（边缘图 / 高频增强 / 变化幅值门 / 纹理能量门）——已被数据否决
   （F14：手工通道 AUC 上限 0.5992）；
3. 不得修改设计文档 §4.2 的 K 门数值、§5.4 的 K7 选一规则、§7.3 的层 B/C 阈值；
   **不得用 test 选层、选特征或选阈值**；
4. 不得改损失 / 增强 / 阈值 / 种子 / EMA；不得多 seed；不得用 Run4/Run5 权重 warm start；
5. 不得删除或覆盖任何历史产物（含失败的与作废的）；
6. 不得把论文硬目标（98.00/92.50/85.00/95.00）当作 Run6 的成功条件；不得声称未实测的结论；
7. 不得编造文献；不得把第三方笔记仓库当作官方代码。

## 7. 产物索引

```text
（S3 全量前测完成后在此登记 sha256；当前为空，PREFLIGHT_PENDING）
```

决策锁：`train_scripts/CASA-TViM/Run6/SELECTION_LOCK.json`（`decision=PREFLIGHT_PENDING`、
`training_runs_executed=0`）；前测结束后按结果更新为 `PASS_...` / `NO_80K`。
