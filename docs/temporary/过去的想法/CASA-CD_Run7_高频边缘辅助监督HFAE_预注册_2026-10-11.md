# CASA-CD Run7 预注册：前两层频率分解高频 → 边缘检测辅助监督（HFAE）

> **文档属性**：可执行的预注册规格书（**不是**结果报告）。
> **冻结仓库版本**：本文件提交时的 `main` HEAD。
> **前置结论（Run6）**：`NO_80K` —— 冻结 M1 特征里**存在**可分辨信息（C2 probe test AUC 0.9724 /
> AP 0.9216），但相对完整基线只 **+0.0080**，且 `baseline_head` 单独就有 AUC **0.9565**：
> **判别信息已在模型自己的 head 置信度里，读出层结构性增益已穷尽**。
> **本次研究政策（2026-10-11 修订，README「研究目标与实验纪律」）**：
> 硬目标（98.00/92.50/85.00/95.00）是**努力方向而非必达门**；
> **每轮改进都上服务器跑完整 80K**；**训练前诊断从宽**，有效性由**服务器 80K 消融**判定。

---

## §1 一句话机制

**从编码器 stage0（1/4）与 stage1（1/8）的频率分解高频分支接出证据，经中高频增强后做一个
"变化边缘检测"辅助监督任务；边缘监督的梯度回流到高频分支，从而重塑表征（而非再读一路证据）。**

**为什么这是新机制而不是重试旧方案**

| 已被数据否决 | 否决内容 | 与本机制的区别 |
|---|---|---|
| Run5（`NO_80K`） | 把原生 `local_conv(x_high)` 作为**无监督证据 tap** 接进 TAR；small 尺度与冻结 tap 几乎无差（margin 0.00697 vs 0.00685），对 FP-like(0.2636) 与真实漏检(0.2970) 响应几乎相同 | 本机制**有监督**：边缘 GT 的梯度直接作用于高频分支；且 tap 在 **1/4**（Run5 从未接过 1/4） |
| F14 | **手工**边缘/高频/幅值/纹理通道作为判据（最佳单项 AUC 0.5992、`change_abs` 0.4779 方向相反） | 本机制不引入手工判据通道；边缘头是**可学习模块**且只用于训练目标 |
| Run2 FRH / Run4 FET1 | 末端（refine 之后/head 附近）旁路，两次 `MECHANISM-FAIL` | 本机制作用在**编码器最细两级**的表征上，不在末端读出 |
| Run6 | TAR/DCR 之间的**读出层门控**（K4b FAIL） | 本机制**改变主干算出来的东西**，不改变读出结构 |

**承重部件是"监督"**：去掉边缘监督即退化为已被否决的无监督注入，因此辅损不可选。

---

## §2 架构依据（实测，非推测）

TinyViM 频率解耦在 `SS2D.forward_core`：
`x_low, x_high = split(x); x_high = local_conv(x_high)`（高频分支，**全分辨率、不经 pool**）；
`index<3` 时另有 Laplace 残差 `res = x_low − upsample(pool(x_low))`。

| stage | 分辨率 | d_model | 低频 ch | **高频 ch** | HF 占比 | SS2D 池化 | 扫描分辨率 |
|---|---|---|---|---|---|---|---|
| **stage0** | **1/4 (64²)** | 48 | 12 | **36** | **75%** | 8× | 8² |
| **stage1** | **1/8 (32²)** | 64 | 32 | **32** | 50% | 4× | 8² |
| stage2 | 1/16 | 168 | 84 | 84 | 50% | 2× | 8² |
| stage4 | 1/32 | 224 | 168 | 56 | 25% | 1× | 8² |

**stage0 是全网络最"高频主导"的阶段**（75% 通道给 HF，低频被 8× 池化压到 8²）——
这就是"前两层、高分辨率、小目标清晰"的架构依据。
（交叉验证：上表 32ch@1/8 与 84ch@1/16 与 Run5 冻结记录的 `Pair-2`/`Pair-3` tap 形状逐一吻合。）

---

## §3 实现规格（冻结）

**tap**：`encoder.network[0][2].op.local_conv`（36ch@1/4）+
`encoder.network[2][2].op.local_conv`（32ch@1/8），**只读 forward hook**，不改 `ss2d.py` 任何算子。

**边缘头 `HFEdgeHead`**（每尺度 1×1 → 对齐相加 → DW3×3 → 1×1 出边缘 logit）：

- 输入按双时相配对：`cat([A, B, |A−B|])`（tap 为 `(2B,C,H,W)`，前 B = A、后 B = B）
  ⇒ 输出 batch = **B**（**每对样品一张变化边缘图**，而非给 A/B 各一份）；
- 输出 `edge_logits (B,1,64²)`（监督用）与边缘特征 `z (B,32,64²)`（注入用）；
- 参数：`aux_only` head **7,041**；`aux_inject` 再加注入 1×1 → head **10,209**。

**边缘监督目标**：`E = dilate₄(gt) \ erode₄(gt)`（4 邻接结构元，内外各 1px），
用 `adaptive_max_pool2d` 降到 64² **保底**（1–16px 小对象的 1px 轮廓不被吃掉）；A/B 共享。
损失 = `λ · (BCE + 1 − Dice)`（复用 `BCEDiceLoss`），**λ = 0.2 固定，禁止搜索**。

**两个部署形态**（实验前固定，四库同一值）：

| variant | `--hf_edge_form` | 推理期 | **deploy 参数** | 增量 |
|---|---|---|---|---|
| `M1_R7CTRL` | `--hf_edge 0` | 与 Run1 `M1_FULL` 同构 | **4,880,190** | — |
| `E9_HFAE_A` | `aux_only` | `switch_to_deploy()` **删除整个边缘头** | **4,880,190** | **+0（零推理代价）** |
| `E10_HFAE_B` | `aux_inject` | 保留 proj/dw + 零初始化注入（丢监督专用 `edge_out`） | **4,890,366** | +10,176 |

**零初始化**：注入 1×1 的 weight/bias 全零（**必须在 `apply` 之后设置**，否则被覆盖 ⇒ T2 自检）
⇒ epoch-0 与关闭开关**逐位相同**。
**RNG 纪律**：边缘头在 `torch.random.fork_rng(devices=[])` + 固定局部种子 16 内构造，
不推进全局随机流 ⇒ CTRL 与 EXP 的既有模块初始化逐位一致（实测 1298 共有键 0 处不一致）。

**CLI（fail-closed，写入 `arch.json` / `run_manifest.json` / `last.pth`）**：
`--hf_edge {0,1}`、`--hf_edge_form {none,aux_only,aux_inject}`、
`--hf_edge_taps {1_4_1_8,1_4}`、`--hf_edge_lambda`（默认 0.2）；
`protocol_version = "run7_hf_edge_exact80k_v1"`；与 `--fine_tap` 互斥。

---

## §4 受控消融（全部服务器完整 80K）

| 变体 | 唯一变量 | 四库 |
|---|---|---|
| `M1_R7CTRL` | 无（同期同代码对照） | CDD/LEVIR/SYSU/WHU 各 80,000 step |
| `E9_HFAE_A` | + 边缘监督（train-only，deploy 不变） | 同上 |
| `E10_HFAE_B` | + 边缘监督 + 零初始化注入 DCR 1/4 输入 | 同上 |

协议：256×256、BCE+Dice(+λ·edge)、Adam(2e-4, 0.9/0.99, wd=1e-4)、poly^0.9 + 200 iter warmup、
严格 80,000 `optimizer.step`、batch 32、seed 16、test-as-val、`prob > 0.5`、
四库各从同一 ImageNet 权重**独立从头训练**；`--exact_max_steps 1`。

**判据（按新政策：无多层硬门，看方向与一致性）**

- **主**：四库 F1 vs 同期 `M1_R7CTRL`；
- **次**：SYSU small(1–255px) pooled Recall / Hit@25（**F04 冻结值 0.1974 / 0.1896**）、
  未匹配小预测 CC 数（冻结 2430）、ObjPrecision/ObjRecall（冻结 0.5320 / 0.7674）；
- **守门**：中/大对象 pooled Recall 不显著回退；CDD/WHU 不得明显恶化；
- **机制自检**（训练期即验证，不依赖最终指标）：
  ① `[HFAE-GRAD]` 首步 `hf_tap_grad_norm > 0`（**边缘梯度确实进了高频分支**）；
  ② epoch-0 `torch.equal(CTRL, EXP)`；③ 真实 batch 折叠 disagreement = 0；
  ④ deploy 参数门 ≤5M；⑤ `[HFAE-LOSS]` 逐 epoch 下降（辅助任务确实被学会）。
- **接受标准（预注册）**：`E9`/`E10` 至少一个在 **≥3/4 库 F1 ≥ CTRL** 且 **SYSU F1 > CTRL**，
  且 `[HFAE-GRAD]` 与折叠门全过 ⇒ 记为"机制有效"。否则记**负结果 + 归因**（同样写进论文）。

---

## §5 预期与风险（诚实的先验）

**预期方向**：收益应在**小目标召回/定位**（SYSU、LEVIR 的建筑小目标），不在 CDD/WHU 的大面积伪变化。
依据：F04 显示 SYSU small **pooled Recall 仅 0.1974**（漏检主导）；Run6 显示 FP 判别信息已在 head
（⇒ 判别侧无空间）；F10 显示 32% 的 FP 是贴边碎片（更准的边缘有助于减少碎片）。

**风险**

| ID | 风险 | 应对 |
|---|---|---|
| R1 | 判据仍是"信息已在 head"（Run6 的教训） | 本机制**改变表征**而非重读特征；T5 强制验证梯度进高频分支 |
| R2 | R8（EdgeRefNet, TGRS 2026）已占"边缘引导细化+边缘监督" | 定位为 **STR 内部增强/消融**，不得抬成独立第三贡献；投稿前须重新核验 R8 方法细节 |
| R3 | λ 是新超参，易被质疑"调参" | **λ=0.2 预注册固定，禁止搜索**；论文写明未搜索 |
| R4 | 边缘目标在很小对象上可能稀薄 | `adaptive_max_pool2d` 保底 + `[HFAE-LOSS]` 监控 |
| R5 | 1/4 尺度参数量/MAC 上升导致不公平 | deploy 门与 fvcore 实测并列报告；`aux_only` 形态零推理代价 |
| R6 | 四库迁移性不一致 | 逐库报告，不平均掩盖；负面结论照实记录 |

---

## §6 冻结承诺

本文档的机制、两形态、λ=0.2、四库 80K 预算、判据与接受标准在**见结果之前**固定；
结果出来后只能**另附**结果报告，不得事后修改上述任何一项。
Run1–Run6 产物、`models/` 既有算子（`ss2d.py`/`caacp_ss2d.py`/`str_dcr.py`/`str_reparam.py`）
一律不动；新代码只新增 `models/model/str_hf_edge.py` 并在
`casa_tvim_str_net.py` / `train.py` / `eval.py` / `run7_dry_run.py` / `test_run7_hf_edge.py` 中接线。

---

## §7 实施澄清与逐条对拍（**追加于见结果之前**，2026-10-11 10:20）

> **性质声明**：本节是**追加**的，§1–§6 一字未改（§6 要求冻结项不得事后修改）。
> 本节**不新增判据、不放宽任何阈值**，只把 §3/§4 的散文式表述与**机器可执行的裁决代码**
> 对齐，并如实披露一处"代码比散文**更严**"的地方。所有阈值仍以
> `train_scripts/CASA-TViM/Run7/SELECTION_LOCK.json` 的
> `acceptance_criteria_frozen_numeric` 为准（唯一规范形式）。

### 7.1 冻结时序（可核验）

- `SELECTION_LOCK.json` 的 `_frozen_at = 2026-10-11T09:35:00+08:00`。
- 09:27 启动的是**同期对照** `M1_R7CTRL`（波一）；**没有任何 E9/E10 run 存在**。
- 冻结时**全部 12 个 run 都还没有任何完整 TEST 区块**（TEST 区块只在 80,000 步收尾时写出；
  实测到 10:00 波一仍是 `NO_COMPLETE_TEST_BLOCK`）⇒ 冻结时**无从观察到任何结果**。
- 因此"判据在任何 E9/E10 结果之前固定"这一条**成立且可复核**。

### 7.2 规范阈值（逐字取自 SELECTION_LOCK.json）

| 键 | 值 | 对应 §4 的哪一句 |
|---|---|---|
| `min_datasets_ge_ctrl` | 3 | 「≥3/4 库 F1 ≥ CTRL」 |
| `f1_tie_tolerance` | 0.0 | 「F1 ≥ CTRL」为**严格** ≥（无容差） |
| `deploy_param_limit` | 5,000,000 | §4 机制自检 ④ |
| `grad_norm_min` | 0.0 | §4 机制自检 ①（要求 `> 0`） |
| `guardrail_recall_drop_max` | 0.02 | 「中/大对象 pooled Recall 不显著回退」（沿用 Run6 §7.3 已冻结先例） |
| `guardrail_objrecall_drop_max` | 0.02 | ⚠️ **见 7.3**：散文只写了 pooled Recall，代码另压了目标级 ObjRecall |
| `guardrail_f1_datasets` | `[CDD-CD-256, WHU-CD-256]` | 「CDD/WHU 不得明显恶化」 |
| `guardrail_f1_drop_max` | 0.002 | 同上（0.20pp 相对底线） |
| `verdict_labels` | MECHANISM-PASS / MECHANISM-FAIL / GUARDRAIL-FAIL / INVALID / INCOMPLETE | 裁决取值域 |

### 7.3 唯一一处「代码比散文更严」的披露

§4 散文的守门只写了「中/大对象 pooled Recall 不显著回退；CDD/WHU 不得明显恶化」，
而代码另加了 **`ObjRecall_loose` 回退 > 0.02 即触发守门**（目标级，loose 口径）。
**这是额外从严、不是放宽**，且与其余阈值同时冻结（09:35，见结果之前），
故不违反预注册精神；此处如实披露以免评审时被当作"事后加码"。
若课题负责人认为应当严格按散文执行，正确做法是**另开一次编号实验**，而不是改本节或改锁。

### 7.4 §4 判据 → 执行位置的逐条映射（已实测）

| §4 判据 | 落在哪里 | 状态 |
|---|---|---|
| 主：四库 F1 vs 同期 CTRL | `run7_verdict.py::decide` `primary_ok`（`n_ge ≥ 3` **且** SYSU **严格**提升 **且** 机制自检全过） | ✅ 与散文逐字一致（已读码核对） |
| 次：SYSU small pooled Recall / Hit@25 / 未匹配小预测 / ObjP / ObjR | `run7_verdict.py::decide` 的 `detail["small_object_report"]` + 三类结局的 `attribution` + `verdict.md` 小目标表 | ✅ 2026-10-11 补齐（此前只被要求"存在"、从未输出；见 Run7 README §7.7） |
| 守门：中/大 pooled Recall、CDD/WHU F1 | `guardrail_ok()`；缺失对象级指标时判 `INCOMPLETE`（**不假 PASS**） | ✅ |
| 机制自检 ① `[HFAE-GRAD]` 首步 `hf_tap_grad_norm > 0` | `mech[...]["grad_reaches_hf_branch"]` + `check_run.py` 的 `[HFAE-GRAD]` 断言 | ✅ |
| 机制自检 ② epoch-0 `torch.equal(CTRL, EXP)` | **T2**（`test_run7_hf_edge.py`，1298 共有键 0 处不一致）+ **dry-run 门**（三形态均 true）；由 `fork_rng` + 注入零初始化在**构造期**保证 | ✅ 属结构性性质，**不从日志逐 run 复推**（如实说明） |
| 机制自检 ③ 真实 batch 折叠 disagreement = 0 | 训练收尾 `[REPARAM-REAL-ARGMAX-DISAGREE]`，由 `run7_verdict`（`fold_disagree_zero`）、`check_run.py`、`independent_audit.py` 三处独立检查 | ✅ |
| 机制自检 ④ deploy ≤ 5M | `deploy_within_limit` + `check_run.py` + 锁的 `deploy_param_limit` | ✅ |
| 机制自检 ⑤ `[HFAE-LOSS]` 逐 epoch 下降 | `aux_loss_present_and_decreasing`（`len(aux) ≥ 2` 且 `aux[-1] ≤ aux[0]`） | ✅ |
| 接受标准 | `winners = [v for v in EXPERIMENTS if primary_ok]`；取第一个；无 winner ⇒ `MECHANISM-FAIL` + 归因 | ✅ |
| 收口链 | `finish_run7.sh`（6 步）→ 12 run 已 DRY 排练 `FINISH_RC=0`（Run7 README §7.9） | ✅ |

### 7.5 §6 不改动既有算子的核验（实测）

`git log` 显示本节写作时（见结果之前）Run7 期间的模型侧改动**恰好**是 §6 承诺的范围：

- **新增**：`models/model/str_hf_edge.py`、`models/run7_dry_run.py`、`models/test_run7_hf_edge.py`；
- **接线修改**：`models/train.py`、`models/eval.py`、`models/model/casa_tvim_str_net.py`；
- **未改动**（最近一次提交均在 Run7 之前）：`layers/ss2d.py`(2026-10-03)、
  `layers/caacp_ss2d.py`(2026-10-05)、`str_dcr.py`/`str_reparam.py`(2026-10-02)、
  `tinyvim_s_slim.py`(2026-10-05)、`str_fine_tap.py`(2026-10-10)。

### 7.6 折叠门「刀锋像素」澄清（⚠️ **见结果之后**作出，须在论文中显式披露）

**触发事实**：波一对照臂 `M1_R7CTRL/CDD-CD-256` 的**真实数据 batch** 折叠检查给出
`[REPARAM-REAL-ARGMAX-DISAGREE] 9.537e-07`，而 §4 机制自检 ③ 与 §7.4 记的是
「真实 batch disagreement = **0**」⇒ 严格读法下该 run 判 FAIL，收口链会 `exit 4` **不产出裁决**。

**定位证据**（`tools/archive/run7_fold_knife_edge.py`，用**真实 best ckpt** 复现 train.py 收尾的同一 batch）：

| 项 | 值 |
|---|---|
| 翻转像素 | 恰 **1 / 1,048,576**（16×256×256） |
| 该像素 `p_train` / `p_deploy` | 0.49999690 / 0.50000036 |
| `\|p_train − 0.5\|` | **3.099e-06** |
| 该次折叠最大绝对误差 | **6.199e-06** |
| 全批处于误差带（`\|p−0.5\| < 6.199e-06`）内的像素 | **2 / 1,048,576** |
| 同源性自检 | `max_abs` 与日志相对差 **1.9e-05**、`disagree` 相对差 **2.7e-05**（日志只印 4 位有效数字）⇒ **同一 batch** |
| 判定（与 Run4 §10 同定义） | **KNIFE_EDGE_ARTIFACT** |

**澄清内容（替代 §4③ 在该情形下的字面读法）**

> 真实数据 batch 的折叠门 = **二值化逐像素一致**；**唯一例外**是该像素的
> `|p_train − 0.5|` **小于该次折叠自身的最大绝对误差** —— 此时该像素落在阈值的**数值误差带之内**，
> 二值化在该点上**本就不适定**，属数值现象而非折叠实现错误。
> 该例外**必须**附可定位证据文件 `KNIFE_EDGE_<dataset>_<variant>.json`
> （`verdict=KNIFE_EDGE_ARTIFACT`、`n_flipped>0`，且 `disagree`/`max_abs_diff` 与日志吻合）；
> **证据缺失或数值不吻合 ⇒ 一律判 FAIL**（三处独立实现均为 fail-closed）。

**与 Run4 先例的关系**：Run4 §10 已就**合成随机 batch** 作出同一定义，但当时把**真实 batch**
保留为严格 `== 0`（Run4 的真实 batch 恰好无翻转，此情形从未被实际决定）。本节是把该判据
**显式扩展到真实 batch**——因为真实 batch 同样是"部署分布上的固定 16 张"，其不适定性来自同一个阈值问题。

**诚实披露（必须写进论文）**

1. 本节是在**看到波一（对照臂）结果之后**作出的——因为它是由真实运行**触发**的，事先无从预知；
2. 它**不依赖任何性能指标**（不是按 F1 放宽），且**对称适用于全部 12 个 run**（含 E9/E10），
   不偏袒任何变体；CTRL 与实验组用**同一条**规则；
3. 严格读法的代价是：因 1 个像素在离阈值 3.1e-06 处翻转而**判整个 run 无效**；
4. 判据、阈值、λ、seed **一概未改**；本节只澄清"数值不适定像素"的处理，
   §4 的**接受标准与守门阈值**（`SELECTION_LOCK.json`）**完全不动**。

**已实现的三处独立校验**（互不 import，各自实现）：
`check_run.py`（逐 run 硬门）· `run7_verdict.py`（`fold_ok_knife`，裁决机制自检）·
`independent_audit.py`（A6，独立复算）。三处均在"无证据/不吻合"时判 FAIL。

### 7.7 性能缺陷（hook 注册非幂等）与其处置：整轮重跑（⚠️ 须在论文中显式披露）

**触发事实**：波二（`E9_HFAE_A`，`hf_edge=1`）自 13:35 起跑，20:31 复核时四个库的单 epoch 耗时
均**随训练推进单调上升**（CDD 65 s→282 s、SYSU 85 s→350 s、LEVIR 45 s→200 s、WHU 35 s→126 s），
而波一 CTRL 全程平坦（CDD 40~240 epoch 恒在 56~63 s）。到封存时刻波二仅完成 65.7K/66.7K/63.4K
步，预计剩余时间被拖长 4~5 倍。

**根因（已定位到行，非服务器/GPU 问题）**：`models/model/casa_tvim_str_net.py::_ensure_hf_hooks()`
的幂等判据查错了对象——

```python
attached = any(getattr(h, "_hfae_key", None) == key
               for h in getattr(mod, "_forward_hooks", {}).values())   # ← dict 的 value 是 hook 函数
h = mod.register_forward_hook(...); h._hfae_key = key                 # ← 标记却打在 RemovableHandle 上
```

`nn.Module._forward_hooks` 是 `{handle_id: hook_fn}`，而 `register_forward_hook()` 返回的
`RemovableHandle` 是**另一个对象**、不在该 dict 里（服务端实测：`_forward_hooks` 的 value 类型为
`function`，其上无 `_hfae_key`）。于是 `attached` **恒为 False**，每次 `forward()` 都重复注册
1 个 hook/模块（实测 +2/次，永不停止）。

**它为什么会线性拖慢**：单个 hook 的成本 ≈ **2.0 μs/forward**（用真实模型人为挂 10k→30k 个
hook 实测：2.02 / 1.97 μs）。每个 epoch 的 forward 数 `F = 训练步数 + 全测试集验证 forward 数`
（`val` 每 epoch 都跑），因此 epoch 耗时对 epoch 数的斜率 = `2cF²`。用**独立测得的 c** 预测四个库，
无自由参数：

| 数据集 | train/test | F（forward/epoch） | 预测斜率 `2cF²` | **实测斜率** | 比 |
|---|---|---|---|---|---|
| CDD | 10000/3000 | 313+188 = 501 | 1.00 s/epoch | **1.10** | 1.10 |
| LEVIR | 7120/2048 | 223+128 = 351 | 0.49 | **0.55** | 1.12 |
| SYSU | 12000/4000 | 375+250 = 625 | 1.56 | **1.72** | 1.10 |
| WHU | 5947/744 | 186+47 = 233 | 0.22 | **0.25** | 1.15 |

并预测 CDD epoch 171 的耗时 = **247 s**，实测 **245 s**。四库一致（隐含 c ≈ 2.2 μs）⇒ 该缺陷
**独自**解释了全部劣化，不存在第二个未知成因。

**纯性能缺陷的证据（关键）**：重复注册的 hook 全部执行 `self._hf_taps[key] = o`，写入的是
**同一个 key、同一个 tensor 对象**，字典赋值幂等 ⇒ 下游 tap 取值逐位不变。新增回归测试
**T12(c)** 直接证明：同一权重、同一输入下，人为加入 40 个重复 hook 后的输出与 aux logits
与加之前**逐位相同**（`max|d_out| = max|d_logits| = 0.00e+00`）。
`hf_edge=0` 的 CTRL 更是在该函数**首行即 return**，属不可达代码 ⇒ **波一 CTRL 与修复版逐位同一**。

**修复内容（最小改动，仅 1 个身份文件 + 1 个测试文件）**：

1. 标记改打在 **hook 函数**上（`fn._hfae_key = key`），幂等判据与 `_forward_hooks` 的 value 同类型；
2. 新增 **fail-closed 不变式**：同一 tap 上属于本 key 的 hook > 1 立即 `RuntimeError`
   （该缺陷曾**静默**运行 7 小时，新判据使其在第 2 次 forward 就炸）；判据对 `deepcopy` 路径安全；
3. 新增诊断访问器 `hf_edge_hook_counts()`（使"幂等性"成为可断言、可回归的量）；
4. 新增 **T12** 三向回归：连续 25 次 forward 后 hook 数恒为 1/模块、handles 恒定；人为重复注册
   必被拒；（c）数值中性如上。

**处置（用户裁定）**：波一/波二全部产物**原样封存不删除**，修复后**整轮 12 个 run 从零重跑**，
以保持 `independent_audit.py` A8「12 run 代码身份唯一」。封存目录后缀
`_SUPERSEDED_20261011_2324_prehookfix`（含 `_SUPERSEDED_README_*.txt` 记述封存时各 run 步数与实测曲线）。
新身份：`models/model/casa_tvim_str_net.py = e125e7ac622bff0b…`（`SOURCE_SHA256.txt` 27/27 与服务器
逐字节一致）；`T0–T12 = 13 checks / 0 FAIL`；`PROTOCOL_VERSION`、`SELECTION_LOCK.json`、
λ、seed、batch、80K 步数**一字未改**。重跑起点：**2026-10-11 23:27**（波一启动）。

**诚实披露（必须写进论文）**

1. 本节是在**看到波一 CTRL 完整结果、以及波二 E9 的部分中途曲线之后**才作出的——缺陷由慢速
   **触发**，事先无从预知；
2. 但重启的**唯一**依据是**与任何精度指标无关的墙钟缺陷**：一条用独立实测常数 `c`、
   无自由参数地预测四库斜率（误差 10~15%）的定量定律；且 §4 的接受标准与守门阈值
   （`SELECTION_LOCK.json`）**完全未动** ⇒ 重启**在逻辑上不可能**改变接受与否的判定规则；
3. 被丢弃的中途数字（含"E9 在同阶段四库均 ≥ CTRL"的观察）**不进入**任何结论、表格或裁决；
   新 campaign 从零产生自己的全部数字；
4. 代价与影响面诚实列出：波一 4 个已完整跑完的 CTRL run（各 80K 步）被作废重跑；
   波二 E9 丢弃 65,730 / 66,677 / 63,375 / 80,000 步四个 run（全部封存可追溯）；
5. 该缺陷同样是无界**内存泄漏**（80K 步会累积约 20 万个 hook 与约 40 万个 handle 元组），
   修复本身就是必要的工程正确性修正，而非为提速而做的取舍。

**唯一被改动的身份文件**：`models/model/casa_tvim_str_net.py`（缺陷所在文件本身）。
其余 8 个 `SOURCE_IDENTITY_FILES` 与全部判据/阈值文件字节未变。

