# CASA-CD × STR-RepNet 融合：STR-Fusion Run1 设计与预注册方案

> 日期：2026-10（新对话第一轮交付稿，**未经用户确认前不改任何代码**）
> 位置：本文件存于 CASA-CD 工作区 `docs/temporary/`，作为融合主线的预注册文档。
> 一句话：把 CASA-CD 的 **ChangeViT-T（冻结）+ CASAA token 思想** 与 STR-RepNet 的
> **TAR（时空代数重参数化 bridge）+ DCR（解码器全程组合重参数化）** 拼成新模型，
> 在 CASA-CD 工作区、按 CASA-CD 协议（80K / BCE+Dice / seed 16 / GPU1）实现与实验。
>
> 硬约束（全部必须同时满足）：
> - 四数据集 test F1：SYSU ≥ 85 / LEVIR ≥ 92.5 / WHU ≥ 95 / CDD ≥ 98；
> - 轻量化（CASA-CD 口径）：**有效推理参数 < 3M**（工程目标 ≤ 2.20M；与 STR 侧
>   28.83M 锚点是两套口径，本工作以 <3M 为准）；
> - 创新必须是**结构重参数化本身**；禁止 loss 调参 / 增广 / threshold tuning /
>   多 seed 包装成创新；
> - 预注册 gate + 失败停止规则；任何新结构候选先过零训练 audit 再花正式训练预算。

---

## 0. 结论先行

1. **多尺度特征接口（设计问题 1）**：采用「深度即尺度」token 金字塔
   （Depth-as-Scale token pyramid）。冻结 ViT4 的四个 block 状态
   `B1→64×64、B2→32×32、B3→16×16、B4→8×8`（固定无参数重采样）构成 TAR/DCR
   所需的四尺度二时相输入。**唯一不依赖被否决路线的可行接口**（P0 重建 / B4-only /
   O-PRE 全部已被 R6/R8/R9-D0 证伪），且它与 R8-D0 的关系必须由预注册的
   SF-D0 零训练 gate 正面裁决（见 §6），**gate 不过不训练**。
2. **预算（设计问题 2）**：ViT4 冻结 1.977M + TAR/DCR/head 部署约 0.620M ≈
   **2.596M < 3M**（估算，机器审计为准）；对照 baseline ChangeViT-T 11.754M /
   26.32G，融合部署 FLOPs 估算 ≈ 2.4–2.6G（约 −90%）。工程目标 ≤2.20M 为**预注册
   单值旋钮**（D 由机器审计一次确定，非 sweep）。
3. **训练协议（设计问题 3）**：完全沿用 CASA-CD trainer 与协议（BCE+Dice、
   Adam(2e-4, 0.9/0.99, wd=1e-4)、poly 0.9 + 200 warmup、80000 steps、batch 16、
   256×256、seed 16、test-as-val、GPU1 单卡）。STR 的 CE+2×Lovász/300 epoch 仅为
   历史参考。折叠等价性按 STR `test_reparam_equivalence.py` 范式移植为
   T0/T1/T2（FP64 组合、一次 FP32 cast、部署二值化 disagreement = 0）。
4. **不破坏既有资产（设计问题 4）**：Run1–Run9 代码/checkpoint/log 一律不动；
   新代码全部放新文件（`str_*.py`），`train.py`/`eval.py`/`smoke_test.py` 只做
   加性分支（旧路径逐字节语义不变）；新主线目录 `train_scripts/STR-Fusion/Run1/`。
5. **预注册实验（设计问题 5）**：唯一变量 = 训练期结构重参数化（C0 plain 单路径
   vs M1 全 rep，**部署函数类完全相同**）；SF-D0 零训练 gate 先行；SYSU 决策实验
   （C0+M1 两个 80K）→ 仅 PASS 后 LEVIR M1 → WHU M1；CDD M1 为第 5 个（超预期
   预算，需用户批准）。判据数字全部预注册（§7），失败即停不救。

---

## 1. 证据表（两仓库正 / 负资产 + 四数据集 gap）

### 1.1 STR-RepNet 侧资产（机制来源）

| 类别 | 事实 | 对本融合的含义 |
|---|---|---|
| **正：折叠原语** | TAR（concat+sum+signed-diff 三路 → 单 1×1）、DCR（RepDW3 / RepPW1x1 / RepPairFuse1x1 全部可折叠）、每线性分支独立 BN + 可折叠 α 残差（BN-FR） | 直接移植原语；这是「结构重参数化本身」创新的物质基础 |
| **正：折叠纪律** | FP64 组合全程、最后一次性 cast FP32；九轮实测部署 argmax disagreement = 0；训练期独有分支零初始化（BN γ=β=0），epoch-0 输出与主路径逐位一致 | 移植为 T0/T1/T2 + smoke 硬门槛（§5/§6） |
| **正：规模事实** | STR D=160 中间+Decoder ≈ 0.84M 参数 / 0.92G FLOPs（encoder dims 96/192/384/768）；本融合全部输入 192 维，同 D 下更小（估算 0.62M，§3） | 预算可行性依据 |
| **正：BN-FR 与 DCR 有效性** | BN-FR 零部署增量换 LEVIR +0.61 / SYSU +1.28（Run2 vs Run1）；DCR 四数据集稳定 Full > Plain | DCR 复现于新 encoder 的机制动机 |
| **负：八轮参数化全失败** | Run3–Run10（Edge-Basis / IBAS / BOTR / NSCR / PBRU / MPCR / BiFTR / PFDR）全部 FAIL/WEAK，共同签名「置信度锐化」（Recall↓/Precision↑）；分支均学到非零结构但无净益 | 移植时**不**把「训练分支越花哨越好」当默认假设；M1 的分支集 = Run2 原版三路（有可证伪动机），不加任何 Run3–Run10 机制 |
| **负：decoder 小目标保真** | Run6 诊断：LEVIR small ≤502px pixel recall 80.8%、24% 完全漏检；D0 full-encoder 不修复 → 瓶颈在 decoder 侧 | 融合必须正面回答「细节/边界从哪来」（§2.4） |
| **负：旧 baseline 从未被反超** | HAM-CD LEVIR 92.11，STR 最好 91.44（Run2 last2）；SYSU 83.45（VMamba-Tiny, 300ep, CE+Lovász, seed 2333） | STR 数字**不能**当 CASA 锚点（协议/encoder 均不同），仅作「该解码器家族可训练性」参考 |

### 1.2 CASA-CD 侧资产（当前工作区 / 协议 / 纪律）

| 类别 | 事实 | 对本融合的含义 |
|---|---|---|
| **正：ChangeViT-T 冻结框架** | corrected DeiT loader（patch 原位继承、pos 14×14→16×16 bicubic、深度截断后 blocks.N+ 不存在）；**ViT 必须冻结**（官方协议会把 ViT 训成精确零权重，吸收态不可恢复） | 融合 ViT4 一律冻结 + 校验 checksum；复用现有 loader，零改动 encoder.py |
| **正：健康参考与轻量锚点** | 健康 full12 frozen SYSU **83.14**（11.754M，超预算，参考上限）；ViT4 + ResNet + 旧头 **82.77**（8.195M，最强已验证轻量结构）；LEVIR frozen A1 **91.84** | 四数据集 gap 的基准（§1.4） |
| **正：CASAA token 思想** | A1（冻结 ViT + K=64 content 聚类压缩 K/V，零新增参数）基本无损（LEVIR −0.01 / SYSU +0.02） | 「token 计算非均匀性」是融合叙事的 token 侧支撑；A1 式 content 压缩作为**可选后续机制落点**（M2，需另行预注册），Run1 不动它 |
| **正：四数据集 token depth 曲线** | B4 稳健最优、B12 全面劣化、最优 depth dataset-dependent；FDAM/ViT-CoMer/ResCLIP 文献支撑「浅层多细节、深层语义、inner-patch 受限」 | 多尺度接口取 B1–B4 四个深度状态的实证依据 |
| **正：零训练 raw gate 纪律** | Run4–Run9 七轮负结果、正式 80K 总消耗始终只有 1 个；run7/run8 audit 工具链 + SYSU R4-1 对照自检（0.6535/0.5948 ±0.005）可直接复用 | SF-D0 gate 复用 run8 审计的 `boundary_band`/`pixel_pr_auc`/`rank_normalize_np` 与对照自检 |
| **负：CASAA router 主线终止** | Top32 0.39→0.63 未转化为 F1（A4-D −0.12）；oracle 上限 +1.48 不可部署 | **不得未重新预注册复活 router 主线**；Run1 完全不碰 routing |
| **负：detail-free 重建七连败** | R4 Light/PSD 三连败；R5 Mobile raw gate FAIL；R6 P0 重建 FAIL（P0 PR 0.3711）；R7 B2 截断 FAIL；R8 B4-only dense FAIL（边界带 lift 3/4 数据集 <0.02）；R9 O-PRE FAIL（G0/G1 全 0/4） | **R8/R9 postmortem 结论**：冻结 plain ViT 的 token 表征无法靠「重采样/重建」补回 dense 边界；可行路径需要**训练过的、任务适配的空间表征**。融合设计的正面回应见 §2.4 |
| **负：ViT 训练崩溃** | 统一 lr=2e-4 训练 DeiT → 零权重吸收态 | 融合 ViT 冻结是唯一选项，不设 `encoder_train` 旋钮 |

### 1.3 STR vs CASA 协议差异（以 CASA 为准）

| 项 | STR（历史参考） | CASA（本工作采用） |
|---|---|---|
| 损失 | CE + 2×Lovász | **BCE + Dice** |
| 优化器 | — | **Adam(2e-4, 0.9/0.99, wd=1e-4)** |
| 调度 / 预算 | 300 epoch | **poly(0.9)+200 warmup，80000 steps** |
| batch / 输入 / seed | 16 / 256×256 / 2333 | **16 / 256×256 / seed 16** |
| 验证协议 | test-as-val | **test-as-val**（相同） |
| GPU | GPU0（STR 仓库记载） | **GPU1 单卡**（CASA 交接文档 + 服务器 nvidia-smi 实况为准，不动别人卡） |
| 结果口径 | 最后一个完整 TEST RESULTS 区块 | **相同** + arch.json sidecar 校验 |

> 融合模型不是 swap-symmetric（TAR concat 主分支与时相顺序有关，与 ChangeViT
> baseline、STR 全系一致）；CASA Run7–9 头部的「严格时间交换对称」是那些头的设计
> 选择而非项目级铁律。Run1 在 smoke 中**测量** |pred(A,B)−pred(B,A)| 做记录
> （不设 gate），论文阶段如实声明。

### 1.4 四数据集 gap（CASA 口径锚点）

| 数据集 | Train/Val/Test | 当前最佳验证 F1（结构，参数） | 硬目标 | 缺口 |
|---|---|---:|---:|---:|
| SYSU | 12,000/4,000/4,000 | 83.14（full12 冻结，11.75M）/ 82.77（ViT4+ResNet+旧头，8.20M） | **85.0** | −1.86 / −2.23 |
| LEVIR | 7,120/1,024/2,048 | 91.95（trained baseline，死 ViT）/ 91.84（frozen A1） | **92.5** | −0.55 / −0.66 |
| WHU | 5,947/743/744 | 94.84（baseline） | **95.0** | −0.16 |
| CDD | 10,000/2,998/3,000 | 97.75（baseline） | **98.0** | −0.25 |

> 决策数据集 = SYSU（缺口最大，先行）；STR 侧锚点（Run2：CDD 98.42 / WHU 95.14 /
> LEVIR 91.44 / SYSU 83.45）协议与 encoder 不同，不混入 CASA 判据。

---

## 2. 多尺度特征接口（设计问题 1 的最终选择与逐层定义）

### 2.1 候选与排除（事实，不拍脑袋）

| 候选 | 排除依据（代码/日志事实） |
|---|---|
| (a) ViT 中间 block token 重排 + 上采样组金字塔 | **选定**（见 2.2） |
| (b) patch 重建（P0/patchembed 反投影） | R6-D0：P0 token PR 0.3711 < 0.44 门槛（SYSU）→ H6-A 证伪；R8-D0 M4 unembed 亦弱 |
| (c) 单尺度 16×16 语义 + head 侧重建 | 即 R8-D0 B4-SPE 路线：边界带 lift 3/4 数据集 <0.02 → 已被预注册否决 |
| (d) O-PRE 重叠采样格变密 | R9-D0：G0=0/4、G1=0/4 → H9 证伪 |
| (e) 另加轻量 detail 支路（ResNet/LightDetail/Mobile） | R4/R5 负结果链 + ResNet ≤layer3 ≈ 2.78M 直接超 <3M 预算；且违反「encoder = ChangeViT」融合边界 |

### 2.2 最终选择：「深度即尺度」token 金字塔（Depth-as-Scale）

冻结 ViT4（256×256 输入，patch 16 → 16×16 token，192 维，blocks 0–3）。
用 `DinoVisionTransformer.get_intermediate_layers(n=[1,2,3,4], reshape=True, norm=True)`
取各 block 输出过 final-LN 的状态（与 R7-D0 `token_sources` 口径一致），双时相
各自前向一次（冻结、no_grad），得到 (B,192,16,16) 的 B1/B2/B3/B4，再做**固定
无参数重采样**：

| 尺度（STR 命名） | 来源 ViT 状态 | 取法 | 空间分辨率 | 通道 | 新增参数 |
|---|---|---|---|---|---|
| t1（最细，接 fuse1） | **B1**（blocks[0] 输出） | bilinear ×4 | 64×64 | 192 | 0 |
| t2（接 fuse2） | **B2**（blocks[1] 输出） | bilinear ×2 | 32×32 | 192 | 0 |
| t3（接 fuse3） | **B3**（blocks[2] 输出） | 原样 | 16×16 | 192 | 0 |
| t4（最粗） | **B4**（blocks[3] 输出，ViT4 终 token） | avgpool 2×2 | 8×8 | 192 | 0 |

- **为什么 t4 用 B4**：R7-D0 四数据集 B4 稳健最优（变化敏感语义源）。
- **为什么 t1 用 B1 而不是 P0**：P0 变化判别力最弱（R6-D0），B1 保留一个 block
  混合后的浅层纹理+空间状态；FDAM/ResCLIP 文献证据支持浅层空间定位更强。
- **逐尺度通道对齐**：全部 192 维 → TAR 的 `TemporalRep1x1(2×192→160)` 统一投影
  到 D（部署折叠为单个 1×1）。四尺度重采样本身 0 参数，投影参数计入 TAR 预算。
- **Siamese**：pre/post 各自前向冻结 ViT4 → 各自金字塔 → TAR 二时相 bridge。

### 2.3 与 R8-D0「ranking ≠ dense 重建」证据的关系（必须正面回答）

R8-D0 证伪的是「**无训练**地把 16×16 语义 token bilinear 到像素域做重建」；
R8/R9 postmortem 明确写出可行路径需要「**训练过的、任务适配的空间表征**」。
本方案的结构性回应分两层：

1. **训练过的任务适配空间表征 = TAR+DCR 本身（约 0.62M 可训练部署参数）**：
   R8-D0 的失败图（bilinear + 固定阈值）没有任何可学习空间算子；本融合的
   bridge+decoder 是 STR 九轮折叠纪律验证过的可训练卷积解码器，在 64/32/16/8
   四个尺度做 DW/PW 局部建模 + 跨尺度 concat/sum/diff 融合，负责把多深度 token
   证据重建成边界锐利的 256×256 概率图。
2. **但「多深度 token 证据是否真的非冗余、真的比 B4-only 多携带边界带证据」是
   一个可证伪、且训练前就能测的假设** —— 这就是 SF-D0 gate（§6.1）：用 R8-D0
   同款边界带 PR-AUC 度量测「B1–B4 融合分 vs B4-only」的零训练 lift。gate 不过，
   说明多深度接口同样继承 inner-patch 局限，**不花任何 80K**。

> 诚实风险声明（写入预注册）：如果 SF-D0 gate FAIL，本融合主线按规则终止；
> 届时任何补救（O-PRE 作侧向源 / 加训练 stem / 放弃融合）都必须**重新预注册**，
> 不得在本轮方案内自动执行。

---

## 3. 参数 / FLOPs 预算表（设计问题 2；估算 → 机器审计为准）

口径（CASA）：EFFECTIVE = forward 图内全部参数（冻结照计、死参数不计）；
FLOPs 输入 2×3×256×256，fvcore + unsupported_ops 同时上报。

### 3.1 部署图（deploy，D=160，估算）

| 组件 | 计算 | 参数 |
|---|---:|---:|
| ViT4-192（冻结） | 已核实（B4SPEHead 文档口径） | 1,976,832 |
| TAR 时间投影 ×4 | 4 × (2·192·160 + 160) | 246,400 |
| TAR 局部块 ×4 | 4 × (DW3 160·9+160 + PW 160·160+160) | 109,440 |
| DCR fuse ×3 | 3 × (320·160 + 160) | 154,080 |
| DCR block ×4 | 4 × 27,360 | 109,440 |
| head Conv(160→1,1) | 160 + 1 | 161 |
| **合计 deploy EFFECTIVE** | | **≈ 2,596,353（2.596M）< 3.0M ✓** |

- 部署 FLOPs 估算：ViT4 双时相 ≈ 1.4G（DeiT-Tiny 12b@224=1.26G × (256/196)² ×
  (4/12) × 2）＋ TAR/DCR ≈ 0.9G（STR 口径同规模）＋ 重采样/头少量 ≈ **2.4–2.6G**。
  对比 baseline 26.32G 约 −90%；**以机器审计实测为准，论文不引估算值**。
- **机器审计实测（2026-10，run1_strfusion_budget.py，GPU1）**：
  - deploy 图（C0 与 M1 完全相同）：TOTAL = EFFECTIVE = **2,596,353（2.596M）< 3M ✓**，
    TRAINABLE 0.6195M，FLOPs **2.2136G**（fvcore，unsupported=8，与 baseline
    同口径上报告）；手算 2,596,353 与机器逐位一致；
  - 训练图 M1（full）：TOTAL 3.1229M / TRAINABLE 1.1461M / FLOPs 3.0625G；
  - 训练图 C0（plain）：TOTAL 2.6001M / TRAINABLE 0.6232M / FLOPs 2.2266G。
- 训练图（M1）：主分支 + sum/diff/aux 分支 + 各分支 BN + α，TOTAL/TRAINABLE 由
  机器审计报告（实测见上；TRAINABLE 口径 = requires_grad）。
- **工程目标 ≤2.20M**：预注册**单值旋钮** D*（非 sweep）：若用户确立 ≤2.20M 为
  硬门槛，则 D* = 机器审计下 deploy EFFECTIVE ≤ 2.20M 的最大 D，一次性确定
  （粗略 D*≈95–100，纯估算不作数）。D 只允许因**预算门槛**而调整，**禁止**用
  D 扫描救 F1。
- ViT12（5.536M）+ DCR 必超 3M → 不采用；ViT2（1.087M）只有两个深度、不足以组
  四尺度金字塔 → 不采用。**ViT4 是唯一满足「四尺度 + <3M」的组合**。

### 3.2 预算审计工具

新增 `analyse/run1_strfusion_budget.py`：构建 C0/M1 的 train-graph 与 deploy-graph，
输出 TOTAL / TRAINABLE / EFFECTIVE（train 与 deploy 双口径）+ FLOPs(G) +
unsupported_ops；硬断言：
- deploy EFFECTIVE ≤ 3.0M（G4 硬 gate）；
- C0 与 M1 的 deploy Params **逐位相等**（同一函数类）；
- 若启用 ≤2.20M 工程门槛 → 另断言 ≤ 2.20M。

---

## 4. 折叠等价性方案（设计问题 4/3 的移植口径）

### 4.1 移植范围（只移植 Run2 世代原语）

| 新文件（CASA 侧） | 内容（从 STR 移植） | 明确不移植 |
|---|---|---|
| `models/model/str_reparam.py` | `fold_conv_bn`、`RepDW3`、`RepPW1x1`、`RepPairFuse1x1`、`switch_module_to_deploy` | NSCR / PBRUHead / MPCR / PFDR（Run6–10） |
| `models/model/str_tar.py` | `TemporalRep1x1`、`TARStage`、`MultiScaleTAR` | BOTR（Run5） |
| `models/model/str_dcr.py` | `RepLocalBlock`、`DCRDecoder`（Run2 路径） | NSCR / MPCR / PFDR 挂点 |

训练图（每算子）：
- `TemporalRep1x1`：concat（主）+ sum + signed-diff（Q−P）三路，各带独立 BN
  （aux 零初始化 γ=β=0）+ 无 α（保持 STR 原版）→ 部署单 1×1（2·192→160, bias）。
- `RepDW3`：DW3 + DW1×3 + DW3×1（各 BN，aux 零初始化）+ α·x（BN-FR）→ 单 DW3。
- `RepPW1x1`：PW 主 + 串行低秩（r=D/4）+ diag + α·x（BN-FR）→ 单 1×1。
- `RepPairFuse1x1`：concat（主）+ sum + diff（各 BN，aux 零初始化）+ α·L → 单 1×1。
- head：`Conv2d(D→1,1,bias)`（单通道概率输出，不折叠；CASA 输出契约 = sigmoid 后
  (B,1,256,256) ∈ [0,1]）。

### 4.2 折叠纪律（STR P0 口径）

- 组合全程 FP64，**最后一次性 cast FP32**；
- 训练期独有分支零初始化（BN γ=β=0）→ epoch-0 输出与主路径**逐位一致**；
- `switch_to_deploy()` 后 state_dict 不得再含任何 BN/aux 键（断言）；
- 部署后对固定输入二值化（0.5 阈值）disagreement = 0（单通道概率图的 argmax
  类比，即 STR 的 [REPARAM-ARGMAX-DISAGREE] 口径，硬性）。
- RNG 纪律（STR Run8/9/10 教训）：所有模块的**主分支先构造、aux 后构造**，C0
  （aux 全关）与 M1 共享同一构造顺序 → C0/M1 epoch-0 主路径权重逐位一致，由
  smoke 断言 max_diff = 0。

### 4.3 分层测试（移植 STR test_reparam_equivalence.py 范式，去 yacs 依赖）

新增 `models/test_str_reparam_equivalence.py`（独立脚本，固定配置）：

- **T0（FP64 纯代数）**：BN 折叠公式；concat+sum+diff → 单 1×1（W_P=W_cP+W_s−W_d、
  W_Q=W_cQ+W_s+W_d）；DW 1×3/3×1 pad 嵌入；PW 串行低秩+diag；PairFuse 跨尺度
  sum/diff。阈值：**目标 1e-12、上限 1e-10**（按实测量级校准并注释，事后不改）。
- **T1（block 级）**：各原语 train→deploy 折叠，FP32 **tol=2e-5**；分支扰动到
  训练后量级（conv std 0.05、BN γ std 0.5、running stats 随机化）保证折叠
  非平凡；另做 FP64 代数对拍（<1e-10）。
- **T2（全模型）**：C0/M1 整网 train 图 vs deploy 图，固定随机输入（seed 固定、
  TF32 off、cudnn deterministic）：记录 max_abs_error（**断言 <2e-4**，STR 口径）；
  二值化 disagreement 判定口径（训练前澄清）：**随机初始化下输出聚集在 0.5 阈值
  附近，任何合法的 1e-5 级折叠误差都可能翻转个别像素——这是初始化属性而非折叠
  缺陷**。T2 的硬 gate 因此是：翻转像素必须全部落在折叠误差带内（|y0−0.5| ≤
  2e-4）且翻转比例 ≤ 2e-4（原始值如实记录）；**训练后正式模型的
  `[REPARAM-ARGMAX-DISAGREE] == 0`（TEST RESULTS 区块）仍是操作硬 gate**，训练
  后概率图极化（STR 九轮实测 0），若出现 >0 即按 SF-1 硬条件失败处理。
  deploy Params/FLOPs 与预算审计一致；打印 deploy 模块图确认分支删除。
- **T2b（全模型 · 活分支，训练前补充）**：T2 在初始化态 aux γ=0（平凡折叠），
  T2b 把全部 rep 分支扰动到训练后量级（conv std 0.05、BN γ/β std 0.5、running
  stats 0.5–2.0 健康区间）再测整网折叠：**断言 <2e-4、disagreement ≤2e-4 且翻转
  全在误差带内**（实测 ~1e-6 量级、disagree=0）。smoke 的 3-step/batch-2 折叠
  读数因 BN running stats 退化（1/σ 放大 FP32 舍入）只作记录、不设 gate。

---

## 5. 逐文件修改清单（设计问题 4，全部绝对路径）

### 5.1 新增文件（核心代码，全部在 CASA-CD 工作区）

| 文件（绝对路径） | 内容 |
|---|---|
| `F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\str_reparam.py` | §4.1 折叠原语（STR reparam.py 的 Run2 子集） |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\str_tar.py` | TemporalRep1x1 / TARStage / MultiScaleTAR（encoder_dims 固定 (192,192,192,192)） |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\str_dcr.py` | RepLocalBlock / DCRDecoder（Run2 路径，nscr="none"） |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\str_encoder.py` | `STRViT4Encoder`：内部构造 `Encoder('tiny', vit_depth=4, detail_mode='none_b4', pretrained_path=...)`（**复用现有 corrected DeiT loader，零改动 encoder.py**），`forward(x)` 用 `vit.get_intermediate_layers(n=[1,2,3,4], reshape=True, norm=True)` 产出 [t4←B4↓2, t3←B3, t2←B2↑2, t1←B1↑4]（按 §2.2 固定重采样） |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\str_fusion.py` | `STRFusionNet(nn.Module)`：`__init__(pretrained_path, dim=160, rep_mode='plain'\|'full')`（C0=plain、M1=full）；encoder 冻结；tar→decoder→head；`forward(pre,post)` → sigmoid(B,1,256,256)；`train()` override 保持 encoder.eval()；`switch_to_deploy()` |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\models\test_str_reparam_equivalence.py` | §4.3 T0/T1/T2 |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\analyse\run1_strfusion_budget.py` | §3.2 预算审计（G4） |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\analyse\run1_strfusion_interface_audit.py` | §6.1 SF-D0 零训练接口 gate（复用 run8 审计的 boundary_band / pixel_pr_auc / rank_normalize_np 与 SYSU R4-1 对照自检） |

### 5.2 既有文件的最小加性改动（旧路径语义不变）

| 文件（绝对路径） | 改动（全部加性，不触碰旧分支行为） |
|---|---|
| `F:\Code_Repositories_2\CursorCode\CASA-CD\models\train.py` | ① 新 CLI：`--arch strfusion`、`--str_dim 160`、`--str_rep_mode plain\|full`；② `ChangeViTTrainer.__init__` 加 `arch=='strfusion'` 分支直接构建 `STRFusionNet`（断言 ViT 冻结）；③ `test_best()` 对 strfusion：加载 best → 固定 batch 记录 train-graph 输出 → `switch_to_deploy()` → 记录 `[REPARAM-MAX-ABS-ERROR]` / `[REPARAM-ARGMAX-DISAGREE]` / `[DEPLOY-PARAMS]` / `[DEPLOY-FLOPS]` / `[STR-REP-MODE]` / `[STR-DIM]` → 在 **deploy 图**上跑正式 test；④ arch.json 增加 `arch/vit_depth/str_dim/str_rep_mode` 键 |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\models\eval.py` | 加性分支：`--arch strfusion` 构建 + arch.json mismatch 拒绝 + 加载 best 后 `switch_to_deploy()` 再测（TEST 区块字段与 train 一致） |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\models\smoke_test.py` | 新增 `t_run1_strfusion(...)`（§6.2 清单）+ main 的 `--mode strfusion` 分发；既有 T0–T-R9 函数与分发不动 |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\.claude\_deploy.py` | **无需改动**（UPLOAD 已整树覆盖 models/ + train_scripts/ + analyse/） |

### 5.3 新训练脚本与文档

| 路径 | 内容 |
|---|---|
| `F:\Code_Repositories_2\CursorCode\CASA-CD\train_scripts\STR-Fusion\Run1\README.md` | 本方案摘要 + 判据 + 结果记录（启动时创建） |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\train_scripts\STR-Fusion\Run1\C0_Plain\{SYSU,LEVIR,WHU,CDD}\train_*.sh` | C0（`--arch strfusion --str_rep_mode plain`）四数据集脚本，按 gate 顺序启动 |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\train_scripts\STR-Fusion\Run1\M1_TAR-DCR\{SYSU,LEVIR,WHU,CDD}\train_*.sh` | M1（`--arch strfusion --str_rep_mode full`） |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\train_scripts\STR-Fusion\Run1\dryrun_SYSU.sh` | 60-step scratch dry run（模板同 Run4–Run9） |
| `F:\Code_Repositories_2\CursorCode\CASA-CD\docs\temporary\CASA-CD_STR融合_Run1_设计与预注册方案.md` | 本文件 |

> 服务器路径（沿用 CASA 惯例，不新建目录体系）：checkpoint
> `/share_datasets/yqwang/checkpoints/CASA-CD/STR-Fusion/Run1/<group>/<dataset>/`；
> 日志 `/home/yqwang/outputs/CASA-CD/STR-Fusion/Run1/<group>/<dataset>/train_log.txt`；
> 诊断 `/home/yqwang/outputs/CASA-CD/diagnostics/STR-Fusion/Run1/`。

---

## 6. 预注册 gate 与停止规则（设计问题 5/6）

### 6.1 SF-D0 零训练接口 gate（先于一切训练；四数据集）

复用 R8-D0 方法学：冻结 ViT4（corrected loader，checksum 逐位校验），完整 test 集
前向；每源 rank-normalize → 重采样 256×256；指标 = **边界带 PR-AUC**（R8-D0 的
M2 口径）。

| 代号 | 内容 | 阈值 |
|---|---|---|
| G0（审计有效性） | 冻结 ViT4 checksum 逐位一致；SYSU R4-1 ResNet 1/8 对照 PR 0.6535 / Top32 0.5948（±0.005）；B4-only 边界带 PR-AUC 复现 R8-D0 记录（±0.01） | 不过 → AUDIT-INVALID（修审计，不判机制） |
| G1（多深度非冗余） | PRbnd(fuse4) ≥ PRbnd(B4) + 0.02，fuse4 = mean(rank{B1,B2,B3,B4} 各重采样后) | **SYSU 必过** 且 ≥2/4 数据集 |
| G2（细尺度不毁语义） | PRbnd(fuse2) ≥ PRbnd(B4) − 0.02，fuse2 = mean(rank{B1,B2}) | ≥3/4 数据集 |
| G4（预算） | 机器审计 deploy EFFECTIVE ≤ 3.0M（§3） | 硬性（与 G1/G2 独立） |

**裁决**：G0/G4 不过 → 修复审计后重跑（不算机制失败）；G1 或 G2 不过 →
**[SF-D0-GATE] FAIL → 不启动任何 80K，融合主线按预注册终止**。补救方向
（O-PRE 侧向源 / 训练 stem / 放弃）需重新预注册，本轮禁止自动执行。
**SF-D0 PASS → 进入 SF-T。**

### 6.2 SF-T 训练前硬门槛（Phase 0，全过才允许 80K）

1. T0 纯 FP64 代数（§4.3 阈值，实测校准、事后不改）；
2. T1 block 折叠 FP32 < 2e-5 + FP64 对拍 + deploy 无分支断言；
3. T2 全模型 < 2e-4 + 二值化 disagreement（翻转全在误差带内且比例 ≤2e-4；
   训练后正式模型 [REPARAM-ARGMAX-DISAGREE] = 0 为操作硬 gate）+ deploy 预算断言；
4. smoke（`t_run1_strfusion`）：
   - C0/M1 epoch-0 输出**逐位一致**（max_diff = 0.0）；
   - aux 分支 epoch-0 输出严格为 0；aux BN γ 梯度非零；γ nudge 后 aux conv 梯度出现；
   - 冻结 ViT4 checksum（复用 vit_pretrain_audit 口径 depth=4）；
   - `switch_to_deploy()` 后 state_dict 无 BN/aux 键；C0/M1 deploy Params 逐位相等；
   - 3-step 前向+反向 loss 有限；|pred(A,B)−pred(B,A)| 记录（不设 gate）；
5. SYSU 60-step dry run：exit 0、loss 有限、[TOTAL/EFFECTIVE/TRAINABLE/FLOPS] 行、
   冻结 ViT checksum、arch.json 正确、TEST 区块正常；跑完删 scratch。

任何一项硬失败 → 修复实现重过（P0 问题），**不**直接进入 80K。

### 6.3 SF-1 SYSU 决策实验（正式 80K × 2）

唯一变量 = 训练期结构重参数化；C0 与 M1 的部署函数类完全相同（单路径同形状
卷积），固定：rep_mode 外的全部协议（§0.3）、batch 16、seed 16、RandomExchange
等增广原样、80K、test-as-val。A0 参考 = 历史 R4-0（83.14）/ R4-1（82.77），不重跑。

| 组 | 定义 | deploy 图 |
|---|---|---|
| C0_Plain | rep_mode=plain（无 sum/diff/aux，BN-FR α 保留） | 单路径 convs |
| M1_TAR-DCR | rep_mode=full（TAR 三路 + DCR 三原语全开，aux 零初始化） | 与 C0 同函数类（折叠后单路径） |

归因：Topology = C0 − R4-1（跨解码器换代的系统级参考，**非受控对比**，只作上下文）；
**Rep = M1 − C0**（受控唯一变量）；System = M1 − R4-1。

**预注册判据（SYSU，只读同一 train_log.txt 最后一个完整 `=== TEST RESULTS ===` 区块）**：

- **PASS**：M1 F1 ≥ 0.8500 且 M1−C0 ≥ +0.10pp 且 deploy ≤3M 且二值化 disagreement = 0。
- **WEAK**：M1 F1 ≥ 0.8500 但 M1−C0 ∈ [0, +0.10pp) → 目标达成但 rep 不可辨识；
  结果归档为「部署拓扑性能点」，**不得**宣称结构重参数化创新，不扩展。
- **FAIL**：M1 F1 < 0.8500，或 M1−C0 < 0，或 M1−C0 由 Recall↓/Precision↑ 构成
  （「置信度锐化」签名，STR 八轮失败模式，无论 F1 高低都判 rep-not-supported）
  → **停止融合主线，不救机制**。
- **禁止**（任何结果下）：D/深度/branch 扫描、改 loss、threshold tuning、
  seed 重跑、增广改动。M1−C0 阈值 +0.10pp 依据：SYSU test N=4000、C0/M1 共享
  除训练分支外一切随机性，配对比较噪声底 ≈ 0.07–0.10pp（CASA 历史 ±0.15 口径
  按 √(2048/4000) 折算），阈值置于噪声顶。

### 6.4 SF-2 扩展实验（仅 SF-1 PASS 才允许）

| 阶段 | 实验 | 判据 | 失败停止线 |
|---|---|---|---|
| SF-2a | LEVIR M1（1 个 80K） | F1 ≥ 0.9250 | F1 < 0.9184（A1 锚点 91.84）→ 停 |
| SF-2b | WHU M1（1 个 80K） | F1 ≥ 0.9500 | F1 < 0.9484（基线 94.84）→ 停 |
| SF-2c | CDD M1（1 个 80K） | F1 ≥ 0.9800 | F1 < 0.9775（基线 97.75）→ 停 |

- **正式 80K 预算 = SYSU×2 + LEVIR×1 + WHU×1 = 4 个**（符合「预期 ≤3–4」）；
  CDD M1 为第 5 个，仅当前三阶段全部 PASS **且用户明确批准**才启动。
- 最终声明「四数据集硬目标完成」仅当 M1（deploy 图）同时满足 85/92.5/95/98
  且 deploy ≤ 3M。

---

## 7. smoke / dry run / 审计执行清单（设计问题 7，顺序固定）

```text
[0]  本地：py_compile 全部新文件（沙箱 python 查语法）
[1]  本地：test_str_reparam_equivalence.py 的 T0（纯 FP64 代数，CPU 可跑）
[2]  部署：python -B .claude/_deploy.py（无需改白名单）
[3]  服务器 GPU1（先 nvidia-smi 确认空闲、单任务）：
       smoke_test.py --mode strfusion（§6.2-4 全项）
       test_str_reparam_equivalence.py（T0/T1/T2）
       analyse/run1_strfusion_budget.py            → G4 裁决
       analyse/run1_strfusion_interface_audit.py   → [SF-D0-GATE] 裁决（四数据集）
[4]  SF-D0 FAIL → 写终止记录（README/快照/git push），不训练；PASS → [5]
[5]  生成 train_scripts/STR-Fusion/Run1 全部脚本（LF 行尾字节校验）
[6]  SYSU dryrun（60 steps，scratch 目录，跑完删除）
[7]  SYSU C0 正式 80K（GPU1 单任务，~3–4.5h）→ 完成后 SYSU M1（串行，勿重复启动）
[8]  收集日志（_collect.py）→ 只读最后完整 TEST RESULTS 区块 → SF-1 一次性裁决
[9]  PASS → SF-2a（LEVIR M1）→ SF-2b（WHU M1）→（用户批准）SF-2c（CDD M1）
[10] 全部结束后：extract_metrics_to_excel + models_to_txt + README + git push
```

服务器运行命令范式（经 `.claude/_ssh.py` 文件模式，防 pwsh 引号问题）：

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate casacd
cd /home/yqwang/projects/CASA-CD/models
CUDA_VISIBLE_DEVICES=1 python smoke_test.py --pretrained_weight_path pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth --gpu_id 0 --mode strfusion
CUDA_VISIBLE_DEVICES=1 python test_str_reparam_equivalence.py --pretrained_weight_path pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth
cd ../train_scripts/STR-Fusion/Run1 && bash -n **/*.sh
# 正式训练（SYSU C0 完成后才启动 M1；单任务纪律）
cd C0_Plain && nohup bash train_SYSU-CD-256.sh > /dev/null 2>&1 &
```

---

## 8. 论文故事（设计问题 8：创新性 / 可解释性 / 统一叙事）

**候选模型名（论文阶段定）**：内部代号 `STRFusion`；候选
① *Token-Depth Algebraic Re-parameterization for Ultra-Lightweight Change Detection*、
② *STR-CASA*。

**统一叙事（一条主线，两个仓库各贡献一半）**：

> **「非均匀计算」决定该省什么，「结构重参数化」决定省下的参数该怎么放、怎么部署。**
>
> CASA-CD 用五轮负结果证明了 token 与 depth 两个维度的计算冗余（A1：K/V 压缩到
> 25% 基本无损；R4-0→R4-1：12→4 block 仅 −0.37 F1；B4>B12），并证明「冻结 plain
> ViT 的 token 证据无法被无训练重建补成 dense 边界」（R8-D0）——语义证据在 token
> 里，但**细节/边界必须由训练过的、任务适配的空间算子生产**。STR-RepNet 恰好
> 提供了一套这样的算子：TAR 把二时相交互（concat/sum/signed-diff）用代数折叠成
> 单个 1×1，DCR 把整个 top-down 解码器的 DW/PW/跨尺度融合全部解析折叠为单路径
> 静态卷积，FP64 组合 + 一次 FP32 cast，部署与训练图在二值决策上逐位一致
> （argmax=0）。融合模型 = 冻结 ViT4 的多深度 token 金字塔（change-aware 语义源）
> ＋ 0.62M 可训练且可折叠的 TAR+DCR（任务适配空间表征源），部署总参 <3M。

**创新点 = 结构重参数化本身**（论文主张，三条腿缺一不可）：
1. **Temporal Algebraic Re-parameterization on token pyramids**：四深度二时相
   bridge 的三路代数（含 signed-diff 的方向敏感变化基）训练期解耦优化、部署期
   精确折叠——这是 TAR 在「token 深度金字塔」这一新接口上的首次落位；
2. **Decoder-wide Compositional Re-parameterization as the trained spatial
   adapter**：正面回应 R8-D0 的「ranking ≠ dense 重建」——细节生产者不是任何
   detail 支路，而是全程可折叠的解码器组合（DW3/PW/fuse 各自的多分支训练 → 单核
   部署），用 R8-D0 同款边界带度量预注册裁决（SF-D0）；
3. **可解释性**：(a) 部署图是单路径静态卷积，折叠后的每个 1×1/DW 核可直接检查
   ——signed-diff 分支折叠贡献可可视化为「变化方向滤波器」，sum 分支为「共同
   语义投影」，机制可被权重级审计；(b) 折叠是**解析等价**（FP64 代数 <1e-10、
   二值化 disagreement=0），训练增益只可能来自训练期多分支优化景观，而非部署
   函数类改变——这给「重参数化到底改了什么」一个干净的归因通道（C0 vs M1 同
   函数类消融）；(c) CASAA 的 token 思想以「depth 维度的非均匀变化证据」形式
   融入：浅层 token 供细节、深层 token 供变化判别（R7-D0 曲线支撑），冻结
   encoder 的 Full-Q 位置证据保留（A1 结论支撑），省下的全部训练预算集中在
   可折叠中间层。

**可证伪性（写进论文的负结果空间）**：SF-D0 gate 可能证伪「多深度 token 证据
非冗余」（→ inner-patch 局限无法用深度维度绕过）；M1−C0 可能再现 STR 八轮的
「置信度锐化」签名（→ 训练分支只有优化效应、无判别信息增量）。任一发生，
主线按预注册终止，负结果如实归档——这本身就是课题纪律的一部分。

**不允许的写法**：不得把 loss/增广/threshold/多 seed 写成贡献；不得用
「≤3M 本身」当创新（MixCDNet 0.32M 已存在）；不得称单 seed 统计显著；与
HAM-CD/轻量 SOTA 比较时协议必须逐项对齐并诚实声明。

---

## 9. 已确认决策记录（2026-10，用户第二轮「开始吧」）

1. **判据数字**：按本稿执行——M1−C0 ≥ +0.10pp（SYSU）；SF-D0 的 G1（PRbnd(fuse4) ≥
   PRbnd(B4)+0.02，SYSU 必过且 ≥2/4）、G2（PRbnd(fuse2) ≥ PRbnd(B4)−0.02，≥3/4）。
2. **预算口径**：<3M 为硬门槛（G4）；≤2.20M 仅作工程目标记录；**Run1 不启用 D* 旋钮**
   （D=160 固定，唯一变量优先，禁止用 D 扫描救 F1）。
3. **CDD 第 5 个 80K**：仅当 SF-1/SF-2a/SF-2b 三阶段全部 PASS 才启动。
4. **命名**：内部代号 `STRFusion`、目录 `train_scripts/STR-Fusion/Run1`（用户确认）。
5. **swap 对称性**：只测量不设 gate（与 ChangeViT/STR 一致），论文如实声明。
6. **GPU**：GPU1 单卡纪律（STR 文档的 GPU0 记载作废）。
7. **定位**（用户重申）：研究生论文课题，核心 = 方法创新（变化感知 CASAA/非对称
   token 建模 + 极轻量结构 + 跳层连接/解码器时空结构重参数化思想）；不做工程化堆叠，
   不把 loss 调参 / 训练技巧包装成创新贡献。

> 判据阈值自本文档确认后**不再修改**。

---

## 附录 A：本方案依据的文件清单（已通读）

STR-RepNet 侧：README.md；docs/temporary/STR-RepNet_交接文档_新对话启动.md；
models/changedetection/models/{reparam,tar,dcr_decoder,STRRepNet,Mamba_backbone}.py；
models/changedetection/script/test_reparam_equivalence.py；
docs/temporary/STR-RepNet_Run10_PFDR_设计与实验方案.md；train_scripts/TAR-DCR/Run10/README.md。
CASA-CD 侧：README.md；docs/temporary/过去的想法/CASA-CD_交接文档_2026-09-30.md；
docs/ChatGPT_Project_Settings.md；docs/temporary/CASA-CD_Run9_R9-D0结果与B4-OPRE路线终止.md；
docs/temporary/CASA-CD_Run8_R8-D0结果与B4-only路线终止.md；
models/model/{encoder,decoder,trainer,sgdp_head,depth_pyramid_head,b4_spe_head}.py；
analyse/{run7_depth_source_audit,run8_b4_dense_recoverability_audit}.py；
docs/参考文献/文献索引.md；另核实 models/train.py、models/eval.py、models/smoke_test.py、
.claude/_deploy.py。
