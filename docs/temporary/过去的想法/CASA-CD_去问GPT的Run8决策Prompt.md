# CASA-CD 下一步决策 Prompt（Run8：最终路线决策，带去网页 GPT，附 GitHub 链接）

> 仓库：https://github.com/YuqiWang-code/CASA-CD （main 分支已是最新，HEAD=`7fb2aa9`）
> 用法：把下面整段直接粘贴给网页 GPT。让它先读仓库里的指定文档再回答。

---

我是硕士课题「极轻量遥感二值变化检测」的负责人。仓库在
https://github.com/YuqiWang-code/CASA-CD （main 分支已是最新）。

请先读仓库里这几份文档（按顺序），它们是我项目的事实权威：
1. `README.md`（全部结果表 + 约定）
2. `docs/temporary/CASA-CD_Run7_CSDP方案与预注册.md`（上一轮你的方案，已执行完）
3. `docs/temporary/CASA-CD_Run7_R7-D0结果与CSDP停止.md`（**本轮最重要：四数据集
   depth gate 结果 + CSDP 停止记录 + 下一步决策叉 §6**）
4. `docs/temporary/CASA-CD_Run6_R6-D0结果与Run6停止.md`（Run6 终局）
5. `docs/temporary/过去的想法/CASA-CD_Run5_R5-D0结果与Run5停止.md`（Run5 终局）
6. `docs/temporary/过去的想法/CASA-CD_Run4_R4-D0审计结果与R4-2d启动.md` 与
   `CASA-CD_Run4_R4-2d_PSD结果与轻量detail路线终止.md`（Run4 终局）
7. `docs/参考文献/文献索引.md`（本轮调研的 11 篇文献 PDF 清单 + 官方链接 +
   参考代码清单；PDF 都在 `docs/参考文献/2024-2026主证据/`）
代码重点：`models/model/{encoder,trainer,decoder,sgdp_head,depth_pyramid_head}.py`、
`analyse/{run4_detail_interface_audit,run5_postmortem_and_mobile_audit,
run6_semantic_token_audit,run7_depth_source_audit}.py`。

## 一句话现状

目标：二值变化检测，4 数据集（CDD/LEVIR/SYSU/WHU，256×256，label 阈值 gray≥128），
最终模型有效推理参数 **<3M（工程目标 ≤2.20M）**，在 2M 量级超过轻量 SOTA
（RFANet 2.86M / SeCoR 2.50M / Lighter 1.10M / CGLNet 0.99M；MixCDNet 已做到
0.32M/1.59G，所以「小参数」本身不是创新）。
已按预注册规则走完五轮证伪：CASAA change-router（Run1-3）、自定义轻量 detail×3
（Run4）、MobileNet 预训练 prefix（Run5）、PatchEmbed token 重建（Run6）、
B1+B2 深度截断语义源（Run7）。**每轮都靠零训练 raw gate 在花 80K 之前证伪，
所以正式训练只消耗过 1 个（R4-2d）**。现在需要你基于完整证据链做 Run8 的
最终路线决策。

## 必须遵守的硬约束（不要违反）

- 方法创新导向；**不改 loss、不把训练技巧当创新**；单 seed=16（不做多 seed）。
- 每个实验**唯一变量 + 预注册 gate**；任何新结构候选**必须先过零训练 raw gate**
  才允许花 80K；gate 不过不进入下一步；不做 sweep；失败后不调阈值救场。
- 训练协议固定：BCE+Dice、Adam 2e-4、poly+200 warmup、80000 steps、batch 16、
  256×256、seed 16、test-as-val；ViT 一律冻结（corrected DeiT loader）。
- 服务器只有 GPU1；SYSU 一个 80K 约 3-3.5h、LEVIR ~4.4h——**正式 80K 总数
  ≤3-4 个就要到定稿**。
- 参数量口径：forward 图内全部参数（冻结照计、死参数不计）；不把量化/剪枝/蒸馏
  当核心贡献；不再搜索新的独立 detail branch（Run5 §12/§48 预注册禁令仍有效，
  要推翻需在方案里明确论证）。

## 完整证据链（全部数字最新，可直接引用）

### A. 全部正式端到端结果（SYSU，F1 / IoU）

| Run | 结构 | F1 | IoU | Params | FLOPs |
|---|---:|---:|---:|---:|
| baseline Run1 | ChangeViT-T 官方协议（ViT 部分死权重） | 82.48 | 70.19 | 11.754M | 26.32G |
| R4-0 | 健康 full12 frozen + corrected loader | **83.14** | 71.14 | 11.754M | 26.32G |
| R4-1 | ViT4 + ResNet detail + 旧 FI/decoder | 82.77 | 70.61 | 8.195M | 24.10G |
| R4-2 / 2b | ViT4 + LightDetail 32(48)/64(96)/128(160) + adapters + 旧 head | 82.30 / 82.32 | 69.92 / 69.95 | 5.49M | 10.7-11.0G |
| R4-2d | ViT4 + PSD-Detail 0.078M + 旧 head | 82.02 | 69.52 | 5.491M | 11.52G |

（LEVIR：baseline 91.95；冻结 A1 91.84；最终目标 ≥91.50。）

### B. 零训练 raw gate 的全部关键数字（完整 test 集，1−cos token/feature ranking）

R4-D0（SYSU，feature 级 PR-AUC）：ResNet 1/8=0.6535；Light32 深层 0.32 崩坏。
R5-D0（SYSU）：MobileNetV3-Small prefix D4=0.4831<0.50（差 0.017，3/4 子门槛过）；
H2 证实 PSD stem 被重写（rel_L2 0.639 vs R4-1 0.211）；H1 否定 adapter 关键
（TileAdapter ΔF1 −0.013）；**旧 FI 零权重吸收**（R4-1 的 FI 只有 1/8 一路注入
活跃，1/2、1/4 权重精确归零）。
R6-D0（SYSU，token 级）：P0=0.3711、B1=0.5349、B2=0.6008、B3=0.6108、
B4=0.6127、B12=0.5143。
R7-D0（四数据集，token 级 PR-AUC）：

| source | CDD | LEVIR | SYSU | WHU |
|---|---:|---:|---:|---:|
| B2 | 0.4164 | 0.2947 | 0.6008 | 0.2660 |
| B3 | 0.4524 | 0.3017 | 0.6108 | **0.3508** |
| B4 | **0.4594** | 0.3009 | **0.6127** | 0.3449 |
| B12 | 0.3980 | 0.2465 | 0.5143 | 0.1693 |

结论：**B4 是四数据集稳健最优语义源**（CDD/SYSU 第一、LEVIR/WHU 第二）；
full12（B12）全面劣化（支持「深未必好」）；B2 只在 SYSU/LEVIR 够用
（CSDP-B2 路线被 C2=2/4 证伪）；最优 change-sensitive depth 是 dataset-dependent。

### C. 已就绪、尚未被自身 gate 否决的资产（重要）

1. **SGDP head**（`sgdp_head.py`，115,267 参数，difference-first + 语义门控 +
   coarse-to-fine，FLOPs 1.68G，时间交换对称）——smoke 全绿，但**从未被训练**：
   Run5 的 gate 链要求 Mobile detail 先过 raw gate，detail 失败导致 SGDP 连带
   没机会上场。**「ViT4 + B4 + SGDP（无独立 detail）」这个组合从未被端到端
   检验过**，而它的两个输入证据都是正面的（B4=0.61 四数据集稳健；head 复杂度
   达标）。
2. **CSDP head**（`depth_pyramid_head.py`，90,832 参数，ViT2+head=1,177,936、
   FLOPs 0.673G）——smoke 全绿，但因 B2 gate 失败按规则未训练。若改成消费
   B3/B4 会违反「不改 B2→B3 救场」的停止规则，需要你明确是否重新预注册。
3. **A1 content 压缩**（CASAA Run2）：冻结 ViT + K=64 content 聚类基本无损
   （LEVIR −0.01 / SYSU +0.02 vs 冻结 baseline），零新增参数——论文 analysis 资产。
4. 零训练审计工具链 4 个脚本（feature/token 级 raw gate 全套，可复用于任何候选）。

### D. 预算事实

ViT4-192 = 1,976,832（占总预算 90%）；ViT2 = 1,087,104。旧 FI+decoder =
3.435M（必须退出最终模型）。SGDP/CSDP head 均为 0.09-0.12M，因此
「ViT4 + 轻量 head」≈ 2.08-2.10M、「ViT2 + 轻量 head」≈ 1.18M，
两者都满足 <3M 且后者有更大 head 余量——但 R7-D0 已证 ViT2 的 B2 源不足。

## Run8 决策叉（请你裁决；也可给出更好方案，但要过同样的证据标准）

1. **「ViT4 + B4 + SGDP」首次端到端检验**（我当前最倾向）：B4 是唯一通过全部
   raw gate 的语义源，SGDP 是现成 head——组合从未被训练，且参数 ≈2.09M、
   FLOPs ≤2.0G 达标。风险：无独立 detail 的 head 重建能力未经检验
   （ViT-CoMer 提示 inner-patch 交互不足）；需要你给定预注册 gate 与失败出路。
2. **depth-adaptive 多源聚合**：把「最优 depth 依赖数据集」本身变成机制
   （B2-B4 多 depth token 的轻量可学习聚合），而不是硬截断——但需论证与
   RFL-CDNet「intermediate feature 全利用 + deep supervision」的实质区别，
   否则创新性不足；且这是新的多变量结构，raw gate 定义更难。
3. **以 Run4 已验证组合收尾**：ViT4 + ResNet detail + 旧 head（82.77 已有）+
   A1 K=64 压缩做论文主线，把 depth/detail 五轮负结果作为系统性分析
   （budget allocation study）。风险最低、故事完整，但「轻量」只到 8.2M/5.5M
   级别，不满足 <3M 的最终模型要求——需要你判断这能否被接受为论文结局，
   或与候选 1 合并（先 1 后 3）。

## 请你输出的内容（一次性给全，要可执行、可预注册）

1. **证据判读**：综合 R7-D0 四数据集 depth 曲线 + 文献（FDAM 频率衰减、
   LaViT attention saturation、ResCLIP 中层定位、ViT-CoMer inner-patch 不足、
   LiFT 轻量 densify、EoMT plain-ViT dense），选哪个候选、为什么；
   明确「无独立 detail 的 B4+轻量 head」是否值得最后一个 SYSU 80K。
2. **Run8-D0 零训练审计**（如需要）：定义、指标、预注册阈值（具体数字）。
3. **Run8 唯一主方案**：完整逐层定义 + 参数预算表（≤2.20M，最好 ≤2.10M）+
   FLOPs 粗估；若复用 SGDP/CSDP head，说明改动边界。
4. **预注册 gate**：每个 80K 的唯一变量、reference（R4-1 82.77 / R4-0 83.14，
   注明用哪个）、F1/IoU/Δ/复杂度阈值；SYSU 终 gate 沿用 F1≥82.30 & IoU≥69.92，
   LEVIR ≥91.50 & IoU≥84.33。
5. **停止规则**：每个 gate 失败后的唯一出路（明确「永久停止 X，只能进入 Y」）。
6. **执行顺序与实验预算**：≤3-4 个正式 80K 到定稿；四数据集铺开条件。
7. **论文机制故事与 CASAA/A1 的最终位置**：如果方案成立，主贡献怎么写
   （现有素材：depth redundancy 12→4 仅 −0.37 F1；四数据集 depth 曲线；
   五轮负结果的 budget allocation study；change-evidence-first head）；
   CASAA 是否仍保留为「token 维度 + depth 维度的统一叙事」的一部分。

请直接给出完整方案文档，我会按之前 Run5/6/7 的格式执行（存进 docs/temporary/，
先零训练 raw gate，再 dry run，再 GPU1 的 80K，逐 gate 判定）。
