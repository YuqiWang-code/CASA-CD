# CASA-CD 新对话交接 Prompt：STR-RepNet × CASA-CD 融合（换 baseline 与新主线）

> 用法：把下面代码块中的全部内容复制粘贴到 **CASA-CD 工作区
> （F:\Code_Repositories_2\CursorCode\CASA-CD）新建的对话**开头。
> 所有文件路径均为绝对路径，已在 2026-10-02 核实存在。

```text
这是一个跨工作区融合任务的交接。先完整阅读以下文件（全部绝对路径），再输出设计方案，
经用户确认后再动手改代码。

# 一、任务一句话

STR-RepNet 硕士课题换 baseline：把原 baseline（HAM-CD）换成 CASA-CD 系，并开始一条
新的融合主线——
「encoder / backbone 部分：用 CASA-CD 的 ChangeViT + CASAA 的 token 思想；
跳层连接与解码器部分：保留 STR-RepNet 的时空结构重参数化（TAR + DCR）思想」。
即：把两个项目各自的正面资产拼成一个新模型，在 CASA-CD 工作区实现与实验。

必须同时满足（硬约束）：
- 四数据集 test F1 硬目标：SYSU ≥ 85 / LEVIR ≥ 92.5 / WHU ≥ 95 / CDD ≥ 98；
- 轻量化（CASA-CD 口径：有效推理参数 <3M，工程目标 ≤2.20M——与 STR 侧旧锚点
  28.83M 是两套口径，本工作以 <3M 为准）；
- 创新性、故事性、可解释性；创新必须是「结构重参数化本身」，
  禁止把 loss 调参/数据增广/threshold tuning/多 seed 包装成创新；
- 预注册 gate + 失败停止规则；任何新结构候选先过零训练 audit 再花正式训练预算。

# 二、两个仓库的事实权威（先读，按此顺序）

STR-RepNet 侧（机制来源）：
1. F:\Code_Repositories_2\CursorCode\STR-RepNet\README.md
2. F:\Code_Repositories_2\CursorCode\STR-RepNet\docs\temporary\STR-RepNet_交接文档_新对话启动.md
3. F:\Code_Repositories_2\CursorCode\STR-RepNet\models\changedetection\models\reparam.py
4. F:\Code_Repositories_2\CursorCode\STR-RepNet\models\changedetection\models\tar.py
5. F:\Code_Repositories_2\CursorCode\STR-RepNet\models\changedetection\models\dcr_decoder.py
6. F:\Code_Repositories_2\CursorCode\STR-RepNet\models\changedetection\models\STRRepNet.py
7. F:\Code_Repositories_2\CursorCode\STR-RepNet\models\changedetection\models\Mamba_backbone.py
   （本次要替换掉的旧 encoder，仅供对照）
8. F:\Code_Repositories_2\CursorCode\STR-RepNet\models\changedetection\script\test_reparam_equivalence.py
   （T0/T1/T2 折叠等价性测试范式）
9. F:\Code_Repositories_2\CursorCode\STR-RepNet\docs\temporary\STR-RepNet_Run10_PFDR_设计与实验方案.md
   （最新一轮负结果；预注册判据/门槛审计/判据裁决的格式范本）
10. F:\Code_Repositories_2\CursorCode\STR-RepNet\train_scripts\TAR-DCR\Run10\README.md

CASA-CD 侧（当前工作区，代码/协议/审计工具链在这里）：
1. F:\Code_Repositories_2\CursorCode\CASA-CD\README.md
2. F:\Code_Repositories_2\CursorCode\CASA-CD\docs\temporary\过去的想法\CASA-CD_交接文档_2026-09-30.md
3. F:\Code_Repositories_2\CursorCode\CASA-CD\docs\ChatGPT_Project_Settings.md（项目长期指令）
4. F:\Code_Repositories_2\CursorCode\CASA-CD\docs\temporary\CASA-CD_Run9_R9-D0结果与B4-OPRE路线终止.md
   （最新终局：七轮负结果链 + 下一步提示）
5. F:\Code_Repositories_2\CursorCode\CASA-CD\docs\temporary\CASA-CD_Run8_R8-D0结果与B4-only路线终止.md
6. F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\encoder.py
7. F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\decoder.py
8. F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\trainer.py
9. F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\sgdp_head.py
10. F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\depth_pyramid_head.py
11. F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\b4_spe_head.py
12. F:\Code_Repositories_2\CursorCode\CASA-CD\analyse\run7_depth_source_audit.py
13. F:\Code_Repositories_2\CursorCode\CASA-CD\analyse\run8_b4_dense_recoverability_audit.py
14. F:\Code_Repositories_2\CursorCode\CASA-CD\docs\参考文献\文献索引.md

# 三、STR 侧要移植的机制（摘要，细节以 reparam.py/tar.py/dcr_decoder.py 为准）

- **TAR**（Temporal Algebraic Re-parameterization）：四尺度二时相 bridge。
  每个尺度训练期 Concat + Sum + signed-Diff 三路（各带独立 BN，BN-FR），
  部署折叠为单个 1×1（`TemporalRep1x1` / `TARStage` / `MultiScaleTAR`）。
- **DCR**（Decoder-wide Compositional Re-parameterization）：top-down 多尺度解码器
  （fuse3/block3 → fuse2/block2 → fuse1/block1 → refine），全部算子可折叠：
  `RepDW3`（DW3+DW1×3+DW3×1+αI → 单 DW3）、`RepPW1x1`（主 PW+低秩串行+diag+αI → 单 1×1）、
  `RepPairFuse1x1`（跨尺度 concat+sum+diff+αL → 单 1×1）。每个线性分支独立 BN + 可折叠
  α 残差（BN-FR）。
- **折叠纪律**：组合全程 FP64、最后一次性 cast FP32；部署后 argmax disagreement = 0
  （实测）；部署 Params/FLOPs 做机器预算审计；训练期独有分支零初始化（BN γ=β=0），
  epoch-0 输出与主路径逐位一致。
- **规模事实**（STR，D=160）：中间+Decoder ≈ 0.84M 参数 / 0.92G FLOPs；
  整网 29.5M 中 encoder 占绝大多数（VMamba-Tiny 冻结）。
- **STR 侧负结果教训**（Run3–Run10 八种参数化全部 FAIL/WEAK，共同签名「置信度锐化」
  Recall↓/Precision↑；Run6 诊断：LEVIR small≤502px recall 80.8%、24% 完全漏检，
  瓶颈在 decoder 侧小目标保真）：移植 DCR 时不要把「训练分支越花哨越好」当默认假设，
  任何新分支必须有可证伪机制动机。

# 四、CASA-CD 侧要保留/接合的东西（摘要，细节以 CASA-CD 文档为准）

- **ChangeViT-T**（corrected DeiT loader，冻结 ViT 优先）。参数事实：ViT12 ≈ 5.536M、
  ViT4-192 ≈ 1.977M、ViT2 ≈ 1.087M（forward 图内全量口径）。
- **CASAA token 思想**：router 主线已按预注册终止（Top32 0.39→0.63 未转化为 F1），
  但 **A1（冻结 ViT + K=64 content 聚类压缩，零新增参数）基本无损**是正面资产，
  可作为可选机制落点；不得未重新预注册就复活 router 主线。
- **端到端正资产**：健康 full12 frozen SYSU F1 83.14（参考上限）、
  ViT4+ResNet+旧头 82.77（最强已验证轻量结构）、LEVIR 冻结 A1 91.84。
- **四数据集 token depth 曲线**（run7_depth_source_audit.py）：B4 稳健最优、
  B12 全面劣化、最优 depth dataset-dependent；R8-D0 进一步证明
  「token ranking 好 ≠ 无 detail 的 dense 重建好」（边界带 lift 几乎消失）。
  → 融合设计必须正面回答「细节/边界从哪来」。
- **协议与纪律**：BCE+Dice、Adam(2e-4, 0.9/0.99, wd=1e-4)、poly(power 0.9)+200 warmup、
  80000 steps、batch 16、256×256、seed 16、test-as-val；**GPU1 单卡**（注意：STR 仓库
  文档写的是 GPU0，两仓库记载不一致，以 CASA-CD 交接文档与服务器 nvidia-smi 实况为准，
  不要动别人的卡）；新结构先零训练 raw gate；正式结果只认同一 train_log.txt 最后一个
  完整 `=== TEST RESULTS ===` 区块；arch.json sidecar 校验；不做 sweep。

# 五、融合必须解决的设计问题（先给方案再动代码，逐条回答）

1. **多尺度特征接口**：TAR/DCR 需要四尺度金字塔（STR 里是 64/32/16/8 分辨率、
   统一 D=160 通道），ChangeViT 只给 16×16 token。可选项（不替你拍板，列出事实）：
   ViT 中间 block token（B1–B4）重排 + 上采样组成金字塔；patch 重建（P0/R8-D0 已证
   ranking 弱）；单尺度 16×16 语义 + head 侧重建。必须给出每级尺度从哪个 ViT 状态取、
   如何得到、参数量多少，并说明与 R8-D0「ranking≠dense 重建」证据的关系。
2. **预算**：ViT4(1.977M) + DCR D=160(≈0.84M) ≈ 2.82M < 3M（可行），但 TAR 二时相
   输入若需要额外 stem/投影要另算；ViT12 + DCR 超 3M。最终以代码实测
   TOTAL/TRAINABLE/EFFECTIVE Params + FLOPs 为准，D 与 ViT depth/width 可作预算旋钮
   （预注册，不 sweep）。
3. **训练协议**：默认沿用 CASA-CD 工作区的 trainer 与协议（80K、BCE+Dice、seed 16、
   GPU1）；STR 侧 CE+2×Lovász/300 epoch 只是历史参考，不要混用。折叠加折叠校验
   （T0/T1/T2）按 STR 的 test_reparam_equivalence.py 范式移植。
4. **不破坏既有资产**：CASA-CD 的 RunN 代码/checkpoint/log 一律不动；融合新代码放
   新文件（建议 `F:\Code_Repositories_2\CursorCode\CASA-CD\models\model\str_tar.py`、
   `str_dcr.py`、`str_reparam.py` 等，命名由你定）；新 run 目录建议
   `F:\Code_Repositories_2\CursorCode\CASA-CD\train_scripts\` 下新建主线目录
   （如 STR-Fusion/Run1，最终命名等用户确认）。
5. **预注册实验设计**：唯一变量、C0 对照 + M1 主实验、零训练 raw gate 先行
   （复用 CASA-CD 的 run4–run9 audit 工具范式）、SYSU 先行再 LEVIR→WHU→CDD、
   判据给具体数字、失败即停不救。预期正式 80K 总数 ≤3–4 个。

# 六、你（新对话 AI）第一轮要做的

1. 通读上述全部文件；
2. 输出一份《STR×CASA 融合 设计与预注册方案》md（存到
   F:\Code_Repositories_2\CursorCode\CASA-CD\docs\temporary\），至少包含：
   ① 证据表（两仓库正/负资产、四数据集 gap）；② 多尺度特征接口的最终选择与逐层定义；
   ③ 参数/FLOPs 预算表（<3M 硬门槛）；④ 折叠等价性方案（FP64/一次 cast/argmax=0）；
   ⑤ 逐文件修改清单（绝对路径）；⑥ 预注册 gate 与停止规则；⑦ smoke/dry run/审计清单；
   ⑧ 论文故事（创新性/可解释性论证，CASAA token 思想与 STR 时空重参数化如何合成
   一个统一叙事）；
3. 等用户确认后再动代码。用户会像以往一样按轮次驱动。
```

---

## 附：本交接撰写时的两个仓库关键事实速查（供核稿）

- STR 部署锚点（Run2 机器实测）：28.828706M / 12.6062G（旧口径，本融合不用）；
  DCR 中间+decoder ≈0.84M / 0.92G @D=160；STR 训练协议 seed 2333 / 300 epoch /
  batch 16 / CE+2×Lovász。
- CASA-CD 硬目标已写入其 README（SYSU ≥85 / LEVIR ≥92.5 / WHU ≥95 / CDD ≥98，
  有效推理参数 <3M，工程目标 ≤2.20M）；正式 80K 历史总消耗仅 1 个（R4-2d）；
  七轮证伪（CASAA router / 自定义 detail×3 / Mobile prefix / P0 重建 / B2 截断 /
  B4-only / B4-OPRE）全部由零训练 gate 在花 80K 前拦截。
- 两仓库 GPU 卡号记载冲突（STR 文档 GPU0、CASA-CD 文档 GPU1），以 CASA-CD 工作区
  纪律与服务器实况为准。
