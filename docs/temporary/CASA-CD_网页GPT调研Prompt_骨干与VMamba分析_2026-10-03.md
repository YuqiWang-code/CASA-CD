# CASA-CD 网页 GPT 调研分析 Prompt（骨干强度 / VMamba / 四向扫描 CASAA）

> 生成于 2026-10-03。用途：把下面完整 prompt 复制到网页版 GPT（ChatGPT / Claude /
> Gemini 等具备联网或丰富知识库的模型）进行详细网络调研、分析与判断。
> 配套材料：`docs/experiment_metrics.xlsx`、
> `docs/temporary/models_and_metrics_CASA-STR_Run1.txt`、
> 上游快照 `models_and_metrics_TAR-DCR_Run2.txt`（STR-RepNet 的 VMamba-Tiny 实现）。

---

你是深度学习与遥感变化检测（Change Detection, CD）方向的资深研究员。请对下述
硕士课题做一次**详细的网络调研、证据分析与架构判断**，全程区分「文献/论文事实」与
「你的推理判断」，给出可直接执行的结论。

## 一、课题背景与硬约束

- 任务：极轻量遥感**二值变化检测**，输入双时相 256×256 影像（A/B 两张），输出变化
  概率图，标签阈值 gray≥128。
- 四数据集硬目标（F1，必须同时全部达到）：**CDD ≥98、LEVIR ≥92.5、SYSU ≥85、
  WHU ≥95**（IoU 与 F1 同方向）；**有效推理参数 ≤5M**。
- 固定训练协议（不允许改）：BCE+Dice 损失、Adam(2e-4)、poly(0.9) 学习率 +200
  warmup、80,000 steps、batch 16、seed 16、test 集当验证集选 best。**禁止**损失
  调参、数据增强、阈值调参、多 seed 包装；**创新必须是结构性的**。
- 现状：当前架构 CASA-STRNet = SHViT-S1（CVPR 2024，单头注意力、部分通道 FFN，
  ImageNet-1K 预训练）截断主干 + CASAA@1/16（变化感知非对称注意力：完整 Query、
  K=64=Kc32 变化 token 直保留+Kb32 背景共享聚类聚合）+ TAR 二时相 bridge +
  DCR 可折叠解码器（STR 重参数化）。deploy 2.42M 参数 ≤5M。

## 二、现有实验证据（16 个完整 80K run，双卡并行，全部从头训练）

（格式：CDD / LEVIR / SYSU / WHU 的 F1）

1. ChangeViT-T（Pattern Recognition 2025）完整复现基线（20.66M，26.32G）：
   **97.75 / 91.95 / 82.48 / 94.84**。
2. CASA-STR Run1 用 SHViT-S1 截断主干（deploy 2.38–2.42M）：
   - A0 基线（无创新模块）：**94.67 / 89.90 / 82.46 / 93.70**——比 ChangeViT-T
     低 CDD −3.08、LEVIR −2.05、WHU −1.14，SYSU 持平；
   - M1 双创新（CASAA+STR-rep）：**95.54 / 90.37 / 83.02 / 93.72**（Δ +0.87 /
     +0.47 / +0.56 / +0.02）；
   - A1 只开 CASAA：94.54 / 89.87 / 82.49 / 93.65（≈A0）；
   - A2 只开 STR-rep：95.53 / 90.28 / 82.49 / 93.41（CDD/LEVIR 增益主载体）。
3. **上游 STR-RepNet 的 TAR-DCR Run2（同样的 TAR/DCR 头 + 冻结 VMamba-Tiny 主干，
   29.57M 总量 / 1.58M 可训练 / 12.61G FLOPs）**：
   - A0_Plain（全冻结）：97.32 / 90.09 / 81.14 / 93.55；
   - full_last2（末两阶段可训练）：**98.42 / 91.44 / 83.45 / 95.14**；
   - 其 36.08M 完整基线：98.79 / 92.11 / 82.99 / 95.00。

我们自己的判断（请你验证或反驳）：
(a) 同一个 TAR/DCR 头，换 VMamba-Tiny 主干比 SHViT-S1 主干高 **CDD +2.88、WHU
+1.42、LEVIR +1.07、SYSU +0.43 pp**——骨干表征强度是主要瓶颈，而非训练策略；
(b) 我们的结构性创新（CASAA + STR-rep）在弱骨干上仍有 +0.5~+0.9pp 稳定增益，
创新有效，可平移到强骨干；
(c) 弱骨干下 CDD 缺口最大（95.5 vs 98.4）——CDD 是抗伪变化数据集，对语义判别
质量最敏感；
(d) 训练策略健康（loss 曲线正常、重参数化折叠等价性门槛全过、协议一致）。

## 三、需要你调研并回答的三个问题

### Q1：是训练策略问题，还是骨干太弱？（判断题）
- 结合上述证据判断瓶颈归属。特别调研：ImageNet-1K 精度/骨干容量与遥感 CD 表现
  的相关性证据（例如 ViT 系与 Mamba 系骨干在 LEVIR/SYSU/WHU/CDD 上的对照文献）；
  SHViT-S1 这类超轻骨干（单头注意力 + 部分通道 FFN + BN）是否存在已知的表达力
  短板（如窗口/单头注意力语义建模弱、BN 对遥感小目标分布敏感）。
- 若「训练策略」也有嫌疑（如 80K steps 欠拟合强骨干、backbone lr×0.1 过低），
  给出证据与修正建议，但**不得建议违反上述固定协议**。

### Q2：VMamba 能否作为 backbone？（可行性调研）
- VMamba（NeurIPS 2024，SS2D 四向交叉扫描 + VSSM）：Tiny/Small/Base 各型号的
  **参数、FLOPs、ImageNet-1K Top-1 精度**，官方 ImageNet-1K 预训练权重是否公开
  可下载（仓库 MzeroMiko/VMamba）；SS2D 的 4 方向扫描 token 数膨胀与实现开销。
- 已用于遥感变化检测的 Mamba 系工作（2024–2026）：VMambaCD、ChangeMamba、
  RSMamba、MambaBCD、RS-Mamba 等——它们的骨干、参数量、LEVIR/SYSU/WHU/CDD 上的
  F1 数字（尽量给到论文报告值），以及与我们三张证据表的可比性。
- **预算矛盾**：VMamba-Tiny ≈27–28M 超出 ≤5M 硬约束。请调研可行解：
  ① 更小的 Mamba 系预训练骨干（如 Vim-Ti 7M、EfficientVMamba、MambaVision、
  LocalMamba-T、VSSD 压缩版等）的参数/ImageNet-1K 精度/预训练权重可得性；
  ② 对 VMamba-Tiny 做深度/宽度截断+微调（例如只保留前 2–3 个 stage）是否仍保留
  预训练收益，截断后参数量估算；③ 蒸馏；④ 说明如果 ≤5M 内无解，放宽预算到多少
  能保住 VMamba-Tiny 的收益（给证据链，供导师决策）。
- 输出一张「候选骨干对比表」：型号 | 参数 | FLOPs | ImageNet-1K | 预训练权重 |
  遥感 CD 已知表现 | 适配 ≤5M 的路径。

### Q3：Mamba 四向扫描的 token 能不能做「变化感知压缩」（CASAA 化）？（创新点可行性 + novelty 检查）
- 我们想保留的机制：完整 Query（逐位置判别）+ 只压缩 K/V 上下文（K=64），疑似变化
  token 直保留、稳定背景 token 共享聚类聚合；变化分数由 1/8 双时相特征 cosine 得到
  （参数自由）。
- 技术问题：SS2D 把 2D 特征沿 4 个方向展平成序列（token 数 ×4）。请调研/推理：
  ① 压缩应作用于「4 个方向序列分别」还是「合并后的 token」？SSM 对序列顺序敏感
  （scan order 是归纳偏置），token 压缩会改变序列长度与顺序——SSM 本身可吃变长
  序列吗？压缩会不会破坏四向扫描的上下文语义？
  ② 有无先例：Mamba 系 token pruning / token merging / 稀疏扫描（2024–2025 的
  Sparse Mamba、Mamba token selection、VSSD 的 token+channel 双重压缩等），哪些
  可直接借鉴？
  ③ 「变化感知扫描」作为新创新点：在 scan 前用变化分数重排/加权 token（把疑似
  变化 token 放到序列前端或沿 scan 方向加权），是否有人做过？SSM 的选择性遗忘
  （selective state）与变化感知的天然契合点在哪？
  ④ 与现有「Mamba+CD」工作的差异性与 novelty 风险。
- 给出你认为最有论文故事性、且工程可实现（基于官方 VMamba 代码 + selective scan
  CUDA kernel，我们已有可运行快照）的 1–2 个具体设计，并做参数量估算（必须 ≤5M）。

## 四、输出格式要求

1. 每问给「结论一句话 + 证据/文献 + 风险与不确定性」。
2. 附文献清单：论文名 / 年份 / 会议或期刊 / 关键数字（参数、F1、ImageNet-1K）/
   链接（arXiv 或 GitHub）。
3. 最后给「下一步推荐」：一个明确的主干+创新组合方案（含参数量与预期收益区间），
   以及备选方案。
4. 若某事实你不确定，请明确标注「待核实」并给出查询关键词，不要编造数字。
