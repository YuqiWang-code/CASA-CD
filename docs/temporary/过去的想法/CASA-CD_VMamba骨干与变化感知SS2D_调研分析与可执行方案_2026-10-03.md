# CASA-CD：VMamba / TinyViM 骨干、变化感知 SS2D 与下一轮主线重构调研

> **日期：2026-10-03**  
> **研究任务：** 极轻量全监督遥感二值变化检测（CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256）  
> **固定硬约束：** F1 同时达到 CDD ≥ 98、LEVIR ≥ 92.5、SYSU ≥ 85、WHU ≥ 95；有效推理参数 ≤ 5M；固定 BCE+Dice / Adam(2e-4) / poly(0.9)+200 warmup / 80K steps / batch 16 / seed 16 / test-as-val。  
> **证据截止：** 2026-10-03。  
> **当前仓库：** `YuqiWang-code/CASA-CD`，本文审阅时 `main` HEAD = `0378647b776c4d29c1516c6a8a8bb19a07fe805e`。  
> **证据标记：**  
> - **[代码事实]**：来自当前 CASA-CD / STR-RepNet / 官方参考仓库源码、README、正式 TEST RESULTS。  
> - **[文献事实]**：来自 2024–2026 CCF-A 主会或 IEEE TGRS / JSTARS 等权威论文、官方论文页、官方 GitHub。  
> - **[推理判断]**：基于上述事实作出的架构判断，不等同于论文已证明结论。  
> - **[待核实]**：当前没有足够权威一手来源支持，禁止当作论文事实引用。

---

# 0. 结论先行

## 0.1 对当前 CASA-STR Run1 的总判断

**结论：当前主要问题确实更像“截断后的 SHViT-S1 表征上限太低”，而不是明显的训练代码/折叠实现错误；但“训练策略完全没问题”仍然说得过头，因为 `backbone_lr_ratio=0.1` 是否导致预训练主干适配不足还没有被定量排除。**

最关键的四条证据是：

1. **[代码事实] A0 在没有 CASAA、没有 STR-rep 时就已经明显低于 ChangeViT-T。**  
   A0：94.67 / 89.90 / 82.46 / 93.70；  
   ChangeViT-T：97.75 / 91.95 / 82.48 / 94.84。  
   差值 CDD −3.08、LEVIR −2.05、SYSU −0.02、WHU −1.14 pp。  
   因而主创新尚未进入模型时，CDD/LEVIR/WHU 的大部分缺口已经存在。

2. **[代码事实] 当前真正用于 CASA-STR 的 SHViT-S1 不是完整 S1，而只是约 1.861M 的截断预训练 trunk。**  
   当前实现只保留 `patch_embed + blocks1 + blocks2`，并通过 patch-embed 中间层构造 1/4、1/8 特征；完整 SHViT-S1 后续 320-channel 最终 stage 被删除。  
   因此不能用“完整 SHViT-S1 ImageNet Top-1=72.8%”代表当前 CD trunk 的真实语义能力；**截断 trunk 的 ImageNet Top-1 未知，理论上只会更弱而不会更强。**

3. **[代码事实] Run1 的结构增益不足以填平 backbone floor。**  
   M1 相对 A0：+0.87 / +0.47 / +0.56 / +0.02 pp。  
   即使把这组增益全部保留，仍填不回 A0 在 CDD/LEVIR/WHU 上的 1–3 pp 主干缺口。

4. **[代码事实] 当前 scheduler 已经修正为按 optimizer group 的 `lr_scale` 更新，0.1× backbone LR 确实生效；折叠等价、预训练 key 继承、β=0 初始化、梯度链均已经通过 smoke。**  
   因此目前没有发现足以解释 2–3pp 性能差距的 P0 实现错误。

**但**，0.1× 主干学习率意味着 SHViT trunk 实际 base LR 为 `2e-5`。这可能是合理的预训练微调设置，也可能使一个已经被截断的 trunk 过度“近冻结”。当前只有 loss 正常并不能证明 backbone 得到了充分任务适配。因此下一轮在不改训练协议的情况下，必须增加 **stage-wise gradient/update/weight-drift 审计**。

---

## 0.2 对“直接换完整 VMamba-Tiny”的判断

**结论：完整 VMamba-Tiny 能作为“强骨干上限/导师决策参考”，但在当前 ≤5M 硬约束下不能作为最终主干。**

- **[文献事实] VMamba v2 Tiny：约 30M 参数、4.9G FLOPs、ImageNet-1K Top-1 82.6%。**
- Small：50M / 8.7G / 83.6%。
- Base：89M / 15.4G / 83.9%。
- 官方 ImageNet 预训练权重可直接获得。

你的 STR-RepNet 已经实证过：约 29.57M 的 VMamba-Tiny + TAR/DCR，`full_last2` 可以做到  
**98.42 / 91.44 / 83.45 / 95.14**。这说明强 VMamba 确实能明显抬高 CDD/WHU，并提升 SYSU；但它仍然 **LEVIR < 92.5、SYSU < 85**。所以：

> **放宽到约 30M 是获得完整 VMamba 表征的必要条件之一，却不是达到四个硬目标的充分条件。**

将 30M VMamba 硬裁到 ≤5M 需要去掉约 80% 参数，风险远高于当前任务需要；尤其**宽度裁剪会破坏预训练权重 shape，深度裁剪虽可原位保留剩余块，但大幅裁剪后无法假设 82.6% ImageNet 表征仍存在。**

---

## 0.3 本文首选新主干

**首选不是完整 VMamba-Tiny，而是 TinyViM-S（ICCV 2025）做“预训练保留优先”的受控瘦身。**

TinyViM-S 的理由：

- **[文献事实]** ICCV 2025，CCF-A；
- 官方模型仅 **5.6M / 0.9 GMAC**；
- ImageNet-1K Top-1 **79.2%（300e）/ 80.3%（1000e）**；
- 官方代码、checkpoint、训练日志公开；
- 天然 1/4、1/8、1/16、1/32 四级特征；
- 官方源码本身就使用 `selective_scan_cuda` 和四向 `CrossScan/CrossMerge`；
- 更关键：它已经采用 **“低频上下文下采样后做 SS2D + 高频/局部残差保留”** 的频率解耦设计，和你最初“完整判别位置 + 压缩上下文”的 CASA 思想高度契合；
- 相比完整 SHViT-S1 72.8%，其完整模型 ImageNet Top-1 高约 7.5pp；相比 VMamba-Tiny 82.6%，它只低约 2.3pp，但参数小一个数量级。

---

## 0.4 本文首选新创新一：不要把 CASAA 的 TopK 聚类硬搬进 SS2D

**建议把创新一从“Full-Q + compressed-K/V”抽象成更一般的：**

> **Full Dense Decision Lattice + Change-Aware Compressed Context**  
> 完整保留最终需要逐位置判别的稠密空间路径，只对提供全局上下文的 SSM 路径做结构保持的变化感知压缩。

在 TinyViM 上，最自然的实现是：

# **CAACP-SS2D：Change-Aware Asymmetric Context Pooling for SS2D**
中文：**变化感知非对称上下文聚合 SS2D**

不是 TopK 删除 token，不是把变化 token 重排到序列前面，也不是四方向分别聚类。

而是：

1. 在二维网格上先按固定 `2×2` cell 进行**结构保持压缩**；
2. 使用双时相共享 change score，在每个 cell 内做变化感知加权聚合；
3. 得到规则 `8×8` 上下文网格；
4. 再由这个规则网格生成四个方向序列；
5. SS2D 扫描后上采样回完整 `16×16`；
6. TinyViM 原有高频/局部残差路径始终保持完整。

它是“CASAA 思想在 SSM 中的正确等价物”，而不是字面复制 Q/K/V。

---

## 0.5 为什么不能做“变化 token 置前 / TopK 后再四向扫描”

**[文献事实 + 代码事实] 不推荐。**

- SS2D 的状态转移是顺序敏感的；
- `CrossScan` 的四条序列有明确二维几何意义；
- `CrossMerge` 依赖同一个 `H×W` 网格才能把四路恢复回二维；
- ICML 2026 的 STORM 明确指出：**spatially agnostic token reduction 会破坏结构化 Mamba 的二维扫描前提，并导致严重性能坍塌**；
- EfficientVMamba 采用的是固定、结构化的 atrous skip sampling，而不是任意 TopK；
- TinyViM 采用的是规则低频池化，而不是不规则删 token。

所以 Q3 的核心答案是：

> **压缩应发生在二维规则网格上，再生成四方向序列；不要四个方向各自独立 TopK，也不要先把变化 token 重排到前面。**

---

# 1. 当前 CASA-CD 仓库证据审查

## 1.1 当前主线实际状态

**[代码事实]** `train_scripts/CASA-STR/Run1/README.md` 当前标记为：

- 16/24 个完整 80K run 已完成；
- 完成：A0 / M1 / A1 / A2 × 四数据集；
- C1 Full Attention 对照被中断；
- C2 content-SAA 尚未启动；
- 当前主线暂停，原因是距离硬目标仍有约 1.3–2.5pp，准备重新调研更强 backbone / Mamba。

当前正式完成结果：

| 变体 | CDD | LEVIR | SYSU | WHU |
|---|---:|---:|---:|---:|
| A0_BASE_PLAIN | 94.67 | 89.90 | 82.46 | 93.70 |
| A1_CASAA_PLAIN | 94.54 | 89.87 | 82.49 | 93.65 |
| A2_STR_ONLY | 95.53 | 90.28 | 82.49 | 93.41 |
| M1_CASAA_STR | **95.54** | **90.37** | **83.02** | **93.72** |

硬目标：

| 数据集 | M1 | 目标 | 差距 |
|---|---:|---:|---:|
| CDD | 95.54 | 98.00 | −2.46 |
| LEVIR | 90.37 | 92.50 | −2.13 |
| SYSU | 83.02 | 85.00 | −1.98 |
| WHU | 93.72 | 95.00 | −1.28 |

---

## 1.2 当前 SHViT trunk 是什么

**[代码事实]**

`models/model/shvit_s1_trunc.py` 并不是直接调用完整官方 SHViT-S1，而是重建出其前部：

```text
Input 256×256
  ↓
patch_embed intermediate taps
  F1: 1/4,  C=32,  64×64
  F2: 1/8,  C=64,  32×32
  ↓
blocks1
  F3: 1/16, C=128, 16×16
  ↓
CASAA pair @ 1/16
  ↓
blocks2
  F4: 1/32, C=224, 8×8
```

完整官方 SHViT-S1 配置是：

```python
embed_dim   = [128, 224, 320]
depth       = [2, 4, 5]
partial_dim = [32, 48, 68]
types       = ["i", "s", "s"]
```

当前实现**没有保留完整最终 `blocks3` / 320-channel stage**。

预训练审计显示：

- retained trunk 参数约 **1,861,296**；
- retained keys **246/246 原位继承**；
- checkpoint 本身没有加载错误。

因此：

**[推理判断]** 当前问题不是“SHViT 权重没加载”，而是“为了 ≤5M 预算，保留下来的预训练网络部分本身可能不足以承担最终语义判别”。

---

## 1.3 CASAA 当前真正贡献有多大

不要只看 `M1-A0`。

用 2×2 因子设计拆开：

### CASAA 在 plain STR 下的主效应

\[
A1-A0
\]

- CDD −0.13
- LEVIR −0.03
- SYSU +0.03
- WHU −0.05

**结论：CASAA 单独几乎等于 0。**

### STR 在无 CASAA 下的主效应

\[
A2-A0
\]

- CDD **+0.86**
- LEVIR **+0.38**
- SYSU +0.03
- WHU −0.29

**结论：CDD / LEVIR 的主要增益来自 STR-rep。**

### CASAA 在 STR 已开启时的边际效应

\[
M1-A2
\]

- CDD +0.01
- LEVIR +0.09
- SYSU **+0.53**
- WHU **+0.31**

这比单独 CASAA 有意思。

对应的交互项：

\[
(M1-A2)-(A1-A0)
\]

约为：

- CDD +0.14
- LEVIR +0.12
- SYSU **+0.50**
- WHU **+0.36**

**[推理判断]** 目前更合理的说法不是“CASAA 稳定带来 +0.5~0.9pp”，而是：

> **CASAA 本身没有被证明能独立提升；它可能在 STR 已改变时空融合表征后，对 SYSU/WHU 有条件性互补。**

由于单 seed，且 C1/C2 对照尚未完成，这个“交互效应”只能作为下一轮机制假设，不能写成普适结论。

---

# 2. 先校正你提出的四个判断

## (a) “同一 TAR/DCR 下 VMamba 明显强于 SHViT，因此 backbone 是主要瓶颈”

### 结论

**方向上基本支持，但你当前给出的比较并不是严格的 apples-to-apples，不能直接把全部差值归因于 backbone。**

### 为什么

你给出的：

- VMamba `full_last2`：98.42 / 91.44 / 83.45 / 95.14
- CASA M1：95.54 / 90.37 / 83.02 / 93.72

差值恰好是：

- CDD +2.88
- LEVIR +1.07
- SYSU +0.43
- WHU +1.42

但这两组不只换了 backbone：

- VMamba：STR full、只解冻末两 stage；
- SHViT：CASAA + STR full、截断 trunk、全部 retained trunk 按 0.1× LR 微调；
- 两个 encoder 的 feature geometry / channel / pretraining / architecture 都不同。

如果更接近“STR-only 对 STR-only”：

- VMamba `full_last2` vs CASA `A2_STR_ONLY`：

| 数据集 | VMamba full_last2 | SHViT A2 | 差 |
|---|---:|---:|---:|
| CDD | 98.42 | 95.53 | +2.89 |
| LEVIR | 91.44 | 90.28 | +1.16 |
| SYSU | 83.45 | 82.49 | +0.96 |
| WHU | 95.14 | 93.41 | +1.73 |

仍然支持 VMamba 更强。

但如果比较“冻结 backbone + plain 头”：

- STR-RepNet A0_Plain frozen VMamba：97.32 / 90.09 / 81.14 / 93.55
- CASA A0：94.67 / 89.90 / 82.46 / 93.70

则：

- CDD +2.65
- LEVIR +0.19
- SYSU **−1.32**
- WHU **−0.15**

说明“VMamba 在所有数据集都天然强”并不成立。

### 最终可写结论

**[推理判断]**

> 当前最强证据支持的是：**CDD 与 WHU 对更强的预训练层次化主干特别受益；SYSU/LEVIR 的收益还受到任务适配、可训练深度和融合方式影响。当前 SHViT 截断 trunk 是主要嫌疑之一，但不是已经严格隔离出的唯一因果变量。**

---

## (b) “CASAA + STR 的 +0.5~0.9pp 可以平移到强骨干”

### 结论

**反驳“可平移”这一强表述。**

- M1 对 A0 的确 4/4 非负；
- 但 WHU +0.02 属于可忽略量级；
- A1 CASAA 单开 4 数据集约等于 A0；
- CDD/LEVIR 的收益由 STR 主导；
- CASAA 目前只显示出“与 STR 的可能交互”，不是稳定独立增益。

所以应改为：

> **[待验证假设 H-transfer]：在更强且具结构化全局建模能力的 backbone 上，变化感知上下文压缩可能与 STR 形成互补；是否保留 +0.3~0.8pp 必须重新做完整四数据集 2×2 消融。**

禁止把 SHViT Run1 的增益直接加到新 backbone 上预测最终分数。

---

## (c) “CDD 缺口最大，因为 CDD 抗伪变化、最依赖语义质量”

### 结论

**前半句有文献支持；后半句是合理推断，但还没有被你的实验直接证明。**

**[文献事实]**

CDD 是季节变化显著的数据集，包含季节、光照、背景外观差异；这些变化本身通常不应被判为真实目标变化，因此确实会产生大量 pseudo-change 干扰。

**[推理判断]**

更强语义表征有助于区分：

```text
appearance change
≠
semantic/structural change
```

所以“CDD 更依赖语义稳定性”是合理机制解释。

但：

- CDD 同时还包含尺度、道路、车辆、建筑等多类变化；
- 数据 split /增强 /边界也影响分数；
- 当前只看到 backbone 更换和解冻与 CDD 分数相关，不能推出唯一因果。

建议增加**不改变训练协议**的 post-hoc 分析：

1. 在 CDD test 上按 `FP / FN` 输出 hard sample；
2. 人工抽样 200 张；
3. 标记：
   - seasonal vegetation
   - illumination/shadow
   - snow/soil appearance
   - geometric misalignment
   - small structural change
4. 比较 A0 / M1 / ChangeViT / 新 backbone 的 FP 来源。

如果新 backbone 的主要收益来自 seasonal/illumination FP 明显下降，才真正支持这条论文叙事。

---

## (d) “训练策略健康”

### 结论

**P0 实现层面基本健康；优化适配层面仍未完全排除。**

已经支持“健康”的证据：

- SHViT checkpoint retained key 246/246；
- β=0 前向等价；
- CASAA β/qkv/proj 梯度非零；
- STR fold FP64→FP32 等价；
- real batch argmax disagreement≈0；
- loss 正常；
- scheduler 当前代码按 `lr_scale` 保留 backbone/new module LR 比例；
- 16 个 80K run 均不是明显训练崩溃。

但还缺：

- backbone 各 stage 的梯度范数；
- 实际参数 update / param ratio；
- pretrained→best 的相对 L2 drift；
- BN running stats 漂移；
- best epoch 时 backbone 是否几乎没离开预训练点。

因此本文不建议先改 LR，而建议**先测，不改协议**。

---

# 3. Q1：是训练策略问题还是 backbone 太弱？

## 一句话结论

> **主嫌疑是“截断后的 SHViT-S1 表征容量/语义深度不足”；没有发现足以解释大幅掉点的训练故障，但 0.1× backbone LR 是否造成欠适配仍是一个未排除共因。**

---

## 3.1 ImageNet 表征强度与 CD 的证据应该怎么理解

不能简单认为：

\[
\text{ImageNet Top-1 高} \Rightarrow \text{CD F1 一定高}
\]

更合理的是：

> ImageNet Top-1 是“通用预训练表征强度”的一个 proxy；在**相近多尺度接口、相近下游训练条件**下具有参考价值，但不是 CD 性能的充分条件。

### ChangeMamba 给出的直接证据

**[文献事实] ChangeMamba / MambaBCD（IEEE TGRS 2024）** 使用 ImageNet 预训练的 Visual Mamba encoder。

论文结果显示：

SYSU F1：

- Tiny 81.29
- Small 82.83
- Base 83.11

WHU F1：

- Tiny 93.33
- Small 94.06
- Base 94.19

这两组表现出较明显的容量收益。

但 LEVIR-CD+ 并不是严格单调：

- Tiny ≈ 88.0
- Small ≈ 87.8
- Base ≈ 88.4

说明：

> **更大/更高 ImageNet 精度通常能提高表示上限，但下游 CD 的融合接口、任务适配、数据特征仍然决定最终增益。**

而且 ChangeMamba 训练协议与 CASA-CD 不同：

- AdamW 1e-4；
- wd 5e-3；
- SYSU 20k，其余 50k iterations；
- 包含旋转/flip；
- LEVIR 使用 **LEVIR-CD+**，不是你当前 LEVIR-CD-256。

所以只能做机制参考，不能横向当正式 SOTA 数字。

---

## 3.2 SHViT-S1 的已知设计取舍

**[文献事实] SHViT（CVPR 2024）** 的目标就是极致消除冗余：

- 更激进 patchify；
- early stages 不做 self-attention；
- later stages 使用 single-head attention；
- 只有部分通道参与 attention；
- 用 BN/Conv friendly macro design 提高实际吞吐。

完整 SHViT-S1：

- 6.3M
- 241M FLOPs
- ImageNet-1K Top-1 72.8%

这个设计非常适合“移动端分类吞吐”，但对于你当前任务存在两个潜在冲突：

### 冲突 1：CD 是 dense prediction

二值 CD 的最终目标是：

- 小建筑；
- 狭长道路；
- 边界；
- 时相差异；
- pseudo-change 抑制。

这些需求比单标签分类更依赖：

- 多尺度语义；
- 局部细节；
- 全局上下文；
- 跨时相稳定表征。

### 冲突 2：你又进一步删除了完整 S1 的最后 stage

当前 trunk 只有 1.861M。

所以你实际上不是在比较：

```text
完整 SHViT-S1
vs
VMamba-Tiny
```

而是在比较：

```text
极度截断 SHViT-S1 prefix
vs
较完整 VMamba-Tiny
```

这个差距很可能比“72.8 vs 82.6”还大。

---

## 3.3 80K 是否不足？

没有证据表明 80K 本身不足。

80K 是固定协议，不应修改。

真正需要查的是：

> 80K 中，backbone 有没有被有效更新？

当前：

```text
new modules LR = 2e-4
backbone LR    = 2e-5
```

### 下一轮必须增加以下日志，不改变任何训练策略

每 500 或 1000 step 记录：

```text
[BACKBONE-ADAPT]
stage_name
grad_norm
param_norm
grad_norm / param_norm
optimizer_update_norm
update_norm / param_norm
relative_L2_from_pretrain
```

在 best checkpoint 记录：

\[
D_l=\frac{\|\theta_l-\theta_l^{pre}\|_2}
{\|\theta_l^{pre}\|_2+\epsilon}
\]

同时记录 BN：

```text
running_mean drift
running_var drift
```

### 判读

**[推理判断 / 启发式，不是文献硬阈值]**

如果 80K 后大多数 retained backbone 层：

- 相对 L2 drift 仍接近 0；
- update/param 长期比 new head 小几个数量级；
- 梯度也极小；

那么“0.1× LR 近似冻结”应被列为共因。

如果 backbone 有稳定梯度和明显权重漂移，而 F1 仍受限，则“表征上限不足”的证据更强。

---

## 3.4 Q1 风险与不确定性

1. ImageNet Top-1 与 CD F1 不存在可直接回归的线性关系。
2. SHViT 当前是截断版本，没有 ImageNet 验证精度。
3. VMamba upstream 的训练可见层数与 CASA 不一致。
4. 单 seed 下 0.1–0.3pp 不应过度解释。
5. 当前 C1 full-attention / C2 content-SAA 未完成，所以 CASAA 自身的“压缩无损性”在新 SHViT 主线没有完整闭环。

---

# 4. Q2：VMamba 能不能做 backbone？

## 4.1 完整 VMamba：可以，但和 ≤5M 直接冲突

### 官方 VMamba（NeurIPS 2024，CCF-A）

| Backbone | ImageNet-1K Top-1 | Params | FLOPs | 官方预训练 |
|---|---:|---:|---:|---|
| VMamba-T | **82.6** | **30M** | **4.9G** | 有 |
| VMamba-S | 83.6 | 50M | 8.7G | 有 |
| VMamba-B | 83.9 | 89M | 15.4G | 有 |

> 注：早期 VMamba v1 / 不同配置会看到约 22–28M 的 Tiny 数字；当前官方 v2 主表约 30M。论文比较时必须注明版本，不能混用。

### 可行性

- 算法上：完全可行；
- 你现有 selective scan 快照：具备工程基础；
- 参数上：**不符合 ≤5M**。

---

## 4.2 如果导师允许放宽预算，需要放宽到多少

**[推理判断]**

完整 VMamba-T + 轻量 TAR/DCR 至少需要约：

```text
≈ 30M 量级
```

才谈得上基本保留官方 VMamba-T 的完整 ImageNet 表征。

而已有 STR-RepNet：

```text
~29.57M
full_last2
CDD   98.42 ✓
LEVIR 91.44 ✗
SYSU  83.45 ✗
WHU   95.14 ✓
```

所以：

> **如果导师把参数上限放到 30M，CDD/WHU 已显示明显收益，但 LEVIR/SYSU 仍不自动过线。**

这也是为什么本文不建议为了 VMamba 本体直接放弃“极轻量”定位。

---

## 4.3 更小的 Mamba / Hybrid Mamba 候选

### 候选表

| Backbone | 年份/层级 | Params | FLOPs / GMAC | ImageNet-1K Top-1 | 预训练 | 对本课题判断 |
|---|---|---:|---:|---:|---|---|
| **TinyViM-S** | ICCV 2025 / **CCF-A** | **5.6M** | **0.9G** | **79.2 / 80.3** | 有 | **首选** |
| EfficientVMamba-T | AAAI 2025 / **CCF-A** | 6M | 0.8G | 76.5 | 有 | 备选 |
| Vim-Ti | ICML 2024 / **CCF-A** | 7M | ~1.5G | 76.1 / 78.3* | 有 | 不层次化，多尺度适配差 |
| MobileMamba-T2 | CVPR 2025 / **CCF-A** | 8.8M | 0.255G@192 | 73.6 / 76.9† | 有 | 参数仍偏大，精度不占优 |
| MSVMamba-Nano | NeurIPS 2024 / **CCF-A** | ~7M | ~0.9G | ~77.3 | 有 | 仍超预算，次选 |
| VSSD-Tiny | ICCV 2025 / **CCF-A** | 28M | 5.0G | 83.8 | 有 | 强但远超预算 |
| MambaVision-T | CVPR 2025 / **CCF-A** | 31.8M | 4.4G | 82.3 | 有 | 强但远超预算 |
| LocalVim-T | 2024 / **Preprint** | ~8M | ~1.5G | ~77.8 | 有/部分 | 层级较弱，且非主会 |
| LocalVMamba-T | 2024 / **Preprint** | ~26M | ~5.7G | ~82.7 | 有/部分 | 超预算 |

\* Vim 的 “+” 是官方额外 finer-granularity fine-tune 结果，不能和普通训练表混为一谈。  
† MobileMamba 的 † 版本使用其增强训练设定，表中必须保留该标记。

---

## 4.4 为什么首选 TinyViM-S 而不是 EfficientVMamba-T

### TinyViM-S

**[文献事实]**

- ICCV 2025；
- 5.6M；
- 0.9 GMAC；
- 79.2 / 80.3 Top-1；
- four-stage hierarchy；
- 官方 checkpoint；
- 源码包含 selective scan CUDA；
- 同时兼顾 Conv local path 和 Mamba global path。

### EfficientVMamba-T

**[文献事实]**

- AAAI 2025；
- 6M；
- 0.8G；
- 76.5 Top-1；
- official checkpoint；
- Atrous Selective Scan 非常适合做“结构保持稀疏扫描”参考。

### 选择理由

**[推理判断]**

当前最大问题是 backbone floor，不是再省 0.1G。

所以优先：

```text
TinyViM-S 80.3
>
EfficientVMamba-T 76.5
```

而不是为了更低 FLOPs 再牺牲通用预训练表征。

---

# 5. TinyViM-S 如何压进 ≤5M：预算重分配方案

## 5.1 不建议宽度裁剪

如果把通道：

```text
48 / 64 / 168 / 224
```

直接改窄，则：

- Conv shape 变化；
- SSM A/B/C/Δ 投影 shape 变化；
- pretrained checkpoint 不能原位加载；
- 只能做 channel slicing 或重新初始化。

这会重复你之前“为了轻量把预训练能力一起切掉”的问题。

**默认否决。**

---

## 5.2 推荐：只做深度裁剪 + 缩 STR head

官方 TinyViM-S：

```text
widths = [48, 64, 168, 224]
depths = [3, 3, 9, 6]
```

最后 stage（1/32）：

```text
Local
Local
Local
Local
Local
TViM / SS2D
```

推荐保留：

```text
Local[0]
Local[1]
Local[2]
TViM[5]
```

删除：

```text
Local[3]
Local[4]
```

也就是：

> **保留最终 global SS2D，只删两个重复 local block。**

### 源码级参数估算

基于官方 TinyViM-S 源码逐层公式估算：

- classifier 前 feature trunk ≈ **5.458M**；
- 1 个 224-channel LocalBlock ≈ **0.407M**；
- 删除 2 个 ≈ −0.814M；
- slim trunk ≈ **4.644M**。

然后把当前 STR hidden dim：

```text
D = 160
```

改为：

```text
D = 96
```

在 TinyViM encoder dims `[48,64,168,224]` 下，deploy TAR+DCR+head 参数公式约为：

\[
P_{head}
=
14D^2 + 2D\sum C_i + 96D + 1
\]

其中：

\[
\sum C_i=48+64+168+224=504
\]

代入 `D=96`：

\[
P_{head}\approx 235,009
\]

所以：

```text
slim trunk  ≈ 4.644M
STR head    ≈ 0.235M
CAACP       ≈ 1 scalar + 无参 score
--------------------------------
总计粗估     ≈ 4.88M
```

### 重要说明

这是**源码级预算估算，不是正式报告数字**。

实现后必须由服务器：

```text
DEPLOY-PARAMS
DEPLOY-FLOPS
```

重新实测。

由于安全余量只有约 0.12M：

> 如果机器实测 >4.95M，**先把 STR D=96 → 80**，不要继续砍预训练 backbone。

因为本文的中心判断就是“预算要优先留给 backbone”。

---

## 5.3 为什么“保留更多 backbone + 缩 decoder”比“继续砍 backbone + 保持 D160”更合理

当前证据：

- SHViT A0 backbone floor 明显；
- STR head 已经只有约 0.5M；
- STR 的作用主要是训练多分支、部署折叠，而不是依赖 D160 才成立。

因此：

**[推理判断]**

把约 0.30M 参数从 D160 decoder 移回预训练 backbone，保留一个额外 224-channel pretrained LocalBlock，更符合当前证据链。

这必须在下一轮一开始固定，不能根据单数据集结果来回搜索。

---

# 6. Q3：四向扫描 token 能否做变化感知压缩？

# 6.1 SSM 能不能吃变长序列？

### 一句话

**Selective Scan kernel 本身可以接受运行时不同的 `L`；但 VMamba/TinyViM 的二维 CrossScan/CrossMerge 语义不能把 `L` 当成任意 token bag。**

### 代码层原因

官方 TinyViM/VMamba：

```text
X: B,C,H,W
 ↓
CrossScan
 ↓
4 × (H·W) sequence
 ↓
SelectiveScan
 ↓
CrossMerge(H,W)
 ↓
B,C,H,W
```

`selective_scan_cuda` 的 `L` 是运行时维度，所以“变长”本身不是 kernel 禁止项。

但 `CrossMerge` 明确依赖：

```text
H
W
row-major
column-major
reverse row-major
reverse column-major
```

因此：

> **规则 8×8 可以扫；任意 64 个 TopK 像素不能直接被当作 8×8 扫。**

---

# 6.2 为什么 SSM 比 Attention 更怕 token 重排

Attention：

\[
Y_i=\sum_j \mathrm{softmax}(q_i^\top k_j)v_j
\]

保留完整 Q 后，压 K/V 并不会改变 query 的空间顺序。

SSM 则存在递归状态：

\[
h_t=\bar A_t h_{t-1}+\bar B_t x_t
\]

所以如果：

```text
x1,x2,x3,...,xN
```

被 TopK / score 排序成：

```text
x17,x5,x200,...
```

状态传播路径已经变了。

变化 token “置前”会改变：

- 邻接关系；
- 路径距离；
- 隐状态累计顺序；
- 四方向之间的几何对应。

这不是普通 token pruning 的小改动。

---

# 6.3 STORM 对这一问题给出了非常直接的证据

**[文献事实] STORM，ICML 2026**

论文明确指出：

> spatially agnostic token reduction 会违反 selective scanning 所需的二维结构前提，使结构增强型 Vision Mamba 出现严重性能崩溃。

STORM 的解法不是任意 TopK，而是：

- structured spatial units；
- localized constraints；
- grid topology preservation；
- neighborhood coherence。

这直接支持本文的判断：

> **如果要在 VMamba 上做 CASA，压缩必须先尊重二维网格，再谈 change awareness。**

---

# 6.4 应该“每个方向分别压缩”吗？

**不建议。**

假设四方向分别产生自己的 TopK：

```text
left-right  : I_lr
right-left  : I_rl
up-down     : I_ud
down-up     : I_du
```

则四路 context token 不再对应同一二维结构。

这会引入：

- 不一致 token 集；
- 不一致邻接；
- CrossMerge 很难严格逆映射；
- 方向差异和内容选择差异混在一起。

### 推荐

先在二维平面生成：

```text
C ∈ R^{B×C×H'×W'}
```

再：

```text
CrossScan(C)
```

由同一规则网格产生四个方向。

即：

> **先压二维 context lattice，再展平四方向；而不是四方向各自压 token。**

---

# 6.5 可以把四方向 token 合并后再压吗？

也不建议。

四方向不是冗余复制，而是 SS2D 的主要归纳偏置之一。

先合并会消掉：

- LR / RL；
- UD / DU；

的方向状态差异。

如果要降低冗余，更安全的是降低：

```text
H×W
```

而不是先把 direction 维度抹掉。

---

# 7. 已有先例与 novelty 风险

## 7.1 EfficientVMamba：结构化稀疏扫描

**[文献事实] AAAI 2025**

EfficientVMamba 提出 Atrous Selective Scan：

- 通过 skip sampling 降低全局扫描成本；
- 保持固定空间规律；
- 同时保留卷积分支的局部信息。

它说明：

> “结构化下采样后再扫”是可行先例。

但是它不是双时相，也不是 change-aware。

---

## 7.2 TinyViM：低频压缩扫描 + 高频保留

**[文献事实] ICCV 2025**

TinyViM 的 SS2D 本身已经：

1. 把一部分通道作为低频/global branch；
2. 对低频 branch 做规则 pooling；
3. 在更小网格上做四向 selective scan；
4. 保留高频 residual/local branch；
5. 上采样后恢复原空间输出。

这几乎天然给出了：

> full dense local decision path + compressed global context

因此 TinyViM 比 VMamba-T 更适合把 CASA 思想“自然地”落到 SSM。

---

## 7.3 CAM-CD：最需要规避的直接 novelty 碰撞

**[文献事实] CAM-CD: Change-Aware Mamba for Remote Sensing Change Detection，IEEE JSTARS 2026（SCI，非 CCF-A）**

官方 GitHub `xjkgis/CAM-CD` 的 `VSSBlock_CAAS.py` 可以看到：

- 先进行完整 `cross_scan_fn`；
- 完整 `selective_scan_fn`；
- 生成 4 路扫描输出 `ys`；
- Change Prior Module 产生四方向权重；
- 用 change prior **重新加权四方向 scan response**；
- 再 `cross_merge_fn`。

也就是说 CAM-CD 已经做了：

> **change-aware directional scan response reweighting**

所以以下想法不适合作为你的主创新：

```text
change score → 四向权重
change score → 哪个方向更重要
change-aware scanning gate
change prior → scan output modulation
```

这些已经高度接近 CAM-CD。

---

## 7.4 Mamba-CD 2026 的碰撞

**[文献事实] Mamba-CD，IEEE JSTARS 2026（SCI，非 CCF-A）**

题目已经明确：

> Change Region-Aware Attention + Recursive Context Refinement

即使它不等价于本文的 context pooling，也说明：

> “加一个 change region-aware 模块”本身已经不是足够强的新颖性表述。

---

# 8. 首选创新设计：CAACP-SS2D

# 8.1 名称

**CAACP-SS2D**  
**Change-Aware Asymmetric Context Pooling for SS2D**  
中文：**变化感知非对称上下文聚合 SS2D**

核心原则：

> **完整保留 dense decision lattice，只替换 TinyViM 原有“均匀低频压缩”的内容选择方式。**

---

# 8.2 放在哪里

只改 TinyViM-S 的：

```text
Stage 3（0-based index=2）
1/16
final TViMBlock
```

而不是所有 stage 全改。

原因：

- 1/4、1/8 token 太多，干预风险大；
- 1/32 已经只有 8×8，没必要再压；
- 1/16 恰好是 16×16=256 token；
- 2×2 pool 后正好 8×8=64 context token；
- 与你原 CASAA 的 N=256 / K=64 完全对应；
- 单点修改最利于归因。

---

# 8.3 原 TinyViM 的低频路径

简化写：

\[
X_l \in \mathbb{R}^{B\times C_l\times16\times16}
\]

原模型：

\[
C_{avg}=\mathrm{AvgPool}_{2\times2}(X_l)
\]

得到：

\[
C_{avg}\in\mathbb{R}^{B\times C_l\times8\times8}
\]

再：

\[
R=X_l-\mathrm{Up}(C_{avg})
\]

\[
G=\mathrm{Up}(\mathrm{SS2D}(C_{avg}))
\]

\[
Y_l=G+R
\]

因此输出仍为完整 16×16。

---

# 8.4 变化感知上下文聚合

对双时相：

\[
X_l^A,\; X_l^B
\]

计算参数自由 score：

\[
s_i=1-\cos(X_{l,i}^A,X_{l,i}^B)
\]

在每张图内做 rank normalization：

\[
\hat s_i=\mathrm{ranknorm}(s_i)
\]

对每个固定 2×2 cell \(\Omega_j\)：

\[
w_{j,i}=
\frac{\epsilon+\hat s_i}
{\sum_{k\in\Omega_j}(\epsilon+\hat s_k)}
\]

两个时相共享同一组权重：

\[
C_{ca,j}^A=\sum_{i\in\Omega_j}w_{j,i}X_{l,i}^A
\]

\[
C_{ca,j}^B=\sum_{i\in\Omega_j}w_{j,i}X_{l,i}^B
\]

这样：

- 疑似变化位置在 cell 内贡献更高；
- 稳定背景被强聚合；
- A/B 使用相同空间 assignment；
- T1/T2 交换 score 不变；
- 输出仍是规则 8×8 lattice；
- 不进行任意 TopK；
- 不破坏四向 CrossScan 几何。

---

# 8.5 如何 100% 兼容 TinyViM 预训练初态

这是本方案最关键的一点。

不要直接把 `AvgPool` 换掉。

定义：

\[
C_\beta=C_{avg}+\beta(C_{ca}-C_{avg})
\]

初始化：

\[
\beta=0
\]

则 epoch 0：

\[
C_\beta=C_{avg}
\]

**完全恢复官方 TinyViM 原功能。**

然后：

\[
R=X_l-\mathrm{Up}(C_\beta)
\]

\[
G=\mathrm{Up}(\mathrm{SS2D}(C_\beta))
\]

\[
Y_l=G+R
\]

训练后 β 自动决定是否需要 change-aware context。

### 梯度是否会死？

不会。

\[
\frac{\partial C_\beta}{\partial\beta}
=
C_{ca}-C_{avg}
\]

只要两者不完全相等，β 在初始化时就有梯度。

这比把新 Conv 权重全零更安全。

---

# 8.6 为什么这仍然是“非对称建模”

Transformer 版：

```text
Full Query
Compressed K/V
```

SSM 没有独立 Q/K/V。

所以不能硬说“Full Query”。

本文新的统一表述应该是：

```text
Full Dense Decision Lattice
+
Compressed Context Lattice
```

TinyViM 中：

- full dense path：16×16 local/high-frequency residual；
- context path：8×8 SS2D global context；
- 只改变 context 的聚合；
- 最终空间输出仍 16×16。

因此核心思想没有变：

> **不压最终判别位置，只压提供上下文的信息。**

只是从 Attention 版的 Q/K/V 非对称，推广成 SSM 版的 **dense/context path 非对称**。

这比强行把 Mamba 叫作 token attention 更严谨。

---

# 8.7 与已有工作的实质区别

### vs SAT / 原 CASAA

SAT/CASAA：

```text
full Q × compressed K/V
```

CAACP：

```text
full dense residual path
+
change-aware compressed 2D context
→ SS2D
```

没有 Q/K/V。

---

### vs EfficientVMamba

EfficientVMamba：

```text
固定 atrous skip pattern
```

CAACP：

```text
固定 cell topology
+
双时相 change-aware cell aggregation
```

空间结构固定，但保留的信息由双时相变化决定。

---

### vs TinyViM

TinyViM：

```text
uniform average low-frequency pooling
```

CAACP：

```text
change-aware shared weighted pooling
```

并且 β=0 精确回到 pretrained TinyViM。

---

### vs STORM

STORM：

- 通用单图 token reduction；
- topology-aware；
- training-free；
- 目标是让 token pruning 不破坏 Mamba。

CAACP：

- 双时相 CD；
- change-aware；
- 固定 cell；
- 直接改变 SS2D 的 context content；
- dense decision lattice 始终完整。

碰撞风险：**中等**，需要论文投稿前继续检索 2026 新工作。

---

### vs CAM-CD

CAM-CD：

```text
full scan
→ 4 directional response
→ change prior directional reweight
```

CAACP：

```text
change-aware 2D context aggregation
→ smaller regular lattice
→ four-direction scan
```

一个是**scan 后方向响应加权**；一个是**scan 前上下文内容压缩**。

这是当前最关键的新颖性边界。

---

# 9. 备选创新：CA-ΔSSM（只作备选，不建议主线）

可以保持完整 L，使用变化 score 调制：

- Δ；
- input gate；
- state update strength。

例如：

\[
\Delta_t'=\Delta_t\cdot(1+\alpha\hat s_t)
\]

初始化 α=0。

优点：

- 顺序完全不变；
- 几乎零参数；
- 易改官方 CUDA 前后的 tensor。

问题：

1. Mamba 本身的核心就是 input-dependent selective state update；
2. CAM-CD 已经做 change-aware scan modulation；
3. “change score 调 SSM 状态” novelty 风险很高；
4. 不再体现“压缩冗余背景”的原 CASA 主线。

所以：

> **仅作为 CAACP 失败后的机制对照，不建议作为论文主创新。**

---

# 10. 新主线架构建议

# **CASA-Mamba-STR / CASA-TViM-STR**

建议工作名：

**CASA-TViM-STRNet**

结构：

```text
A/B
 │
 ├──────────── shared TinyViM-S-Slim ────────────┐
 │                                               │
 │ Stage1 → F1 1/4,  C48                         │
 │ Stage2 → F2 1/8,  C64                         │
 │ Stage3 prefix → 1/16, C168                    │
 │              │                                │
 │       CAACP-SS2D final TViM                   │
 │       full dense 16²                           │
 │       context 8²=64                           │
 │              ↓                                │
 │             F3                                │
 │                                               │
 │ Slim Stage4: Local×3 + final TViM             │
 │              ↓                                │
 │             F4 1/32, C224                     │
 └───────────────────────────────────────────────┘
          ↓ per-time F1/F2/F3/F4
 MultiScale TAR, encoder_dims=(48,64,168,224), D=96
          ↓
 DCR decoder D=96
          ↓
 1×1 head
          ↓
 sigmoid 256×256
```

---

# 11. 训练图与推理图

## 11.1 训练图

```text
TinyViM-S-Slim pretrained
  +
CAACP beta scalar
  +
TAR multi-branch
  +
DCR multi-branch
```

Loss：

```text
BCE + Dice
```

不新增：

- auxiliary loss；
- distillation loss；
- edge loss；
- router supervision；
- teacher。

---

## 11.2 推理图

```text
TinyViM-S-Slim
+
CAACP-SS2D（真实推理模块）
+
folded TAR
+
folded DCR
+
head
```

CAACP 不删除，因为它本身就是创新一。

TAR/DCR：

```text
train multi-branch
→ deploy single Conv/DWConv
```

---

# 12. 为什么暂时不推荐蒸馏

蒸馏理论上可以：

```text
large VMamba teacher
→ tiny student
```

但当前固定协议明确：

```text
BCE + Dice
```

如果加入 KD：

- feature KD；
- logit KD；
- teacher consistency；

都会改变 objective。

这与本课题“不改 loss、不把训练技巧当创新”的纪律冲突。

所以：

> **当前不把蒸馏列入主方案。**

除非导师明确修改“固定训练协议”。

---

# 13. 代码怎么改

## 13.1 新增文件

### `models/model/tinyvim_s_slim.py`

来源：

```text
xwmaxwma/TinyViM
```

只提取必要组件：

- Conv2d_BN
- RepDW
- LocalBlock
- TViMBlock
- SS2D
- CrossScan
- CrossMerge
- TinyViM-S backbone

不要把 classification head 带进最终模型。

实现：

```python
out_indices = [stage1, stage2, stage3, stage4]
```

输出：

```text
[48,64,168,224]
```

Slim Stage4：

```text
orig index 0
orig index 1
orig index 2
orig index 5 (TViM)
```

加载器必须显式做：

```text
old key -> new key
```

并打印 retained key 数与 SHA/checksum。

---

## 13.2 新增 `models/model/layers/caacp_ss2d.py`

建议提供：

```python
rank_normalize_2d(score)
change_score_cosine_2d(x1, x2)
change_weighted_pool2x2(x, score)
CAACPSS2DPair
```

注意：

- score routing 可以 `no_grad`；
- feature weighted sum 必须对 feature 保留梯度；
- A/B 使用同一 score；
- score 必须严格 T1/T2 symmetric；
- 不要 TopK；
- 不要改变 cell 顺序；
- 不要给四方向分别选 token。

---

## 13.3 修改 TinyViM stage3 final TViM

不要重新写一个完全独立随机 SS2D。

应复用 pretrained SS2D 的：

- in_proj；
- local branch；
- x_proj；
- dt_proj；
- A_logs；
- Ds；
- out_proj。

只改：

```python
c_avg = pool(x_low)
```

为：

```python
c_avg = pool(x_low)
c_ca  = change_weighted_pool(x_low, score)
c     = c_avg + beta * (c_ca - c_avg)
```

然后保持原 scan。

这是保证预训练兼容性的关键。

---

## 13.4 新增 `models/model/casa_tvim_str_net.py`

负责 paired data flow：

```text
cat(A,B) as 2B
→ shared BN backbone blocks
→ 到 CAACP 前 split
→ compute shared score
→ paired CAACP
→ cat 2B
→ remaining backbone
→ split features
→ TAR/DCR
```

### 重要

BN 层尽量继续使用：

```text
2B concatenate forward
```

而不是：

```python
fa = backbone(A)
fb = backbone(B)
```

分开两次 forward。

否则同一 shared Siamese backbone 会在一个 iteration 内被两次不同 batch statistics 更新，和预训练/现有实现不一致。

---

## 13.5 TAR / DCR

复用当前：

```text
str_tar.py
str_dcr.py
str_reparam.py
```

只改：

```python
encoder_dims=(48,64,168,224)
dim=96
```

不要在同一轮再加入：

- BOTR；
- NSCR；
- PFDR；
- MPCR；
- TASS；
- EdgeGate。

本轮只证明：

```text
更强 tiny Mamba backbone
+
CAACP
+
已有 STR
```

避免模块堆叠。

---

## 13.6 `models/train.py`

增加：

```text
--arch casa_tvim_str
--tinyvim_pretrained_weight_path
--caacp 0/1
--str_dim 96
--rep_mode plain/full
```

仍保持：

```text
BCE+Dice
Adam
2e-4
poly
80K
batch16
seed16
test-as-val
```

不改变现有数据协议。

---

# 14. 预训练兼容验证：必须比 SHViT 更严格

实现后先跑以下审计。

## T0 checkpoint 身份

记录：

```text
filename
file size
sha256
source release
```

---

## T1 retained key 审计

必须打印：

```text
total pretrained keys
retained backbone keys
exact-shape loaded keys
missing expected keys
unexpected keys
intentionally dropped keys
```

要求：

```text
retained expected keys 100% exact load
```

---

## T2 tensor checksum

随机抽取：

- stem；
- stage1 Local；
- stage2 TViM；
- stage3 SS2D A_logs；
- stage4 retained Local；
- final SS2D；

逐 tensor 检查：

```text
max_abs_diff == 0
```

---

## T3 β=0 功能回归

同一个 slim backbone：

```text
original avg-pool SS2D
vs
CAACP beta=0
```

必须：

```text
max_abs_diff <= FP32 tolerance
```

理想 CPU/FP64 reference 可到 1e-7~1e-6。

---

## T4 gradient

验证：

```text
beta.grad != 0
pretrained SS2D grad != 0
upstream feature grad != 0
```

---

## T5 temporal swap

输入：

```text
(A,B)
(B,A)
```

change score 必须：

```text
exact/near exact equal
```

最终模型若任务定义对时相交换应保持不变，则输出也应做 swap consistency smoke。

---

# 15. selective_scan CUDA / RTX 5090 注意事项

你已有 VMamba 可运行快照，这是优势。

但 TinyViM 官方源码使用自己的：

```text
selective_scan_cuda
```

版本。

不要直接假设：

```text
旧 wheel
+
PyTorch 2.14
+
CUDA 13.2
+
sm_120
```

一定兼容。

推荐：

1. **优先复用当前 RSML-3 已验证可运行的 selective scan kernel**；
2. 对 TinyViM wrapper 做接口适配；
3. 跑：
   - forward；
   - backward；
   - AMP；
   - batch16；
   - 256²；
4. 检查：
   - NaN；
   - illegal memory access；
   - backward consistency；
   - nrows；
   - force_fp32。

不要为了“官方源码完全原样”牺牲当前已验证的 5090 kernel 环境。

---

# 16. 参数 / FLOPs / latency 审计

## 参数

正式结果必须报告：

```text
TRAIN-GRAPH PARAMS
TRAINABLE PARAMS
DEPLOY PARAMS
```

≤5M 只看：

```text
DEPLOY PARAMS
```

但训练参数也应透明报告。

---

## FLOPs

Mamba 的 CUDA selective scan 经常不被 fvcore 完整识别。

所以必须同时报告：

```text
fvcore supported FLOPs
unsupported_ops
SSM analytical FLOPs
```

不能在 unsupported selective scan 时把 fvcore 数字直接称作“总 FLOPs”。

---

## 延迟

虽然论文硬约束不是 latency，但既然主张 efficient scan，建议增加：

```text
RTX5090
batch1
256×256 A/B pair
warmup 100
measure 500
median latency
P90 latency
peak memory
```

这是补强“轻量”故事，不改变训练。

---

# 17. 下一轮实验：最小 2×2 因子设计

所有正式实验仍然：

```text
CDD
LEVIR
SYSU
WHU
```

四个数据集完整 80K。

## Phase A：先锁 backbone

### B0：TinyViM-S-Slim + plain STR graph

```text
CAACP = off
rep   = plain
```

4 个数据集全部跑完。

它的作用不是消融，而是回答：

> “这个 ≤5M 新 backbone floor 到底够不够？”

### Backbone 放行判据

**建议用相对判据，不用单数据集小波动：**

相对当前 SHViT A0：

- 四数据集 Macro F1 至少 +0.8pp；
- CDD / LEVIR / WHU 至少 2/3 提升 ≥1.0pp；
- SYSU 不允许下降 >0.3pp；
- deploy ≤5M。

如果做不到：

> 不要再花 12 个 80K 给 CAACP/STR，直接判定 TinyViM-S-Slim 仍不足以解决 backbone floor。

这仍满足“每个实验完整跑四数据集”。

---

## Phase B：2×2 消融

| Variant | CAACP | STR-rep | 作用 |
|---|---|---|---|
| A0_TVIM_PLAIN | ✗ | ✗ | 新 backbone baseline |
| A1_CAACP | ✓ | ✗ | 创新一 |
| A2_STR | ✗ | ✓ | 创新二 |
| M1_FULL | ✓ | ✓ | 完整方法 |

共：

```text
4 variants × 4 datasets = 16 个 80K
```

其中 A0 已在 Phase A 完成，所以若放行，还需 12 个。

---

# 18. 创新一成功/失败判据

CAACP 相对 A0：

### 成功

建议：

- 3/4 数据集 F1 ≥ +0.30pp；
- 另一个数据集不低于 −0.10pp；
- IoU 同方向；
- CDD 或 SYSU 至少一个 ≥ +0.40pp；
- β 最终明显离开 0；
- change-aware pool 的 context 与 avg-pool 有非零差异。

### 失败

- 4 个数据集平均增益 <0.15pp；
- 或 ≥2 数据集下降 >0.20pp；
- 或 β 长期回到≈0；
- 或新 module 主要只提高 Precision、Recall 大幅下降，表现为简单 confidence sharpening。

失败就不要再加 score head / 新 loss 去救。

---

# 19. STR 在新 backbone 上如何判断

A2 vs A0：

如果仍出现：

```text
CDD / LEVIR positive
SYSU / WHU variable
```

说明此前 STR 的数据集依赖性跨 backbone 保持。

如果 A2 全面失效：

> 说明 STR 的增益并不具备 backbone-independent transferability，应重新审视“第二创新点固定保留”的必要性。

这一步非常重要。

不能因为 STR 是导师原始思路就默认它一定在新 backbone 上继续有效。

---

# 20. 完整模型 M1 的成功标准

最终硬目标不变：

```text
CDD   >= 98
LEVIR >= 92.5
SYSU  >= 85
WHU   >= 95
Deploy Params <= 5M
```

同时：

```text
IoU 与 F1 同方向
reparam equivalence 通过
pretrained retained keys 100%正确
```

如果模型只做到：

```text
97.8 / 92.3 / 84.8 / 95.1
```

从论文研究角度已经非常接近，但按你的课题约定仍然是：

```text
HARD TARGET FAIL
```

不要事后降低门槛。

---

# 21. “预期收益区间”应该怎么写才科学

这里不能给一个假装精确的预测。

以下只是**工程预期/立项阈值，不是文献事实**。

## Backbone 替换

TinyViM-S-Slim vs 当前 SHViT A0：

```text
目标区间：
CDD/LEVIR/WHU  +0.8 ~ +2.0pp
SYSU           +0.0 ~ +1.0pp
```

依据：

- full TinyViM-S ImageNet 表征明显强于 full SHViT-S1；
- 当前 SHViT 又是强截断版本；
- 但 TinyViM 也需要裁 2 个 LocalBlock，CD 实际收益未知。

---

## CAACP

相对 TinyViM A0：

```text
目标：
平均 +0.2 ~ +0.8pp
```

不是承诺。

如果它需要 +2pp 才能救 backbone，说明主干本身仍选错了。

---

## STR

参考当前 CASA-STR：

```text
CDD / LEVIR:
历史上约 +0.4 ~ +0.9pp

SYSU / WHU:
不稳定
```

不能直接相加。

---

# 22. 为什么不建议继续当前 C1/C2 再跑完后才换 backbone

当前 C1/C2 的科学价值仍存在：

- full attention；
- content-SAA；
- 可进一步区分“压缩 vs change-aware”。

但当前 hard target 差距主要来自 A0 backbone floor。

所以优先级上：

```text
新 backbone feasibility
>
继续补 8 个 SHViT 对照
```

已有 C1 checkpoint 保留即可。

如果未来论文最终仍使用 SHViT 版 CASAA，则再恢复 C1/C2。

如果最终转 TinyViM/SSM，C1/C2 不再是最终架构的核心消融。

---

# 23. Mamba-CD 文献对你的直接启示

## ChangeMamba — TGRS 2024

**文献事实：**

- 最早系统把 Visual Mamba 用于 RS change detection 的代表作之一；
- ImageNet pretrained VMamba encoder；
- MambaBCD Tiny/Small/Base；
- SYSU 随模型变大明显提高；
- WHU 同样有容量收益；
- Tiny BCD 模型仍约 17.13M / 45.74G，远不满足 ≤5M。

**启示：**

> Mamba 的 global context 对 CD 有价值，但“用 Mamba”本身不等于“轻量”。

---

## CDMamba — TGRS 2025

**文献事实：**

题目直接强调：

> Incorporating Local Clues Into Mamba

论文动机指出，单纯强化 global scan 会忽视 dense prediction 的局部信息。

**启示：**

这与 TinyViM 的设计高度一致：

```text
global SSM
+
local/high-frequency path
```

也支持本文不选择“只有压缩 scan、丢 dense path”的方案。

---

## ST-Mamba — TGRS 2025

核心：

- Mamba feature extraction；
- spatio-temporal synergistic module；
- background feature unification；
- multi-scale/channel fusion。

**启示：**

“抑制 pseudo-change / 统一稳定背景”已经是 Mamba-CD 中重要主题。

因此 CASA 的论文表述必须聚焦：

> **冗余背景上下文如何被结构保持地压缩**

而不是泛泛地说“让 Mamba 更关注变化”。

---

## CAM-CD — JSTARS 2026

是当前 novelty 最危险的近邻。

必须在 Related Work 中明确对比：

```text
CAM-CD:
change prior → directional response reweight

Ours:
bi-temporal score → structured context aggregation before CrossScan
dense local path unchanged
```

---

# 24. 文献清单（2024–2026 主证据）

## A. 视觉 Mamba / 高效 backbone

### 1. VMamba: Visual State Space Model
- 年份：2024
- Venue：NeurIPS 2024，**CCF-A**
- Tiny：82.6 Top-1 / 30M / 4.9G（当前官方 v2 主表）
- 官方论文：https://proceedings.neurips.cc/paper_files/paper/2024/hash/baa2da9ae4bfed26520bb61d259a3653-Abstract.html
- 官方代码：https://github.com/MzeroMiko/VMamba
- 权重：官方 release 可下载

### 2. TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba
- 年份：2025
- Venue：ICCV 2025，**CCF-A**
- TinyViM-S：5.6M / 0.9G / 79.2（300e）/ 80.3（1000e）
- 官方论文：https://openaccess.thecvf.com/content/ICCV2025/html/Ma_TinyViM_Frequency_Decoupling_for_Tiny_Hybrid_Vision_Mamba_ICCV_2025_paper.html
- 官方代码：https://github.com/xwmaxwma/TinyViM
- 权重：官方 repo 表格提供

### 3. EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba
- 年份：2025
- Venue：AAAI 2025，**CCF-A**
- T：6M / 0.8G / 76.5 Top-1
- S：11M / 1.3G / 78.7
- B：33M / 4.0G / 81.8
- 官方论文：https://ojs.aaai.org/index.php/AAAI/article/view/32690
- 官方代码：https://github.com/TerryPei/EfficientVMamba

### 4. Vision Mamba (Vim)
- 年份：2024
- Venue：ICML 2024，**CCF-A**
- Vim-Ti：7M / 76.1 Top-1；官方 finer-granularity variant 78.3
- 官方论文：https://proceedings.mlr.press/v235/zhu24f.html
- 官方代码：https://github.com/hustvl/Vim

### 5. Multi-Scale VMamba
- 年份：2024
- Venue：NeurIPS 2024，**CCF-A**
- Tiny 约 82.8–83.0 Top-1；官方还提供 Nano/Micro
- 官方论文：https://proceedings.neurips.cc/paper_files/paper/2024/hash/2d69e771d9f274f7c624198ea74f5b98-Abstract.html
- 官方代码：https://github.com/YuHengsss/MSVMamba

### 6. MambaVision
- 年份：2025
- Venue：CVPR 2025，**CCF-A**
- T：82.3 / 31.8M / 4.4G
- 官方论文：https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html
- 官方代码：https://github.com/NVlabs/MambaVision

### 7. VSSD
- 年份：2025
- Venue：ICCV 2025，**CCF-A**
- Tiny：83.8 / 28M / 5.0G
- 官方论文：https://openaccess.thecvf.com/content/ICCV2025/html/Shi_VSSD_Vision_Mamba_with_Non-Causal_State_Space_Duality_ICCV_2025_paper.html
- 官方代码：https://github.com/YuHengsss/VSSD

### 8. MobileMamba
- 年份：2025
- Venue：CVPR 2025，**CCF-A**
- T2：8.8M / 255M @192 / 73.6；增强训练版 76.9
- 官方论文：https://openaccess.thecvf.com/content/CVPR2025/html/He_MobileMamba_Lightweight_Multi-Receptive_Visual_Mamba_Network_CVPR_2025_paper.html
- 官方代码：https://github.com/lewandofskee/MobileMamba

### 9. STORM: Spatial-Aware Reduction Framework
- 年份：2026
- Venue：ICML 2026，**CCF-A**
- 关键结论：结构增强 Vision Mamba 对 spatially-agnostic token reduction 极敏感；必须保持二维拓扑/邻域一致性
- 官方论文：https://proceedings.mlr.press/v306/lv26e.html

---

## B. 遥感变化检测 Mamba

### 10. ChangeMamba
- 年份：2024
- Venue：IEEE TGRS 2024，**SCI 权威期刊**
- DOI：10.1109/TGRS.2024.3417253
- MambaBCD：
  - Tiny：17.13M / 45.74G
  - Small：49.94M / 114.82G
  - Base：84.70M / 179.32G
- SYSU F1：81.29 / 82.83 / 83.11
- WHU F1：约 93.33 / 94.06 / 94.19
- LEVIR 使用 **LEVIR-CD+**，不可与本课题 LEVIR-CD-256 直接横比
- arXiv：https://arxiv.org/abs/2404.03425
- 官方 GitHub：https://github.com/ChenHongruixuan/ChangeMamba

### 11. CDMamba
- 年份：2025
- Venue：IEEE TGRS 2025，**SCI 权威期刊**
- DOI：10.1109/TGRS.2025.3545012
- 核心：将 local clues 引入 Mamba，平衡 global/local dense representation
- 官方代码：https://github.com/zmoka-zht/CDMamba
- **精确四数据集 F1：本文不引用未经原表核验的二手数字。**

### 12. ST-Mamba
- 年份：2025
- Venue：IEEE TGRS 2025，**SCI 权威期刊**
- DOI：10.1109/TGRS.2025.3579617
- 核心：Mamba encoder + spatio-temporal background unification + multi-scale/channel fusion
- **精确参数/F1：待从 IEEE 正文表格逐项核验后再进入论文对比表。**

### 13. CAM-CD: Change-Aware Mamba for Remote Sensing Change Detection
- 年份：2026
- Venue：IEEE JSTARS 2026，**SCI，非 CCF-A**
- DOI：10.1109/JSTARS.2026.3713960
- 核心：Change Prior + CA-SS2D，重加权四方向 scan responses
- 官方 GitHub：https://github.com/xjkgis/CAM-CD
- 这是本文 innovation novelty 的重点近邻。

### 14. Mamba-CD
- 年份：2026
- Venue：IEEE JSTARS 2026，**SCI，非 CCF-A**
- DOI：10.1109/JSTARS.2026.3652135
- 核心：Change Region-Aware Attention + Recursive Context Refinement
- 与“泛化的 change-aware Mamba”叙事存在概念碰撞。

---

## C. 原 CASA 思路锚点

### 15. SAT: Selective Aggregation Transformer for Image Super-Resolution
- 年份：2026
- 状态：CVPR 2026 / 项目材料中的论文版本
- arXiv：https://arxiv.org/abs/2604.07994
- 官方代码：https://github.com/PhuTran1005/SAT
- 关键思想：完整 Query、聚合 K/V。

### 16. SHViT
- 年份：2024
- Venue：CVPR 2024，**CCF-A**
- S1：6.3M / 241M / 72.8 Top-1
- 官方论文：https://openaccess.thecvf.com/content/CVPR2024/html/Yun_SHViT_Single-Head_Vision_Transformer_with_Memory_Efficient_Macro_Design_CVPR_2024_paper.html
- 官方代码：https://github.com/ysj9909/SHViT

---

# 25. 用户点名但需要纠正/待核实的文献名

## “RSMamba”

**需要纠正。**

常见的 RSMamba：

> RSMamba: Remote Sensing Image Classification With State Space Model

主要是**遥感图像分类**，不是 BCD 方法。

另有面向 dense prediction 的 RS-Mamba，但也不是你当前所说的“BCD 方法”。

所以不要在论文中写：

```text
RSMamba 是 Mamba-based change detection SOTA
```

除非明确指出哪一篇。

---

## “VMambaCD”

**[待核实]**

当前高可信检索没有发现一个应当被直接称为“VMambaCD”的权威主论文，与 ChangeMamba / MambaBCD 名字可能混淆。

查询关键词：

```text
"VMambaCD" remote sensing change detection
"VMamba-CD"
visual mamba change detection binary
```

论文正式写作前必须拿到 DOI / publisher page 再引用。

---

# 26. 立即执行顺序

## 第 1 步：不要先训练

下载并固定：

```text
TinyViM-S official 1000e checkpoint
```

记录 SHA256。

---

## 第 2 步：本地实现 TinyViM-S-Slim

固定结构：

```text
[48,64,168,224]
[3,3,9,(3 Local + final TViM)]
```

只删 stage4 两个 LocalBlock。

---

## 第 3 步：把 STR D 固定为 96

先机器测：

```text
deploy params
```

必须：

```text
<5M
```

若 >4.95M：

```text
D 96 → 80
```

不要再砍 backbone。

---

## 第 4 步：先做纯 backbone smoke

检查：

- feature shapes；
- exact retained checkpoint；
- pair 2B BN；
- 5090 selective scan forward/backward；
- AMP；
- no NaN。

---

## 第 5 步：实现 CAACP

只改：

```text
stage3(index2) final TViM SS2D
```

先不碰其他 scan。

---

## 第 6 步：β=0 regression

必须证明：

```text
CAACP(beta=0)
==
official uniform-pool slim TinyViM
```

---

## 第 7 步：STR 接入

复用已经验证的：

```text
TAR
DCR
reparam
```

不改它们的机制。

---

## 第 8 步：完整 smoke + dry run

必须验证：

- loss finite；
- β nonzero grad；
- SSM grads；
- STR aux grads；
- fold；
- real batch binary disagreement；
- last.pth resume；
- manifest；
- pretrain SHA。

---

## 第 9 步：A0 四数据集

先把：

```text
TinyViM-S-Slim + plain
```

4 个数据集都完整跑完。

---

## 第 10 步：决定是否继续 2×2

若 backbone floor 明显提高，再启动：

```text
A1 CAACP
A2 STR
M1 Full
```

每个都是四数据集完整 80K。

---

# 27. 仍需补充的证据

1. **TinyViM-S-Slim 的真实 ImageNet Top-1。**  
   当前没有重新做 ImageNet 验证，因此不能把完整 80.3% 写到 slim 版本头上。

2. **TinyViM-S-Slim 的正式 deploy params / FLOPs。**  
   本文 4.88M 是源码公式估算；服务器测量才是正式数值。

3. **TinyViM 在这四个 CD 数据集上的 A0 结果。**  
   这是决定新路线是否成立的第一关键实验。

4. **backbone 0.1× LR 的适配强度。**  
   增加梯度/update/drift 日志即可，不需要改协议。

5. **CDD false-positive 来源。**  
   如果要写“抗伪变化/语义稳定性”故事，必须做 case audit。

6. **CAM-CD 与投稿时 2026 新文献的 novelty 再检索。**  
   特别关键词：
   ```text
   change-aware selective scan
   change-aware token reduction mamba
   bi-temporal token merging mamba
   spatial-aware mamba pruning
   remote sensing change detection sparse scan
   adaptive pooling selective scan
   ```

---

# 28. 最终推荐

## 主方案

# **TinyViM-S-Slim + CAACP-SS2D + TAR/DCR**

### 参数目标

```text
TinyViM slim trunk  ≈ 4.64M
TAR/DCR D96         ≈ 0.235M
CAACP               ≈ 0M
--------------------------------
Deploy              ≈ 4.88M  （待机器核验）
```

### 机制故事

> 高分辨率二时相遥感中，逐位置变化判别需要完整空间格，而全局上下文存在大量稳定背景冗余。不同于 Transformer 中 Full-Q/Compressed-KV，SSM 不存在独立 Q/K/V，且四向扫描依赖二维拓扑。因此我们在 TinyViM 的频率解耦 SS2D 中保留完整 dense local lattice，仅对 global context lattice 进行双时相变化感知、拓扑保持的 cell-wise 聚合，再进行四向 selective scan。训练时以 β=0 精确继承 ImageNet 预训练功能；推理时不引入 teacher，STR 的时相交互和解码多分支继续折叠为单路径。

这是当前最符合：

- 原导师“完整判别位置 + 压缩冗余背景”思想；
- ≤5M；
- 强预训练；
- Mamba scan；
- STR 第二创新；
- novelty 风险可控；

的一条路线。

---

## 备选方案

### **EfficientVMamba-T slim + structured change-aware atrous context + STR**

优点：

- AAAI 2025；
- Atrous Selective Scan 和 topology-preserving compression 更直接；
- 原模型 6M，距离预算不远。

缺点：

- ImageNet Top-1 76.5，低于 TinyViM-S 80.3；
- 当前首要矛盾是 backbone floor，所以不应先选更弱 general representation。

只有 TinyViM-S-Slim A0×4 仍然不够时，再转这个备选。

---

# 29. 最后一句判断

**现在不应该继续“在 SHViT 上救 CASAA”，也不应该把 30M VMamba-Tiny 暴力裁成 5M。**

下一轮最合理的问题应该改成：

> **在 ≤5M 的真正强 tiny pretrained Mamba backbone 上，能否把 CASA 的“完整判别格 + 压缩上下文”从 Attention 的 Q/K/V 非对称，推广为 SS2D 的 dense/context 非对称，并与已经验证可折叠的 TAR/DCR 形成互补？**

这条问题链比：

```text
换个更强 backbone
+
继续原 CASAA
```

更完整，也更符合你最初课题的学术主线。

