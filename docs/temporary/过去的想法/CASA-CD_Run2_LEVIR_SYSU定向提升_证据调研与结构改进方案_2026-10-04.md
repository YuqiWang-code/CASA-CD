# CASA-CD Run2：LEVIR / SYSU 定向提升——证据调研、代码审查与结构改进方案

> **日期：2026-10-04**  
> **仓库：** https://github.com/YuqiWang-code/CASA-CD  
> **审阅版本：** `main`，本次审阅时 HEAD = `96f93b49fc4735a070c5c545fc8fe802a3818995`  
> **当前主线：** CASA-TViM-STRNet = TinyViM-S-Slim + CAACP-SS2D + TAR/DCR  
> **任务：** CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256，全监督二值变化检测  
> **固定协议：** BCE+Dice；Adam(2e-4, β=(0.9,0.99), wd=1e-4)；poly(0.9)+200 warmup；80K；batch 32；seed 16；test-as-val；阈值 0.5；输入 256×256；A/B/label 同步几何增强。  
> **硬目标：** CDD ≥ 98、LEVIR ≥ 92.5、SYSU ≥ 85、WHU ≥ 95；有效 deploy Params ≤ 5M；IoU 与 F1 同向。  
> **本轮短期目标：** 在 **CDD 不低于 97.18、WHU 不低于 95.07** 的前提下，LEVIR ≥ 92、SYSU ≥ 84。  
> **实验纪律：** 每个模型改动必须四数据集完整从头训练；不做单数据集 gating；不改 loss / augmentation / threshold / seed 包装。凡训练策略变化均标记 **「需导师批准」**。
>
> 文中证据类型：
> - **[代码事实]**：当前 CASA-CD 主分支代码、Run1 快照、正式 TEST 汇总直接支持。
> - **[文献事实]**：2024–2026 CCF-A 或 IEEE TGRS/JSTARS/ISPRS JPRS 等正式论文及官方代码支持。
> - **[推理判断]**：根据代码和实验现象提出的机制解释，尚需新实验验证。
> - **[待核实]**：未从一手论文表格或官方日志得到足够证据，不作为正式数字使用。

---

# 0. 结论先行

## 0.1 现在最值得改的不是 backbone，而是「CAACP 的置信度逻辑」和「DCR 输出端的细尺度恢复」

当前 TinyViM-S-Slim 已经把此前 SHViT 截断主干的主要表征缺口补掉了一大部分：

| 模型 | CDD F1 | LEVIR F1 | SYSU F1 | WHU F1 | Deploy Params |
|---|---:|---:|---:|---:|---:|
| CASA-STR / SHViT M1 | 95.54 | 90.37 | 83.02 | 93.72 | 2.416M |
| **CASA-TViM M1** | **97.18** | **91.07** | **83.47** | **95.07** | **4.880M** |

因此下一步不建议再次大换 backbone。当前问题已经从“骨干明显太弱”变成更具体的两个结构瓶颈：

1. **LEVIR：当前 CAACP 会把 Recall 拉高，但同时明显损伤 Precision。**  
   A2_STR → M1_FULL：
   - Recall：89.59 → **90.33**（+0.74pp）
   - Precision：92.84 → **91.82**（−1.02pp）
   - F1：91.19 → **91.07**（−0.12pp）

   **[推理判断]** 这和当前 CAACP 的 `rank_normalize` 很吻合：即使一幅双时相图几乎没有真实变化，纯 rank 也会强制生成从低到高的相对“变化排序”，从而在 LEVIR 大量零变化/极稀疏变化样本中人为放大弱差异，增加 FP。

2. **SYSU：当前主要缺口是“高 Recall 与高 Precision 不能同时保持”，并且输出细节恢复仍偏弱。**  
   A2_STR → M1_FULL：
   - Recall：81.24 → **84.41**（+3.17pp）
   - Precision：85.29 → **82.55**（−2.74pp）
   - F1：83.21 → **83.47**（+0.26pp）
   - IoU：71.25 → **71.63**

   CAACP 明显提高了变化敏感性，但把更多复杂背景一起判成变化；而当前 DCR 所有可学习解码都结束在 **1/4（64×64）**，最后只是 `1×1 conv + bilinear → 256×256`。对 SYSU 的密集、多类别、不规则边界来说，这是一个非常具体的结构短板。

---

## 0.2 本轮最推荐的两个改动

### **首选 1：CP-CAACP —— Confidence-Preserving CAACP**

只改 CAACP 的无参 score→cell weight 映射，不增加参数、不改 SS2D、不改位置、不改 topology。

当前：

\[
s_i = 1-\cos(f_i^A,f_i^B)
\]

先对整张图做 rank normalization：

\[
r_i=\operatorname{RankNorm}(s_i)
\]

然后 2×2 cell 内：

\[
w_i \propto \epsilon+r_i
\]

问题在于：**rank 只保留顺序，不保留绝对置信度。**  
一张“所有位置都只有很小 cosine difference”的零变化图，仍然会被强制排出 0～1 的显著性差异。

建议改成：

\[
q_i = s_i \cdot r_i
\]

\[
w_i=\frac{1+q_i}{\sum_{j\in\Omega}(1+q_j)}
\]

其中 \(\Omega\) 是固定 2×2 cell。

这样：

- 当整幅图的绝对变化证据很弱：\(s_i\approx0\)，自然退化为近似 uniform pooling；
- 当真实变化证据较强：rank 仍负责区分 cell 内相对重要位置；
- 无阈值、无可学习参数；
- A/B 对称；
- 不改变 8×8 context lattice；
- 不改变 CrossScan 顺序；
- 不引入新 loss。

**最直接针对 LEVIR Precision 下滑，同时尽量保留 SYSU Recall 增益。**

---

### **首选 2：FRH —— Fine-resolution Reparameterized Head**

不再让 DCR 在 64×64 后立刻做 `1×1→1 channel + ×4 bilinear`。

建议改为：

```text
DCR output: B×96×64×64
      ↓ bilinear ×2
B×96×128×128
      ↓ 训练期：
   base 1×1
   + γ · [RepDW3(96) → 1×1]
      ↓
1×128×128 logits
      ↓ bilinear ×2
1×256×256
```

其中：

- `γ=0` 初始化；
- RepDW3 仍为现有 `DW3 + DW1×3 + DW3×1 + identity` 训练结构；
- **不加激活函数**，使整个 head 在线性意义下可解析折叠；
- 部署时可以把：
  - base 1×1；
  - γ×DW3×PW1×1
  折叠成一个 **D→1 的 3×3 Conv**。

Deploy 额外参数约只有：

\[
96\times9+1-(96+1)\approx768
\]

即总参数大约仍为：

\[
4.880M + 0.000768M \approx 4.881M
\]

远低于 5M；额外 FLOPs 约数十 MFLOPs 量级。

它直接针对：

- LEVIR 小建筑轮廓；
- SYSU 密集边界/碎片区域；
- 当前 decoder 学习终点只有 1/4 的问题。

而且它不是再堆一个“边界模块”，而是**替换当前过于简单的最终预测头**，仍然属于 STR 的“训练丰富、部署折叠”逻辑。

---

## 0.3 暂时不要做的三件事

1. **不要恢复 Stage4 一个完整 LocalBlock。**  
   当前 TinyViM trunk ≈ 4.645M，一个 224-channel LocalBlock 约 +0.407M。  
   单 backbone 就会约：

   \[
   4.645+0.407=5.052M>5M
   \]

   即使把 TAR/DCR 的 D 缩到接近 0，也已经超预算。因此在 ≤5M 硬约束下直接否决。

2. **不要把 CAACP 扩到 Stage4。**  
   TinyViM Stage4/index=3 本来就没有 low-frequency pooling，CAACP 的“只压上下文、不压 dense path”前提不再成立；强行增加 pooling 会改变预训练结构本质。

3. **不要现在引入 deformable alignment / flow / frequency module。**  
   BiFA、DgFA、CAM-CD、SeCoR 等已经覆盖“显式对齐 / change-aware correction / scan guidance”相邻空间；不仅 novelty 风险高，也很容易破坏当前 ≤5M 和“只保留两个核心创新”的论文主线。

---

# 1. 当前仓库证据表

## 1.1 当前正式 Run1 指标

**[代码事实]** `docs/temporary/models_and_metrics_CASA-TViM_Run1.txt` 当前汇总的 CASA-TViM 正式 TEST 结果如下：

| Variant | CDD F1 / IoU | LEVIR F1 / IoU | SYSU F1 / IoU | WHU F1 / IoU |
|---|---:|---:|---:|---:|
| A0_TVIM_PLAIN | 96.76 / 93.73 | 91.08 / 83.62 | 82.39 / 70.06 | 94.78 / 90.08 |
| A1_CAACP | 96.75 / 93.70 | 91.03 / 83.53 | 82.11 / 69.65 | 94.84 / 90.19 |
| A2_STR | 97.11 / 94.39 | **91.19 / 83.81** | 83.21 / 71.25 | 94.53 / 89.63 |
| **M1_FULL** | **97.18 / 94.51** | 91.07 / 83.60 | **83.47 / 71.63** | **95.07 / 90.60** |

当前 M1 Recall / Precision：

| Dataset | Recall | Precision | 诊断 |
|---|---:|---:|---|
| CDD | 97.03 | 97.33 | 已平衡，接近饱和 |
| LEVIR | 90.33 | 91.82 | Recall 尚低，但 CAACP 已开始牺牲 Precision |
| SYSU | 84.41 | 82.55 | Recall 明显升高，Precision 成主要损失源 |
| WHU | 93.56 | 96.62 | 高 Precision，整体已经过当前短期线 |

---

## 1.2 当前参数与预训练状态

**[代码事实]**

- TinyViM-S 官方完整模型：约 5.6M / 0.9G@224；
- 当前 Slim：
  - widths `[48,64,168,224]` 不改；
  - Stage4 由 Local×5+TViM 缩成 Local×3+TViM；
  - retained trunk ≈ **4.645M**；
  - official 1000e EMA checkpoint 的保留层按 shape/key 原位继承；
  - Run1 README 记录 744 个保留 key 逐位继承；
- TAR/DCR D=96；
- 完整 deploy ≈ **4.880M**；
- 当前还剩约 **0.120M** 参数空间。

---

## 1.3 当前 CAACP 的代码行为

**[代码事实]**

位置：`models/model/layers/caacp_ss2d.py`

当前核心：

```text
1/16 feature: 16×16
↓
cosine change score
↓
global per-image rank normalization
↓
fixed 2×2 cell weighted pooling
↓
8×8 context lattice
↓
SS2D four-direction CrossScan
↓
upsample
↓
dense residual path
```

关键公式当前实际上是：

\[
c = c_{avg}+\beta(c_{ca}-c_{avg})
\]

并且 residual 也是基于这个 `c`：

\[
r=x-\operatorname{Up}(c)
\]

最终低频分支：

\[
y=\operatorname{Up}(\operatorname{SS2D}(c))+x-\operatorname{Up}(c)
\]

等价于：

\[
y=x+\operatorname{Up}(\operatorname{SS2D}(c)-c)
\]

这意味着 CAACP 对 context 的改变会同时：

- 送进 SS2D；
- 又被从 residual 中减掉。

**[推理判断]** 这形成一定的“自抵消”机制，是 β 训练后仍只有约 `5.6e-3~3.9e-2` 且 A1 主效应接近 0 的一个合理解释，但还不是已证实因果。

---

## 1.4 当前 DCR 的代码行为

**[代码事实]**

DCR：

```text
8² → 16² → 32² → 64²
```

每个阶段：

```text
bilinear up
→ RepPairFuse1x1
→ RepLocalBlock
```

到 64×64 后：

```text
Conv1×1: 96→1
→ bilinear 64→256
```

因此从 64² 到 256² 之间没有任何 learnable spatial operation。

这给 Q1 的“小目标/边界恢复不足”提供了非常直接的代码证据。

---

# 2. 一个必须先指出的 P0：当前代码并非严格 80,000 optimizer steps

这是本次完整审查中最重要的工程/协议问题。

当前 `train.py`：

```python
max_batches = len(train_loader)
max_epochs = ceil(max_steps / max_batches)

for epoch in range(max_epochs):
    train_epoch(...)   # 整个 loader 全跑完
```

`train_epoch()` 内没有：

```python
if global_iter >= max_steps:
    break
```

因此会跑到**最后一个完整 epoch 结束**，而不是严格 80,000 step。

按当前 batch=32、`drop_last=False` 和仓库数据规模：

| Dataset | Train samples | iters/epoch | ceil epochs | 实际 step | 超过 80K |
|---|---:|---:|---:|---:|---:|
| CDD | 10,000 | 313 | 256 | **80,128** | +128 |
| LEVIR | 7,120 | 223 | 359 | **80,057** | +57 |
| SYSU | 12,000 | 375 | 214 | **80,250** | +250 |
| WHU | 5,947 | 186 | 431 | **80,166** | +166 |

幅度只有约 0.07%–0.31%，所以**不太可能解释当前 1pp 级差距**，而且同数据集的 A0/A1/A2/M1 内部比较仍然基本公平。

但是从论文协议真实性上：

> 现在不能严格写“所有实验都是恰好 80,000 steps”。

### 建议

**P0 协议修复：需导师批准。**

最规范的做法是：

```python
if global_iter >= args.max_steps:
    break
```

并让 scheduler 同样按真正的 global step 结束。

但如果从下一轮才修，Run1 与 Run2 会出现轻微训练预算差异。因此需要导师在两种方案中明确选一个：

- **方案 A（推荐，最严格）**：修成 exact 80K；后续关键 baseline/M1 在最终定稿阶段按 exact 80K 重新跑四数据集。
- **方案 B（省算力但表述弱）**：冻结现有“ceil-to-full-epoch”作为真实协议，论文如实写“approximately 80K, completed to the end of the last epoch”，以后继续保持完全一样。

由于你的硬约束明确写的是 80,000 steps，科学上更推荐 A。

---

# 3. Q1：为什么恰好 LEVIR / SYSU 落后？

# Q1 一句话结论

> **LEVIR 的第一嫌疑是“稀疏/零变化场景中的置信度失真 + 小目标细节不足”；SYSU 的第一嫌疑是“高类别/背景复杂度下 Recall–Precision 冲突 + 1/4 解码终点造成的密集边界恢复不足”。单纯归因于 backbone、配准或类别不平衡都不完整。**

---

## 3.1 假设①：特征分辨率 / 小目标感受野不足

### 证据强度

- **LEVIR：强**
- **SYSU：中—强**
- **CDD：弱**
- **WHU：中，但当前已较高**

### [代码事实]

当前最后 learnable dense refinement 只有 64×64。

LEVIR 是建筑变化，孤立小建筑占比高；当一个目标在 256² 只有几到十几像素宽时，在 64² 上可能只剩 1–4 个 feature cell。

SYSU 则含更多类型、更复杂形状和密集区域；IoU 只有 71.63，说明区域完整性仍明显不足。

### [文献事实]

**CDMamba, IEEE TGRS 2025**  
题目即为 *Incorporating Local Clues Into Mamba for Remote Sensing Image Binary Change Detection*。论文把“纯 Mamba/global modeling 缺少 local clues”作为核心问题；在 LEVIR-CD 上报告 F1 90.75，强调 local branch 对小变化和边缘细节的重要性。

- IEEE: https://ieeexplore.ieee.org/document/10902569
- DOI: https://doi.org/10.1109/TGRS.2025.3545012
- Official GitHub: https://github.com/zmoka-zht/CDMamba

**SChanger, IEEE JSTARS 2025**  
通过 spatial consistency 和大核空间建模增强变化区域结构；其消融中 spatial consistency attention 对 LEVIR 有正增益。SChanger-small 在 LEVIR-CD 达 92.45，参数仅 0.607M；base 92.87 / 2.37M。

- IEEE: https://ieeexplore.ieee.org/document/10945386/
- DOI: https://doi.org/10.1109/JSTARS.2025.3555849
- Official GitHub: https://github.com/zhouziyu-cn/SChanger

### 风险

WHU 同样是 building dataset，但你已经 95.07，说明“所有建筑数据都只缺分辨率”显然不成立。LEVIR 的稀疏性/零变化图比例与 score FP 问题必须一起考虑。

---

## 3.2 假设②：时相配准误差与 pseudo-change

### 证据强度

- **LEVIR：中**
- **SYSU：中—强**
- **CDD：中，但当前模型处理得已经较好**
- **WHU：中**

### [文献事实]

**BiFA, IEEE TGRS 2024** 明确把：

- illumination difference；
- perspective difference；
- bitemporal spatial misalignment

视为 pseudo-change 来源，并通过 differential flow field 做显式空间对齐。

- IEEE: https://ieeexplore.ieee.org/document/10471555/
- DOI: https://doi.org/10.1109/TGRS.2024.3376673
- Official GitHub: https://github.com/zmoka-zht/BiFA

**DgFA, IEEE JSTARS 2025** 同时处理 heterogeneous appearance 与 affine geometric difference，并估计 transformation field 做空间配准。

- IEEE: https://ieeexplore.ieee.org/document/10830007/
- DOI: https://doi.org/10.1109/JSTARS.2025.3526795

**ST-Mamba, IEEE TGRS 2025** 的研究动机也是复杂成像条件造成的 pseudo-change，通过 spatio-temporal synergy 统一背景特征。

- DOI: https://doi.org/10.1109/TGRS.2025.3579617

### [推理判断]

当前 TAR 有 signed difference：

\[
Q-P
\]

对真实变化敏感，也会对微小错位敏感。

但目前没有你自己数据上的“offset / registration error”量化，因此不能直接说 LEVIR/SYSU 的主要问题就是配准。

### 结论

**现在不优先做 deformable/flow alignment。**

原因：

1. 需要新参数/大量 FLOPs；
2. novelty 已被 BiFA/DgFA 占得很近；
3. 很容易成为第三个主模块；
4. 当前更简单的 CAACP score / decoder 问题已有直接代码证据。

---

## 3.3 假设③：类别多样性与密集边界质量

### 证据强度

- **SYSU：强**
- LEVIR：中
- CDD：中
- WHU：弱—中

### [代码事实]

SYSU：

- A0 IoU 70.06；
- A2 IoU 71.25；
- M1 IoU 71.63；

而 WHU M1 IoU 90.60。

这不是单纯“分类阈值”问题，而是 region overlap 本身明显更难。

更重要的是：

A2 → M1 时 SYSU：

```text
Recall +3.17pp
Precision -2.74pp
```

说明模型已经“看见更多变化”，但无法稳定区分：

```text
true dense change
vs
complex background / pseudo-change
```

### [文献事实]

**SCAM, IEEE JSTARS 2025/vol.19 2026**：

- channel-adaptive state-space scan；
- difference fusion；
- lightweight CNN multiscale decoder；
- LEVIR-CD F1 91.01；
- SYSU-CD F1 **83.60**；
- WHU-CD F1 93.14。

它同样卡在大约“LEVIR 91 / SYSU 83.6”的区域，说明这一性能段在轻/中型 Mamba CD 中并不罕见。

- IEEE: https://ieeexplore.ieee.org/document/11282994/
- DOI: https://doi.org/10.1109/JSTARS.2025.3641390

**SChanger-small** 则达到 SYSU 84.58，同时 LEVIR 92.45，说明 tiny model 并非天然达不到；它更强调 spatial consistency 和语义预训练。

### 结论

**SYSU 下一步应优先提升“局部结构恢复 + false positive control”，而不是进一步单纯提高 change sensitivity。**

---

## 3.4 假设④：样本不平衡 / 零变化对

### 证据强度

- **LEVIR：强**
- SYSU：弱—中
- WHU：中
- CDD：低—中

### [项目数据事实]

你当前统计：LEVIR 约 **54% patch 为零变化**。

### [代码事实]

当前 CAACP：

```python
s = 1 - cosine
s = rank_normalize_2d(s)
```

rank normalization 会让每幅图都有完整相对排序。

例如：

```text
真实变化丰富图：
s = [0.02 ... 0.65] → rank 0~1

零变化图：
s = [0.001 ... 0.010] → 仍然 rank 0~1
```

绝对 evidence 相差几十倍，但 CAACP 路由看到的“排序强度”接近。

### 与实际 LEVIR 结果的吻合

A2_STR → M1：

- Recall 提高；
- Precision 更明显下降；
- F1 反而微降。

这是当前最强的机制吻合证据之一。

### 结论

不允许改 loss 的情况下，最值得做的不是 focal/class weight，而是：

> **让结构本身保留 absolute confidence，不要把任何弱差异都强行相对显著化。**

这就是 CP-CAACP 的核心。

---

# 4. Q2：结构改进候选排序

> 下表的 Δ 是**工程预期区间 / 立项判断，不是论文事实、不是保证值**。任何候选是否成立，以四数据集完整 80K TEST 为唯一裁决。

| 优先级 | 改动 | 主要机制 | 预期 ΔLEVIR | 预期 ΔSYSU | CDD / WHU 风险 | Deploy 参数增量 | Novelty 风险 |
|---|---|---|---:|---:|---|---:|---|
| **P1** | **CP-CAACP：绝对置信度保留的 rank×magnitude score** | 弱绝对差异时自动接近 uniform pooling，减少零变化 FP | **+0.20~+0.50** | 0~+0.30 | **低**；CDD/WHU 可能近似不变 | **0** | **低—中** |
| **P2** | **FRH：1/2 分辨率可折叠预测头** | DCR 后在 128² 做线性 RepDW spatial refinement，部署折成单 3×3 head | **+0.30~+0.70** | **+0.30~+0.60** | 低—中；可能轻微改变 CDD/WHU 边界 | **约 +768** | **低** |
| **P3** | **RA-CAACP：Reference-Anchored residual** | residual 永远基于 `c_avg`，CAACP 只改变 context scan，不在 dense residual 中反向抵消 | −0.10~+0.30 | **+0.20~+0.50** | 中；可能放大 CAACP 导致 LEVIR FP | **0** | **低—中** |
| **P4** | **Fine-STR Rank Expansion** | 仅提高 DCR 细尺度 `RepPW1x1` 训练期低秩 branch rank，部署仍同一个 1×1 | +0.10~+0.35 | +0.15~+0.40 | **低** | **0 deploy** | 低；更像第二创新内部增强 |
| **P5** | **D=96→120** | 用剩余预算提高 TAR/DCR统一通道容量 | +0.10~+0.40 | +0.15~+0.50 | 中；可能过拟合/改变四数据集平衡 | **约 +99.1K** | **无创新性** |
| **P6** | **1/8 score source → Stage3 CAACP** | score 从 32² shallow feature 得到，再规则 downsample 到16²，保留小建筑差异 | +0.15~+0.45 | 0~+0.25 | 中；shallow appearance noise 可能伤 CDD/WHU | **0** | 中 |
| **P7** | **Stage2 CAACP（4×4 cell→8×8）** | 在 1/8 stage 的官方低频 pool 路径做 change-aware structured pooling | 0~+0.25 | +0.10~+0.35 | **中—高**；现有 CAACP 独立主效应已弱 | 约 +1 scalar | 中 |
| **P8** | **Parameter-free Local Support Temporal Correction** | TAR 前从异时相 3×3 邻域选可靠局部支持以减轻错位 | +0.10~+0.40 | +0.20~+0.50 | **高**；可能抹掉真实小变化 | 0~极小 | **高：SeCoR/BiFA/DgFA 邻近** |

---

# 5. P1：CP-CAACP 详细设计

## 5.1 当前问题

当前：

\[
w_i = \frac{\epsilon+r_i}{\sum(\epsilon+r_j)}
\]

其中 \(r_i\) 仅表示 rank。

它不能区分：

```text
强变化 image
vs
所有 cosine difference 都很小的 image
```

---

## 5.2 建议公式

保留原始 cosine magnitude：

\[
s_i=\operatorname{clamp}(1-\cos(f_i^A,f_i^B),0,2)
\]

保留 rank：

\[
r_i=\operatorname{RankNorm}(s_i)
\]

组合：

\[
q_i=s_i r_i
\]

cell 内：

\[
w_i=\frac{1+q_i}{\sum_{j\in\Omega}(1+q_j)}
\]

### 为什么加 1

这是关键。

如果没有 1：

\[
w_i\propto q_i
\]

在所有 \(s_i\) 都非常小时，归一化后仍会形成强相对选择。

加 1 后：

\[
s_i\rightarrow0\Rightarrow w_i\rightarrow1/|\Omega|
\]

自然回退到 avg-like pooling。

---

## 5.3 数据流

```text
xa, xb @16×16
  ↓
1-cos
  ↓
absolute s
  ├── rank → r
  └────────────┐
               × → q=s*r
               ↓
        fixed 2×2 cell
               ↓
        w=(1+q)/sum(1+q)
               ↓
      c_ca(A), c_ca(B)
```

A/B 继续共享同一个 `w`。

---

## 5.4 参数 / FLOPs / 预训练兼容

- 新参数：0；
- token/lattice 数：不变；
- selective scan FLOPs：不变；
- β=0 仍然：
  \[
  c=c_{avg}
  \]
  所以 epoch0 仍精确恢复 pretrained path；
- T1/T2 对称：保持。

---

## 5.5 可证伪假设

> **H-CP：** 如果 LEVIR 的主要问题之一是 rank-only score 在弱变化/零变化图上制造人工显著性，则 CP-CAACP 应主要提高 Precision，且 Recall 不应大幅下降；SYSU 应保留大部分 Recall 增益。

### 成功判据

四数据集完整训练后，至少满足：

- LEVIR：
  - Precision ≥ 当前 M1 91.82；
  - F1 ≥ 91.35（首轮结构有效门槛）；
- SYSU F1 不低于 83.40；
- CDD ≥ 97.18 − 0.10；
- WHU ≥ 95.07 − 0.10。

注意：这是“机制值得继续”的门槛，不是最终硬目标。

---

# 6. P2：FRH 详细设计

## 6.1 当前 head

```python
x = decoder(feats)          # B,96,64,64
logits = head(x)            # B,1,64,64
logits = interpolate(...256)
```

最后 ×4 上采样完全参数自由。

---

## 6.2 推荐 FRH

```text
x64: B×96×64×64
 ↓ bilinear ×2
x128: B×96×128×128

branch A:
  Conv1×1(96→1)

branch B:
  RepDW3(96)   # linear; train rich / deploy DW3
  Conv1×1(96→1)
  × gamma      # gamma=0

sum
 ↓
logit128
 ↓ bilinear ×2
logit256
```

不要在 RepDW 与 output PW 之间插入 SiLU，否则整个 head 无法合并成一个线性 3×3 kernel。

---

## 6.3 部署折叠

RepDW3 deploy 后：

\[
y_c=K_c*x_c+b_c
\]

后接 1×1：

\[
z=\sum_c a_c y_c+b
\]

可合成：

\[
K^{out}_{c,:,:}=a_cK_c
\]

base 1×1 只需 pad 到 3×3 中心：

\[
K_{final}=K_{corr}+K_{base}^{pad}
\]

最终部署：

```text
bilinear 64→128
→ single Conv3×3(96→1)
→ bilinear 128→256
```

仍然是极简路径。

---

## 6.4 参数

当前 head：

\[
96+1=97
\]

新 deploy 3×3 head：

\[
96\times9+1=865
\]

净增：

\[
768
\]

模型：

\[
4,880,190+768\approx4,880,958
\]

仍远低于 5M。

---

## 6.5 文献机制依据

- CDMamba/TGRS 2025：local clues 对小变化和边缘有效；
- SChanger/JSTARS 2025：spatial consistency / decoder spatial modeling 在 LEVIR 有明确增益；
- SeCoR/JSTARS 2026：轻量模型的低层 ambiguous feature correction 仍有显著价值。

但 FRH 与它们不同：

> 它不是再增加 attention/edge module，而是把当前 1/4-only output head 改成 **STR-style foldable half-resolution predictor**。

Novelty 风险低，适合作为第二创新（STR decoder）的自然深化，而不是第三独立模块。

---

# 7. P3：RA-CAACP（Reference-Anchored）为什么值得保留为第三候选

当前：

\[
r=x-\operatorname{Up}(c)
\]

而 \(c\) 包含 CAACP。

建议：

\[
r_{ref}=x-\operatorname{Up}(c_{avg})
\]

\[
c=c_{avg}+\beta(c_{ca}-c_{avg})
\]

\[
y=\operatorname{Up}(\operatorname{SS2D}(c))+r_{ref}
\]

这样：

- dense/high-frequency reference path 保持官方 TinyViM 定义；
- 只有 context scan 真正被 CAACP 改；
- β=0 与官方模型仍完全等价；
- “完整判别格 + 改上下文”的论文故事更严格。

但它会让 CAACP 的作用更强，因此在 LEVIR 可能放大 FP。建议 **CP-CAACP 先跑，RA 后跑**，不要一开始两个一起改，否则无法判断是哪一个起作用。

---

# 8. P4：Fine-STR Rank Expansion——0 deploy 参数的容量提升

当前 `RepPW1x1(D=96)` 的 serial low-rank branch：

```text
96 → r=24 → 96
```

训练后解析折到：

```text
96 → 96 single 1×1
```

因此可以只在：

- DCR block1（64²）；
- final refine（64²）

把：

```text
r = 24 → 48
```

甚至 64。

训练期新增参数，但 deploy：

```text
仍然 96×96 Conv1×1
```

所以：

- Deploy params = 0 增量；
- Deploy FLOPs = 0 增量；
- 只增加训练时拟合自由度；
- 完全符合 STR 第二创新的逻辑。

**[推理判断]** 这比“再加一个模块”干净得多，但预期幅度不会太大，更适合作为 FRH 之后的低成本补强。

---

# 9. P5：D=96→120 的预算核算

当前 TAR+DCR+head 的 deploy 参数近似：

\[
P(D)=14D^2+1104D+1
\]

当前：

\[
D=96\Rightarrow P\approx235,009
\]

总模型约：

\[
4,880,190
\]

不同 D：

| D | Head/TAR/DCR params | 总模型估算 | 比当前增加 |
|---:|---:|---:|---:|
| 96 | 235,009 | 4,880,190 | 0 |
| 104 | 266,241 | 4,911,422 | +31,232 |
| 112 | 299,265 | 4,944,446 | +64,256 |
| **120** | **334,081** | **4,979,262** | **+99,072** |
| 124 | 352,161 | 4,997,342 | +117,152 |

因此理论最大约 D=124。

### 建议

如果走这个方向，只选 **D=120**，不要搜索 104/112/120/124。

原因：

- 仍留约 20K 安全余量；
- 不做参数搜索；
- 变量明确。

但它没有论文创新性，因此不应作为 Run2 第一实验。

---

# 10. CAACP 是否扩到 Stage2 / Stage4？

## 10.1 Stage2（1/8）：可以做，但不是第一优先

TinyViM 原 index=1 的 global low-frequency path 并不是 2×2 pooling，而是：

\[
32\times32 \xrightarrow{4\times4\ pool} 8\times8
\]

因此如果加 CAACP，应严格保持：

```text
fixed 4×4 cell
→ 8×8 context
```

而不是改成 2×2 得到 16×16。

后者会：

- 改 selective scan sequence length；
- 改 FLOPs；
- 改 pretrained computation；
- 不能用当前 β interpolation 直接对齐形状。

### 风险

一个 4×4 cell 合并 16 个位置，对 LEVIR 小建筑可能反而更粗。

所以优先级低于“用 1/8 feature 生成 score，但仍只修改 Stage3 2×2 context”。

---

## 10.2 Stage4（1/32）：不建议

index=3 没有 low-frequency pooling。

增加 CAACP 意味着从无 compression 变有 compression，已经不是：

```text
官方 context pool → change-aware context pool
```

而是：

```text
完整 SS2D → 新压缩 SS2D
```

预训练兼容、机制风险都明显更高。

---

## 10.3 1×2 / 2×1 cell：暂不做

Stage3 当前：

```text
16×16 --2×2--> 8×8
```

若改：

```text
1×2 → 16×8
2×1 → 8×16
```

context shape 与 `c_avg` 不同，无法继续：

\[
c=c_{avg}+\beta(c_{ca}-c_{avg})
\]

除非再增加 reshape/pool/merge。

这会把一个干净的机制变成复杂组合，不符合“不要模块堆叠”。

---

# 11. 1/8 feature 是否应作为 change score 来源？

## 结论

> **值得作为 P6，但不优先于 CP-CAACP。**

当前 score 来自 1/16 stage2 prefix：

- semantic 更强；
- resolution 只有16²。

LEVIR 小建筑在 1/8 32² 上通常保留更好。

可以：

```text
F2_A/F2_B @32²
→ cosine difference
→ 2×2 average downsample score to16²
→ CP rank×magnitude
→ Stage3 2×2 CAACP
```

参数 0。

### 风险

浅层 feature 对：

- 光照；
- 纹理；
- seasonal appearance

更敏感，可能直接伤 CDD。

而且 CASA-CD 历史 CASAA Run3 已出现过：

> 更好的 shallow/detail router ranking 未必能转成最终 F1。

因此只放 P6。

---

# 12. Q3：固定协议内还有什么合理空间？

# Q3 一句话结论

> **当前没有证据支持靠 EMA、BN 冻结或改 backbone LR 来救指标；这些都不是结构创新，并且多数触及固定协议。当前最有价值的是“只加诊断，不改变优化行为”。**

---

## 12.1 可以立即做、不改变训练协议的事项

### A. 把 BACKBONE-ADAPT 改成“只统计真正从 checkpoint 原位继承的 key”

当前 `_backbone_ref` 保存整个 encoder state，包括：

- pretrained inherited tensors；
- 新建 norm；
- CAACP β 等新模块。

所以 README 中 9%–14% `rel_L2` 不能严格解释成：

> “ImageNet pretrained weights 全部漂移了 9%–14%”。

建议日志拆成：

```text
[PRETRAIN-DRIFT/stem]
[PRETRAIN-DRIFT/stage0]
[PRETRAIN-DRIFT/stage1]
[PRETRAIN-DRIFT/stage2]
[PRETRAIN-DRIFT/stage3-retained]
```

只统计 744 个 exact-loaded inherited keys。

这是诊断，不改变训练。

---

### B. 记录每个数据集 CAACP 的真实作用强度

β 本身不是 contribution。

应记录：

\[
E_{ca}=
\frac{\|\beta(c_{ca}-c_{avg})\|_2}
{\|c_{avg}\|_2+\epsilon}
\]

同时按样本统计：

```text
score_abs_mean
score_abs_p95
cell_weight_entropy
context_delta_rms
```

并区分：

```text
GT change ratio == 0
GT change ratio > 0
```

如果 LEVIR 零变化图的 `cell_weight_entropy` 仍然很低，就直接支持 CP-CAACP。

---

### C. Error decomposition

不改模型，只对 test prediction 做：

- zero-change pair FP rate；
- changed pair Recall；
- boundary ±2/±4px F1/IoU；
- connected-component 按面积 small/medium/large F1；
- FP 与 shadow / vegetation / registration hard cases 的人工抽样。

这比继续猜“LEVIR 为什么低”更有价值。

---

## 12.2 BN 相关

当前 A/B 先 concat 成 2B 再过 shared TinyViM：

```text
[A;B] → 2B
```

这是好的。

它保证 Siamese BN 每个 step 看到两个时相的共同分布，而不是：

```python
encoder(A)
encoder(B)
```

分两次更新 running statistics。

batch 32 时，backbone BN 实际一次 forward 看约 64 temporal images，统计已经相对稳定。

### 不建议

- freeze BN：**需导师批准**
- SyncBN：当前四数据集是独立进程/任务，并非一个 DDP model，不适用
- post-hoc BN recalibration：**需导师批准**
- BN→LN/GN：属于结构改变，而且破坏 TinyViM 预训练 macro design

---

## 12.3 EMA

TinyViM 的 ImageNet checkpoint 本身使用 EMA weights，但 **CD fine-tune 阶段新增 EMA** 会改变：

- 训练状态；
- checkpoint 定义；
- test-as-val best selection口径。

因此：

> **新增 EMA：需导师批准，不建议本轮做。**

---

## 12.4 backbone_lr_ratio 0.1

现在：

```text
backbone 2e-5
new modules 2e-4
```

已有 global drift 9%–14%，至少说明 backbone 并非近冻结。

因此没有足够证据说 0.1× 太低。

改变：

```text
0.1 → 0.2 / 0.5 / 1.0
```

属于训练策略搜索：

> **需导师批准。**

当前不建议。

先做 exact-loaded per-stage drift 审计再谈。

---

## 12.5 其它训练策略

以下全部不属于当前允许空间：

- gradient clipping：**需导师批准**
- optimizer 改 AdamW：**需导师批准**
- weight decay 分组：**需导师批准**
- batch size 改动：**需导师批准**
- stochastic depth 调整：结构/训练共同变化，当前不做
- threshold tuning：明确禁止
- TTA：明确不建议
- multi-seed 只挑最好：明确禁止

---

# 13. Q4：2024–2026 对标文献

> **重要：不同论文数据处理、loss、optimizer、pretraining、crop、test protocol 不同。下面只用于结构/量级参考，不可直接声称公平 SOTA 对比。**

| 方法 | 年份 / Venue | Params | LEVIR-CD F1 | SYSU-CD F1 | 主要结构 | 与本文的关系 |
|---|---|---:|---:|---:|---|---|
| **CASA-TViM M1** | 本项目 | **4.880M** | **91.07** | **83.47** | TinyViM-S-Slim + CAACP + STR | 当前基准 |
| **SChanger-small** | 2025 JSTARS | **0.607M** | **92.45** | **84.58** | semantic pretraining + temporal fusion + spatial consistency | 非 Mamba；证明极小模型也可到 92/84，但其 pretraining strategy 不能直接照搬 |
| **SChanger-base** | 2025 JSTARS | **2.370M** | **92.87** | 84.17 | 同上 | LEVIR 强，FLOPs 18.275G |
| **SCAM** | 2025 JSTARS | **待核实** | **91.01** | **83.60** | channel-adaptive scan + diff fusion + light CNN decoder | 与“scan importance”有 novelty 邻近 |
| **CDMamba** | 2025 TGRS | **11.90M** | **90.75** | 论文未以 SYSU 为主表/待核实 | Mamba + local clues | 支持 local detail 必要性 |
| **ChangeMamba MambaBCD-S/B** | 2024 TGRS | 49.94/84.70M（paper variants） | **使用 LEVIR-CD+，不能与 LEVIR-CD 横比** | 82.83 / 83.11（paper） | VMamba + spatio-temporal SSM | 说明容量提升对 SYSU 有一定收益，但仍未天然到85 |
| **CAM-CD** | 2026 JSTARS | **51.67M** | **90.78** | **82.46** | Change Prior + CA-SS2D direction response reweight | **与 change-aware scan 高碰撞；本项目必须保持“scan前 structured context pooling”差异** |
| **Mamba-CD** | 2026 JSTARS | **27.94M** | **87.50（来自后续文献汇总；主表待再核）** | 待核实 | change-region-aware attention + recursive context refinement | 与泛化“change-aware attention / decoder refinement”相邻 |
| **SeCoR** | 2026 JSTARS | **2.50M / 2.66G** | 原摘要称四数据集强，**精确F1待核** | **精确F1待核** | reliability-aware local support correction + prototype-prior repair | 与显式跨时相局部纠偏高度邻近 |
| **CFNet** | 2025 JSTARS | 待核实 | **92.18** | **82.89** | cosine-distance focuser + content-aware strategy | 与 cosine score reweight 相邻；CP-CAACP需强调“structured context compression”而非普通 focuser |
| **ST-Mamba** | 2025 TGRS | **待核实** | **待核实** | **待核实** | Mamba encoder + spatio-temporal background unification | 支持 pseudo-change/background consistency 动机 |

---

## 13.1 ChangeMamba：为什么“83 左右”很有参考意义

**[文献事实]** ChangeMamba 原论文的 SYSU：

- Tiny F1 ≈ 81.29
- Small ≈ 82.83
- Base ≈ 83.11

官方仓库后续重整代码的 Small checkpoint 报告约 83.36，但作者明确说明重训结果可能与原论文不同。

这说明：

> 单纯把 VMamba 从 Tiny 增大到 Small/Base，并没有把 SYSU 自动推到85。

所以你现在 4.88M 得到 83.47，已经非常接近甚至超过大量更大 Mamba 模型的公开量级。

关键不是再追 backbone 大小，而是解决：

```text
Precision / boundary / pseudo-change
```

---

## 13.2 SChanger：最值得借鉴的不是它的预训练策略，而是“空间一致性”

SChanger-small：

- 0.607M；
- LEVIR 92.45；
- SYSU 84.58。

这是当前最危险也最有价值的对标之一。

但其强结果包含：

- 单时相语义 supervised pretraining；
- Semantic Change Network fine-tuning。

直接照搬将改变你的训练协议：

> **需导师批准，而且不建议作为本课题主线。**

可借鉴的是结构思想：

> dense CD 不能只靠高层语义，空间一致性/局部结构建模对 LEVIR 和 SYSU 都重要。

FRH 正好取这一“局部结构恢复”的方向，但保持你的 STR story。

---

## 13.3 CAM-CD：CAACP novelty 边界必须写清楚

CAM-CD 当前官方代码的 CA-SS2D：

```text
full CrossScan
→ full selective_scan
→ 4 directional outputs
→ Change Prior → direction weights
→ weighted directional response
→ CrossMerge
```

你的 CAACP：

```text
dense lattice preserved
→ bi-temporal score
→ fixed-cell structured context aggregation
→ smaller regular context lattice
→ CrossScan / selective scan
→ dense residual restoration
```

实质差异：

| CAM-CD | CASA CAACP |
|---|---|
| scan 后对 4 个方向响应加权 | **scan 前压缩 context 内容** |
| 不做 structured token/context reduction | **固定 cell topology-preserving reduction** |
| change prior 决定 direction importance | change evidence 决定 cell 内 context contribution |
| full scanned lattice | dense decision lattice完整、context lattice压缩 |

因此：

- 不要把下一版写成 “change-aware directional scan”；
- 不要让 score 去控制四方向权重；
- 不要做“变化区域优先扫描顺序”。

这些会靠近 CAM-CD / adaptive-scan 文献。

---

## 13.4 STORM 对 token/context compression 的约束

**STORM, ICML 2026（CCF-A）** 指出，结构化 Vision Mamba 对 spatially agnostic token reduction 很敏感，随意删 token 会破坏二维扫描拓扑。

- PMLR: https://proceedings.mlr.press/v306/lv26e.html

这继续支持当前 CAACP 的核心设计：

```text
fixed 2×2 cells
→ regular 8×8 lattice
```

因此本轮**不要**改回 TopK / arbitrary merge。

---

# 14. Novelty 碰撞图

| 想法 | 近邻工作 | 风险 | 本项目应如何保持差异 |
|---|---|---|---|
| change-aware direction gating | CAM-CD / adaptive scan | **高** | 不碰方向权重；只在 scan 前做 context aggregation |
| arbitrary token pruning/merging | STORM 等 | 中—高 | 固定 cell，保持规则 lattice |
| explicit optical-flow/deform alignment | BiFA / DgFA | **高** | 暂不作为主创新 |
| opposite-temporal local correction | SeCoR | **高** | 暂不做或仅作后备 |
| cosine-based reweight | CFNet | 中 | 强调“compression of context lattice”，不是普通 feature gate |
| local clue Mamba | CDMamba | 中 | FRH 位于 STR decoder，不改 Mamba backbone local/global decomposition |
| general boundary module | 大量 CD 工作 | 高 | FRH 定义为**foldable prediction head replacement**，不是独立 EdgeGate |
| train-rich/deploy-simple decoder | STR-RepNet / 本项目 | 低（自有主线） | 继续沿结构重参数化统一叙事 |

---

# 15. Q5：最少实验路径

# Q5 一句话结论

> **先做 2 个四数据集实验：E1=CP-CAACP，E2=FRH。不要同时改，先隔离机制；两者都有效后才做组合。这样第一轮只需要 8 个 full 80K run。**

---

## 15.1 在训练前先做 0-cost diagnostics

这不是模型改动，不消耗80K，也不违反“四数据集完整训练”的规则。

对当前 M1 best checkpoint 四数据集各跑一次：

### D1：score 置信度

记录：

```text
abs_cos_score mean / p50 / p95
rank score entropy
2x2 cell weight entropy
beta
context_delta_rms
```

分别按：

```text
GT change ratio = 0
0 < ratio <= 1%
1% < ratio <= 5%
ratio > 5%
```

分组。

如果 LEVIR zero-change 组：

```text
abs score 很低
但 rank/cell entropy 仍形成明显选择
```

则 CP-CAACP 机制证据非常强。

### D2：boundary / object size

当前 A2 与 M1 输出做：

- boundary ±2px F1；
- boundary ±4px F1；
- small connected components；
- medium；
- large。

如果 LEVIR/SYSU 的 loss 主要集中在 boundary/small component，而 CDD/WHU较好，则 FRH 证据增强。

---

## 15.2 实验 E1：CP-CAACP

保持：

```text
TinyViM-S-Slim
TAR/DCR full
D=96
CAACP Stage3 only
beta logic不变
```

唯一变量：

```text
rank-only weight
→
absolute-confidence-preserving rank×magnitude weight
```

四数据集各80K。

### 通过标准

相对当前 M1：

- CDD：≥97.18（你的强要求；若允许统计容差则需预注册，当前按严格不退）
- WHU：≥95.07
- LEVIR：期望 ≥91.4
- SYSU：≥83.47，最好 ≥83.7

如果 LEVIR Precision明显恢复，即使 F1只+0.2，也说明方向成立。

---

## 15.3 实验 E2：FRH

基于**当前 M1**，而不是 E1 结果，单独换 output head。

这样 E1/E2 是严格两条单变量证据。

四数据集各80K。

### 通过标准

- CDD ≥97.18
- WHU ≥95.07
- LEVIR ≥91.5
- SYSU ≥83.7
- boundary F1 提高；
- deploy params <4.89M；
- fold test 通过。

---

## 15.4 如果 E1 / E2 都通过

再做：

### E3：CP-CAACP + FRH

四数据集完整80K。

因为 E1/E2 已经各自有单变量证据，E3 才允许测试组合。

短期目标：

```text
CDD >= 97.18
WHU >= 95.07
LEVIR >= 92.0
SYSU >= 84.0
```

如果 E3 做不到，不建议继续盲加模块。

---

## 15.5 如果 E1 失败

说明：

> LEVIR Precision 问题并不主要来自 rank-only confidence。

则不再调 score 公式。

路线：

```text
E2 FRH
→
若有增益，再考虑 Fine-STR Rank Expansion
```

不要搜索更多 score 归一化。

---

## 15.6 如果 E2 失败

说明：

> 1/4→full 的简单 learnable spatial refinement 不是当前主要瓶颈。

则下一候选是：

```text
RA-CAACP
```

而不是继续加更复杂 edge decoder。

---

# 16. 逐文件修改清单

## E1：CP-CAACP

### `models/model/layers/caacp_ss2d.py`

建议新增：

```python
def confidence_preserving_score(abs_score, rank_score):
    return abs_score * rank_score
```

cell pool：

旧：

```python
w = eps + rank_score
w = w / w.sum(...)
```

新：

```python
q = abs_score * rank_score
w = 1.0 + q
w = w / w.sum(...)
```

### `models/model/casa_tvim_str_net.py`

当前只把 rank score 传进 CAACP。

应让 CAACP 同时得到：

```text
abs_score
rank_score
```

或者在外部生成 `q`。

保持 `no_grad` routing。

### 日志新增

```text
[CAACP-SCORE-MODE] confidence_preserving_rank_magnitude
[CAACP-ABS-MEAN]
[CAACP-WEIGHT-ENTROPY]
```

---

## E2：FRH

### 新文件建议

`models/model/str_fine_head.py`

实现：

```text
Train graph:
up2
base 1x1
+
gamma * PW(RepDW3(x))
sum
up2

Deploy graph:
up2
single Conv3x3(D→1)
up2
```

必须实现：

```python
get_equivalent_kernel_bias()
switch_to_deploy()
```

并写 standalone fold unit test。

### `models/model/casa_tvim_str_net.py`

替换：

```python
self.head = nn.Conv2d(str_dim,1,1)
...
logits = self.head(x)
logits = F.interpolate(logits, size=input_size)
```

为：

```python
self.head = STRFineHead(...)
...
logits = self.head(x, target_size=input_size)
```

### `switch_to_deploy`

追加：

```python
self.head.switch_to_deploy()
```

---

# 17. Smoke / Dry-run 必测

## CP-CAACP

1. T1/T2 swap：
   \[
   score(A,B)=score(B,A)
   \]
2. zero evidence：
   若 `A==B`，cell weights 应接近严格 uniform 1/4；
3. β=0：
   新 CP-CAACP 与 A0 pretrained path 数值等价；
4. gradient：
   β grad 非零；
5. shape：
   dense16² / context8² 不变。

---

## FRH

1. train→deploy max abs；
2. 0.5 binary disagreement；
3. real batch disagreement；
4. γ=0 epoch0 correction确实为0；
5. fold 后训练分支删除；
6. deploy params；
7. FLOPs；
8. 128→256输出尺寸严格一致。

---

# 18. 当前代码还应修的两个非方法问题

## P0：exact max_steps

前文已经说明。  
**需导师批准**后修。

---

## P2：`last.pth` 在 best 更新之前保存

当前：

```python
_save_last(...)
if F1 > best:
    self.best_f1 = ...
    save best
```

所以 `last.pth` 内的：

```text
best_f1
best_epoch
```

会比当前 epoch 落后一拍。

正常完整训练不影响最终 best 文件，但崩溃恢复时可能造成：

- best metadata 回退；
- 旧 best 文件删除/命名逻辑不一致。

建议改成：

```python
if improved:
    update best metadata
    save best

_save_last(...)
```

这是工程正确性修复，不是训练策略调参。

---

# 19. FLOPs 口径也要统一

当前仓库：

- README / smoke 的某些位置记录约 2.9G；
- `models_and_metrics_CASA-TViM_Run1.txt` 的 fvcore 汇总约 **2.6747G**；
- selective scan CUDA 可能存在 unsupported/解析 FLOPs 口径差异。

正式论文不要同时出现两个数。

建议最终同时输出：

```text
fvcore_supported_flops
unsupported_ops list
analytical_selective_scan_flops
total_reported_flops
```

并在所有模型采用同一口径。

---

# 20. 文献清单

## 20.1 TinyViM

**Ma et al., TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba, ICCV 2025（CCF-A）**

- TinyViM-S：5.6M / 0.9G，ImageNet-1K 79.2（300e）/80.3（1000e）
- 核心：low-frequency Mamba context + high-frequency local path
- CVF: https://openaccess.thecvf.com/content/ICCV2025/html/Ma_TinyViM_Frequency_Decoupling_for_Tiny_Hybrid_Vision_Mamba_ICCV_2025_paper.html
- GitHub: https://github.com/xwmaxwma/TinyViM
- arXiv: https://arxiv.org/abs/2411.17473

---

## 20.2 ChangeMamba

**Chen et al., ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model, IEEE TGRS 2024**

- DOI: 10.1109/TGRS.2024.3417253
- MambaBCD SYSU paper F1：
  - Tiny 81.29
  - Small 82.83
  - Base 83.11
- 论文 BCD 参数量：
  - Tiny 17.13M
  - Small 49.94M
  - Base 84.70M
- 注意：其建筑 benchmark 是 **LEVIR-CD+**，不是本课题 LEVIR-CD-256，不能直接横比。
- IEEE: https://ieeexplore.ieee.org/document/10565926
- GitHub: https://github.com/ChenHongruixuan/ChangeMamba
- arXiv: https://arxiv.org/abs/2404.03425

---

## 20.3 CDMamba

**Zhang et al., CDMamba: Incorporating Local Clues Into Mamba for Remote Sensing Image Binary Change Detection, IEEE TGRS 2025**

- LEVIR-CD F1：90.75
- Params：11.90M
- 关键点：Mamba global feature + local clue extraction
- IEEE: https://ieeexplore.ieee.org/document/10902569
- DOI: https://doi.org/10.1109/TGRS.2025.3545012
- GitHub: https://github.com/zmoka-zht/CDMamba

---

## 20.4 SChanger

**Zhou et al., SChanger: Change Detection From a Semantic Change and Spatial Consistency Perspective, IEEE JSTARS 2025**

- small:
  - Params 0.607M
  - LEVIR 92.45
  - SYSU 84.58
  - CDD 95.75
  - WHU 93.15
- base:
  - Params 2.370M
  - LEVIR 92.87
  - SYSU 84.17
  - CDD 97.62
  - WHU 93.20
- base FLOPs：18.275G；small 6.242G
- 核心：semantic pretraining + temporal fusion + spatial consistency
- **语义预训练/fine-tune recipe 若引入本项目属于「需导师批准」**
- IEEE: https://ieeexplore.ieee.org/document/10945386/
- DOI: https://doi.org/10.1109/JSTARS.2025.3555849
- GitHub: https://github.com/zhouziyu-cn/SChanger
- arXiv: https://arxiv.org/abs/2503.20734

---

## 20.5 SCAM

**Zhang et al., SCAM: Scan Channel Attention Mamba-Based Network for Remote Sensing Change Detection, IEEE JSTARS 2025（卷期为 2026 vol.19）**

- LEVIR 91.01
- SYSU 83.60
- WHU 93.14
- CLCD 78.84
- 关键：channel adaptive state-space scan + difference fusion + lightweight CNN decoder
- Params：**待从正式 Table 进一步核实**
- IEEE: https://ieeexplore.ieee.org/document/11282994/
- DOI: https://doi.org/10.1109/JSTARS.2025.3641390

---

## 20.6 CAM-CD

**CAM-CD: Change-Aware Mamba for Remote Sensing Change Detection, IEEE JSTARS 2026**

- Params：约 51.67M
- LEVIR F1：约 90.78
- SYSU F1：约 82.46
- 核心：Change Prior + change-aware directional SS2D response weighting
- 与本项目 CAACP 的主要 novelty 边界：**它是 full scan 后 direction weighting；我们是 scan 前 structured context compression**
- DOI: https://doi.org/10.1109/JSTARS.2026.3713960
- GitHub: https://github.com/xjkgis/CAM-CD

---

## 20.7 ST-Mamba

**Zhao et al., ST-Mamba: Spatio-Temporal Synergistic Model for Remote Sensing Change Detection, IEEE TGRS 2025**

- 核心：
  - Mamba Feature Extraction Module
  - Spatio-Temporal Synergy Module
  - 背景特征统一 / pseudo-change suppression
- 精确 Params / LEVIR / SYSU 数字：**待从 IEEE 正式表格核实，不在本文编造**
- DOI: https://doi.org/10.1109/TGRS.2025.3579617

检索关键词：

```text
"ST-Mamba" "Table" "LEVIR-CD" "SYSU-CD" "parameters"
"ST-Mamba" "3579617" PDF
```

---

## 20.8 Mamba-CD

**Mamba-CD: Mamba-Based Change Detection Network for Remote Sensing Images With Change Region-Aware Attention and Recursive Context Refinement Mechanism, IEEE JSTARS 2026**

- Params：27.94M
- FLOPs：13.08G
- 后续公开文献汇总给出的 LEVIR F1 约87.50；**正式主表数字建议投稿前从 IEEE PDF 再核一次**
- 核心：Change Region-Aware Attention + Recursive Context Refinement
- DOI: https://doi.org/10.1109/JSTARS.2026.3652135

---

## 20.9 SeCoR

**SeCoR: Evidence-Guided Selective Correction for Lightweight Remote Sensing Change Detection, IEEE JSTARS 2026**

- Params：2.50M
- FLOPs：2.66G
- half-width：0.99M / 1.99G
- 核心：
  - reliability-aware local support correction
  - adaptive prototype-prior repair
- IEEE 摘要称四数据集均有很强 accuracy-efficiency trade-off；**精确 F1 本文不写，待从正式表核实**
- IEEE: https://ieeexplore.ieee.org/document/11646457/
- DOI: https://doi.org/10.1109/JSTARS.2026.3721954

---

## 20.10 CFNet

**CFNet: Optimizing Remote Sensing Change Detection Through Content-Aware Enhancement, IEEE JSTARS 2025**

- LEVIR F1：92.18
- SYSU F1：82.89
- 核心：content-aware strategy + cosine-distance Focuser
- 与 CAACP 的 proximity：都使用双时相 cosine evidence；区别必须落在 **structured context compression / fixed lattice / SSM context path**，不能只说“cosine change awareness”
- IEEE: https://ieeexplore.ieee.org/document/11016006/
- DOI: https://doi.org/10.1109/JSTARS.2025.3574173

---

## 20.11 BiFA

**Zhang et al., BiFA: Remote Sensing Image Change Detection With Bitemporal Feature Alignment, IEEE TGRS 2024**

- 显式处理：
  - illumination；
  - perspective；
  - temporal channel alignment；
  - differential flow spatial alignment；
  - multiscale alignment
- IEEE: https://ieeexplore.ieee.org/document/10471555/
- DOI: https://doi.org/10.1109/TGRS.2024.3376673
- GitHub: https://github.com/zmoka-zht/BiFA

---

## 20.12 DgFA

**Dual-Granularity Feature Alignment for Change Detection in Remote Sensing Images, IEEE JSTARS 2025**

- 同时处理 heterogeneous appearance 与 affine geometric difference；
- transformer features + semantic alignment + transformation field spatial alignment；
- DOI: https://doi.org/10.1109/JSTARS.2025.3526795
- IEEE: https://ieeexplore.ieee.org/document/10830007/

---

## 20.13 STORM

**STORM: Spatial-Aware Reduction Framework, ICML 2026（CCF-A）**

- 关键事实：结构化 Vision Mamba 不适合 spatially agnostic arbitrary token reduction；
- 必须保留二维 topology / neighborhood coherence；
- 对 CAACP “规则 cell 再 CrossScan”有直接理论/实验支持。
- PMLR: https://proceedings.mlr.press/v306/lv26e.html

---

# 21. 下一步推荐

## 第一优先：先完成 0-cost diagnosis，再立刻跑 CP-CAACP × 4

原因不是“它最可能单独涨最多”，而是：

1. 它直接修当前第一创新的最明显逻辑弱点；
2. 0 参数；
3. 不改变 lattice / SS2D / backbone；
4. LEVIR Recall↑Precision↓的现象与该弱点高度一致；
5. 即使失败，也能快速证明“CAACP score calibration 不是主要问题”，从而停止在这个方向继续搜索。

预期：

```text
LEVIR: +0.2 ~ +0.5pp
SYSU :  0   ~ +0.3pp
CDD  : 约不变
WHU  : 约不变
```

---

## 第二优先：FRH × 4

原因：

1. 当前 DCR 学习终点只有 1/4 是明确代码事实；
2. 同时针对 LEVIR 小目标与 SYSU 密集边界；
3. Deploy 只 +~768 params；
4. 可解析折叠；
5. 不改变 backbone；
6. 不与 CAM-CD / alignment 文献抢 novelty。

预期：

```text
LEVIR: +0.3 ~ +0.7pp
SYSU : +0.3 ~ +0.6pp
CDD  : -0.1 ~ +0.2pp
WHU  : -0.1 ~ +0.3pp
```

这只是工程期望，不能相加当最终保证。

---

## 如果二者都正向

第三步只跑：

```text
CP-CAACP + FRH
```

四数据集。

本轮合理期待区间：

```text
CDD   97.2 ~ 97.5
LEVIR 91.6 ~ 92.2
SYSU  83.9 ~ 84.4
WHU   95.0 ~ 95.3
```

**这仍不是最终硬目标。**

如果组合能稳定到 LEVIR≈92 / SYSU≈84 且不退 CDD/WHU，说明主线是健康的，再考虑：

- RA-CAACP；
- Fine-STR rank expansion；

而不是再次换 backbone。

---

# 22. 最终架构判断

当前 CASA-TViM 并不是“效果不理想所以整条路线失败”。

相反，已有证据说明：

1. **TinyViM backbone 选择是成功的。**  
   4.88M 已经接近 29.57M VMamba+TAR/DCR 的量级，并把 CDD/WHU 推到 97.18/95.07。

2. **STR 仍然是真正稳定的性能来源之一。**  
   尤其 SYSU +0.83，CDD +0.35。

3. **CAACP 的思想没有被证明无效，但当前实现方式没有把“变化感知”转换成稳定主效应。**  
   它在 SYSU/WHU 与 STR 有正交互，却在 LEVIR出现“Recall提升、Precision下降”的典型 false-positive signature。

4. **下一轮不应该再堆模块，而应该把两个核心创新各自修到更纯：**
   - CAACP：从“强制 relative rank”变成“absolute-confidence-preserving context compression”；
   - STR decoder：从“64²后纯插值”变成“128²可折叠空间预测”。

如果这两个改动仍无法把 LEVIR推近92、SYSU推过84，才有充分证据进入更激进的：

```text
alignment / backbone budget reallocation / Stage2 CAACP
```

而不是现在就同时全部加入。

---

# 23. 立即执行顺序

1. **先不要改训练参数。**
2. 修正/确认 `max_steps` P0，是否采用 exact 80K —— **需导师批准**。
3. 写四数据集 post-hoc diagnostic：
   - zero-change FP；
   - score magnitude；
   - cell entropy；
   - boundary F1；
   - object-size F1。
4. 实现 **CP-CAACP**。
5. smoke：
   - A/B swap；
   - `A==B` uniform weights；
   - β=0 pretrained equivalence；
   - gradient；
   - params/FLOPs。
6. CP-CAACP 四数据集完整80K。
7. 实现 **FRH**。
8. 独立 fold unit test。
9. FRH 四数据集完整80K。
10. 只有 E1/E2 都通过时，才跑组合 E3×4。
11. 更新 `docs/experiment_metrics.xlsx` 时仍只从每个 `train_log.txt` **最后一个完整 `=== TEST RESULTS ===` 区块**读取。
12. 最后再决定是否进入 RA-CAACP / Fine-STR，而不是继续模块搜索。

---

# 24. 仍需补充证据

1. LEVIR “约54% zero-change patch”的统计脚本与最终数值应写入仓库，便于论文复现。
2. Current CAACP 的 score-GT PR-AUC / zero-change score 分布尚未直接测。
3. 当前 β 的绝对数值不等于模块贡献，需要 context residual energy。
4. ST-Mamba 精确 Params / LEVIR / SYSU 主表数字需从 IEEE 正文核验。
5. SCAM 参数量需从正式论文复杂度表核验。
6. SeCoR 四数据集精确 F1 需从正式表核验。
7. Mamba-CD 的 LEVIR/SYSU精确主表数字应在投稿前由 IEEE 正文复核。
8. CASA-TViM FLOPs 当前仓库存在约2.67G与约2.9G两个口径，应统一 selective scan FLOPs 计数。
9. exact 80K 协议必须在下一轮启动前裁决，避免最终论文出现训练预算口径冲突。

