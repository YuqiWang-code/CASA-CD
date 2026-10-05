# CASA-CD Run2 复盘：SYSU 极小目标瓶颈与下一轮结构实验设计

> **日期：2026-10-05**  
> **仓库：** `https://github.com/YuqiWang-code/CASA-CD`  
> **本次直接依据：**
> - `docs/temporary/run2_report.md`
> - `docs/temporary/run2_zero_cost_diag.json`
> - `docs/temporary/models_and_metrics_CASA-TViM_Run2.txt`
> - `models/model/layers/caacp_ss2d.py`
> - `models/model/str_fine_head.py`
> - `models/model/str_dcr.py`
> - `models/model/str_reparam.py`
> - `models/model/casa_tvim_str_net.py`
>
> **任务硬约束：** 256×256 双时相全监督二值变化检测；CDD / LEVIR / SYSU / WHU 四数据集；BCE+Dice、Adam 2e-4、poly+200 warmup、80K、batch 32、seed 16、test-as-val、threshold=0.5 固定；每次结构改动四数据集完整从头训练；有效 deploy Params ≤5M；论文只保留两个结构创新，不靠 loss / augmentation / threshold / 多 seed 包装。
>
> **最终硬目标：**
>
> \[
> F1_{\text{CDD}}\ge98,\quad
> F1_{\text{LEVIR}}\ge92.5,\quad
> F1_{\text{SYSU}}\ge85,\quad
> F1_{\text{WHU}}\ge95
> \]
>
> 本文严格区分：
> - **[直接事实]**：由 Run1/Run2 指标、D1/D2、当前源码直接支持；
> - **[强推断]**：多条独立证据指向同一机制，但尚未由单变量实验直接证实；
> - **[待验证假设]**：下一轮需要四数据集 80K 实验裁决；
> - 所有“预期 ΔF1”都是**实验设计先验区间，不是结果承诺**。

---

# 0. 结论先行

## 0.1 Run2 并没有出现“两个模块互相打架”，而是两个改动在 SYSU 上都把模型推向了错误的 Precision–Recall 方向

这是本轮最重要的结论。

Run1 M1 在 SYSU：

- Recall = **84.41**
- Precision = **82.55**
- F1 = **83.47**

Run2：

| 变体 | Recall | Precision | F1 | 相对 M1 |
|---|---:|---:|---:|---:|
| M1 | **84.41** | 82.55 | **83.47** | — |
| E1 CP-CAACP | 80.94 | **85.43** | 83.13 | −0.34 |
| E2 FRH | 80.10 | **86.05** | 82.97 | −0.50 |
| E3 CP+FRH | 80.69 | 85.12 | 82.85 | −0.62 |

**E1、E2、E3 的共同失败签名不是“Precision 太低”，而是 Recall 被压掉了 3.5～4.3pp。**

因此：

> **SYSU 下一轮不应该再做“更保守、更锐化、更抑制弱响应”的结构，而应该在保持当前 CAACP 的高 Recall 优势前提下，保护/恢复极小变化目标的细粒度正证据。**

---

## 0.2 SYSU small-component F1=0.1796 与当前网络尺度之间存在非常直接的几何矛盾

D2 定义：

```text
small component < 256 input pixels
```

若近似成正方形，则边长 `<16 px`。

映射到当前 backbone：

| 尺度 | 特征尺寸 | small component 的典型面积上限 |
|---|---:|---:|
| input | 256² | <256 px |
| 1/4 | 64² | <16 feature cells |
| 1/8 | 32² | <4 feature cells |
| **1/16** | **16²** | **<1 feature cell** |
| 1/32 | 8² | <0.25 cell |

而当前 CAACP 恰恰位于 **Stage3 1/16**。

也就是说：

> **D2 中定义的“small”目标，在 CAACP 所在层平均已经是 sub-token / sub-cell 目标。**

这比“边界不够锐”更根本。

FRH 位于 decoder 最末端，即使把输出从 64² 提到 128²，它仍然只能重新加工已经存在的特征，不能重新创造在 1/16 或更早阶段已经丢失的极小目标语义。

这很好地解释了：

- FRH 对 LEVIR 有轻微收益；
- 对 SYSU 却 Recall 大幅下降；
- SYSU small F1 仍是全项目最大的单一缺口。

---

## 0.3 当前 CAACP 不是“完全失败”，而是呈现非常清楚的**尺度选择性**

用 Run1 的 A2_STR（无 CAACP）对 M1（有 CAACP）做 D2 对比：

### SYSU

| 指标 | A2_STR | M1 | Δ |
|---|---:|---:|---:|
| band2 | 0.6490 | **0.6710** | **+0.0220** |
| band4 | 0.6872 | **0.7081** | **+0.0209** |
| small | **0.1838** | 0.1796 | **−0.0042** |
| medium | 0.4152 | **0.4620** | **+0.0468** |
| large | 0.7910 | **0.8273** | **+0.0363** |

同时像素级：

```text
A2_STR : Recall 81.24 / Precision 85.29 / F1 83.21
M1     : Recall 84.41 / Precision 82.55 / F1 83.47
```

这说明当前 Stage3 CAACP：

- **确实增强了变化敏感性**；
- 对 medium / large change 很有效；
- 对整体 boundary 也有效；
- 但对 `<256 px` 的极小目标反而略有伤害。

所以真正的问题不是：

> “change-aware context 不应该做。”

而是：

> **当前 change-aware context 的作用尺度太晚、太粗；它适合 medium/large context，却没有覆盖 SYSU 最困难的 sub-token small changes。**

这直接决定了下一轮应该改“尺度与残差路径”，而不是继续改 score 校准。

---

# 1. Q1：E1 / E2 / E3 的消融模式到底说明了什么？

# 1.1 先看 2×2 因子关系：没有证据支持“优化冲突”

定义：

\[
\Delta_1=F1(E1)-F1(M1)
\]

\[
\Delta_2=F1(E2)-F1(M1)
\]

若两改动完全独立相加，则组合预测：

\[
\Delta_{\text{add}}=\Delta_1+\Delta_2
\]

实际组合交互项：

\[
I=\Delta(E3)-\Delta_1-\Delta_2
\]

得到：

| Dataset | E1 Δ | E2 Δ | E3 Δ | 交互项 \(I\) |
|---|---:|---:|---:|---:|
| CDD | −0.04 | −0.01 | 0.00 | **+0.05** |
| LEVIR | +0.07 | +0.14 | +0.22 | **+0.01** |
| SYSU | −0.34 | −0.50 | −0.62 | **+0.22** |
| WHU | +0.11 | −0.36 | −0.09 | **+0.16** |

如果 E1 与 E2 存在明显的**破坏性优化冲突**，应更容易看到：

\[
I<0
\]

即组合比两个单项负效应简单相加还差。

但四个数据集的 \(I\) 都是非负。

特别是 SYSU：

```text
若完全相加：-0.34 + -0.50 = -0.84
实际 E3：                    -0.62
```

组合反而比简单相加**少坏 0.22pp**。

### 结论

**[强推断]：SYSU 的失败不是“E1 和 E2 互相打架”。**

更合理的解释是：

> **E1 和 E2 独立地都把模型推向“高 Precision / 低 Recall”的保守解；二者作用方向相似、部分冗余，所以组合没有出现额外崩溃。**

也不支持“参数容量竞争”：

- E1 = **0 新参数**；
- E2 = **仅 +768 deploy 参数**；
- 总模型仍 4.881M；
- 不存在明显的参数预算挤占。

所以三选一：

1. 优化冲突 —— **证据弱**；
2. 容量竞争 —— **基本可排除**；
3. **机制错配 / 作用尺度错配 —— 证据最强。**

---

# 1.2 E1 CP-CAACP：诊断对象真实存在，但它不是主瓶颈

## 原假设

D1 发现 LEVIR：

```text
zero-change images = 1113 / 2048 ≈ 54%
zero group cell entropy = 1.2504
>5% group entropy        = 1.2624
uniform ln4             = 1.3863
```

即零变化图反而被 rank-only 分配得更不均匀。

所以 E1：

```text
rank-only:
w ∝ ε + rank

CP:
w ∝ 1 + s·rank
```

希望弱绝对变化时退化到 uniform。

这个诊断本身没有错。

但是训练结果告诉我们：

### LEVIR

| | Recall | Precision | F1 |
|---|---:|---:|---:|
| M1 | 90.33 | 91.82 | 91.07 |
| E1 | **90.64** | 91.64 | **91.14** |

如果原假设“主要问题是 zero-change FP”是主导机制，那么更期待：

```text
Precision ↑
Recall ≈
```

实际却是：

```text
Recall +0.31
Precision -0.18
```

这不是原先期待的错误修复签名。

### 因此应如何解读？

**[直接事实]** CP 让 LEVIR F1 +0.07，但没有按“Precision 修复”机制工作。

**[强推断]**：

> D1 找到了一个真实现象，但它只是 LEVIR 的**次要误差源**，不足以解释 1.4pp 左右的目标缺口。

不能因为 E1 最终 +0.07 就写：

> “CP 成功抑制零变化 false positives”。

当前 Recall / Precision 数据不支持这一表述。

---

# 1.3 为什么 LEVIR 对 E1 / E2 / E3 都略正，但幅度全小于 0.25pp？

这是一个非常稳定的模式：

```text
E1 +0.07
E2 +0.14
E3 +0.22
```

而且：

\[
0.07+0.14=0.21\approx E3(0.22)
\]

说明两项确实在修**相对独立的小问题**，但都没碰到 LEVIR 的主要上限。

## 原因一：LEVIR 的 small / boundary 有缺口，但没有 SYSU 那么灾难性

D2：

| Dataset | small F1 | band2 |
|---|---:|---:|
| CDD | 0.660 | 0.857 |
| LEVIR | **0.627** | **0.801** |
| SYSU | **0.180** | **0.671** |
| WHU | 0.447 | 0.832 |

LEVIR 的 small / boundary 确实偏低，所以 FRH 能有 +0.14。

但它不是“几乎检测不到”的状态。

因此一个末端 128² 局部修正可以改善少量边界/FP，却不会带来 +1pp 以上跳跃。

---

## 原因二：E1 的实际扰动幅度天然很小

CAACP：

\[
c=c_{avg}+\beta(c_{ca}-c_{avg})
\]

LEVIR 训练后 β 约 0.018～0.023。

也就是说 E1 只是在一个已经被约 2% 标量门控的 context perturbation 中，再改变：

```text
cell weighting rule
```

它不是 backbone feature 的大改造。

因此 E1 的合理量级本来就更像：

```text
0.0x ~ 0.2x pp
```

而不是 1pp 级主提升。

---

## 原因三：E1 / E2 都是“后端修正”，不能创造早期漏掉的小建筑语义

- E1：Stage3 / 1/16；
- E2：decoder 后 / 64→128。

而 LEVIR small component 同样在 1/16 接近 sub-token。

所以这两个机制更多在做：

```text
已有 evidence 的重加权 / 整形
```

不是：

```text
恢复原本没有编码出来的 tiny-change evidence
```

### 结论

> **LEVIR 的 +0.22 说明 E1/E2 有小量互补，但同时也明确告诉我们：继续在 score normalization 或最终 logits sharpening 上搜索，边际收益大概率已经耗尽。**

---

# 1.4 为什么 SYSU 对 E1 / E2 都负，而且组合更差？

核心不是“组合更差”本身，而是要看 Recall / Precision。

## E1：CP-CAACP 把当前 CAACP 最有价值的 SYSU 特性压掉了

Run1：

```text
A2_STR:
Recall 81.24
Precision 85.29
F1 83.21

M1 rank-CAACP:
Recall 84.41   (+3.17)
Precision 82.55 (-2.74)
F1 83.47       (+0.26)
```

**当前 rank-CAACP 在 SYSU 的主要价值就是大幅提高 Recall。**

E1 后：

```text
Recall 80.94
Precision 85.43
```

它几乎把模型重新推回了 A2_STR 的“保守高 Precision”区域，而且 Recall 甚至比 A2 还低 0.30pp。

### 机制解释

SYSU D1：

```text
zero images = 0
0-1% images = 0
1-5% = 968
>5%  = 3032
```

也就是说：

> **CP-CAACP 针对的“零变化 / 极弱变化图”问题，在 SYSU 几乎不存在。**

对这种真实变化密集的数据，rank-only 的强相对选择本来就是 CAACP 提升 Recall 的来源之一。

CP 把权重往 uniform 拉，等于：

```text
减少变化上下文的相对突出
→ 弱/小变化更难激活
→ Recall 下滑
→ Precision 上升
```

这和结果完全一致。

因此 E1 对 SYSU 不是“优化没训好”，而是**设计目标本身与数据集错误类型相反**。

---

## E2：FRH 同样是典型“提高置信度、牺牲弱正样本”的机制

SYSU：

```text
M1 : Recall 84.41 / Precision 82.55
E2 : Recall 80.10 / Precision 86.05
```

这是非常大的交换：

- Recall **−4.31pp**
- Precision **+3.50pp**

FRH 并没有提升 tiny-target semantic recall。

它更像：

> 把现有 64² decoder evidence 在 128² 上做局部空间重整，使高置信区域更干净，但把原本就很弱的小变化响应进一步筛掉。

这与 D2：

```text
small F1 = 0.1796
```

高度吻合。

如果目标在 decoder 输入中已经只有非常弱的通道响应：

```text
3×3 local refinement
```

并不能凭空恢复语义，反而容易把孤立弱正响应视作局部噪声。

---

## E3 为什么是 −0.62 而不是崩得更厉害？

因为 E1 和 E2 的作用方向相似：

```text
都提高 Precision
都降低 Recall
```

它们不是互相“打架”，而是**部分重复做同一种保守化**。

所以 E3：

```text
Recall 80.69
Precision 85.12
```

依旧落在同一象限。

但其 −0.62 比简单相加 −0.84 好，说明二者有一定冗余/补偿。

### 最终判断

**证据排序：**

1. **机制错配 / 尺度错配：强**
2. late-stage correction 无法恢复 tiny semantics：强
3. spatial over-refinement / weak-positive suppression：中—强
4. 优化冲突：弱
5. 参数容量竞争：极弱

---

# 2. 从 D2 进一步得到一个比“small F1 很低”更重要的结论

# 2.1 CAACP 对 SYSU medium / large 非常有效，却对 small 无效

再次看：

```text
small  : -0.0042
medium : +0.0468
large  : +0.0363
band2  : +0.0220
```

这几乎是一个教科书式的“context scale mismatch”签名。

如果 change score 本身完全错误，那么通常 medium / large 也不应该同时明显改善。

所以不应该因为 Run2 CP 失败就否定 CAACP。

更合理的论文结论是：

> **Stage3 CAACP 已证明 change-aware context 对可解析的中大变化区域有效；其主要失败点是 1/16 网格无法为 sub-token small change 提供足够独立的空间证据。**

这给下一版创新一一个非常清楚的升级方向：

```text
不是“更强 score”
而是“更早/更细粒度的 context preservation”
```

---

# 2.2 SYSU 的主要任务不是“再提升 Precision”

M1 已经：

```text
P=82.55
R=84.41
```

E1/E2 能轻易把 P 推到 85~86，但代价是 R 跌到80。

所以模型并不缺“变得保守”的能力。

它缺的是：

> **在不把背景一起点亮的情况下，把 small/weak true change 留住。**

这决定了后续候选的设计标准：

### 应鼓励

- 在 1/4、1/8 层保留 tiny temporal evidence；
- 减少 tiny evidence 被 coarse context / spatial smoothing 消掉；
- 在现有 CAACP recall 优势上修 small target；
- fine-scale temporal structure。

### 不应再做

- score 再压平；
- logits 再锐化；
- 输出端再加 edge filter；
- 单纯提升 threshold-like selectivity；
- 后处理式 suppression。

---

# 3. Q2：下一轮 5 个结构候选排序

## 总排序

| 优先级 | 候选 | 命中的主要证据 | Deploy 参数变化 | 预期 SYSU | 预期 LEVIR | 风险 |
|---|---|---|---:|---:|---:|---|
| **1** | **RA-CAACP：Residual-Anchored CAACP** | CAACP medium/large↑但 small↓；当前 residual 会把 change-aware pooled signal 从 dense path 再减掉 | **0** | **+0.15~+0.45** | −0.05~+0.20 | **低** |
| **2** | **FS-TAR：1/4 Fine-Scale Spatial-Temporal Rep-TAR** | SYSU small<256px 在1/4仍有<16 cells，而1/16已<1 cell；需要更早的 temporal-local evidence | **约 +0.074M** | **+0.35~+0.80** | **+0.15~+0.45** | 中 |
| **3** | **S2-HCAACP：Stage2-only Hierarchical CAACP relocation** | 当前 Stage3 CAACP medium/large强、small弱；需把 change-aware aggregation 前移到1/8 | **≈0** | **+0.25~+0.65** | 0~+0.35 | 中 |
| **4** | **SP-DCR：Spatial-Preserving DCR endpoint** | FRH 额外 spatial refine 使 SYSU Recall −4.31；怀疑末端 spatial mixing 继续抹弱小目标 | **减少参数** | +0.15~+0.45 | 0~+0.25 | 中 |
| **5** | **Fine-STR rank r24→48** | 只可能改善训练期 channel optimization；不改变 spatial evidence 或 deploy function class | **0 deploy** | 0~+0.25 | 0~+0.15 | **低，但收益上限低** |

> 上述区间全部是**待验证先验**。  
> SYSU 从 83.47 到85还差1.53pp，不能期待一个微改动必然一次补齐；但前四个候选至少直接命中 D2 指出的结构瓶颈，Fine-STR rank expansion 则没有。

---

# 4. 候选一：RA-CAACP —— 当前最干净、最值得先跑的实验

## 4.1 当前 CAACP 的关键公式

当前源码：

\[
c=c_{avg}+\beta(c_{ca}-c_{avg})
\]

然后：

\[
res=x-\operatorname{Up}(c)
\]

最终：

\[
y=\operatorname{Up}(SS2D(c))+res
\]

因此：

\[
y=x+\operatorname{Up}(SS2D(c)-c)
\]

---

## 4.2 为什么这可能特别伤 small target？

设一个 2×2 cell 中只有一个位置属于极小真实变化。

CAACP 会让 \(c_{ca}\) 更偏向这个 changed location。

但随后 residual 又使用：

\[
x-\operatorname{Up}(c)
\]

这意味着：

> **越被 CAACP 选中的局部变化证据，越有一部分被 coarse context 从 dense residual 中减掉。**

对 medium / large 变化：

- 目标跨多个 cell；
- 即使局部高频 residual 减弱，SS2D context 仍能表达完整结构。

对 small target：

- 本来只剩一个或不到一个 Stage3 token；
- 再把其局部 contrast 吸收到 coarse context 中并从 residual 扣掉；
- 很容易直接消失。

这与 D2：

```text
medium +0.0468
large  +0.0363
small  -0.0042
```

高度吻合。

---

## 4.3 RA 改法

保持：

\[
c=c_{avg}+\beta(c_{ca}-c_{avg})
\]

但 residual 永远锚定官方 uniform context：

\[
res_{ref}=x-\operatorname{Up}(c_{avg})
\]

最终：

\[
y=\operatorname{Up}(SS2D(c))+res_{ref}
\]

即：

\[
y=x+\operatorname{Up}(SS2D(c)-c_{avg})
\]

### 关键性质

- β=0 时仍与 TinyViM 官方路径完全一致；
- 不加参数；
- 不改 8×8 context lattice；
- 不改 CrossScan；
- 不改 score；
- 不改 Stage；
- 不改 backbone；
- 唯一变量就是：**change-aware context 是否应该参与 dense residual subtraction**。

这是一个非常干净的机制实验。

---

## 4.4 它命中哪条瓶颈？

**直接命中：**

> CAACP 对 medium/large 有益，但对 small 轻微负向。

RA 的目标不是“提升 CAACP 强度”，而是：

> **保留 CAACP 对 context 的作用，同时保护 original dense high-frequency residual。**

因此它比 CP-CAACP 更对症。

---

## 4.5 预期

### SYSU

**待验证先验：+0.15~+0.45pp**

理想签名：

- Recall 恢复/继续提高；
- Precision 不应发生 E1/E2 那种 +3pp / Recall−4pp 的巨大交换；
- small-component F1 明显高于0.1796；
- medium/large 不丢。

### LEVIR

**−0.05~+0.20pp**

因为 LEVIR 不是明显的 CAACP small-collapse 数据集，收益可能较小。

---

## 4.6 失败信号

以下任一出现，RA 路线立即停：

1. SYSU F1 ≤83.47；
2. SYSU small F1 ≤0.180；
3. medium / large 的 CAACP 现有优势明显回退；
4. SYSU Recall仍低于 M1 84.41；
5. CDD <97.18 或 WHU <95.07（按你的严格不退纪律）。

---

## 4.7 最小消融

只需要一个新变体：

```text
M1_RA
TinyViM-S-Slim
+ rank CAACP
+ RA residual
+ 当前 TAR/DCR
```

四数据集完整80K。

不要：

- 同时用 CP；
- 同时用 FRH；
- 改 β init；
- 改 score；
- 改 D。

这样结果才能真正回答：

> “small loss 是否来自 change-aware context 同时侵入 dense residual？”

---

# 5. 候选二：FS-TAR —— 我认为最有机会真正抬 SYSU 上限的 STR 改法

## 5.1 机制动机

D2 small：

```text
<256 input px
```

在：

```text
1/4  → <16 feature cells
1/8  → <4 cells
1/16 → <1 cell
```

所以如果目标是提升 tiny target Recall，最合理的位置不是：

```text
1/16 CAACP之后
或者 128² logits末端
```

而是：

> **在 1/4 temporal fusion 时就把“局部空间模式 + 双时相差异”编码进去。**

---

## 5.2 当前 TAR stage1

当前 `TemporalRep1x1`：

```text
[P,Q]
 ├ concat 1×1
 ├ sum 1×1
 └ signed-diff 1×1
       ↓
     D=96
       ↓
  RepLocalBlock
```

所有时相代数首先都是**逐位置 1×1**。

空间邻域要等 temporal fusion 后才由 RepDW3 处理。

---

## 5.3 建议：只把最细 stage1 改成 TemporalRep3x3

训练图：

```text
main:
[P,Q] → concat 1×1 + BN

aux:
(P+Q) → 1×1 + BN               # 可保留现有
(Q-P) → 3×3 Conv + BN           # 新：tiny-change spatial-temporal branch

sum
↓
SiLU
↓
现有 RepLocalBlock
```

新 signed-diff spatial branch zero-init，epoch0 与当前 M1 完全一致。

### deploy

所有线性 branch 可以解析折成：

```text
single Conv3×3(2*C1 → D)
```

stage1：

```text
C1=48
2C1=96
D=96
```

当前 deploy temporal 1×1：

\[
96\times96=9,216
\]

新 3×3：

\[
96\times96\times9=82,944
\]

净增：

\[
73,728 \approx 0.0737M
\]

总 deploy：

\[
4.880M+0.0737M\approx4.954M
\]

仍 ≤5M，剩余约46K安全空间。

### FLOPs

64² 上增加约0.30G量级 MAC/FLOPs 口径，整体仍约3G左右，属于当前可接受量级，但必须服务器实测。

---

## 5.4 为什么它比 FRH 更匹配 D2？

FRH：

```text
已有 decoder evidence
→ 128² spatial correction
```

FS-TAR：

```text
A/B fine-scale feature
→ spatial-temporal difference modeling
→ 后续 semantic/context
```

前者只能“修图”。

后者是在小目标还没有被下采样成 sub-token 之前：

> **先提取真正的 tiny temporal evidence。**

这正是 D2 指向的结构位置。

---

## 5.5 为什么还符合第二创新的论文故事？

它不是第三个模块。

它就是 TAR 从：

> Temporal Algebraic Reparameterization

进一步升级为：

> **Fine-scale Spatial-Temporal Algebraic Reparameterization**

训练多分支：

```text
concat / sum / spatial signed-diff
```

部署：

```text
单个3×3 Conv
```

依旧是完全一致的：

> **training-rich, inference-simple**

故事反而比单纯 r24→48 更强。

---

## 5.6 预期

### SYSU

**+0.35~+0.80pp**

它是本文五个候选里，我认为最可能出现 >0.5pp SYSU 改善的一个。

希望看到：

- Recall ↑；
- small component ↑显著；
- Precision不发生大幅牺牲；
- medium/large不退。

### LEVIR

**+0.15~+0.45pp**

小建筑也可能受益，但其目标稀疏，局部 signed-diff 对配准/纹理更敏感，因此不应预期过大。

---

## 5.7 风险

主要风险：

- CDD seasonal/pseudo-change 被3×3时相差异放大；
- WHU配准误差导致局部边缘响应；
- deploy FLOPs增加约0.3G。

所以必须坚持：

```text
只 stage1 用3×3
stage2-4 保持1×1
```

不要四尺度一起改。

---

## 5.8 失败信号

立即停止条件：

1. SYSU small F1 提升 < +0.015；
2. SYSU F1 <83.47；
3. Recall没有提高；
4. CDD / WHU 任一明显回退；
5. deploy >5M；
6. fold disagreement 不为0/超现有数值门槛。

---

## 5.9 最小单变量消融

```text
M1 baseline
vs
M1_FS_TAR
```

唯一变量：

```text
stage1 TemporalRep1x1
→
stage1 TemporalRep3x3
```

其余完全不动。

---

# 6. 候选三：S2-HCAACP —— 把变化感知上下文前移，而不是继续改 score

## 6.1 为什么值得做？

当前 Stage3 CAACP：

```text
16² → 2×2 weighted pool → 8² context
```

对 small target：

```text
<1 Stage3 feature cell
```

已经太晚。

TinyViM index=1（Stage2 / 1/8）原始 low-frequency path 等效：

```text
32² → pool factor4 → 8²
```

因此可以在**保持最终 8² SS2D context lattice 不变**的前提下，把 change-aware selection 前移。

---

## 6.2 推荐形式：Stage2-only Hierarchical CAACP

不要 Stage2+Stage3 叠加。

**直接把 CAACP 从 Stage3 移到 Stage2。**

官方：

\[
c_{avg}=\operatorname{AvgPool}_{4\times4}(x)
\]

建议 change-aware candidate：

```text
32²
→ change-aware 2×2 weighted pool
→ 16²
→ uniform 2×2 pool
→ 8²
```

得到：

\[
c_{ca}^{S2}
\]

再：

\[
c=c_{avg}+\beta(c_{ca}^{S2}-c_{avg})
\]

β=0 时精确恢复官方 Stage2。

---

## 6.3 为什么不是直接“Stage2再加一个 CAACP”？

因为论文只保留两个创新，不应变成：

```text
S2 CAACP + S3 CAACP + TAR + DCR + ...
```

正确实验应是：

```text
Stage3 CAACP
↓ 替换
Stage2-only CAACP
```

测试的科学问题是：

> “change-aware context 的最优作用尺度是不是应该与 tiny change 的空间分辨率对齐？”

---

## 6.4 它命中什么证据？

最直接命中：

```text
SYSU small 0.180
Stage3 CAACP medium/large ↑，small ↓
```

在1/8：

```text
small <4 feature cells
```

至少仍可形成真实的局部结构，而不是 sub-token。

---

## 6.5 预期

### SYSU

**+0.25~+0.65pp**

如果“CAACP尺度错位”是主要原因，应该首先看到 small component 明显上升。

### LEVIR

**0~+0.35pp**

但浅层 feature 更受：

- texture；
- shadow；
- radiometry；
- minor misregistration

影响，所以 LEVIR/CDD 风险高于 RA。

---

## 6.6 失败信号

1. SYSU small F1 无明显提升；
2. SYSU Recall不升；
3. CDD Precision明显下降；
4. LEVIR zero-change FP 增加；
5. overall F1 没有跨数据集净收益。

如果失败，不要继续做：

```text
S2+S3 multi-CAACP
```

那会变成模块堆叠。

---

# 7. 候选四：SP-DCR —— 不是“继续增强边界”，而是减少末端过度空间平滑

## 7.1 FRH 失败告诉我们什么？

FRH 的 SYSU：

```text
Recall 84.41 → 80.10
Precision 82.55 → 86.05
```

这不是“边界还不够锐”。

恰恰说明：

> **继续在末端做空间局部修正，会把弱正样本变得更保守。**

当前 DCR 在 64²：

```text
block1:
DW3 → PW1

refine:
DW3 → PW1
```

也就是 fine scale 已经连续经过两套 spatial local mixing。

---

## 7.2 建议

把最后：

```python
self.refine = RepLocalBlock
```

改为：

```text
Pointwise-only Rep refinement
```

即：

```text
RepPW1x1
→ SiLU
```

删除最后一个 DW3。

这样：

- 保留 channel recalibration；
- 不再对 tiny spatial response 做第二次3×3邻域混合；
- deploy 参数/FLOPs反而下降；
- 仍是 STR reparameterization；
- 不是新增模块。

可命名为：

> **Spatial-Preserving DCR endpoint (SP-DCR)**

---

## 7.3 预期

### SYSU

**+0.15~+0.45pp**

若“末端 over-smoothing”成立，Recall / small F1 应上升。

### LEVIR

**0~+0.25pp**

小建筑也可能受益；但边界的平滑一致性可能略下降。

---

## 7.4 失败信号

1. small F1 不升；
2. band2/band4下降明显；
3. WHU F1 <95.07；
4. CDD precision下降；
5. SYSU Recall没有恢复。

失败则说明：

> FRH 的负效应是 FRH 自身，而不是“当前 DCR 已经过度 spatial refinement”。

应立刻停止该线。

---

# 8. 候选五：Fine-STR rank expansion —— 可以做，但它**没有直接命中 small bottleneck**

这是你预注册的另一个候选，需要明确评价。

## 8.1 当前 RepPW1x1

D=96：

```text
main 96×96
+
serial low-rank:
96 → r=24 → 96
+
diag
+
residual
```

部署后全部折成：

```text
single arbitrary 96×96 1×1 Conv
```

如果 r：

```text
24 → 48
```

deploy 结构和参数**完全不变**。

---

## 8.2 它实际上改变了什么？

只改变：

> **训练期到同一个 deploy 1×1 kernel 的优化参数化路径。**

注意：

当前 `pw_main` 本身已经是完整：

\[
96\times96
\]

所以 deploy function class 本来就是 full-rank。

r24→48：

- 不增加空间分辨率；
- 不增加 spatial receptive field；
- 不恢复 1/16 前丢掉的 tiny evidence；
- 不改变 deploy 表达空间；
- 只可能改善 optimization / regularization。

---

## 8.3 它能命中什么？

可能命中：

- SYSU 类别/纹理多样性导致的 channel mixing optimization；
- medium/large feature transformation；
- 当前 STR training branch 的收敛质量。

**但不能直接命中：**

```text
small-component F1=0.18
```

因为 small bottleneck 是明显的 spatial/evidence preservation 问题。

---

## 8.4 预期

### SYSU

**0~+0.25pp**

### LEVIR

**0~+0.15pp**

如果它突然 +0.5pp 以上，反而说明我们的“空间瓶颈”判断不完整，训练参数化是更大因素。

---

## 8.5 失败信号

很简单：

```text
四数据集 ΔF1 都在 ±0.15 内
```

则说明：

> r24 已足够，继续 r64/r96 没意义。

**不允许再做 rank sweep。**

---

# 9. RA-CAACP 和 Fine-STR 到底谁更值得先跑？

## 结论

> **RA-CAACP 值得跑；Fine-STR rank expansion 暂时不值得作为下一批第一优先。**

---

## 9.1 RA 命中了哪条数据？

它直接解释：

```text
CAACP:
SYSU medium/large ↑
SYSU small ↓
```

并对应当前明确的公式结构：

```text
change-aware c
既送进SS2D
又从dense residual中扣除
```

所以它有：

- 明确症状；
- 明确代码位置；
- 明确机制；
- 0 参数；
- β=0预训练兼容；
- 单变量可证伪。

是一个合格的下一轮科研实验。

---

## 9.2 Fine-STR rank expansion 命中了哪条数据？

没有直接命中 D2 最大缺口。

它更像：

> “现有 STR 的训练参数化也许还不够丰富。”

但现在没有看到：

- train loss 欠拟合；
- deploy PW rank受限；
- channel bottleneck审计；
- r24分支饱和

这样的证据。

所以它目前是：

**低风险，但证据弱。**

如果算力有限，不应该和 RA 同优先级。

---

# 10. 我建议下一轮不要直接跑 Fine-STR，而是这样排

## 第一批：8 个 80K run

### E4：RA-CAACP ×4

验证：

> Stage3 CAACP 的 small failure 是否来自 residual cancellation。

### E5：FS-TAR ×4

验证：

> SYSU tiny change 是否需要在1/4 temporal fusion时就引入 spatial-temporal local evidence。

这两个实验：

- 一个改创新一；
- 一个改创新二；
- 互相正交；
- 都直接由 Run2 / D2 数据推出；
- 都保持论文仍只有两个创新。

---

# 11. 为什么不是先跑 S2-HCAACP？

S2-HCAACP 的潜在收益很高，但风险也高：

- shallow feature 更容易 pseudo-change；
- 改了 CAACP 作用 Stage；
- pool factor从2变4，需要更谨慎的预训练等价 smoke。

RA 则只改一行核心残差逻辑。

所以顺序应该是：

```text
RA
  ↓
若 small/Recall 有明确恢复
  → 保留 Stage3，不必前移

若 RA 无效
  ↓
才说明问题不是 residual cancellation
  → 进入 S2-HCAACP 尺度重构
```

这样不会浪费实验。

---

# 12. 在启动任何新80K前，先做两个零成本复盘

你现在已经有 E1/E2/E3 checkpoint。

**不要马上训练。**

先把 `run2_zero_cost_diag.py` 对 Run2 三个模型也跑一遍。

---

## Z1：E1/E2/E3 的 component profile

输出：

```text
small / medium / large
band2 / band4
```

特别看 SYSU：

### 假设 H-E1

若 E1：

```text
medium/large 保持
small 明显下降
```

则确认：

> CP 的保守化首先杀 tiny changes。

### 假设 H-E2

若 E2：

```text
small 显著下降
medium/large 较稳定
```

则确认：

> FRH 是 weak-positive spatial suppression，而不是优化异常。

这两个结果会进一步提高 RA / FS-TAR 的优先级。

---

## Z2：E1/E2/E3 的 Recall 按 component size 分解

当前只有 component F1。

建议再输出：

```text
small recall / precision
medium recall / precision
large recall / precision
```

因为整体结果已经清楚显示：

```text
E1/E2主要问题 = Recall collapse
```

如果 small recall 的下降远大于 small precision 的改善，则下一轮“tiny evidence preservation”就从强推断升级为近直接证据。

这一步不改训练、不需新80K。

---

# 13. 下一轮预注册建议

## E4_RA_CAACP

### 唯一变量

```text
当前:
res = x - up(c)

RA:
res = x - up(c_avg)
```

### 预注册成功条件

四数据集硬性：

```text
CDD  >= 97.18
WHU  >= 95.07
```

机制成功条件：

```text
SYSU F1 > 83.47
SYSU Recall >= 84.41
SYSU small F1 >= 0.195
```

其中 small 从0.1796→0.195相当于至少 +0.0154 absolute，才算真的命中 small，而不是整体随机波动。

LEVIR：

```text
不要求必须大涨
但不得明显低于91.07
```

### 失败裁决

若：

```text
SYSU small <0.190
或 SYSU F1 <=83.47
```

则停止 RA，不调 β、不叠 CP。

---

## E5_FS_TAR

### 唯一变量

只 Stage1：

```text
TemporalRep1x1
→ TemporalRep3x3
```

其余 Stage2-4 不动。

### deploy

```text
single Conv3×3(96→96 pair-channel input)
```

模型估计：

```text
4.880M + 0.0737M ≈ 4.954M
```

实现后必须机器实测，不用估算值作正式报告。

### 成功条件

```text
CDD  >= 97.18
WHU  >= 95.07

SYSU F1 >= 83.80
SYSU Recall > 84.41
SYSU small F1 >= 0.205

LEVIR F1 >= 91.20
```

如果 SYSU overall涨但 small不涨，则：

> FS-TAR 不是通过目标机制起作用，论文中不能包装成“small-change preservation”。

---

# 14. 两个实验之后的决策树

```text
E4 RA
│
├─ SYSU small↑ + F1↑
│   └─ 保留 RA，创新一升级完成
│
└─ FAIL
    └─ 转 S2-HCAACP，不再调 Stage3 residual

E5 FS-TAR
│
├─ SYSU small↑ + Recall↑ + CDD/WHU不退
│   └─ 保留，作为 STR 的 fine-scale extension
│
└─ FAIL
    ├─ 若 small仍不升 → 不再扩大 spatial kernel
    └─ 下一候选 SP-DCR 或 S2-HCAACP
```

只有当：

```text
RA 和 FS-TAR 各自单变量都成立
```

才允许跑组合。

否则不要重复 Run2 的：

```text
单项都弱/失败
→ 仍然组合
```

虽然 Run2 E3 是为了完成预注册矩阵，但下一轮应更节省实验预算。

---

# 15. 关于“最终要到 SYSU 85”必须现实一点

当前：

```text
83.47 → 85
需要 +1.53pp
```

Run2 已经证明：

- score calibration 不是 1pp 级杠杆；
- late output refinement 不是 1pp 级杠杆；
- 两者组合也不是。

所以想达到85，下一轮至少要命中一个**表示层级上的大瓶颈**。

当前证据最支持的就是：

> **tiny changes 在 1/16 已成为 sub-token，应该在 1/4~1/8 temporal representation 阶段被显式保护。**

这也是为什么我把 FS-TAR / S2-HCAACP 排在 Fine-STR rank expansion 前面。

---

# 16. 对 LEVIR 的独立判断

LEVIR 当前：

```text
91.07 → 92.5
差1.43pp
```

Run2：

```text
CP +0.07
FRH +0.14
组合 +0.22
```

这说明 LEVIR 与 SYSU 不完全是同一个问题。

## LEVIR 的当前错误结构更像：

```text
稀疏小建筑
+
大量 zero-change image
+
small/boundary 中等偏弱
+
false positive 成本高
```

而不是 SYSU 的：

```text
密集变化
+
极小目标严重漏检
+
Recall 是主要矛盾
```

因此不存在一个“越 suppress FP 越好”的统一模块。

### 这也是为什么 Run2 出现：

```text
同一结构
LEVIR 正
SYSU 负
```

不是异常，而是数据分布和错误类型不同。

---

# 17. 下一版方法应该追求“数据自适应”吗？

暂时**不要**。

很容易想到：

```text
LEVIR用保守模式
SYSU用高Recall模式
```

再做一个 learned gate。

但这会：

- 引入第三机制；
- 增加设计自由度；
- 容易变成 dataset-specific tuning；
- 破坏论文简洁性。

更合理的是寻找一个结构性质：

> **大目标保留 context；小目标保留 dense local evidence。**

这正是：

```text
RA-CAACP
+
fine-scale STR
```

共同的机制方向。

它不需要知道当前数据集是谁。

---

# 18. 代码修改级建议

# 18.1 RA-CAACP

文件：

```text
models/model/layers/caacp_ss2d.py
```

当前逻辑约为：

```python
c_avg = self.pool(x_low)
c_ca = change_weighted_pool2x2(...)
c = c_avg + beta * (c_ca - c_avg)

res = x0 - interpolate(c)
x_low = c
...
x_low = scan(x_low)
x_low = interpolate(x_low) + res
```

RA：

```python
c_avg = self.pool(x_low)
c_ca = change_weighted_pool2x2(...)
c = c_avg + beta * (c_ca - c_avg)

# 唯一改动
res = x0 - interpolate(c_avg)

x_low = c
...
```

新增架构 sidecar：

```text
caacp_residual_mode = current | avg_anchor
```

避免 eval 静默加载错结构。

---

# 18.2 FS-TAR

建议不要直接改通用 `TemporalRep1x1`。

新建：

```text
TemporalRepFine3x3
```

只给 `stage1` 使用。

训练：

```text
concat main 1x1
sum aux 1x1
signed-diff aux 3x3
```

辅助分支继续遵守：

```text
zero-init
BN独立
FP64 fold
```

折叠时：

1. concat 1×1 pad center→3×3；
2. sum 1×1 pad center；
3. diff 3×3：
   - 加到 Q half；
   - 负加到 P half；
4. 合成最终：

```text
W: (D, 2C, 3, 3)
b: (D,)
```

最终单 `Conv2d(2C,D,3,pad=1)`。

---

# 19. Smoke 必测

## RA

- β=0 与当前 TinyViM 官方路径逐位等价；
- A/B swap；
- c_avg / c_ca shape；
- gradient：
  - beta 非零；
  - SS2D parameter grad 非零；
- 旧 current mode 行为完全不变。

---

## FS-TAR

- aux zero-init 时：
  - new train graph 与 current stage1 output bitwise/数值等价；
- fold：
  - random input max_abs；
  - real batch max_abs；
  - 0.5 disagreement=0；
- deploy params ≤5M；
- FLOPs；
- stage2-4 权重/代码完全不动。

---

# 20. 最终推荐

## 下一轮第一优先

### **RA-CAACP × 4**

理由：

- 0 deploy 参数；
- 改动最小；
- 直接由“medium/large↑ small↓”导出；
- 保留当前 CAACP 的整体 Recall 优势；
- 机制可证伪性最好。

---

## 下一轮第二优先

### **FS-TAR × 4**

理由：

- 是唯一一个真正把变化建模前移到 small 仍然可解析的 1/4 feature 上的候选；
- 直接攻击 SYSU 的最大单一缺口；
- 仍是 STR 第二创新内部升级；
- deploy 估计约4.954M，满足预算；
- 不增加额外独立模块。

---

## 暂缓

### Fine-STR r24→48

不是说它“错误”，而是：

> **当前证据没有显示训练期 channel-rank 是瓶颈。**

它应排在：

```text
RA
FS-TAR
S2-HCAACP / SP-DCR
```

之后。

如果前面的空间证据保护路线都失败，再拿 Fine-STR 做低风险 optimization-parameterization 对照更合理。

---

# 21. 一句话概括这轮真正学到了什么

> **Run2 最重要的结果不是“CP 和 FRH 没涨”，而是它们共同暴露了当前 SYSU 的错误方向：模型完全有能力通过结构变得更保守、更高 Precision，但真正缺的是在 1/16 之前保住极小真实变化的 Recall。当前 Stage3 CAACP 已经证明 context 对 medium/large 有效，下一步应该从“更强 suppression/更细 output”转向“dense residual preservation + fine-scale spatial-temporal evidence”。**

这条结论比继续搜索 score 公式或 decoder 小模块更有价值，也更适合作为下一轮论文主线迭代的依据。
