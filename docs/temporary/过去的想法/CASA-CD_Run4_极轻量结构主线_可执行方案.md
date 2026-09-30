# CASA-CD Run4：`<3M` 极轻量主线二——可直接执行的架构、实验与论文方案

> **版本**：Run4 / Mainline-II  
> **任务**：全监督遥感二值变化检测；CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256；A/B/label + list；`gray >= 128`。  
> **固定训练协议**：BCE+Dice，Adam(`lr=2e-4, betas=(0.9,0.99), wd=1e-4`)，poly LR(power=0.9)+200 iter warmup，`max_steps=80000`，batch=16，256×256，seed=16，test-as-val，单 seed。  
> **硬约束**：最终**有效推理参数 <3M**；本方案工程目标 **≤2.20M**、论文目标建议 **≤2.8M**；不以 loss、蒸馏、剪枝、量化、训练技巧作为核心贡献。  
> **Run1–Run3 终局前提**：Change-aware router 迭代已经按预注册停止规则结束；Run4 **不继续** alpha / layer / adaptive quota / 新 loss 搜索。

---

# 0. 结论先行

Run4 不建议继续沿着“把 ChangeViT 每个现有模块等比例缩小”的思路做。  
我建议把最终主线二定稿为一个**预算分工明确的双路径极轻结构**：

> **TinyViT4-192 shallow pretrained semantic path**  
> `+`  
> **32/64/128 三尺度 DWConv detail path**  
> `+`  
> **一次 16×16 对齐后的低秩 Cross-Attention 注入**  
> `+`  
> **Difference-First 轻量金字塔 decoder**

核心原则是：

1. **全局语义只在 16×16 token 网格上花参数和计算**：保留 DeiT-Tiny width=192，但只保留前 4 个 Transformer block；
2. **高分辨率细节不再用 ResNet18**：用极小的 depthwise-separable CNN 保留 1/2、1/4、1/8 三尺度；
3. **不保留 ChangeViT 原始 3 路高开销 Feature Injector**：保留它最有效的“`ViT=Q，detail=K/V`”机制，但把三尺度先池化到 16×16，只做**一次** bottleneck cross-attention；
4. **先建模变化，再做解码**：各尺度先计算 `|F1-F2|`，再逐级融合，而不是继续保留原始重型 difference MLP + 三组反卷积；
5. **Run4 默认不带 A1/K=64 content compression**。它不减参数，而且在 4-block、16×16 ViT 中预计不再是主要系统瓶颈；只保留为论文中的机制分析/可选效率消融。

按下面给出的逐层定义估算：

| 组件 | 估算有效参数 |
|---|---:|
| TinyViT4-192 | **1,976,832** |
| Ultra-Light Detail Branch | **37,024** |
| Scale-Aligned Bottleneck Injector | **34,752** |
| Difference-First Pyramid Decoder | **64,776** |
| **合计** | **2,113,384 ≈ 2.113M** |

也就是说，方案不是“勉强卡在 3M”，而是预计约 **2.11M**，为实现误差和后续小幅结构修正保留约 0.69M 余量。

按 MAC 级粗估，双时相 256×256 前向约 **1.65G** 左右；这只是设计阶段估计，正式数字必须由 Run4 代码上的 `fvcore`、unsupported-op 记录和实际 latency/VRAM 给出，不能直接作为论文结果。

---

# 1. 证据纪律与当前仓库状态

## 1.1 Run1–Run3 终局结论

按当前实验结论，CASAA change-aware router 已经完成了一个很完整的可证伪链条：

```text
SYSU / Kc=32
cosine Top32 precision ≈ 0.39  → F1 相对 A1 负收益
detail Top32 precision ≈ 0.63  → A4-D 相对 A1 -0.12
Oracle precision = 1.00        → A3 相对 A1 +1.48
```

这说明不能再把研究预算投入“把 deployable scorer 再优化一点”的路径。

因此 Run4 中：

- **不继续 change-aware routing**；
- **不做 adaptive quota**；
- **不做 Kc/Kb sweep**；
- **不做 scorer MLP**；
- **不增加辅助 loss**；
- A1 只作为 “Full-Q / compressed-context 可以近似无损” 的分析证据。

---

## 1.2 本次 GitHub 读取存在一个需要先核对的版本冲突

本次通过 GitHub connector 读取 `main` 时，返回的 HEAD 仍是：

```text
1ffe9012d4cd44e8f5e26397ab8e5d2c9c2c4bf1
```

并且 connector 当前列出的 `docs/temporary/` 尚未看到你这次描述的 Run3 文档与 Run3 脚本。

这与“GitHub main 已更新到 Run3 终局”的任务说明不一致。考虑到你已经明确给出了 Run3 最终日志事实，本方案：

- **Run1/Run2 与当前模型代码事实**：按 connector 可核验内容；
- **Run3 终局结果**：按你本次给出的最新直接事实处理；
- **不把 connector 当前缺失的 Run3 内容反过来否定现有结论**。

但在修改代码前，服务器上必须先执行：

```bash
cd /home/yqwang/projects/CASA-CD
git status
git rev-parse HEAD
git log -1 --oneline
ls docs/temporary/
ls train_scripts/CASAA/Run3/
```

确认服务器、本地主分支和真正最新 GitHub SHA 一致，再开始 Run4。  
**不要在版本未核对时覆盖 Run3 代码、日志或 checkpoint。**

---

# 2. 为什么 Run4 不能只轻量化 ResNet 与 Decoder

完整 DeiT-Tiny 已经约 5.5M 参数。

所以即使：

```text
ResNet18 → 0.1M
Decoder  → 0.1M
```

只要 12-block DeiT-Tiny 仍在推理图中：

```text
总参数 > 5.5M
```

就不可能满足 `<3M`。

因此主线二的第一约束不是“轻量 decoder”，而是：

> **12-block plain ViT 必须退出最终推理图。**

width=192 的 DeiT-Tiny 每个标准 block 约：

```text
0.444864M
```

在当前 token/embedding 定义下：

| ViT depth | 估算 ViT 参数 | `<2.8M` 后留给其它组件 |
|---:|---:|---:|
| 3 | ~1.532M | ~1.268M |
| **4** | **~1.977M** | **~0.823M** |
| 5 | ~2.422M | ~0.378M |
| 6 | ~2.867M | 基本没有空间 |
| 12 | ~5.54M | 不可能 |

所以 **prefix-4 是首选点**。

同时保留一个唯一的预注册 fallback：

> 若 prefix-4 在健康对照上损失 `0.5 < ΔF1 ≤ 1.0`，只允许补一次 **prefix-5**；不做 3/4/5/6 深度 sweep。

因为按照本方案其它组件约 0.137M，prefix-5 总体仍约：

```text
2.422M + 0.137M ≈ 2.56M
```

仍满足 `<2.8M`。

---

# 3. P0：Run4 开始前必须先把“DeiT 预训练加载”审计清楚

当前 `encoder.py` 对 Tiny 权重的加载逻辑会主动删除：

```python
pos_embed
patch_embed.proj.weight
```

再加载 checkpoint。

这意味着当前代码中“DeiT-Tiny pretrained”并不等于：

```text
patch embedding + position embedding + all transformer blocks
```

全部原位继承。

特别是 Run4 第一阶段默认**冻结 prefix**，如果输入 patch projection 本身不是预训练权重，这会直接影响对“shallow pretrained semantic branch”的判断。

所以 U2 不能只打印一句：

```text
missing_keys / unexpected_keys
```

而要做一个真正的 **Pretrain Load Audit**。

---

## 3.1 Run4 最终推荐的 prefix 加载规则

先读取本地权重：

```text
/home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth
```

逐项核对 shape。

### 可直接原位加载

若 shape 一致：

```text
patch_embed.proj.weight
patch_embed.proj.bias
blocks.0-3.*
norm.*
```

要求：

```text
loaded tensor == source tensor
max_abs_diff = 0
```

### position embedding

DeiT-224 通常对应 14×14 patch grid，并含额外 token；当前 ChangeViT 是：

```text
256×256 / patch16 = 16×16 = 256 tokens
```

因此位置编码不能简单原位复制。

推荐标准处理：

```text
source pos_embed
→ 去掉 class/distill token（以 checkpoint 实际 shape 为准）
→ reshape 14×14
→ bicubic interpolate 到 16×16
→ flatten 为 [1,256,192]
```

这不是论文创新，也不是训练 trick，只是预训练尺寸兼容。

### 重要实验纪律

Run4 的 **A0_FULL12_FROZEN** 和 **V4_PREFIX4_FROZEN** 必须使用**同一种 corrected pretrained-load 规则**。

这样：

```text
A0 → V4
```

唯一变量才真的是：

```text
ViT depth 12 → 4
```

而不是：

```text
depth + pretrain loader
```

一起变化。

此前 A1_FROZEN 继续保留作历史机制对照，但**不能替代这个新的健康 full-attention A0 对照**。

---

# 4. Run4 架构定稿

建议暂时使用实验名：

> **UL-V4：Ultra-Light ViT4 Change Detector**

正式论文命名等四数据集结果出来后再决定。

总体数据流：

```text
                         ┌─────────────────────────────┐
I_t ────────────────────►│ TinyViT4-192               │
                         │ patch16, 16×16, C=192       │
                         │ DeiT prefix blocks 0-3      │
                         └─────────────┬───────────────┘
                                       │ V_t: 16×16×192
                                       │
I_t ─► Light Detail ─► D2,D4,D8       │
      1/2,1/4,1/8       │              │
                         └─pool/project─┤
                                       ▼
                         Scale-Aligned Bottleneck
                         Cross-Attention Injector
                         Q = ViT, K/V = detail
                                       │
                                       ▼
                              E_t: 16×16×192

T1/T2:
  E1,E2 ────────► |E1-E2|
  D8_1,D8_2 ────► |D8_1-D8_2|
  D4_1,D4_2 ────► |D4_1-D4_2|
  D2_1,D2_2 ────► |D2_1-D2_2|
                         │
                         ▼
              Difference-First Pyramid Decoder
                         │
                         ▼
                   256×256 change map
```

---

# 5. 模块 A：TinyViT4-192

## 5.1 结构

保持：

```text
patch size = 16
embed dim  = 192
heads      = 6
MLP ratio  = 4
token grid = 16×16 = 256
depth      = 4
```

保留 DeiT-Tiny：

```text
blocks.0
blocks.1
blocks.2
blocks.3
```

全部结构与原 checkpoint 同形。

---

## 5.2 参数预算

估算：

| 子组件 | Params |
|---|---:|
| PatchEmbed 3→192, 16×16 | 147,648 |
| 16×16 pos embedding | 49,152 |
| mask token | 192 |
| 4 × ViT block | 1,779,456 |
| final LayerNorm | 384 |
| **总计** | **1,976,832** |

最终以代码 `sum(p.numel())` 为准。

---

# 6. 模块 B：Ultra-Light Multi-Scale Detail Branch

ChangeViT 原论文的实验已经说明两点：

1. detail-capture 是 plain ViT 检测小尺度变化的重要补充；
2. 1/2、1/4、1/8 **三尺度联合优于只用单尺度**；
3. 论文 Table 4 中，即便换成极轻量 CNN，detail branch 仍能发挥明显作用。

因此 Run4 **不应该为了省参数直接砍掉多尺度 detail**。

但不再保留 2.7M 的 ResNet C2–C4。

---

## 6.1 定义 `DSConv`

统一使用：

```text
DWConv 3×3
→ BN
→ ReLU
→ PWConv 1×1
→ BN
→ ReLU
```

---

## 6.2 逐层结构

### Stem

```text
Input:  B×3×256×256
Conv3×3, stride=2, 3→32
BN + ReLU
Output: B×32×128×128
```

### Stage D2：1/2

```text
DSConv 32→32, stride=1
Output D2: B×32×128×128
```

### Stage D4：1/4

```text
DSConv 32→64, stride=2
DSConv 64→64, stride=1
Output D4: B×64×64×64
```

### Stage D8：1/8

```text
DSConv 64→128, stride=2
DSConv 128→128, stride=1
Output D8: B×128×32×32
```

两个时相完全共享权重。

---

## 6.3 参数量

| 层 | Params |
|---|---:|
| Stem 3→32 | 928 |
| DS 32→32 | 1,440 |
| DS 32→64 | 2,528 |
| DS 64→64 | 4,928 |
| DS 64→128 | 9,152 |
| DS 128→128 | 18,048 |
| **Detail 总计** | **37,024 ≈ 0.037M** |

这不是“把 ResNet18 机械减宽”，而是明确把高分辨率路径限定为：

> **仅负责局部层级细节；不在高分辨率上承担昂贵全局关系建模。**

全局关系由 16×16 TinyViT4 负责。

---

# 7. Feature Injector：不要原样保留，也不要完全删除

这是 Run4 最重要的结构决策之一。

## 7.1 为什么不能保留原 Feature Injector

当前代码的 `FeatureInjector`：

```text
3 × Cross-Attention block
每个 block：
  CrossAttention
  + MLP ratio=4
三路输出 concat
再 1×1 fuse
```

对 Tiny 版 `dim1=192`，当前代码估算：

```text
Feature Injector ≈ 1.395M
```

而整个当前 Decoder 约：

```text
≈ 3.435M
```

也就是说，**原 Feature Injector 自己就接近整个 Run4 剩余参数预算的两倍**。

此外，`c2=128×128` 直接作为 K/V 时，注意力矩阵本身也是显存大户。

所以原 FI 必须退出最终模型。

---

## 7.2 为什么也不建议直接退化成 Addition

ChangeViT 的 Table 7 对融合方式做了明确实验：

```text
Addition
Concatenation
Transformer Decoder
FI(reverse)
FI: ViT as Q, detail as K/V
```

其中原作者的：

```text
ViT = Q
detail = K/V
```

是最佳配置。

因此完全删除 selective interaction、只做 resize+add，会失去 ChangeViT 中已经得到实验支持的核心 inductive bias。

---

# 8. 首选：Scale-Aligned Bottleneck Injector（SABI）

Run4 推荐只保留**一次**低秩 cross-attention。

机制：

> 三尺度 detail 先在空间上对齐到 ViT 的 16×16 网格，再融合为一个 compact detail context；ViT token 作为完整 Query，只从这个 compact context 中选择性吸收局部细节。

---

## 8.1 三尺度先对齐

对每个时相：

```text
D2: 128×128×32
D4:  64×64×64
D8:  32×32×128
```

先做 parameter-free：

```text
AdaptiveAvgPool2d(16×16)
```

得到：

```text
16×16×32
16×16×64
16×16×128
```

然后：

```text
1×1: 32  →48
1×1: 64  →48
1×1: 128 →48
BN
```

三路：

```text
D16 = ReLU((P2 + P4 + P8) / 3)
```

得到：

```text
D16: B×48×16×16
```

这一步非常关键：

> **必须先池化，再做投影/attention。**

不要先在 128×128 上把 detail 扩到 192 channel，那会浪费 FLOPs。

---

## 8.2 单次 bottleneck cross-attention

对：

```text
V_t:  B×256×192
D_t:  B×256×48
```

使用：

```text
Q: 192 → 48
K: 48  → 48
V: 48  → 48
heads = 4
head_dim = 12
```

然后：

```text
Attn(Q,K,V): B×256×48
OutProj: 48→192
```

残差：

\[
E_t = V_t + W_o\operatorname{Attn}
      (W_q LN(V_t), W_k LN(D_t), W_v LN(D_t))
\]

**不加 Transformer MLP。**

因为这个模块的任务只是“把 detail 注入 semantic”，不是再做一个完整 Transformer block。

---

## 8.3 参数预算

### Detail 三尺度投影

```text
32→48
64→48
128→48
+ BN
```

合计：

```text
11,040
```

### Bottleneck Cross-Attention

包含：

```text
LN(192)
LN(48)
Q 192→48
KV 48→96
Out 48→192
```

合计：

```text
23,712
```

### SABI 总计

```text
34,752 ≈ 0.0348M
```

相对当前 FI：

```text
1.395M → 0.035M
```

参数约缩小 40 倍。

---

# 9. 模块 C：Difference-First Pyramid Decoder（DFPD）

当前 ChangeViT 是：

```text
每个尺度：
[x, y, |x-y|]
→ 较重多层 Conv difference MLP

然后：
Conv1×1 + ConvTranspose 4×4
逐级恢复
```

这部分对于 `<3M` 不合适。

Run4 推荐改成：

> **Temporal difference first，spatial reconstruction second。**

也就是先把两个时相变成“变化证据”，decoder 不再处理两套完整时相特征。

---

# 10. DFPD 逐层结构

## 10.1 Semantic difference：1/16

Injector 后：

```text
E1,E2: B×192×16×16
```

计算：

```text
S16 = |E1-E2|
Conv1×1 192→96
BN + ReLU
```

输出：

```text
B×96×16×16
```

---

## 10.2 Detail differences

### 1/8

```text
C8 = |D8_1-D8_2|       # 128×32×32
1×1 128→96
BN+ReLU
```

### 1/4

```text
C4 = |D4_1-D4_2|       # 64×64×64
1×1 64→64
BN+ReLU
```

### 1/2

```text
C2 = |D2_1-D2_2|       # 32×128×128
1×1 32→32
BN+ReLU
```

---

## 10.3 Progressive reconstruction

### 16→32

```text
bilinear upsample S16 ×2
+ C8
DSConv 96→96
```

输出：

```text
P8: 96×32×32
```

### 32→64

```text
bilinear upsample ×2
DSConv 96→64
+ C4
DSConv 64→64
```

输出：

```text
P4: 64×64×64
```

### 64→128

```text
bilinear upsample ×2
DSConv 64→32
+ C2
DSConv 32→32
```

输出：

```text
P2: 32×128×128
```

### 128→256

```text
bilinear upsample ×2
DSConv 32→24
Conv3×3 24→1, bias=False
Sigmoid
```

输出：

```text
B×1×256×256
```

---

# 11. DFPD 参数预算

| 模块 | Params |
|---|---:|
| semantic 192→96 | 18,624 |
| detail 128→96 | 12,480 |
| detail 64→64 | 4,224 |
| detail 32→32 | 1,088 |
| DS 96→96 | 10,464 |
| DS 96→64 | 7,328 |
| DS 64→64 | 4,928 |
| DS 64→32 | 2,816 |
| DS 32→32 | 1,440 |
| DS 32→24 | 1,168 |
| classifier 24→1 | 216 |
| **Decoder 总计** | **64,776 ≈ 0.0648M** |

---

# 12. 最终参数预算表

| Component | Params | 占比 |
|---|---:|---:|
| TinyViT4-192 | 1,976,832 | 93.54% |
| Light Detail | 37,024 | 1.75% |
| SABI | 34,752 | 1.64% |
| DFPD | 64,776 | 3.06% |
| **TOTAL** | **2,113,384** | **100%** |

设计阶段硬门槛：

```text
effective params <= 2.20M
```

论文硬门槛：

```text
effective deployed inference params < 3.00M
```

建议不要把 3M 预算用满。

---

# 13. 预计计算量

粗略 MAC 估算：

| Component | 2×256 输入预计 |
|---|---:|
| TinyViT4 pair | ~1.18G |
| Detail pair | ~0.18G |
| SABI pair | ~0.03G |
| DFPD | ~0.26G |
| **总计** | **~1.65G** |

这只是结构设计估计。

正式 Run4 日志必须报告：

```text
fvcore FLOPs
unsupported_ops
batch=1 latency
peak VRAM
```

并统一口径：

```text
输入 = A+B 两张 3×256×256
```

不要把单图 FLOPs 和双时相 FLOPs混在一张论文表里。

---

# 14. 三个候选方案比较：只选一个主方案

## C1：No-Attention Pool+Add

```text
TinyViT4
+ Light Detail
+ pool/project/add
+ DFPD
```

优点：

- 最便宜；
- 最简单。

问题：

- ChangeViT Table 7 已经显示单纯 Addition / Concatenation 不如 `ViT=Q, detail=K/V` 的 selective injection；
- 容易变成“普通轻量 CNN + ViT 拼接”。

**不作为主方案。**

---

## C2：SABI（首选）

```text
三尺度先对齐16×16
→ compact detail token
→ 1次 low-rank Q=ViT/KV=detail cross-attention
→ DFPD
```

优点：

- 保留 ChangeViT 有证据支持的语义→细节选择机制；
- 只有一次 16×16 attention；
- 0.035M injector；
- 论文机制容易讲清楚。

**Run4 主方案。**

---

## C3：Three-Scale Low-Rank Cross-Attention

仍分别对三个 detail scale 做 attention，但把通道缩到 48。

优点：

- 更接近原 FI；
- scale-specific interaction 更强。

问题：

- 三个 attention matrix；
- 实现/显存/论文复杂度增加；
- 当前没有证据说明必须保留三套 attention。

**不先做。**

只有 SABI 明确成为瓶颈时再讨论，不进入 Run4 默认矩阵。

---

# 15. 训练协议：冻结 Prefix 阶段

**第一轮仍然完整跑 80K。**

不要因为模型变小就换成 40K / 60K。

原因：

> Run4 首轮最重要的是和既有 protocol 保持训练预算一致，而不是证明小模型收敛更快。

固定：

```text
Loss            BCE + Dice
Optimizer       Adam
base lr         2e-4
betas           0.9 / 0.99
weight decay    1e-4
LR              poly, power=0.9
warmup          200 iterations
max_steps       80000
batch           16
input           256×256
seed            16
test-as-val     yes
```

冻结：

```python
for p in model.encoder.vit.parameters():
    p.requires_grad = False
```

训练：

```text
Light Detail
SABI
DFPD
```

全部：

```text
lr = 2e-4
```

---

# 16. 不改成 epochs

Run4 继续用：

```text
max_steps = 80000
```

而不是固定 epoch。

因为四个数据集 train size 不同。

如果换成 epochs：

```text
CDD / LEVIR / SYSU / WHU
```

实际 optimizer step 数会失去一致性。

---

# 17. 解冻 Prefix：不是默认方案

只有当：

1. prefix-4 frozen 结构已经证明有能力；
2. 最终 UL 模型只差一个可解释的小性能 gap；
3. 且希望判断 shallow semantic branch 是否值得有限 fine-tune；

才允许一次解冻实验。

解冻不属于创新点，只是最终训练 protocol。

---

# 18. 2K-step ViT Health Gate

若解冻 prefix：

```text
rest lr = 2e-4
ViT lr  = 2e-5
vit_lr_ratio = 0.1
```

只先跑：

```text
2000 steps
```

不直接开 80K。

每 200 step 记录：

```text
patch_embed norm
pos_embed norm
block0.attn.qkv norm
block3.attn.qkv norm
block0.mlp.fc1 norm
block3.mlp.fc1 norm

对应 grad norm
ViT token output RMS
zero-tensor count
NaN/Inf count
```

---

# 19. Health Gate 硬阈值

定义：

\[
r_t=\frac{\|W_t\|_2}{\|W_0\|_2}
\]

对以下核心张量：

```text
patch_embed
pos_embed
block0 qkv
block3 qkv
```

### PASS

200、400、…、2000 step：

```text
所有核心 r_t >= 0.80
```

并且：

```text
无 core tensor 全零
无 NaN/Inf
grad finite
qkv grad 非持续为 0
token RMS ratio ∈ [0.5, 2.0]
```

同时：

```text
step1600 → step2000
任一核心 norm ratio 额外下降 <= 0.05
```

### BORDERLINE

```text
0.75 <= min(r_t) < 0.80
```

只有最后 400 steps 已基本变平，才可以进一步观察。

但默认**不直接进入正式 80K**。

### FAIL

任一：

```text
r_t < 0.75
核心 tensor 全零
NaN/Inf
明显持续向0快速坍缩
token RMS collapse
```

立即停止。

之后：

> **永久冻结 prefix，不再做 ViT LR sweep。**

---

# 20. 解冻通过后也只做一个正式 run

若 2K gate PASS：

```text
vit_lr = 2e-5
rest   = 2e-4
max_steps = 80000
```

只跑一次。

不要再扫：

```text
0.05
0.1
0.2
0.5
```

否则研究主线会再次滑向训练技巧搜索。

---

# 21. Run4 实验原则：先 SYSU，后 LEVIR

Run1/2 已经显示：

- LEVIR 对 ViT 是否健康并不敏感；
- SYSU 对健康 ViT 的依赖更强。

因此 Run4 架构筛选全部优先：

> **SYSU-CD-256**

理由：

如果一个 4-block semantic path 在 SYSU 都没有贡献，LEVIR 上“看起来没掉点”并不能证明 shallow ViT 有效。

---

# 22. Run4 必须先补一个真正健康的 A0 控制

不要直接拿：

```text
baseline Run1 SYSU 82.48
```

作为所有 Run4 深度消融的唯一内部参照。

因为 Run1 受 ViT 崩溃影响。

也不要只用：

```text
A1_FROZEN = 82.04
```

作为 depth-12 → depth-4 对照。

因为 A1 已经同时改了 attention。

所以 Run4 首个正式运行应该是：

## R4-0：A0_FULL12_FROZEN

```text
ViT depth = 12
vanilla attention
corrected DeiT load
ViT frozen
原 ResNet18 detail
原 Feature Injector
原 Decoder
```

唯一目的：

> 建立“健康、冻结、full-attention”的结构参考点。

只先跑 SYSU。

---

# 23. Run4 实验矩阵

## R4-0：健康 full12 reference

### 变量

相对既有模型：

```text
只建立 corrected-pretrain + frozen 健康对照
```

这不是最终方法结果，而是 Run4 内部结构 reference。

---

## R4-1：ViT4 Feature Viability

```text
TinyViT12 → TinyViT4
```

其它全部保留：

```text
ResNet18 detail
原 Feature Injector
原 Decoder
frozen
```

这一步总参数仍会远超 3M，**故意如此**。

目的只有一个：

> 判断 shallow pretrained ViT4 是否还能提供足够 semantic information。

估算 effective params：

```text
11.754M - 8×0.444864M
≈ 8.195M
```

最终以日志为准。

---

## R4-1 判据

相对 R4-0：

### PASS

```text
ΔF1 >= -0.50
ΔIoU >= -0.80
```

→ prefix-4 可用。

### BORDERLINE

```text
-1.00 <= ΔF1 < -0.50
```

只允许：

```text
prefix-5
```

补一个 run。

prefix-5 若明显恢复，则最终架构改为 V5；仍可做到约 2.56M。

### FAIL

```text
ΔF1 < -1.00
```

→ “直接截取 DeiT prefix”这条底座策略不成立。

此时不应马上轻量化 detail/decoder，应重新评估更小 pretrained backbone。

---

# 24. R4-2：只替换 Detail Branch

在 R4-1 基础上：

```text
ResNet18 C2-C4
→
LightDetail 32/64/128
```

为了保持**唯一变量**，暂时加三个 compatibility adapters：

```text
32→64
64→128
128→256
```

再喂给**原 Feature Injector + 原 Decoder**。

这些 adapters 是 R4-2/R4-3 的实验脚手架，不进入最终模型。

### 判据

相对 R4-1：

```text
F1 drop <= 0.30
IoU drop <= 0.50
```

通过才能进入下一步。

如果下降 >0.50 F1：

> 说明 32/64/128 detail capacity 过低，先停止 decoder 轻量化，不要让多个失败变量叠加。

---

# 25. R4-3：只替换 Feature Injector

在 LightDetail 已通过的前提下：

```text
原 3× CrossAttn+MLP FeatureInjector
→
SABI
```

仍临时保留原后端 difference/upsampling decoder。

唯一变量：

> 多尺度 detail→semantic 的注入方式。

### 判据

相对 R4-2：

```text
F1 drop <= 0.30
IoU drop <= 0.50
```

如果 SABI 反而有提升，更好，但不把随机 +0.1 包装成普适结论。

---

# 26. R4-4：最终 `<3M` 模型

在 R4-3 基础上：

```text
原 difference MLP + deconv decoder
→
DFPD
```

并删除 R4-2 用的 temporary adapters。

最终：

```text
TinyViT4
LightDetail
SABI
DFPD
```

预计：

```text
2.113M
~1.65G
```

---

# 27. R4-4 两级成功判据

不要把“结构值得继续”和“已经达到投稿最终目标”混成一个门槛。

## Gate-A：结构可行性

SYSU：

```text
effective params <= 2.20M
F1 >= 81.50
IoU 与 F1 同方向
```

如果连这个门槛都过不了：

> 2.1M 结构的能力明显不足，暂时不补四数据集。

---

## Gate-B：论文候选

希望至少达到：

```text
SYSU F1 >= 82.30
```

这意味着：

- 基本接近原 11.754M baseline 的 82.48；
- 高于 A1_FROZEN 82.04；
- 参数从 11.754M → ~2.11M，约减少 82%。

通过后再跑 LEVIR。

---

# 28. LEVIR Gate

最终 R4-4 只在 SYSU Gate-A 通过后跑 LEVIR。

建议：

### Architecture PASS

```text
LEVIR F1 >= 91.20
```

### Paper-candidate target

```text
LEVIR F1 >= 91.50
```

A1_FROZEN 为 91.84，因此最终 2.1M 模型若能落在：

```text
91.5–92.0+
```

会比较有竞争力。

---

# 29. 什么时候补 CDD / WHU

只在：

```text
SYSU >= Gate-A
且
LEVIR >= Architecture PASS
```

之后补：

```text
CDD
WHU
```

不对所有中间 variant 跑四数据集。

最终主模型必须四数据集完整报告：

```text
Recall
Precision
OA
F1
IoU
Kappa
Params
Trainable Params
FLOPs
Latency
Peak VRAM
```

---

# 30. A1_FROZEN 与原 baseline 各自扮演什么对照角色

## 原 ChangeViT-T

角色：

> **accuracy / architecture ancestry reference**

用于说明：

```text
11.754M → ~2.11M
```

性能变化。

但是 Run1 ViT 崩溃意味着不能把所有 Run4 机制归因都建立在 Run1 baseline 上。

---

## 新 R4-0 A0_FULL12_FROZEN

角色：

> **Run4 depth ablation 的主内部对照**

因为它与 prefix-4：

```text
同 loader
同 frozen 状态
同 full attention
同 detail/decoder
```

只有 depth 不同。

---

## A1_FROZEN

角色：

> **CASAA 历史机制分析对照**

说明健康 ViT 下：

```text
Full Query + K=64 content context
```

可近似保持性能。

但它不应成为最终 lightweight architecture 的唯一 baseline。

---

# 31. A1 K=64 要不要保留进 `<3M` 最终模型

**Run4 默认不保留。**

理由：

### 1. 它不减参数

最终最大约束是：

```text
<3M params
```

A1：

```text
params = 不变
```

所以它解决的不是当前首要约束。

### 2. 4-block ViT 已经很浅

12→4 后，self-attention block 数减少 2/3。

再在 block2-3 做复杂 clustering，系统收益可能很有限。

### 3. 16×16 token 本来只有 N=256

最终约 1.65G 的计算中，高分辨率 detail/decoder 和 MLP 也占据可见比例。

### 4. dynamic clustering 有真实 kernel overhead

理论：

```text
N² → NK
```

不等于 GPU latency 一定同比下降。

---

# 32. A1 只在什么情况下回来

最终 R4-4 成功以后做 profiling。

只有同时满足：

```text
ViT attention 占显著 latency / VRAM
```

才允许一个可选效率实验：

```text
R4-5:
blocks 2-3
vanilla attention
→ content-only Full-Q/K64
```

这里只能用：

```text
content-only
```

不能重新复活 change-aware router。

验收必须看真实：

```text
latency
VRAM
FLOPs
F1
```

如果系统收益很小，就完全不进入最终方法。

---

# 33. Smoke Test 清单

## T0：Pretrain exact-load audit

打印所有继承 key：

```text
patch_embed
blocks0-3
norm
```

对可直接加载 tensor：

```text
max_abs_diff == 0
```

position embedding打印：

```text
source shape
drop-token 后 shape
interpolate 前后 shape
```

---

## T1：Depth

```text
len(vit.blocks) == 4
```

并确认模型中：

```text
blocks.4-11
```

完全不存在，而不是 `requires_grad=False`。

否则它们仍计入推理参数。

---

## T2：Feature shapes

```text
ViT : B×192×16×16
D2  : B×32×128×128
D4  : B×64×64×64
D8  : B×128×32×32
D16 : B×48×16×16
E   : B×192×16×16
pred: B×1×256×256
```

---

## T3：Parameter budget

自动按 module 输出：

```text
vit
detail
injector
decoder
total
trainable
```

断言：

```text
effective < 2.20M
```

正式硬断言：

```text
effective < 3.0M
```

---

## T4：Frozen ViT checksum

训练 dry run 前后：

```text
SHA/hash per tensor
max_abs_change
```

要求：

```text
max_abs_change == 0
```

---

## T5：Gradient path

冻结状态：

```text
ViT grad = None
Detail grad finite & nonzero
SABI grad finite & nonzero
DFPD grad finite & nonzero
```

---

## T6：T1/T2 swap

由于：

```text
shared encoder
shared injector
abs difference
```

理论上最终 change map 应具有更强时序交换一致性。

固定 eval 输入测试：

```text
pred(A,B)
pred(B,A)
```

要求：

```text
max_abs_diff < 1e-5
```

若不成立，先检查 BN/eval 和非对称操作。

这可以成为最终轻量结构的附加性质，但只有测试成立后才能写论文。

---

## T7：旧模式回归

新增 Run4 model type 不得破坏：

```text
baseline
saa
casaa
```

旧 checkpoint 的加载与 eval 路径。

---

## T8：Standalone eval

训练脚本和 `eval.py` 必须由同一组 architecture args 构建模型。

避免发生：

> 参数 shape 恰好能加载，但 eval 实际跑了错误结构。

---

# 34. 真实数据 Dry Run

数据集：

```text
SYSU-CD-256
```

单独目录：

```text
/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/Run4/
/home/yqwang/outputs/CASA-CD/_dryrun/Run4/
```

建议：

```text
200–500 iterations
```

检查：

- A/B/label 几何增强同步；
- loss finite；
- prediction 非全0/全1；
- 参数量；
- FLOPs；
- unsupported ops；
- peak VRAM；
- batch16 是否稳定；
- frozen ViT checksum；
- checkpoint save/resume；
- 独立 eval；
- final test block 格式。

通过后才能启动 80K。

---

# 35. 逐文件修改建议

建议不要直接把所有 lightweight 代码塞进现有 `encoder.py/decoder.py`。

新增：

```text
models/model/light_detail.py
models/model/lite_injector.py
models/model/lite_decoder.py
```

修改：

```text
models/model/encoder.py
models/model/trainer.py
models/train.py
models/eval.py
models/smoke_test.py
```

新增分析：

```text
analyse/param_breakdown.py
analyse/vit_pretrain_audit.py
analyse/vit_health_monitor.py
analyse/profile_model.py
```

Run4 脚本：

```text
train_scripts/UltraLight/Run4/
├── README.md
├── smoke.sh
├── dryrun_SYSU.sh
├── train_R4_0_A0_FULL12_FROZEN_SYSU.sh
├── train_R4_1_VIT4_OLDHEAD_SYSU.sh
├── train_R4_2_VIT4_LIGHTDETAIL_SYSU.sh
├── train_R4_3_VIT4_SABI_SYSU.sh
├── train_R4_4_ULFINAL_SYSU.sh
├── train_R4_4_ULFINAL_LEVIR.sh
├── train_R4_4_ULFINAL_CDD.sh
└── train_R4_4_ULFINAL_WHU.sh
```

不要先建“全部自动串行”的 queue。

每个阶段有明确 gate，前一个没通过，不应继续浪费 80K。

---

# 36. `param_breakdown.py` 至少输出什么

```text
ViT:
  patch_embed
  pos_embed
  block0
  block1
  block2
  block3
  norm

Detail:
  stem
  stage2
  stage4
  stage8

SABI:
  scale projections
  q
  kv
  out

DFPD:
  semantic proj
  detail projections
  fusion blocks
  classifier

TOTAL
EFFECTIVE
TRAINABLE
```

effective 的定义必须是：

> **所有实际出现在 inference forward 图中的 learnable parameters。**

冻结参数也计入 effective。

---

# 37. 参数量对外比较的统一口径

论文必须统一：

### Params

- Siamese shared backbone：权重只计一次；
- frozen 参数：**照计**；
- pretrained backbone：**照计**；
- 不参与 forward 的 dead layer：不计 effective，但可另列 total；
- 训练期辅助模块若推理删除：不计 deployed params，但必须注明；
- 不允许拿“trainable params”冒充“model params”。

### FLOPs

固定：

```text
bi-temporal pair:
2×3×256×256
```

自己的模型全部同一工具重算。

文献 FLOPs 如果口径不清楚：

```text
只引用论文原值
标注 “reported by authors”
```

不要和自己的 fvcore 数字直接做百分比结论。

---

# 38. 当前最值得对标的轻量方法

最终论文不应该只和 TinyCD、FC-Siam 这些老方法比较。

2024–2026 更关键的是：

| Method | Venue/Year | Reported Params | Reported FLOPs | 角色 |
|---|---|---:|---:|---|
| **RFANet** | ISPRS JPRS 2024 | **2.86M** | **3.16G** | `<3M` 权威期刊直接竞争者 |
| **Lighter** | IEEE JSTARS 2025/26 | **1.10M** | **2.01G** | 极轻量强竞争者 |
| **SeCoR** | IEEE JSTARS 2026 | **2.50M** | **2.66G** | 最新 `<3M` 核心竞争者 |
| SeCoR half-width | 同上 | **0.99M** | **1.99G** | `<1M` 效率锚点 |
| **CGLNet** | IEEE GRSL 2026 | **0.99M** | **0.62G** | 极低计算参考 |
| **SChanger** | IEEE JSTARS 2025 | ~**2.37M** | 论文口径需复核 | 强 accuracy 参考，训练/预训练策略不同 |
| TinyCD | NCA 2022/23 | ~0.29M | 口径依实现 | 历史 ultra-light anchor |
| ChangeFormer | IGARSS 2022 | ~41M | 很高 | 非 lightweight，仅作 Transformer 历史对照 |

---

# 39. 关于“ChangeFormer-Lite”

本次检索没有核验到一个权威、统一、官方命名且广泛采用的：

```text
ChangeFormer-Lite
```

配置。

官方 ChangeFormer 的标准版本是重型模型，并不属于 `<3M`。

因此论文里：

> **不要先写一个“ChangeFormer-Lite”参数量数字。**

只有后续找到明确论文/官方 repo 中：

```text
variant name
checkpoint
parameter counting
input size
dataset
```

都可核验的配置后才能加入。

---

# 40. 外部 SOTA 比较时不要直接抄数字当“公平胜负”

不同论文可能存在：

- train/val/test split 不同；
- LEVIR vs LEVIR-CD+；
- label preprocessing 不同；
- crop 不同；
- pretraining 不同；
- test-time augmentation；
- FLOPs 输入口径不同。

因此最终论文表建议分两层。

## Table A：原论文报告结果

```text
Reported by authors
```

明确来源与 protocol。

## Table B：统一协议重跑

只对有官方代码、且值得比较的 2–4 个方法：

```text
统一四数据集
统一256
统一split
统一gray>=128
统一metric代码
```

这张表才可以形成最有力的“同协议轻量 SOTA”结论。

---

# 41. 推荐的核心对标优先级

## 第一优先级

```text
RFANet
Lighter
SeCoR
```

因为：

- 2024–2026；
- 权威遥感期刊；
- 参数都在你真正竞争的 sub-3M 区间。

## 第二优先级

```text
CGLNet
SChanger
```

## 历史参考

```text
TinyCD
FC-Siam-diff
BIT
ChangeFormer
ChangeViT-T
```

---

# 42. 需要补充的外部证据

在最终 SOTA 表之前，还要逐篇核对：

```text
RFANet:
  四数据集具体 split 与 F1/IoU
  Params/FLOPs 的双时相口径

Lighter:
  LEVIR/WHU/SYSU exact table
  是否使用额外预训练

SeCoR:
  LEVIR/WHU/SYSU exact F1/IoU
  full/half variant
  training protocol

SChanger:
  small/base 参数口径
  语义预训练/finetune 的额外数据
  CDD/SYSU/WHU protocol
```

**不要在核验完成前把二手表中的数字写进最终论文主表。**

---

# 43. Run4 还需要补的内部证据

## 必须补 1：R4-0 healthy A0

没有这个结果，无法干净量化：

```text
12 block → 4 block
```

的损失。

---

## 必须补 2：Corrected pretrain loader audit

尤其是：

```text
patch_embed
pos_embed
blocks0-3
```

---

## 必须补 3：当前参数分解

Run4 前精确测：

```text
ViT
ResNet detail
Feature Injector
difference MLP
upsample decoder
```

当前代码估算：

```text
current decoder ≈ 3.435M
其中 Feature Injector ≈ 1.395M
```

应由脚本重新验证。

---

## 必须补 4：ViT4 表征损失

R4-1 是主线二最重要的第一个机制实验。

它回答：

> **DeiT 的浅层 pretrained prefix 是否已经足以提供 CD 所需的 coarse/global semantic prior？**

如果答案是否定的，就不应该先花大量时间改 decoder。

---

# 44. “4-block F1 损失多少算可接受”

我建议预注册：

```text
≤0.50 F1：通过
0.50–1.00：只试 prefix-5
>1.00：prefix truncation 失败
```

为什么不是要求完全无损？

因为 R4-1：

```text
参数直接减少 ~3.56M
```

但仍保留原重型 detail+decoder。

若损失只有 0.3–0.5：

> 后续 lightweight fusion/decoder 仍有可能通过更有效的容量分配收回来。

但如果仅仅截掉 8 个 blocks 就损失 >1 F1：

> 说明这不是一个可靠的极轻量底座。

---

# 45. 最终架构的可证伪假设

## H1：Shallow Semantic Sufficiency

> 在强 multi-scale detail branch 存在时，plain ViT 后 8 个 block 存在较强语义冗余；前 4 个 pretrained block 提供的 16×16 semantic prior 已足以支撑变化检测。

证伪：

```text
R4-1 比 R4-0 掉 >1.0 F1
```

---

## H2：High-Resolution Detail Can Be Cheap

> 细粒度变化检测需要的是多尺度层级结构，而不一定需要 2.7M ResNet 容量。

证伪：

```text
R4-2 比 R4-1 掉 >0.5 F1
```

---

## H3：Scale Alignment Before Interaction

> 三尺度 detail 不需要分别和 semantic 做三次昂贵 cross-attention；先空间对齐并压到 compact context，再做一次 selective interaction 可保留主要注入价值。

证伪：

```text
R4-3 比 R4-2 掉 >0.5 F1
```

---

## H4：Difference-First Decoding

> CD 的 decoder 不需要重复重建两套完整时相表示；先形成变化证据再做逐级多尺度恢复，可以显著降低 decoder 容量而保持判别能力。

证伪：

```text
R4-4 比 R4-3 掉 >0.6 F1
```

---

# 46. 论文故事：CASAA 失败后应该怎么重组

## 不建议的写法

不要继续把：

> “Change-Aware Asymmetric Token Modeling”

放成第一贡献，然后把 A4-D 的失败藏在 supplementary。

当前证据不支持这种叙事。

---

# 47. A1 应该降到哪里

A1 的结论仍然有研究价值：

> **健康 ViT 下，late-stage Full Query + K=64 content context compression 基本无损。**

它可以用于说明：

> ChangeViT 的上下文 token 存在冗余。

但如果最终 UL-V4 **没有使用 A1**，那么它应降为：

```text
Motivation / analysis
或
ablation
```

而不是 main contribution。

原因：

> 论文主贡献必须存在于最终 deployable model 中，并对最终性能/复杂度产生可验证作用。

---

# 48. Run4 推荐的主论文贡献结构

如果 R4-4 成功，建议组织成三条。

## Contribution 1：Extreme-Budget Semantic–Detail Allocation

不是简单说：

> “我们把 DeiT 砍成 4 层。”

而是：

> **在严格 sub-3M 预算下，将全局建模容量集中到低分辨率 pretrained semantic path，将高分辨率容量分配给极小的 hierarchical detail path。**

机制动机：

```text
global semantics:
  16×16
  shallow pretrained ViT

fine detail:
  1/2,1/4,1/8
  depthwise CNN
```

这是“按信息类型分配计算”，不是等比例缩网。

---

## Contribution 2：Scale-Aligned Bottleneck Detail Injection

核心不是“又做一个 attention”。

而是：

> **先把多尺度 detail 变成与 semantic token 对齐的 compact context，再用一次低秩 Q=semantic / KV=detail attention 选择性注入。**

与 ChangeViT 原 FI 的实质区别：

```text
ChangeViT:
3 个不同空间尺度直接做 3 个 Cross-Attention + MLP

Run4:
3 scale → 统一16×16 → compact48d
→ 只做1次低秩 Cross-Attention
→ 无额外 Transformer MLP
```

需要 R4-3 消融支撑。

---

## Contribution 3：Difference-First Lightweight Pyramid

> 在时相共享编码后立即形成多尺度 change evidence，再进行单路径 coarse-to-fine reconstruction。

与当前 ChangeViT 重型 decoder 的区别：

```text
current:
per-scale [T1,T2,abs] heavy modeling
+ transpose-conv cascade

Run4:
abs-difference first
+ native scale projection
+ bilinear
+ depthwise pyramid
```

---

# 49. 不要把“浅层 DeiT”单独包装成足够的创新

`depth=4` 本身主要是结构取舍。

如果最终论文只是：

```text
DeiT 12→4
ResNet→DWConv
Decoder→DSConv
```

会显得是常规轻量化组合。

真正需要用实验建立的方法性观点是：

> **哪类信息应该在哪里花预算。**

即：

```text
Global semantics → shallow pretrained low-resolution tokens
Local details     → tiny hierarchical high-resolution branch
Cross-scale       → one aligned bottleneck interaction
Temporal change   → difference-first reconstruction
```

这才形成一个统一机制，而不是模块堆叠。

---

# 50. “极低参数量超轻量 SOTA”应该怎么表述

在结果还没出来前，论文定位建议写成：

> **An ultra-lightweight binary change detector targeting the sub-3M parameter regime, designed to improve the accuracy–complexity Pareto frontier.**

先不要写：

```text
state-of-the-art
```

---

## 最终只有在满足以下条件后才能写轻量 SOTA

1. 参数量口径统一；
2. 数据集版本与 split 可说明；
3. 至少对 2024–2026 的 `<3M` 强方法有完整比较；
4. 最好重跑 2–4 个官方实现；
5. 你的 F1 / IoU 在同协议下确实领先或形成明显 Pareto 优势。

否则更稳妥：

> **outperforms representative lightweight methods while using about 2.1M parameters**

或：

> **establishes a favorable accuracy–efficiency trade-off in the sub-3M regime**

---

# 51. Run4 的论文目标不是单独追求最小 Params

CGLNet / TinyCD 等可以做到 `<1M`。

所以：

```text
2.11M
```

本身并不自动构成创新。

真正目标是：

\[
\text{在约2M参数处获得尽可能高的 F1/IoU}
\]

形成 Pareto 点：

```text
accuracy ↑
params   ↓
FLOPs    ↓
```

如果 2.11M 能明显超过 0.99M/1.10M 模型的准确率，并接近或超过 2.5–2.86M 强模型，论文会更有说服力。

---

# 52. 正式结果读取纪律

所有 Run4 正式数字仍然只能取同一个：

```text
train_log.txt
```

最后一个完整：

```text
=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

必须同时读取：

```text
Recall
Precision
OA
F1
IoU
Kappa
TOTAL-PARAMS
EFFECTIVE-PARAMS
TRAINABLE-PARAMS
FLOPs
```

不能用：

```text
checkpoint 文件名
best epoch 行
README
Excel 汇总
```

替代。

---

# 53. Run4 checkpoint / log 路径

建议：

```text
/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/<VARIANT>/<DATASET>/

/home/yqwang/outputs/CASA-CD/UltraLight/Run4/<VARIANT>/<DATASET>/train_log.txt
```

例如：

```text
R4_0_A0_FULL12_FROZEN
R4_1_VIT4_OLDHEAD
R4_2_VIT4_LIGHTDETAIL
R4_3_VIT4_SABI
R4_4_ULFINAL
```

保留：

```text
last.pth
best_F1=xxx.pth
```

自动恢复仍使用 `last.pth`。

---

# 54. 启动顺序

严格执行：

```text
版本核对
   ↓
U1 参数预算
   ↓
U2 pretrained-load audit
   ↓
Smoke
   ↓
R4-0 A0_FULL12_FROZEN / SYSU
   ↓
R4-1 VIT4_OLDHEAD / SYSU
   │
   ├── drop <=0.5 → PASS
   │
   ├── 0.5~1.0 → only prefix5
   │
   └── >1.0 → stop prefix strategy
   ↓
R4-2 LightDetail / SYSU
   ↓
R4-3 SABI / SYSU
   ↓
R4-4 ULFINAL / SYSU
   │
   ├── F1 <81.5 → stop / redesign
   └── >=81.5
          ↓
       LEVIR
          │
          ├── <91.2 → 不补全
          └── >=91.2
                 ↓
              CDD + WHU
                 ↓
          SOTA统一协议对照
```

---

# 55. 是否先解冻 ViT

**不。**

Run4 第一轮全部：

```text
frozen prefix
```

原因：

- 先验证结构；
- 避免再次把 ViT collapse 与 architecture effect 混起来；
- pretrained prefix 本身就是当前设计的重要先验。

只有 R4-4 已经可用且需要最后恢复性能时：

```text
2K health gate
→ pass
→ one unfreeze80K
```

---

# 56. 如果 prefix-4 失败，下一步是什么

按严重程度：

## `ΔF1 ∈ (0.5,1.0]`

只做：

```text
prefix-5
```

因为最终参数仍约 2.56M。

## `ΔF1 >1.0`

不继续 sweep depth。

此时重新调研：

> `<2M`、有公开预训练权重、适合 16×16 global semantic 的 backbone。

但这属于下一阶段，不要在 Run4 同时引入 MobileViT / EfficientFormer / TinyViT 等多个 backbone sweep。

---

# 57. 如果 LightDetail 失败

不要马上把通道从：

```text
32/64/128
```

扫：

```text
40/80/160
48/96/192
64/128/256
```

先检查：

- feature activation；
- 是否发生分辨率错误；
- compatibility adapter；
- BN；
- 梯度；
- detail-only feature 可视化。

确定是 capacity 问题后，最多允许一个预注册 fallback：

```text
48 / 96 / 160
```

而不是多组 sweep。

---

# 58. 如果 SABI 失败

首先测试一个**必要对照**：

```text
Pool+Projection+Addition
```

它不是候选主模型，而是回答：

> SABI 失败是 selective interaction 本身有问题，还是任何 lightweight fusion 都无法替代原 FI？

只需要 SYSU 一个 run。

不要直接恢复三路 heavy attention。

---

# 59. 如果 DFPD 失败

检查：

```text
semantic-only
detail-only
各scale change activation
```

若 deep semantic difference 在逐级上采样中被淹没，可以只增加一个：

```text
1×1 gate / scalar-free residual
```

但不引入新 attention、新 loss。

---

# 60. 训练图 / 推理图

## 训练图

```text
DeiT prefix:
  frozen in Stage-1

LightDetail:
  trainable

SABI:
  trainable

DFPD:
  trainable

Loss:
  BCE + Dice only
```

没有：

```text
teacher
auxiliary loss
distillation
GT router
extra training branch
```

---

## 推理图

与训练主路径完全相同：

```text
TinyViT4
+ LightDetail
+ SABI
+ DFPD
```

没有部署删除逻辑，也没有推理期额外模块。

这有利于参数/FLOPs 口径透明。

---

# 61. 当前方案的风险排序

## P0：正确性

1. pretrained loader 实际没有完整加载 patch/pos；
2. Run3 GitHub connector 与用户最新状态存在版本冲突；
3. train/eval 架构参数必须严格一致；
4. depth=4 后不能残留 block4-11 参数；
5. effective param 统计必须排除真正 dead layer，但不能排除 frozen layer。

---

## P1：方法瓶颈

1. prefix-4 semantic capacity 是否足够；
2. 0.037M detail 是否能保持多尺度有效信息；
3. 单次 SABI 能否替代原三次 cross-attn；
4. difference-first 是否损失时相联合建模。

---

## P2：实验工程

1. fvcore unsupported ops；
2. 5090 latency kernel behavior；
3. resume；
4. BN / batch16；
5. logging / path；
6. Git hygiene。

---

# 62. Git 与服务器安全

修改前：

```bash
git status
```

提交前：

```bash
git diff --cached --stat
git diff --cached
```

不要提交：

```text
*.pth
outputs/
cache/
data/
secret
```

不要删除/覆盖：

```text
Run1
Run2
Run3
checkpoint
train_log
```

Run4 全部新目录。

---

# 63. 立即执行清单

## 今天先做

1. 核对 Git SHA 与 Run3 是否真的已在服务器/main；
2. 写 `analyse/param_breakdown.py`，得到当前 11.754M 的精确组件分解；
3. 写 `analyse/vit_pretrain_audit.py`，核对本地 DeiT checkpoint 的 patch/pos/block key 和 shape；
4. 实现 `vit_depth=4`，但**暂时不动 detail / FI / decoder**；
5. 建立 corrected loader；
6. smoke：block0-3 exact load + pos interpolate；
7. 运行 `R4-0 A0_FULL12_FROZEN_SYSU`；
8. 运行 `R4-1 VIT4_OLDHEAD_SYSU`；
9. 用最终 TEST block判断 prefix-4 gate；
10. prefix-4 通过后，才实现 LightDetail。

---

## 第二阶段

11. 实现 32/64/128 LightDetail；
12. R4-2 SYSU；
13. 通过后实现 SABI；
14. R4-3 SYSU；
15. 通过后实现 DFPD；
16. 做 `<2.20M` param smoke + dry run；
17. R4-4 SYSU；
18. SYSU Gate-A 通过才跑 LEVIR；
19. LEVIR 通过后补 CDD/WHU；
20. 最后统一跑 lightweight SOTA 对照。

---

# 64. 推荐的 Run4 最小实验表

| ID | ViT | Detail | Injector | Decoder | Dataset | 核心问题 |
|---|---|---|---|---|---|---|
| R4-0 | 12 frozen | R18 | Original | Original | SYSU | healthy reference |
| R4-1 | **4 frozen** | R18 | Original | Original | SYSU | shallow ViT viability |
| R4-2 | 4 frozen | **LightDetail** | Original | Original | SYSU | detail capacity |
| R4-3 | 4 frozen | LightDetail | **SABI** | Original | SYSU | aligned injection |
| R4-4 | 4 frozen | LightDetail | SABI | **DFPD** | SYSU | final ultra-light |
| R4-4 | 同上 | 同上 | 同上 | 同上 | LEVIR | sparse-change generalization |
| R4-4 | 同上 | 同上 | 同上 | 同上 | CDD | final only |
| R4-4 | 同上 | 同上 | 同上 | 同上 | WHU | final only |
| R4-U | 4 unfrozen | 同上 | 同上 | 同上 | SYSU | **仅2K health pass后可选** |

这是我建议的**最小且有因果解释力**的 Run4 矩阵。

---

# 65. 最终建议

主线二不应该再以 CASAA 为中心。

更强的论文逻辑是：

> **CASAA 的实验首先揭示了：ChangeViT 中并不是所有上下文容量都同等必要；但进一步的 change-aware token allocation 对 deployable signal 过于敏感，因此没有被保留为最终机制。基于这一负结果，论文真正的主方法转向“极端预算下的信息类型分工”：浅层 pretrained ViT 只负责低分辨率全局语义，超轻 CNN 负责高分辨率多尺度细节，通过一次 scale-aligned bottleneck interaction 完成语义–细节融合，并用 difference-first pyramid 直接重建变化图。**

这样做有三个优点：

1. 不掩盖 CASAA 的失败；
2. Run1–Run3 的工作仍然转化成了“为什么不需要继续堆 token router”的机制证据；
3. 最终论文贡献全部落在真正 `<3M` 的部署模型里。

**A1 应降为 analysis/ablation，而不是第一贡献。**

真正需要拿下的是：

```text
~2.1M Params
~1.5–2G 级 FLOPs
四数据集高 F1/IoU
同协议超过 2024–2026 sub-3M lightweight competitors
```

只要这个目标成立，论文故事会比“CASAA + 一个轻量 decoder”更加完整，也更符合你的硕士课题标题——**极轻量遥感二值变化检测**。

---

# 66. 外部文献核验清单

以下只作为 Run4 的比较/设计证据，最终论文仍需下载原论文逐表核对。

### ChangeViT
Zhu et al., *ChangeViT: Unleashing Plain Vision Transformers for Change Detection in Remote Sensing Images*, Pattern Recognition 172 (2026), 112539.  
项目已上传 PDF。原文 Table 3–7 支持：detail branch、多尺度 detail、`ViT=Q/detail=K,V` feature injection 的有效性。

### RFANet
You et al., *Robust feature aggregation network for lightweight and effective remote sensing image change detection*, ISPRS Journal of Photogrammetry and Remote Sensing, 215 (2024), 31–43.  
DOI: https://doi.org/10.1016/j.isprsjprs.2024.06.013  
官方代码：https://github.com/Youzhihui/RFANet

### SeCoR
*SeCoR: Evidence-Guided Selective Correction for Lightweight Remote Sensing Change Detection*, IEEE JSTARS, 2026.  
DOI: https://doi.org/10.1109/JSTARS.2026.3721954  
IEEE 摘要报告：2.50M / 2.66G；half-width 0.99M / 1.99G。

### CGLNet
*High-Level Semantic-Guided Lightweight Network for Remote Sensing Image Change Detection*, IEEE GRSL, 2026.  
DOI: https://doi.org/10.1109/LGRS.2026.3726031  
报告 0.99M / 0.62G；LEVIR F1 91.47、WHU F1 90.77。  
官方代码：https://github.com/bobo59/CGLNet

### Lighter
*Lighter: A Lightweight Full-Information and Dual-Guide Network for Remote Sensing Image Change Detection*, IEEE JSTARS.  
DOI: https://doi.org/10.1109/JSTARS.2025.3647926  
报告 1.10M / 2.01G；覆盖 LEVIR、WHU、SYSU 等数据。

### SChanger
Zhou et al., *SChanger: Change Detection From a Semantic Change and Spatial Consistency Perspective*, IEEE JSTARS, 2025.  
DOI: https://doi.org/10.1109/JSTARS.2025.3555849  
官方代码：https://github.com/zhouziyu-cn/SChanger  
注意其语义预训练/训练策略与 CASA-CD 协议并不完全等价，正式比较需单独标注。

### TinyCD
Codegoni et al., *TINYCD: a (not so) deep learning model for change detection*, Neural Computing and Applications.  
DOI: https://doi.org/10.1007/s00521-022-08122-3  
属于历史极轻量参考，不是 2024–2026 Related Work 的核心竞争者。

### ChangeFormer
Bandara & Patel, *A Transformer-Based Siamese Network for Change Detection*, IGARSS 2022.  
官方代码：https://github.com/wgcban/ChangeFormer  
标准 ChangeFormer 是重型模型；本轮没有核验到统一官方 “ChangeFormer-Lite” 配置，因此不应在论文中凭名称自行构造一个轻量基线。

---

# 67. 最后一条执行规则

**不要一次把 TinyViT4 + LightDetail + SABI + DFPD 全部改完后只跑一个最终模型。**

虽然那样最快，但论文最后无法回答：

```text
性能掉在哪里？
哪个轻量机制真的有效？
参数收益来自谁？
```

Run4 应严格保持：

```text
R4-0 → R4-1 → R4-2 → R4-3 → R4-4
```

每次只换一个组件。

这是当前阶段比继续设计更多模块更重要的事情。
