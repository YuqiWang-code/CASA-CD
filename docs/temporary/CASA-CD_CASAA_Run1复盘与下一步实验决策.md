# CASA-CD：CASAA Run1 复盘与下一步实验决策

> 日期：2026-09-28  
> 仓库：`YuqiWang-code/CASA-CD`，已核对 `main` 最新提交 `3b7472f83e60e587b965f7a6797123f15296d598`。  
> 重点读取：`README.md`、`docs/temporary/CASA-CD_研究路线_ChatGPT方案记录.md`、`docs/temporary/CASA-CD_CASAA_Run1_修改与实验设计建议.md`、`models/model/layers/casaa.py`、`models/model/encoder.py`、`models/train.py`、`models/smoke_test.py`、`train_scripts/CASAA/Run1/`、`others/SAT/saa.py`。  
> 外部方法依据：ChangeViT（Pattern Recognition 172, 2026, 112539；在线 2025）与 SAT/SAA（CVPR 2026 / arXiv:2604.07994）。  
> 本文只做**方法有效性决策**，不改 loss、不做训练技巧 sweep、不做多 seed 统计。

---

## 0. 结论先行

**下一步不建议先跑 `keep_ratio=0.5`、只改最后 2 个 block、全部 12 个 block，也不建议立刻把 A1 当成主创新后直接放弃 CASAA。**

我建议 CASAA 最多再给 **1 次“机制判死刑/救活”的诊断 + 1 次可部署 v2**：

1. **优先级 1：Router Headroom / Oracle Routing 实验**  
   先用现有 checkpoint 做无训练的 routing–GT 对齐诊断；随后只新增 `Oracle-CASAA` 两个 run（LEVIR + SYSU），保持 `K=64, Kc=32, blocks 8-11` 完全不变，只把 cosine 排名换成 GT patch occupancy 排名。  
   **目的不是做可部署方法，而是回答：如果“变化位置”知道得足够准，当前“变化 token 直保留 + 背景聚合”机制本身到底有没有收益。**

2. **优先级 2：只有 Oracle 明确显示有 headroom，才做一个可部署 CASAA-v2**  
   - 如果诊断显示 **cosine 排名质量差**：首选“**Detail-guided parameter-free score**”，利用 ChangeViT 已经存在的 1/8 detail feature 产生变化 cue，不增加参数、不加 loss。
   - 如果诊断显示 **cosine 排名已经不错，但固定 `Kc=32` 与真实变化 patch 数明显不匹配**：改做“**adaptive change quota**”，总 `K=64` 不变，只让 `Kc/Kb` 随每张图的 score 分布变化。  
   **这两个分支二选一，不同时做。**

3. **可选优先级 3：只有可部署 v2 在 K=64 真正优于 A1 后，再做一次 `K=32` 的“压缩压力测试”**  
   这比把压缩放宽到 0.5 更能证明 Change-Aware 的价值：如果变化感知的作用是“在稀缺上下文预算下优先保住有用 token”，那么 K 越紧，CASAA 相对 content-only 的优势应该越明显。

**停止规则很重要：**
- Oracle 都不能优于 A1 → **停止继续设计 change-aware router，立即转主线二**；
- Oracle 有明显 headroom，但可部署 v2 仍不能超过 A1 → **不再继续堆 router 技巧，转主线二**；
- v2 能稳定拉开 A1 → 再做 K=32 压力实验并补 CDD/WHU。

我的总体判断是：**CASAA 现在还值得再做一次“有判别力”的机制实验，但不值得再做低信息量的参数 sweep。**

---

# 1. 当前状态：证据表

| 证据级别 | 当前事实 | 结论 |
|---|---|---|
| 代码直接事实 | `casaa.py` 保留 N=256 的完整 Query，只把 K/V 压成 K=64；A2 为 `Kc=32 + Kb=32`；A1 为 K=64 content aggregation | CASAA 的“Full Q + compressed K/V”实现路线成立 |
| 代码直接事实 | blocks 8–11 使用 paired CASAA；前 8 层、ResNet detail branch、Feature Injector、decoder、loss 均不改 | Run1 的变量控制比较干净 |
| 代码直接事实 | `qkv/proj` 参数名、形状与 DeiT-Tiny 保持一致，routing 无 learnable parameter | 预训练兼容与零新增参数成立 |
| 代码直接事实 | A1/A2 都使用双时相共享描述 `Norm((x1+x2)/2)` 做背景 clustering；A2 额外使用 `1-cos(x1,x2)` 选 change token | A1 是“paired content aggregation control”，不是严格的单流 SAT 原版 |
| Run1 工作结果（用户提供 + 仓库汇总） | LEVIR：A0 91.95 / A1 91.94 / A2 91.86；SYSU：A0 82.48 / A1 82.50 / A2 82.35 | A1≈baseline；A2 没有证明 change-aware 额外价值 |
| 复杂度汇总 | baseline FLOPs 26.3246G；CASAA 26.2593G | 当前系统级 FLOPs 只下降约 **0.25%**，不能把它包装成显著整网加速 |
| SAT 机制依据 | SAT 的核心是 full-resolution Query + compressed K/V，并在压缩率曲线上验证不同 K 的质量–计算权衡 | CASA-CD 迁移“非对称 Query/KV”有机制依据，但 Change-Aware 部分必须由自己的实验单独证明 |
| ChangeViT 机制依据 | ChangeViT 明确把 CNN detail branch 用于细粒度/小目标信息，把 plain ViT 用于高层全局语义 | 用已有 detail feature 作为 change routing cue 是有模型机制依据的，而不是凭空加模块 |

> **正式结果纪律**：当前 GitHub 中没有服务器原始 `train_log.txt`。README/汇总可用于本阶段决策，但论文最终表格仍应以同一 `train_log.txt` 最后一个完整 `=== TEST RESULTS ===` → `=== END TEST RESULTS ===` 区块为唯一正式结果来源。

---

# 2. Run1 真正说明了什么

## 2.1 A1≈baseline 是一个正结果，但它证明的是“上下文冗余”，不是“变化感知有效”

A1 在 LEVIR 与 SYSU 几乎等于 baseline：

\[
\Delta F1_{\text{LEVIR}}=-0.01,\qquad
\Delta F1_{\text{SYSU}}=+0.02
\]

在单 seed 语境下，这可以支持：

> **在 ChangeViT-T 的后四个 ViT block 中，完整保留 Query、把 K/V context 从 256 压到 64，几乎不损失检测精度。**

这是有意义的，因为它说明：

- dense prediction 所需的空间判别位置主要由 **full Query** 保住；
- late-stage K/V context 存在较强冗余；
- `Full Q + compact K/V` 这条迁移路线没有被 Run1 否掉。

但它**不能**支持：

> “压缩背景减少背景干扰，因此提升了变化表征质量。”

因为现在没有可测的精度提升。

论文里更稳妥的表述应该是：

> **Late-stage contextual tokens are highly redundant: retaining full spatial queries while compressing 75% of K/V context preserves detection accuracy with negligible degradation.**

而不是：

> “background compression improves representation quality”。

后者要等 CASAA-v2 真正优于 A1 才能写。

---

## 2.2 A2≈A1 且略低，当前最值得怀疑的不是“压得太狠”，而是 router 本身

因为 A1 在**同样 K=64、同样 4 个 block**下没有掉点，所以：

- “K=64 太少”不是目前最强的解释；
- “后四层不适合压缩”也不是目前最强的解释；
- A2 唯一新增的机制——**固定 32 个 cosine TopK 直保留，并把背景 prototype 从 64 减到 32**——才是首要嫌疑。

换句话说，A2 做了一个资源重新分配：

```text
A1:
  64 个 context prototype 共同覆盖全部 token

A2:
  32 个 raw change candidates
+ 32 个 background prototypes
```

如果 `cosine TopK` 中混入较多伪变化，那么 A2 会同时发生两件事：

1. 把错误 token 当成“宝贵 change token”占掉 context budget；
2. 背景 representation budget 从 64 降到 32。

这足以解释为什么 A2 比 A1 略低。

因此下一步最重要的问题不是：

> “要不要把 K 从 64 改成 128？”

而是：

> **“如果 change ranking 是正确的，A2 这个 allocation 机制本身能不能赢？”**

这就是 Oracle experiment 的价值。

---

# 3. 代码审查后，Run2 前必须先修一个效率问题

## P1：当前 `forward_pair()` 并没有真正按 Q / KV 需要切片计算 fused qkv

当前 `casaa.py` 的核心逻辑实际上是：

```python
qkv_full = self.qkv(x_full)      # 计算 Q、K、V，最后只用 Q
qkv_comp = self.qkv(context)     # 计算 Q、K、V，最后只用 K、V
```

即：

- 对 N 个 full token，仍算了不需要的 full K/V；
- 对 K 个 compressed token，仍算了不需要的 compressed Q。

这不影响正确性，但会削弱 CASAA 的真实计算收益。

### 建议在 Run2 前统一改成

仍然只保留一个 pretrained：

```python
self.qkv = nn.Linear(C, 3*C)
```

state dict 完全不变，但 forward 使用权重 view：

```text
Wq  = qkv.weight[0:C]
Wkv = qkv.weight[C:3C]

Q  = Linear(X_full, Wq)
KV = Linear(C_comp, Wkv)
```

这样：

- **不新增参数**；
- checkpoint / DeiT 权重 key 不变；
- 数学结果与原版等价；
- 不再白算 full K/V 与 compressed Q；
- 更符合论文“只对完整 Query 建模、K/V 仅来自 compact context”的实现叙述。

### 验收

对固定随机输入和同一权重：

```text
max_abs_error(old_forward_pair, sliced_forward_pair) < 1e-6
```

通过后再开展 Run2。

> 这项修改应同时用于 A1/A2/后续 v2；它是实现等价优化，不算新的实验变量。

---

# 4. 优先级 1：Router Headroom / Oracle Routing 实验

## 4.1 单一变量假设

### 假设 H1

> Run1 A2 没有收益的主要原因是 `1-cos(x1,x2)` 对真实变化位置的排序质量不足，而不是“变化 token 直保留 + 背景聚合”这个分配机制本身没有价值。

要验证它，必须构造一个**理想变化排序上界**。

---

## 4.2 先做无训练 Router Audit

不要立刻开 80K。

用现有 baseline / A2 checkpoint，在 LEVIR + SYSU test split 上记录 blocks 8–11 的 score：

\[
s_i^{cos}=1-\cos(x_{1,i},x_{2,i})
\]

GT 转成与 ViT token 对齐的连续 patch occupancy：

\[
g_i=
\frac{1}{16\times16}
\sum_{p\in \text{patch}_i}Y_p
\]

其中 label 先严格按：

```text
gray >= 128
```

二值化。

### 至少记录这些诊断量

1. `Spearman(s_cos, g)`；
2. PR-AUC / ROC-AUC（`g_i>0` 或 `g_i>=1%` 两种口径固定一种即可）；
3. `Top32 changed-pixel coverage`：

\[
\frac{\sum_{i\in Top32(s)} \sum_{p\in patch_i}Y_p}
{\sum_pY_p}
\]

4. `Top32 precision`：Top32 中有真实变化的 patch 比例；
5. 每张图真实 change-patch 数分布：P10 / P50 / P90；
6. blocks 8/9/10/11 的 score quality 是否逐层变好/变坏；
7. A2 中 change bank / background bank 的 token norm 均值与方差；
8. 注意力对 32 个 change-bank token 与 32 个 bg prototype 的平均 attention mass。

### 这个诊断回答两个问题

- **score 差不差？**
- **固定 `Kc=32` 是否与真实 change patch 数严重不匹配？**

这是决定“下一步是换 score 还是 adaptive quota”的证据。

---

## 4.3 Oracle-CASAA：正式诊断 run

### 只改一件事

把：

\[
s_i=1-\cos(x_{1,i},x_{2,i})
\]

替换成：

\[
s_i^{oracle}=g_i
\]

即使用 GT patch change occupancy 给 token 排名。

其余全部不变：

```text
dataset              = LEVIR-CD-256 / SYSU-CD-256
model                 = ChangeViT-T
blocks                = 8,9,10,11
N                     = 256
K                     = 64
Kc                    = 32
Kb                    = 32
keep_ratio            = 0.25
change_share          = 0.50
background clustering = deterministic shared density-peak
qkv/proj              = pretrained original weights
loss                  = BCE + Dice
lr                    = 2e-4
lr schedule           = poly, power=0.9 + original warmup
max_steps             = 80000
batch                 = 16
seed                  = 16
validation protocol   = test-as-val, identical to baseline
```

### 重要：Oracle 不是方法

训练和测试都允许用 GT **仅为了构造机制上界**。

它必须在代码/日志/论文里标成：

```text
Oracle routing — diagnostic upper bound only, not deployable
```

绝不能拿 Oracle 结果做 SOTA 对比，也不能作为最终方法结果。

---

## 4.4 期望现象

如果 CASAA 的“变化优先 allocation”本身是对的：

- Oracle 应明显高于当前 cosine A2；
- 更重要的是，Oracle 应高于 A1 content-only；
- LEVIR 应比 SYSU 更容易观察到收益，因为稀疏变化下“把有限 context 留给真正变化位置”更符合机制假设。

这只是**可证伪预期**，不是结果承诺。

---

## 4.5 成败判据

### 通过：说明 CASAA 还有值得继续做的 headroom

满足：

- Oracle 相对 A1：至少一个数据集 `F1 >= +0.30`；
- 另一个数据集不低于 A1 超过 `0.15`；
- IoU 方向一致；
- Oracle 相对现有 A2 有清晰正 gap。

### 强通过

- LEVIR、SYSU 均优于 A1；
- LEVIR Recall 明显改善；
- Oracle–A2 gap ≥ 0.30 F1。

这会非常清楚地说明：

> **机制有价值，当前瓶颈是 deployable change signal。**

### 失败：直接停止 CASAA router 迭代

如果：

```text
Oracle ≈ A1（两数据集都 <= +0.15 F1）
```

甚至低于 A1，那么结论应该是：

> **即使已知真正变化位置，固定 budget 下“raw change tokens 直保留 + background prototypes”也没有给 ChangeViT 带来额外收益。**

此时不应该再做：

- 更复杂的可学习 scorer；
- 多尺度 scorer；
- adaptive quota sweep；
- 12 层 CASAA；
- 更多 loss。

**直接转主线二。**

这是这个实验最大的价值：它能防止继续在错误机制上消耗大量训练预算。

---

# 5. 优先级 2：可部署 CASAA-v2——只在 Oracle 通过后做

Oracle 通过后，根据 Router Audit 的结果二选一。

---

## 5.1 分支 A（首选）：Detail-guided parameter-free change score

### 触发条件

- Oracle 明显优于 A1/A2；
- cosine 的 Top32 coverage / PR-AUC 明显不足。

### 机制假设 H2-A

> ViT late feature 的同位置 cosine difference 对小目标和局部变化不够敏感；ChangeViT 本身已有用于细粒度变化的 detail feature，因此利用该已有 cue 做 routing，可以改善 change ranking，而不增加推理参数。

### 具体定义

保留当前 ViT score：

\[
s_i^v=1-\cos(v_{1,i},v_{2,i})
\]

取 ChangeViT detail branch 的 **1/8 尺度 feature**：

\[
D_1,D_2\in\mathbb R^{32\times32\times C_d}
\]

用固定 `AvgPool2d(kernel=2,stride=2)` 得到 16×16：

\[
\bar D_1,\bar D_2\in\mathbb R^{16\times16\times C_d}
\]

计算：

\[
s_i^d=1-\cos(\bar d_{1,i},\bar d_{2,i})
\]

为了避免两个 score 数值尺度不同，不加 learnable weight，使用每张图的 rank normalization：

\[
R(s_i)=\frac{\operatorname{rank}(s_i)}{N-1}
\]

最终：

\[
s_i^{v2}
=
\frac12R(s_i^v)+\frac12R(s_i^d)
\]

然后仍然：

```text
TopK Kc=32
K=64
Kb=32
blocks=8-11
```

其它全部不变。

### 为什么这个方案比“学一个 scorer MLP”更适合当前论文

- 0 新参数；
- 0 新 loss；
- detail branch 本来就存在，没有引入额外重模块；
- ChangeViT 原论文已经把 detail branch 的作用定位为 fine-grained/local detail；
- 可以自然解释为“semantic change cue + local detail change cue”；
- 后续把 ResNet detail branch 换成极轻量 detail branch 时，routing cue 仍可复用。

### 注意

这要求在 paired ViT blocks 前先得到 detail feature。

因为两个分支本来就是并行的，所以只需要**调整计算顺序/传 cue**，不能改变 baseline 的 feature 值。必须做 baseline equivalence smoke。

---

## 5.2 分支 B：Adaptive Change Quota

### 触发条件

只有当 Router Audit 表明：

- cosine ranking 与 GT 已经有不错相关；
- 但真实 change-patch 数在 LEVIR / SYSU 或样本间变化很大；
- 固定 `Kc=32` 明显经常过多或过少；

才选择这个分支。

### 机制假设 H2-B

> A2 的问题不是“找错了 token”，而是固定给变化区域 32 个 slot，使稀疏样本浪费 change budget、密集样本又不够用。

### 总 K 不变

\[
K=64
\]

只动态决定 `Kc`。

推荐先用无参数的 score-elbow：

将 score 降序排列：

\[
s_{(1)}\ge s_{(2)}\ge\cdots\ge s_{(N)}
\]

在：

\[
j\in[8,48]
\]

里取最大相邻 gap：

\[
K_c=
\arg\max_j\left(s_{(j)}-s_{(j+1)}\right)
\]

然后：

\[
K_b=64-K_c
\]

这样：

- 总 context budget 不变；
- 无 learnable parameter；
- 无固定 change ratio；
- 稀疏图可给少量 change slot；
- 密集图可给更多 change slot。

### Run2 仍然只跑 LEVIR + SYSU

不补 CDD/WHU，直到筛选通过。

---

# 6. 可部署 v2 的成败判据

无论选 Detail-guided 还是 Adaptive quota，都必须用同一套门槛。

### 通过

相对 **A1** 而不是只相对 baseline：

- 至少一个数据集 `F1 >= +0.30`；
- 另一数据集相对 A1 下降 ≤ 0.15；
- IoU 同方向；
- 相对当前 A2 至少有明确恢复；
- 参数量 0 增长；
- Query N=256 不变；
- 总 K=64 不变（adaptive 仅改变 Kc/Kb）；
- 真实 latency / VRAM 不显著恶化。

### 为什么一定要以 A1 为核心对照

因为现在 A1 已经证明：

> **“仅压缩 context”本身基本无损。**

因此未来 CASAA 要成立，必须证明：

> **change awareness 在相同 context budget 下，能比 content-only allocation 做得更好。**

只要 v2 不能超过 A1，就不能把“Change-Aware”写成主要性能来源。

---

# 7. 优先级 3（可选）：K=32 Compression Stress Test

> **只有可部署 v2 在 K=64 已经通过后再做。**

这是我认为你目前候选列表里漏掉的、非常有论文价值的实验。

## 7.1 为什么它比“放宽到 keep_ratio=0.5”更有信息量

现在 K=64 时 A1 已经≈baseline。

这意味着 context budget 可能还比较宽松：

> 即使不懂“什么是变化”，普通 content clustering 也有足够 slot 表示上下文。

那么 Change-Aware 的优势可能只在**预算真正稀缺**时出现。

因此真正应该问的是：

> **当 K 从 64 减到 32 时，CASAA 能否比 content-only 更慢地掉点？**

这直接对应你的机制主张：

> “有限 context budget 应优先分配给疑似变化 token，稳定背景可以更强聚合。”

---

## 7.2 单一变量

只把：

```text
keep_ratio: 0.25 -> 0.125
K:          64   -> 32
```

其它完全不变。

如果 v2 使用固定 share：

```text
Kc=16
Kb=16
```

如果 v2 使用 adaptive quota，则同比缩放为：

```text
Kc in [4,24]
Kb = 32-Kc
```

### 必须跑成 pair

同一 K=32：

- A1-content；
- CASAA-v2；

在：

- LEVIR；
- SYSU。

共 4 个 run。

---

## 7.3 通过判据

除了看绝对 F1，更应该看**抗压缩退化差**：

\[
D_{A1}=F1_{A1,K64}-F1_{A1,K32}
\]

\[
D_{CASAA}=F1_{CASAA,K64}-F1_{CASAA,K32}
\]

希望：

\[
D_{CASAA}<D_{A1}
\]

建议判据：

- CASAA-v2@K32 相对 A1@K32 至少一个数据集 `+0.20 F1`；
- 另一数据集不更差超过 0.15；
- CASAA 的 K64→K32 退化比 A1 至少少 0.20 F1。

如果成立，论文故事会比“某个固定 ratio 上 +0.1”更强：

> **change-aware allocation improves robustness under increasingly constrained contextual budgets.**

这正是机制证据，而不是调参。

---

# 8. 对你列出的候选 a–e 的明确决策

| 候选 | 当前优先级 | 我的意见 |
|---|---|---|
| a) keep_ratio 0.25→0.5 | **低** | A1@0.25 已无损，放宽只会让 A2 更接近 baseline，不能证明 change-aware |
| a) 只改最后 2 block | **低** | 同理，可能“减弱坏 router 的影响”，但不是机制收益 |
| b) adaptive quota | **中高，但必须诊断后做** | 如果 cosine ranking 好、quota mismatch 明显，这是最合理 v2 |
| c) 换变化信号 | **高，但必须有 Oracle headroom** | Oracle 若明显赢，说明值得换；优先使用已有 detail cue，不先上 learnable scorer |
| d) 只改 1 block | **低** | 效应太弱，难作为主方法 |
| d) 所有 12 block | **暂不做** | early features 更低级，跨时相 cosine 更容易受纹理/配准影响，且一次改太多 |
| d) learnable cross-temporal scorer | **暂缓** | 在 parameter-free 机制未证明前上 MLP/attention scorer 容易变成堆参数 |
| e) 接受压缩无损，转主线二 | **很重要，但建议先做一次 Oracle 判定** | CASAA 再给一次高信息量机会；Oracle 若失败就立即转 |
| f) Oracle upper bound | **最高** | 这是当前缺失的关键可证伪实验 |
| f) K=32 压缩压力测试 | **v2 通过后的高价值 ablation** | 比 keep_ratio=0.5 更能证明 change-aware 在稀缺预算下的作用 |

---

# 9. “A1≈baseline”在论文里怎么定位

## 9.1 可以作为机制基础 / 重要 ablation

推荐定位成：

### Finding 1 — Late-stage contextual redundancy

> Plain ViT 的 late-stage K/V context 在 BCD 中存在较强冗余。对于每个时相保留全部 256 个 query，只保留 64 个聚合后的 context token，几乎不降低 F1/IoU。

它支撑：

1. **为什么 Query 不能压**：dense prediction 仍保留所有空间判别位置；
2. **为什么可以压 K/V**：context side 存在冗余；
3. **为什么后续要做 change-aware allocation**：既然 context budget 可缩小，就需要决定“有限 K 应该分给谁”。

---

## 9.2 现在不能把它写成主结果

原因有两个：

### 1. 没有精度收益

A1≈baseline，只能说明近似无损，不说明背景干扰被减少。

### 2. 当前整网 FLOPs 收益太小

当前汇总：

```text
baseline 26.3246G
CASAA    26.2593G
```

降幅约：

```text
0.0653G ≈ 0.25%
```

这说明 ChangeViT-T 当前的主要计算并不在这 4 个 N=256 的 self-attention 上。

所以论文不能写：

> “CASAA greatly reduces overall computational cost”

除非后续：

- 去掉/轻量化当前重的 detail + Feature Injector + decoder；
- 或把非对称压缩应用到真正高 token 数的上下文交互；
- 或在极轻量最终网络里 CASAA 占比变得显著。

现阶段准确表述是：

> **attention interaction is reduced from \(N^2\) to \(NK\), while the current ChangeViT system-level FLOPs reduction remains modest because other modules dominate computation.**

---

# 10. 你漏掉的两个“更能证明 Change-Aware”的实验

## 10.1 Oracle routing：证明“变化感知分配是否有理论 headroom”

这是当前最缺的。

它把问题分成：

```text
机制不行？
还是 scorer 不行？
```

如果没有 Oracle，你换 cosine / L1 / detail / learnable scorer，实验失败后永远不知道是哪一层原因。

---

## 10.2 Compression stress：证明“变化感知在资源稀缺时更有价值”

CASAA 的核心不是：

> “换一种 clustering 后固定 K=64 能涨 0.1”。

而应该是：

> **当上下文预算越来越小，content-only 会先丢掉关键变化信息，而 change-aware allocation 能更稳地保留检测精度。**

这是一个更像论文机制图的结论：

```text
F1
│ baseline ─────────────
│ A1          \
│              \
│ CASAA         \__
│
└──────────────────── K/N
   0.25  0.125  0.0625
```

当前阶段不需要完整 sweep。

先只补 `0.125` 一个点就足够判断趋势。

---

# 11. Run2 前建议增加的诊断日志

不要只记最终 F1。

CASAA 每个 eval epoch 或最终 test 可额外离线输出：

```text
block
K / Kc / Kb
mean/std/max change score
TopK score gap
TopK GT patch occupancy
changed-pixel coverage@Kc
change-bank token norm
bg-prototype token norm
attention mass -> change bank
attention mass -> bg bank
cluster size mean / max / min
```

不需要加入训练 loss，也不需要每 step 打日志。

只在最终 test 的若干 batch 或分析脚本中统计即可。

这些数据以后能直接做：

- routing 可视化；
- 机制表；
- failure case；
- 论文 supplementary。

---

# 12. 逐文件修改建议（只列 Run2 必需项）

## `models/model/layers/casaa.py`

1. 先实现 `qkv` sliced projection 的等价优化；
2. 把 score 计算抽成明确接口：
   ```text
   cosine
   detail
   oracle   # diagnostic only
   ```
3. 保持 routing no-grad；
4. 保持 aggregate 对原 feature 可回传；
5. 增加可选 routing stats，不引入 learnable head。

## `models/model/encoder.py`

如果走 detail-guided：

- 先计算已有 detail 1/8 feature；
- 得到 16×16 parameter-free detail score；
- 把 cue 传给 blocks 8–11；
- baseline mode 必须与旧版数值等价。

## `models/model/trainer.py` / `models/train.py` / `models/eval.py`

Oracle 诊断模式需要显式传 `label` route hint。

必须加醒目保护：

```text
--casaa_router oracle
```

并在日志首行写：

```text
[DIAGNOSTIC-ONLY] oracle routing uses GT and is not deployable
```

最终方法的 `casaa/detail/adaptive` 模式绝不能依赖 label。

## `analyse/casaa_router_diagnostic.py`

建议新增。

输出：

- block-wise score/GT correlation；
- PR-AUC；
- Top32 change coverage；
- true changed-patch count distribution；
- routing bank stats。

## `train_scripts/CASAA/Run2/`

建议不要覆盖 Run1。

例如：

```text
Run2/
  README.md
  train_A3_ORACLE_LEVIR-CD-256.sh
  train_A3_ORACLE_SYSU-CD-256.sh

Run3/   # 只有 Oracle 通过才建立
  train_A4_CASAA_V2_LEVIR-CD-256.sh
  train_A4_CASAA_V2_SYSU-CD-256.sh
```

---

# 13. Smoke / dry run

## Smoke-0：qkv sliced 等价

```text
old vs sliced
max_abs_error < 1e-6
```

## Smoke-1：baseline 回归

`--mode baseline`：

- output shape 不变；
- fixed input output 对齐；
- pretrained missing/unexpected keys 不增加；
- Params 不变。

## Smoke-2：Oracle router

构造假 label：

- patch occupancy 能映射为 256 个 score；
- Top32 index 正确；
- 不进入梯度；
- T1/T2 swap 不改变 routing。

## Smoke-3：detail router（如果 Oracle 通过）

检查：

```text
detail 1/8: 32x32
pool -> 16x16
score -> Bx256
K=64,Kc=32,Kb=32
```

## Dry run

真实 LEVIR：

```text
200~500 iteration
```

检查：

- loss 正常；
- GPU 显存；
- step latency；
- routing stats；
- checkpoint 恢复；
- 独立 eval 能复现同 router。

通过才启动 80K。

---

# 14. 主线二何时开始

我的建议不是“无限救 CASAA”。

按下面的决策树执行：

```text
Run1:
A1≈baseline, A2≈A1且略低
        │
        ▼
Router Audit + Oracle
        │
        ├── Oracle 也≈A1
        │       └── 停止 change-aware router → 主线二
        │
        └── Oracle 明显>A1
                │
                ▼
        只做一个 deployable v2
                │
                ├── v2 仍≤A1
                │       └── 停止 router → 主线二
                │
                └── v2>A1
                        │
                        ▼
                 K=32 stress test
                        │
                        ▼
                 补 CDD/WHU
```

这能避免把硕士课题变成长期 router 超参搜索。

---

# 15. 关于最终 `<3M`：现在必须提前认识到一个硬约束

仓库当前统计中，完整 DeiT-Tiny 本身约 **5.5M 参数**。

因此：

> **只把 ResNet18 detail branch + decoder 轻量化，最终总参数仍不可能 <3M。**

所以主线二最终必须同时处理 ViT 参数底座，例如：

- 减少独立 block 数；
- 跨层参数共享；
- 设计更小的 pretrained-compatible token backbone；
- 或其它能保留预训练知识但降低独立参数量的方案。

这属于主线二，应另开设计，不要塞进 CASAA Run2。

但它意味着：

> **CASAA 不应该继续消耗太多实验预算，因为最终 `<3M` 的真正硬任务还在后面。**

---

# 16. 最终论文故事的两种可能

## 路线 A：CASAA-v2 成功

论文主线可以是：

1. **Asymmetric context redundancy**：Full Q + 25% K/V 基本无损；
2. **Change-aware context allocation**：在相同 K 下，优先保留变化相关 context 比 content-only 更有效；
3. **Budget robustness**：K 越紧，change-aware 的优势越明显；
4. **Ultra-light architecture**：进一步把模型压到极低参数量，并保持/提升 F1/IoU。

这时 CASAA 是真正的第一创新点。

---

## 路线 B：Oracle / v2 失败

不要硬保“Change-Aware”这个名字。

更合理的处理是：

- 把 A1 作为“asymmetric context aggregation”辅助机制；
- 主要创新重心转向极轻量多尺度结构；
- 如果 A1 在最终轻量模型中能显著降低 attention 计算且保持精度，可以作为第二贡献；
- 如果系统 FLOPs 仍几乎不变，则只保留为分析性 ablation，不占主创新位置。

**不要为了保住最初的论文故事而继续堆 router。**

---

# 17. 立即执行顺序

1. 固定当前 Run1 commit / 不覆盖日志与 checkpoint；
2. 修 `casaa.py` 的 qkv sliced projection，并做 `<1e-6` 等价 smoke；
3. 新增 `analyse/casaa_router_diagnostic.py`；
4. 用已有 baseline/A2 checkpoint 在 LEVIR + SYSU 做 router–GT 对齐分析；
5. 统计真实 change-patch 数分布与 Top32 changed-pixel coverage；
6. 实现 `oracle` diagnostic router；
7. 做 smoke + LEVIR 200–500 iter dry run；
8. 启动 **Oracle × LEVIR/SYSU** 两个 80K run；
9. 只读取各自 `train_log.txt` 最后一个完整 TEST RESULTS；
10. 按本文件阈值判断：
    - Oracle 不过 → 主线二；
    - Oracle 过 → 根据 audit 只选 `detail score` 或 `adaptive quota` 一个；
11. 可部署 v2 通过后，才做 K=32 stress；
12. stress 也通过后，再补 CDD/WHU；
13. 然后进入主线二 `<3M` 结构设计。

---

# 18. 最终建议（一句话）

**Run1 已经证明“Full Query + 25% K/V 可以近似无损”，但没有证明“Change-Aware”有效；下一步最有信息量的不是放宽压缩，而是先用 Oracle routing 把“机制问题”和“scorer 问题”拆开。Oracle 有 headroom 才做一个 parameter-free 可部署 v2；Oracle 都没有 headroom 就停止 CASAA router 迭代，把 A1 留作非对称压缩证据，立即把主要实验预算转到真正决定 `<3M` 与轻量 SOTA 的主线二。**
