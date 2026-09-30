# CASA-CD：CASAA-v2（A4）审查、Run3 实验决策与主线二启动方案

> **审查基准**：GitHub `YuqiWang-code/CASA-CD` `main`，HEAD=`1ffe9012d4cd44e8f5e26397ab8e5d2c9c2c4bf1`（2026-09-29）。  
> **重点核对**：`README.md`、`docs/temporary/CASA-CD_CASAA_Run1复盘与下一步实验决策.md`、`docs/temporary/CASA-CD_ViT崩溃发现与Run2修订.md`、`models/model/layers/casaa.py`、`models/model/encoder.py`、`models/model/resnet.py`、`models/model/layers/block.py`、`models/train.py`、`models/smoke_test.py`、`train_scripts/CASAA/Run2/`、`analyse/casaa_router_diagnostic.py`、`docs/temporary/models_and_metrics_CASAA_Run2.txt`。  
> **方法依据**：ChangeViT（Pattern Recognition 172, 2026, 112539）与 SAT/SAA（CVPR 2026 / arXiv:2604.07994）。  
> **约束**：方法创新优先；不改 loss；不把训练技巧包装成创新；单 seed=16；当前机制筛选不做多 seed；CASAA 保持 Full Query、只压缩 K/V；零新增参数优先；最终主线二目标总有效推理参数 `<3M`。

---

# 0. 结论先行

**A4 值得做，但我不建议现在直接开两个 80K。先做一次“Detail Router Audit”，再决定是否启动训练。**

Run2 已经把最关键的问题回答清楚了：

- 冻结健康 ViT 后，A1 content-only：LEVIR **91.84** / SYSU **82.04**；
- Oracle：LEVIR **91.88** / SYSU **83.52**；
- Oracle−A1：LEVIR **+0.04**，SYSU **+1.48 F1 / +2.15 IoU**；
- 因而在 SYSU 上，`K=64, Kc=32, Kb=32` 这个 allocation **本身已有明确 headroom**；
- 当前瓶颈首先是 **deployable change ranking**，而不是必须先改 quota。

所以我的首选顺序是：

1. **先扩展 Router Audit，不训练**：在 Run2 的 `A1_SAA_FROZEN` checkpoint 上同时测  
   `ViT-cosine / detail-only / 0.5-rank-fusion`；
2. 只有 fused score 在 SYSU 上确实比原 cosine 更接近 GT，才启动 **A4-SYSU 80K**；
3. A4-SYSU 达到预注册门槛后，再跑 **A4-LEVIR 80K**；
4. **Run3 不提前做 K=32 压力测试，也不默认加 detail-only 训练对照**；
5. 若 fused audit 不行，但 detail-only audit 明显更好，则允许只做 **一个 SYSU detail-only 救援 run**；否则停止 router 迭代；
6. A4 若仍不能超过 A1，按既定停止规则转主线二，不再堆 scorer、quota、loss 或 block sweep。

另外有一个必须先修正的实现认知：

> **当前仓库里 32×32 的 1/8 detail feature 是 `resnet.layer3` 输出，不是 `layer2`。**

`Encoder.detail_capture()` 当前数据流是：

```text
256×256
  │ conv1 stride=2
128×128
  │ layer1
x2 = 128×128×64   = 1/2
  │ layer2 stride=2
x3 =  64×64×128   = 1/4
  │ layer3 stride=2
x4 =  32×32×256   = 1/8
```

所以你预注册的：

```text
32×32 → AvgPool2d(2) → 16×16
```

应当取 **`x4 / layer3`**。

这是本次最重要的 P0 更正。

---

# 1. 证据表：事实、推断与待验证假设

| 级别 | 证据 | 当前结论 |
|---|---|---|
| 代码直接事实 | `encoder.py::detail_capture()` 未调用 ResNet `maxpool`，`layer1/layer2/layer3` 分别输出 1/2、1/4、1/8 | 32×32 cue 必须来自 `layer3` |
| 代码直接事实 | CASAA blocks=8–11；Full Q=N=256；K=64；A1 为 K=64 content prototypes；A2/A3 为 Kc32 raw tokens + Kb32 prototypes | A4 可以只替换 scorer，不改 attention budget |
| 代码直接事实 | CASAA qkv/proj 参数名和形状不变，并已改成 sliced projection：full X 只算 Q、compressed context 只算 K/V | A4 不需要改 pretrained attention 参数兼容方式 |
| 代码直接事实 | `compute_contexts()` 中 routing index/assignment 在 `torch.no_grad()` 内；聚合仍从原 feature 取值 | A4 的 detail cue 也应保持 no-grad routing，不能偷偷形成新监督路径 |
| 代码直接事实 | Run2 使用 `--freeze_vit 1`；ResNet detail branch 与 decoder 仍训练 | A4 的 detail signal 会随 detail branch 的主任务训练而演化，但不是新增 learnable scorer |
| Run2 结果 | SYSU Oracle 83.52 vs A1 82.04 | 固定 Kc=32 的机制上限足够大，先修 score 比先改 quota 更合理 |
| Run2 结果 | LEVIR Oracle 91.88 vs A1 91.84 | LEVIR 几乎没有 router headroom，主要用于“不退化”约束 |
| Router Audit | SYSU cosine Spearman≈0.23–0.29、Top32 precision≈0.38–0.39；LEVIR 更低 | 当前 late-ViT cosine ranking 明显不够强 |
| Router Audit | SYSU change patch P10=11/P50=50/P90=182；LEVIR 54% 图像零变化 | 固定 Kc 与真实变化量确实失配，但 Oracle 固定 Kc=32 已能在 SYSU +1.48，因此 quota 不是第一优先级 |
| ChangeViT 论文 | Detail-Capture 输出 1/2、1/4、1/8；论文单尺度实验中 1/8 是三种单尺度中表现最强的一档 | 用 1/8 detail cue 有直接机制依据 |
| ChangeViT 论文 | CNN detail branch 偏细粒度/边界，ViT 偏高层全局语义 | `rank(ViT change)+rank(detail change)` 的互补动机成立 |
| SAT 论文 | Full-resolution Query + compressed K/V 将 attention 从 O(N²) 转为 O(NK) | CASA-CD 的非对称设计来源成立；A4 只改变 K/V 的 selection signal |
| 待验证假设 H-A4 | detail 1/8 的双时相差异排序比 late-ViT cosine 更贴近 GT，或与其互补 | 必须先用 Router Audit 验证，不能直接假定成立 |

---

# 2. A4 设计审查

## 2.1 Detail feature：首选 `layer3 / x4 / 1/8 / 32×32×256`

### 建议

**Run3 正式 A4 固定用 `x4`：**

\[
D_t = F^{t}_{C3}\in \mathbb{R}^{B\times256\times32\times32}
\]

然后：

\[
\bar D_t=\mathrm{AvgPool}_{2\times2}(D_t)
\in\mathbb{R}^{B\times256\times16\times16}
\]

flatten 后：

\[
\bar d_{t,i}\in\mathbb{R}^{256},\quad i=1,\dots,256
\]

detail change score：

\[
s_i^d =
1-
\frac{\bar d_{1,i}^{\top}\bar d_{2,i}}
{\|\bar d_{1,i}\|_2\|\bar d_{2,i}\|_2+\epsilon}
\]

### 为什么不是 layer2

当前实际实现中：

```python
x3 = self.resnet.layer2(x2)   # 64×64, 1/4
x4 = self.resnet.layer3(x3)   # 32×32, 1/8
```

如果用 `layer2`，要从 64×64 下采样到 16×16，应是 `AvgPool2d(4)`，不是 2。

这不仅是 shape 问题，也会改变机制：

- 1/4 更低层，更容易响应纹理、阴影、亮度、季节变化；
- 1/8 有更强局部语义，同时仍比 ViT late feature 保留更多细节；
- ChangeViT 的单尺度消融也支持 1/8 作为 detail cue 的更合理起点。

因此 Run3 不建议再同时做 layer2/layer3 sweep；**固定 layer3**，保持唯一变量。

---

## 2.2 Pool 后做 cosine，而不是先做 cosine 再 pool

建议保持你现在的定义：

```text
feature 32×32
→ AvgPool2d(2)
→ 16×16 feature vectors
→ channel cosine
```

而不是：

```text
32×32 pointwise cosine
→ pool score
```

前者的含义是：

> 先把每个 ViT patch 对应的局部 detail descriptor 聚合出来，再判断两个时相 descriptor 是否变化。

这样更接近“给 16×16 ViT token 提供一个同网格的局部变化 cue”，也更抗 1/8 网格中的局部噪声。

---

# 3. Rank normalization：实现细节

## 3.1 正确顺序

A4 应固定为：

\[
s^v \rightarrow R(s^v)
\]

\[
s^d \rightarrow R(s^d)
\]

然后：

\[
s^{A4}
=
0.5R(s^v)+0.5R(s^d)
\]

最后：

```text
TopK(s_A4, Kc=32)
```

**不要先 raw-score 融合再 rank。**

原因是 `s^v` 与 `s^d` 来自不同 feature space，它们的分布、方差、动态范围没有可比性。先做 rank，才真正实现“不引入可调尺度参数”的 1:1 融合。

也**不需要对 fused score 再 rank 一次**：最终只做 TopK，任何严格单调的再排名都不会改变 TopK 顺序。

---

## 3.2 推荐实现

建议新增：

```python
@torch.no_grad()
def rank_normalize_per_image(s):
    # s: [B, N], high score = more change-like
    B, N = s.shape
    order = torch.argsort(s, dim=-1, stable=True)  # ascending
    rank = torch.empty_like(order, dtype=s.dtype)
    base = torch.arange(N, device=s.device, dtype=s.dtype)
    base = base.unsqueeze(0).expand(B, -1)
    rank.scatter_(1, order, base)
    return rank / max(N - 1, 1)
```

得到：

```text
最低 score -> 0
最高 score -> 1
```

### 不建议

```python
argsort(argsort(s))
```

因为应显式使用 stable sort，便于复现实验与解释 tie 行为。

### tie 问题

cosine 通常是连续值，tie 不会很多；但 detail feature 经过 ReLU，极端情况下可能出现零向量或重复分数。

因此 Audit 必须加：

```text
detail token norm < 1e-6 的比例
score unique ratio
tie ratio
```

如果 tie 很少，stable ordinal rank 足够。

**不要为了平均 rank 改成 N×N pairwise comparison。**  
那会把一个本应轻量的 rank 操作变成 O(N²)，反而破坏 CASAA 的复杂度叙述。

---

# 4. 0.5 / 0.5 融合是否合理

**作为 Run3 的预注册默认值，我支持 0.5 / 0.5，而且不建议做训练 sweep。**

理由不是说 0.5 一定最优，而是：

1. 两个 cue 已经 rank-normalized；
2. 0.5/0.5 是无额外超参数偏好的最简单对称组合；
3. 你当前目标是验证“detail cue 能否救回 change-aware routing”，不是调一个最佳 alpha；
4. 单 seed 下再扫 0.25/0.5/0.75，很容易把论文主线变成小样本超参搜索。

但是在**无训练 Audit**里，我建议同时打印三种：

```text
V:  R(sv)
D:  R(sd)
F:  0.5*R(sv) + 0.5*R(sd)
```

这不是 80K training sweep，而是回答 cue 质量问题。

### 决策规则

- `F > V` 且 `F >= D`：按原计划跑 fused A4；
- `D >> F > V`：说明弱 ViT cue 在稀释 detail cue，可考虑把正式 A4 改成 detail-only；
- `D ≈ V` 且 `F ≈ V`：不值得开 80K；
- `D < V`：detail 分支不适合作为 router，停止该分支。

“`>>`”建议预注册成至少一项核心 ranking 指标 **绝对 +0.03**，且其它指标同向。

---

# 5. 先做 Detail Router Audit：我认为这是必须的

**是，建议在任何 A4 80K 之前先做。**

理由很简单：

> Oracle 已经证明 mechanism headroom；现在唯一需要回答的是 detail cue 是否真的提高 ranking。这个问题可以在几分钟/几十分钟的无训练前向中回答，没有必要先花两次 80K。

## 5.1 Primary checkpoint

优先使用：

```text
CASAA/Run2/A1_SAA_FROZEN/<dataset>/best_F1=*.pth
```

而不是 Oracle checkpoint。

原因：

- A1 的 detail branch 没有受 GT router 的训练路径影响；
- 它更接近“正常可部署模型里已有 detail feature 的质量”；
- 用它判断 A4 是否值得做更干净。

Oracle checkpoint 可以作为 secondary diagnostic，但不要用它决定 A4。

---

## 5.2 Audit 必须新增的指标

对每个 dataset、每个 block 8–11：

| 指标 | ViT-only | Detail-only | Fused |
|---|---:|---:|---:|
| Spearman(score, GT occupancy) | ✓ | ✓ | ✓ |
| ROC-AUC (`g>0`) | ✓ | ✓ | ✓ |
| PR-AUC (`g>0`) | ✓ | ✓ | ✓ |
| Top32 changed-pixel coverage | ✓ | ✓ | ✓ |
| Top32 precision | ✓ | ✓ | ✓ |
| Top32 GT occupancy mean | ✓ | ✓ | ✓ |
| score gap K32/K33 | ✓ | ✓ | ✓ |

另外新增：

```text
Top32 Jaccard(V, D)
Top32 Jaccard(V, F)
Top32 Jaccard(D, F)
```

用于判断 detail cue 是真的补充信息，还是与 ViT 完全重复。

---

## 5.3 SYSU 的 Audit Gate

因为 SYSU 才有 Oracle headroom，我建议以 SYSU 为主要门槛。

A4 进入 80K 的最低条件：

```text
Fused 相比 ViT-only：
PR-AUC                 >= +0.03 absolute
且
Top32 precision 或 coverage >= +0.05 absolute
且
Spearman 不下降
```

更理想：

```text
PR-AUC >= +0.05
Top32 precision >= +0.05
coverage >= +0.05
```

这些门槛不是论文统计显著性，而是**资源分配门槛**：如果 ranking 几乎没变，就没有充分理由指望 80K 把 F1 拉回 +0.30。

---

## 5.4 LEVIR Audit 的重点不是“提高”，而是检查伪变化

LEVIR 的 Oracle 只有 +0.04，54% 图像零变化，因此不要要求 A4 在 LEVIR ranking 上制造很大的正 gap。

更重要的是记录：

```text
zero-change images:
  max score
  top32 mean score
  K32-K33 gap
  detail zero-vector ratio
```

如果 detail cue 在无变化图像上产生大量高置信伪差异，A4 很可能牺牲 LEVIR Precision。

---

# 6. A4 能否收回 SYSU Oracle headroom

## 6.1 判断

**有真实可能，但目前只能给“中等可行性”，不能直接认为大概率成功。**

支持它的证据是：

- Oracle 固定 Kc=32 已经从 82.04 拉到 83.52；
- 因此不需要先证明 adaptive quota 才能有收益；
- detail branch 本来就是 ChangeViT 用来补 ViT 细粒度缺陷的分支；
- SYSU 变化密度高，detail cue 有机会提供比 late semantic cosine 更局部的 change sensitivity。

限制它的证据是：

- 现有 cosine ranking 仍很弱；
- detail feature 的“时相差异”不等于“真实变化”，也可能响应配准、光照、纹理、季节；
- detail branch 已经直接进入 decoder，A4 用它做 router 可能提供互补，也可能只是重复已有信息；
- A4 仍固定只保留 32 个 raw change token，对 P90=182 的 SYSU 仍是强截断。

因此：

> **Oracle headroom 证明“好 ranking 有用”，但不证明 detail score 就是那个好 ranking。**

这就是为什么必须先 Audit。

---

# 7. 为什么现在不优先 adaptive quota

虽然真实变化 patch 数与 Kc=32 明显错配，但我仍不建议它成为 Run3 第一方案。

## 原因 1：Oracle 已经用固定 32 证明 +1.48

这说明：

```text
固定 Kc=32 并没有阻止机制产生大增益。
```

换句话说，当前最先该解决的是：

```text
“这 32 个 token 选得准不准”
```

而不是：

```text
“究竟应该选 21 个还是 47 个”
```

---

## 原因 2：per-image adaptive Kc 会破坏当前非常干净的 batch 实现

当前 `casaa.py` 默认：

```python
Kc = scalar
mask -> [B, N-Kc]
view(B, N-Kc, C)
```

如果每张图 `Kc_b` 不同：

```text
sample 1: Kc=10
sample 2: Kc=41
...
```

则 context 变成 ragged sequence。

你需要：

- per-sample loop；或
- pad 到最大 Kc + attention mask；或
- 固定 bank size 后做无效 slot mask。

这会带来新的：

- latency；
- mask 逻辑；
- FLOPs 口径；
- batch correctness；
- fvcore unsupported ops；
- 实验变量。

对当前研究阶段不划算。

---

# 8. 是否加 detail-only 训练对照

## 默认：不加

Run3 默认只做：

```text
A4_DETAIL_FUSED_FROZEN × SYSU
A4_DETAIL_FUSED_FROZEN × LEVIR
```

因为已有：

```text
A1_FROZEN
A3_ORACLE_FROZEN
```

核心因果链已经是：

```text
content-only
    ↓
deployable detail-fused
    ↓
oracle upper bound
```

这是足够清楚的三点链条。

---

## 只有一种情况加 detail-only

若 Audit 出现：

```text
Detail-only 明显优于 Fused，
并且 Fused 仅略优于/不优于 ViT-only
```

例如 SYSU：

```text
D vs F:
PR-AUC       >= +0.03
Top32 prec   >= +0.03
coverage     不下降
```

那么可以不浪费 fused 80K，直接把 A4 改成：

```text
A4-D: detail-only parameter-free router
```

或者先做**一个 SYSU-only** 80K 的 detail-only 救援实验。

不建议 fused + detail-only 两套同时跑 LEVIR/SYSU 共 4 个 80K。

---

# 9. 一个容易忽略的 A4 训练期坑：detail branch 有 Dropout/BatchNorm

当前：

```python
self.drop = nn.Dropout(p=0.01)
...
x2 = self.drop(self.resnet.layer1(x))
x3 = self.resnet.layer2(x2)
x4 = self.resnet.layer3(x3)
```

所以 A4 使用的 `x4` 在训练期不是完全确定的：

- T1/T2 两次 `detail_capture()` 使用不同 dropout mask；
- layer2/layer3 还含 BN；
- 因而 detail score 会带少量 training-mode stochasticity。

p=0.01 很小，我**不建议为了这个问题修改 dropout 或 BN**，否则会改变 baseline detail branch 的训练行为，破坏 Run3 单一变量。

但必须新增一个审计：

### Detail routing stability

对同一个固定 batch：

1. ResNet BN 固定为 eval；
2. 只让现有 `Dropout(p=0.01)` 工作；
3. 重复 10 次；
4. 统计 fused Top32 的 pairwise Jaccard。

建议门槛：

```text
mean Top32 Jaccard >= 0.90
```

若低于 0.90，说明 routing 对这个 1% dropout 过度敏感，需要重新考虑 cue；不要偷偷关 dropout。

---

# 10. A4 的推荐训练图与推理图

## 10.1 训练图

```text
I1 ───────┬──────── Detail Branch ── C1/2, C1/4, C1/8 ───────────┐
          │                              │                        │
          │                              └─detach/no_grad─ sd ─┐ │
          │                                                    │ │
          └──────── ViT blocks 0-7 ── sv(block8) ─ rank ──────┤ │
                                                               ↓ │
I2 ───────┬──────── Detail Branch ── C2/2, C2/4, C2/8 ───── 0.5+0.5
          │                                                    │
          └──────── ViT blocks 0-7 ────────────────────────────┘
                                                               ↓
                                Top32 + background clustering
                                                               ↓
                           CASAA blocks 8-11: Full Q, K/V=64
                                                               ↓
                            ViT feature 16×16×192
                                                               │
detail features ───────────────── Feature Injector / Decoder ───┘
                                                               ↓
                                                        change map
                                                               ↓
                                                          BCE+Dice
```

关键点：

```text
detail feature -> decoder：正常梯度
detail feature -> router：no_grad / detached
ViT：Run3 全冻结
```

所以不会引入新 loss，也不会让 TopK/argsort 形成伪梯度路径。

---

## 10.2 推理图

推理完全相同，只是不需要 GT：

```text
detail branch
   ↓
parameter-free sd
   ↓
rank fusion with per-block sv
   ↓
TopK
   ↓
CASAA
```

没有 teacher、GT、额外网络或可学习 scorer。

---

# 11. 逐文件修改清单

## 11.1 `models/model/layers/casaa.py`

### 修改

新增：

```python
rank_normalize_per_image()
```

router choices 增加：

```text
detail_fused
```

`_change_score()`：

```python
if oracle:
    return score_hint

if detail_fused:
    assert score_hint is not None
    sv = change_score_cosine(x1, x2)
    rv = rank_normalize_per_image(sv)
    rd = score_hint  # 建议 Encoder 已计算为 rank-normalized detail score
    return 0.5 * rv + 0.5 * rd

return change_score_cosine(x1, x2)
```

建议 `_routing` 多记录：

```text
s_v_raw
s_v_rank
s_d_rank
s_fused
```

### 不改

```text
K=64
Kc=32
Kb=32
background clustering
qkv/proj
norm preservation
blocks 8-11
```

**Run3 不要顺手改其它 CASAA 机制。**

---

## 11.2 `models/model/encoder.py`

新增 helper：

```python
@torch.no_grad()
def detail_score_1_8(self, d1, d2):
    d1 = F.avg_pool2d(d1.detach(), kernel_size=2, stride=2)
    d2 = F.avg_pool2d(d2.detach(), kernel_size=2, stride=2)
    d1 = d1.flatten(2).transpose(1, 2)  # B,256,C
    d2 = d2.flatten(2).transpose(1, 2)
    sd = 1.0 - F.cosine_similarity(d1, d2, dim=-1, eps=1e-8)
    return rank_normalize_per_image(sd)
```

### A4 路径

A4 需要先得到 detail feature，再把 score 送到 ViT：

```text
c_x = detail_capture(x)
c_y = detail_capture(y)

rd = detail_score_1_8(c_x[2], c_y[2])  # c_x[2] == layer3 == 32×32

v_x, v_y = vit.forward_pair(x, y, score_hint=rd)
```

### 强烈建议

**只在 `router=detail_fused` 分支改变计算顺序。**

baseline / A1 / Oracle 保持当前 forward order，不要全局重排，以避免给已有结果引入不必要的实现变化。

A4 中 detail branch **只能 forward 一次**，不能：

```text
一次给 router
一次给 decoder
```

否则虽然参数不增，但真实 FLOPs 与 latency 会明显增加，破坏“已有 detail feature 免费复用”的论点。

---

## 11.3 `models/model/trainer.py`

只需要透传 router；不新增 learnable module。

---

## 11.4 `models/train.py`

更新：

```text
--casaa_router choices:
change
content
oracle
detail_fused
```

日志增加：

```text
[CASAA-ROUTER] detail_fused
[CASAA-DETAIL-SCALE] 1/8
[CASAA-DETAIL-FUSION] rank
[CASAA-VIT-WEIGHT] 0.5
[CASAA-DETAIL-WEIGHT] 0.5
```

Run3 仍：

```text
[FREEZE-VIT] 1
```

---

## 11.5 `models/eval.py`

必须同步 router choice。

这是 P0：CASAA 零新增参数，state dict 与 baseline 兼容，如果 eval 架构实例化错了，checkpoint 仍可能“成功加载”但静默跑成错误 attention。

---

## 11.6 `analyse/casaa_router_diagnostic.py`

建议升级为：

```text
--score_set vit,detail,fused
```

primary 模型：

```text
A1_SAA_FROZEN checkpoint
```

新增：

```text
detail 1/8 score
rank fusion
Top32 overlap
zero-vector rate
tie ratio
zero-change-image score stats
GT-change-count stratified coverage
```

---

## 11.7 `models/smoke_test.py`

新增 A4 smoke，见下一节。

---

# 12. A4 必须新增的 smoke / audit

## T6：detail shape alignment

断言：

```text
layer3 output = B×256×32×32
pool          = B×256×16×16
flatten       = B×256×256
score         = B×256
```

---

## T7：rank normalization

随机连续 score：

```text
min(rank)=0
max(rank)=1
shape unchanged
每行排序关系保持
同输入重复运行完全一致
```

---

## T8：swap symmetry

```text
sd(I1,I2) == sd(I2,I1)
sv(I1,I2) == sv(I2,I1)
sfused(I1,I2) == sfused(I2,I1)
Top32 indices identical
```

---

## T9：gradient graph

必须同时验证：

```text
detail score requires_grad == False
routing indices no grad
```

但：

```text
detail branch layer3.weight.grad != None
decoder grad != None
```

即：

> router 不反传，但 detail branch 仍通过原 decoder 路径正常训练。

---

## T10：parameter parity

A1 vs A4：

```text
total params exact equal
effective params exact equal
state_dict key set exact equal
```

CASAA 不得新增 Linear/Conv/Parameter。

---

## T11：pretrained compatibility

blocks 8–11：

```text
qkv.weight
qkv.bias
proj.weight
proj.bias
```

仍全部从 DeiT-Tiny 原位继承，无新增 missing key。

---

## T12：已有模式回归

A4 改代码后必须保证：

```text
baseline output before/after refactor max_abs_err < 1e-6
A1 output before/after refactor       max_abs_err < 1e-6
Oracle output before/after refactor   max_abs_err < 1e-6
```

至少在 eval + fixed input 下成立。

---

## T13：Frozen ViT invariant

真实 dry run 前后：

```text
all encoder.vit params requires_grad=False
ViT state_dict checksum before == after
ViT tensor max_abs_change == 0
```

不要只看 `requires_grad`。

---

## T14：detail routing stability

前述 10 次重复：

```text
Top32 Jaccard >= 0.90
```

并记录：

```text
detail token norm<1e-6 rate
score tie rate
```

---

# 13. 一个现有代码里的 P1：Run3 暂时不要顺手修

当前背景 prototype：

```python
agg1, agg2 = aggregate_with_shared_assignment(x1_bg, x2_bg, ...)
agg1 = norm_preserve(agg1, x1)
agg2 = norm_preserve(agg2, x2)
```

从机制语义上讲，A2/A3/A4 的背景 prototype 更严格的 reference 应该是：

```python
x1_bg / x2_bg
```

而不是包含已直保留 change token 的全体 `x1/x2`。

否则极高 norm 的 change token 可能影响 background prototype 的 norm restoration。

**但是我不建议 Run3 同时改。**

原因：

- Oracle +1.48 已经是在当前 norm 规则下得到；
- A4 的目标是只验证 scorer；
- 此时修 FNR 会让 A4 vs A1 同时改变 scorer + background norm 语义；
- 因果不干净。

建议：

```text
A4 先保持当前行为；
只有 A4 成功后，把 bg-only norm restoration 做成一个最小后续消融。
```

---

# 14. Audit 后的正式 Run3 实验矩阵

## 14.1 已有，不重跑

| ID | Router | K | ViT | LEVIR F1 | SYSU F1 | 作用 |
|---|---|---:|---|---:|---:|---|
| A1 | content-only | 64 | frozen | 91.84 | 82.04 | 可部署压缩对照 |
| A3 | Oracle GT occupancy | 64 | frozen | 91.88 | 83.52 | 不可部署上界 |

---

## 14.2 Run3 默认只新增 A4

| Run | Dataset | 唯一方法变量 | 固定项 |
|---|---|---|---|
| A4-S | SYSU | score=`0.5 Rank(ViT cosine)+0.5 Rank(detail 1/8 cosine)` | K64/Kc32/Kb32, blocks8-11, frozen ViT, BCE+Dice, 80K, bs16, seed16 |
| A4-L | LEVIR | 同上 | 完全相同 |

---

# 15. Run3 启动顺序

## Stage 0：代码 smoke

```bash
cd /home/yqwang/projects/CASA-CD/models
conda activate casacd

python smoke_test.py \
  --model_type tiny \
  --pretrained_weight_path /home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth \
  --gpu_id 1 \
  --mode all
```

新增 A4 mode 后再把它加入 `all`。

---

## Stage 1：Router Audit，不训练

先 SYSU，再 LEVIR。

Primary：

```text
Run2/A1_SAA_FROZEN best checkpoint
```

输出建议：

```text
/home/yqwang/outputs/CASA-CD/CASAA/Run3/AUDIT/SYSU-CD-256/router_audit.txt
/home/yqwang/outputs/CASA-CD/CASAA/Run3/AUDIT/LEVIR-CD-256/router_audit.txt
```

**Audit Gate 不通过 → 直接不启动 80K。**

---

## Stage 2：真实数据 dry run

不要污染正式 Run3：

```text
checkpoint:
/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/CASAA_A4/SYSU-CD-256/

log:
/home/yqwang/outputs/CASA-CD/_dryrun/CASAA_A4/SYSU-CD-256/
```

建议 20–50 steps 即可，检查：

```text
loss finite
F1 pipeline 正常
detail score finite
K/Kc/Kb 正确
ViT checksum 不变
detail/decoder grad 非零
peak VRAM
无 shape error
```

---

## Stage 3：只先跑 SYSU 80K

脚本建议：

```text
train_scripts/CASAA/Run3/train_A4_DETAIL_FUSED_FROZEN_SYSU-CD-256.sh
```

正式路径：

```text
CKPT:
/share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run3/A4_DETAIL_FUSED_FROZEN/SYSU-CD-256/

LOG:
/home/yqwang/outputs/CASA-CD/CASAA/Run3/A4_DETAIL_FUSED_FROZEN/SYSU-CD-256/train_log.txt
```

---

## Stage 4：SYSU 通过后才跑 LEVIR

```text
train_scripts/CASAA/Run3/train_A4_DETAIL_FUSED_FROZEN_LEVIR-CD-256.sh
```

不要一开始用盲目的两任务 queue，因为你已经有明确 stop rule。

---

# 16. Run3 成败判据

## 16.1 SYSU 主判据

A1_FROZEN：

```text
F1 = 82.04
IoU = 69.55
```

### 通过

```text
A4 F1 >= 82.34   （>= +0.30）
IoU > 69.55，方向一致
```

最好同时看到：

```text
Recall 改善
```

因为 Oracle 的主要价值应表现为更好地保住真正变化 patch，而不是纯 Precision 偶然波动。

### 强通过

```text
F1 >= 82.54 （+0.50）
IoU >= +0.7 左右的同向增益
```

### 失败

```text
F1 <= 82.19（<= +0.15）
```

则不建议继续跑 LEVIR 80K，直接停止 fused router。

`+0.15 ~ +0.30` 视为灰区：优先看 IoU、Recall 与 Router Audit 是否同向；不要再开大规模 hyperparameter sweep。

---

## 16.2 LEVIR 保护判据

A1_FROZEN：

```text
F1 = 91.84
```

允许最大下降 0.15：

```text
A4 F1 >= 91.69
```

LEVIR 不要求 A4 明显提升，因为 Oracle 本身只有 +0.04。

它在 Run3 的主要角色是：

> **验证新 cue 不会在稀疏/零变化场景中制造明显伪变化。**

---

# 17. K=32 压力测试：不要提前到 Run3

**我的建议是不提前。**

原因：

1. 现在还不知道 A4 score 是否有效；
2. K=32 必须跑成 A1 vs A4 pair，LEVIR+SYSU 共 4 个 run；
3. 若 A4 在 K=64 都不能赢 A1，K=32 的机制解释价值很低；
4. 只有 A4 在相同 K=64 下已经证明 change-awareness 有额外收益，K=32 才能验证：

\[
\Delta_{\text{drop}}^{A4}
<
\Delta_{\text{drop}}^{A1}
\]

即：

> change-aware allocation 在更紧的 context budget 下更抗压缩。

所以建议把它放到 **Run4**。

---

# 18. Run4（仅 A4 通过后）

```text
K=32
```

同一数据集必须成对：

```text
A1-content K32
A4-detail-fused K32
```

优先 SYSU，再 LEVIR。

核心不是只看绝对 F1，而是看：

\[
D_{A1}=F1(A1,K64)-F1(A1,K32)
\]

\[
D_{A4}=F1(A4,K64)-F1(A4,K32)
\]

如果：

\[
D_{A4}<D_{A1}
\]

并且 A4-K32 仍明显高于 A1-K32，才能有力支持：

> **在稀缺上下文预算下，change-aware routing 能优先保住任务相关 token，而稳定背景可以更激进地聚合。**

这会比简单写“FLOPs 降低”更像论文机制证据。

---

# 19. 参数量 / FLOPs / latency 的报告方式

A4：

```text
新增 learnable params = 0
K 仍为 64
Query N 仍为 256
```

主 attention FLOPs 与 A2/A3 同级。

额外操作只有：

```text
AvgPool 32→16
detail cosine
rank sort
score add
```

这些不会显著增加参数量，但要注意：

> `fvcore` 对 `argsort/topk/dynamic routing` 可能不完整计数。

因此正式报告至少同时保留：

```text
Params
fvcore FLOPs
unsupported_ops
batch=1 latency
peak VRAM
```

论文里不要把 `26.32G → 26.09G` 这类整网约 1% 以内差异包装成巨大加速。

CASAA 当前更强的效率论据应该是：

> **attention interaction 的 N×N → N×K 机制复杂度下降；系统级收益被 ChangeViT 的 detail branch / Feature Injector / decoder 占比稀释。**

这也正是主线二要解决的问题。

---

# 20. A4 最终论文机制如何表述

如果 A4 成功，建议主张不是：

> “detail branch 又增加了一个模块”。

而是：

> **Change-aware asymmetric context allocation reuses an existing fine-detail representation to estimate where contextual fidelity is worth preserving. Full-resolution queries maintain per-location prediction, while only the contextual K/V bank is budgeted asymmetrically: change-suspected tokens are retained explicitly and stable background tokens are represented by shared prototypes.**

与 SAT 的实质差异：

```text
SAT:
  根据单图 token density / high-frequency structure 压缩 K/V，
  目标是 SR 中的 reconstruction efficiency。

CASAA:
  面向 bi-temporal change detection，
  routing signal 显式来自双时相差异；
  change candidates 与 stable background 使用不同 context allocation；
  Query 始终保留每个变化判别位置。
```

A4 再增加的差异是：

```text
semantic cue: late ViT temporal difference
+
detail cue: existing 1/8 CNN temporal difference
```

但不引入 learnable router。

---

# 21. 如果 A4 失败：停止规则

出现任一情况建议结束 CASAA router 迭代：

### Audit 阶段失败

```text
detail/fused ranking 没有明显优于 current cosine
```

→ 不开 80K。

### SYSU 80K 失败

```text
A4−A1 <= +0.15
```

→ 不再做 alpha sweep、layer sweep、adaptive quota、new loss。

### SYSU 有一点点提升但不到门槛

```text
+0.15 ~ +0.30
```

且 IoU/Recall 不一致

→ 仍按机制未证明处理，不靠单 seed 微小差异包装创新。

---

# 22. 主线二：严格 `<3M` 时，完整 DeiT-Tiny 必须退出推理图

这是主线二首先要明确的数学约束：

```text
完整 DeiT-Tiny backbone ≈ 5.5M
```

因此：

> **即使冻结 ViT，它仍然是 5.5M 推理参数；freeze 不会帮助 `<3M`。**

所以最终 `<3M` 模型不能保留完整 12-block DeiT-Tiny。

---

# 23. 主线二的 3 个候选底座

| 候选 | 总体思路 | Pretrain 兼容 | `<3M` 可行性 | 风险 |
|---|---|---|---|---|
| **B1：DeiT-Tiny width192 prefix-4** | 保持 192 维，只截断深度到 4 blocks | **最好：block 0–3 可原位加载** | **高** | 语义深度下降 |
| B2：width128 + 6/8 blocks | 同时缩 width/depth | 只能做权重 slicing / projection，不再严格原位 | 高 | 预训练兼容弱，研究变量多 |
| B3：纯轻量 CNN/混合 backbone | 完全移除 DeiT | 失去 DeiT 兼容 | 最高 | CASAA 主线与最终模型割裂 |

**首选 B1。**

---

# 24. 为什么我推荐 4-block prefix，而不是先换 width

DeiT-Tiny `D=192, MLP ratio=4` 时，每个标准 block 大约：

```text
0.445M params
```

当前 12-block ViT 约：

```text
5.54M
```

保留 prefix：

| ViT 深度 | 估算 ViT 参数 | 留给 detail+decoder 的 `<3M` 预算 |
|---:|---:|---:|
| 3 blocks | ~1.53M | ~1.47M |
| **4 blocks** | **~1.98M** | **~1.02M** |
| 5 blocks | ~2.42M | ~0.58M |
| 12 blocks | ~5.54M | 不可能 |

4 blocks 是比较合理的第一落点：

- width=192 不变；
- qkv/MLP 权重形状不变；
- 直接加载 DeiT 前 4 个 block；
- 不需要搞 channel slicing；
- 仍然是全局 self-attention；
- 给 detail+decoder 留约 1M 参数。

如果 4-block 语义能力不够，再考虑 3-block + 更强轻量 decoder，而不是先做复杂 width transplantation。

---

# 25. 主线二建议的最终预算

目标：

```text
ViT prefix-4             ≈ 1.95–2.00M
ultra-light detail       <= 0.20–0.25M
ultra-light decoder      <= 0.65–0.75M
misc/norm/classifier     <= 0.05M
------------------------------------
effective total          <= 3.00M
```

最好给自己留余量：

```text
target <= 2.8M
```

避免最终统计时因为 bias/BN/投影超预算。

---

# 26. CASAA 在 `<3M` 版本中如何保留

若 A4 在当前 ChangeViT-T 上通过，不要丢掉主线一。

在 4-block ViT 中可改为：

```text
blocks 0-1: vanilla pretrained attention
blocks 2-3: CASAA
```

仍保持：

```text
Full Q
compressed K/V
zero learnable router
output 16×16 token resolution unchanged
```

这样最终论文可以形成非常清楚的两层贡献：

1. **CASAA**：任务感知的非对称上下文建模；
2. **Ultra-Light Change Representation**：把原 11.7M effective ChangeViT 压到 `<3M`。

而不是两个完全无关的模型。

---

# 27. 主线二的 ViT 训练稳定性怎么排

## 27.1 先把“训练健康”变成硬性验收，而不是等 80K 后看 F1

任何新的 trainable ViT variant，在正式 80K 前先做：

```text
2K-step ViT health run
```

每 200 steps 记录：

```text
pos_embed norm
patch_embed norm
block0 qkv norm
last block qkv norm
zero-weight tensor count
grad norm
```

### 失败判据

任一出现：

```text
任一核心 ViT tensor 变成全零
median weight-norm ratio < 0.5
持续单调向 0 快速坍缩
NaN/Inf
```

→ 不允许启动 80K。

---

## 27.2 主线二默认先冻结 pretrained prefix

第一轮架构有效性实验建议：

```text
freeze truncated ViT prefix
train lightweight detail + decoder
```

目的：

> 先回答 `<3M` 结构是否有能力，而不是重新把“ViT 优化崩溃”混进结构实验。

如果 frozen prefix 已经够强，这是最干净方案。

---

## 27.3 若必须解冻

训练技巧不能当创新点，但可以作为稳定训练协议。

只允许做**最小工程性健康对照**：

```text
frozen
vs
small ViT LR
```

不做大量 LR sweep。

并且任何可训练 ViT 都必须先通过 2K health gate。

---

# 28. 主线二开始前建议补一个健康参考：A0_FROZEN

当前 Run2 有：

```text
A1 content-compressed + frozen ViT
A3 Oracle + frozen ViT
```

但没有：

```text
vanilla attention + frozen ViT
```

因此后续论文如果要严格回答：

> “健康 pretrained ViT 上，K/V compression 本身掉不掉点？”

还缺：

```text
A0_FROZEN
```

建议不是现在插入 Run3，而是在：

```text
A4 通过后 / 主线二正式启动前
```

补 LEVIR + SYSU 两个 frozen vanilla baseline。

否则 A1 vs A4 能证明 change-aware 相对 content-only，但不能完整量化：

```text
vanilla full attention
→ content compression
→ change-aware compression
```

这组三段链条。

---

# 29. 主线二立即启动清单（A4 失败时）

## Step U0：冻结当前 CASAA 结论

记录：

```text
A1 / A3 / A4
正式 final TEST block
router audit
参数/FLOPs
停止原因
```

不要再改旧 Run 的日志与 checkpoint。

---

## Step U1：参数预算脚本

给每个组件单独报：

```text
ViT patch embed
ViT block 0...11
detail layer1/2/3
Feature Injector
decoder each stage
classifier
dead params
```

目标是得到精确预算，而不是只看总参数。

---

## Step U2：建立 `TinyViT4-192` 编码器

```text
depth = 4
embed_dim = 192
patch = 16
token grid = 16×16
```

加载 DeiT：

```text
blocks.0–3 原位
norm / qkv / mlp 原位
```

验证：

```text
missing/unexpected keys 白名单
每个已加载 tensor checksum
pretrained vs instantiated tensor max_abs_diff == 0
```

注意：当前 ChangeViT-T 代码本来就删除 `pos_embed` 与 `patch_embed.proj.weight` 后再加载，因此主线二必须把“哪些权重实际继承”写清楚，不要只写一句“使用 DeiT 预训练”。

---

## Step U3：先做 ViT4 feature viability

先暂时保留现有 detail+decoder（虽然总参数还 >3M），只验证：

```text
12 blocks → 4 blocks
```

在 SYSU / LEVIR 的性能损失是否可接受。

这是为了避免你先花大量时间设计 `<1M` decoder，最后发现 4-block ViT 本身不够用。

---

## Step U4：轻量 detail

目标：

```text
<=0.25M
```

保留 1/2、1/4、1/8 三尺度，但用：

```text
DWConv + 1×1 PWConv
```

或轻量 inverted bottleneck。

不要再使用完整 ResNet18 子网。

---

## Step U5：轻量 decoder

当前 decoder 是主参数大户之一。

目标：

```text
<=0.75M
```

核心只保留：

```text
bi-temporal difference
multi-scale progressive fusion
upsample
classifier
```

不堆新 attention。

---

## Step U6：再把 CASAA 插回 TinyViT4

只有主线一 A4 通过时做：

```text
block2-3 CASAA
```

否则主线二独立推进，不强行把失败的 router 绑进最终模型。

---

# 30. Run3 脚本建议

目录：

```text
train_scripts/CASAA/Run3/
├── README.md
├── audit_A4_SYSU.sh
├── audit_A4_LEVIR.sh
├── dryrun_A4_SYSU.sh
├── train_A4_DETAIL_FUSED_FROZEN_SYSU-CD-256.sh
└── train_A4_DETAIL_FUSED_FROZEN_LEVIR-CD-256.sh
```

**暂时不要建自动 `run_queue.sh` 连跑两个 80K。**

因为 SYSU 是明确的 stage gate。

---

# 31. Run3 A4 核心启动参数

保持：

```bash
--model_type tiny
--mode casaa
--casaa_layers 8,9,10,11
--casaa_keep_ratio 0.25
--casaa_change_share 0.50
--casaa_router detail_fused
--freeze_vit 1
--lr 2e-4
--lr_mode poly
--max_steps 80000
--batch_size 16
--seed 16
```

`--vit_lr_ratio 0.1` 在 `freeze_vit=1` 下实际上不参与 ViT 更新，可以继续保留以保持脚本风格，但日志解释时不要把它当实验变量。

---

# 32. Checkpoint / Log / 恢复规则

A4：

```text
checkpoint:
/share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run3/A4_DETAIL_FUSED_FROZEN/<dataset>/

log:
/home/yqwang/outputs/CASA-CD/CASAA/Run3/A4_DETAIL_FUSED_FROZEN/<dataset>/train_log.txt
```

保留：

```text
last.pth
best_F1=*.pth
```

崩溃恢复：

```text
重新运行同一脚本
→ train.py 自动从 ckpt_dir/last.pth 恢复 optimizer / epoch / best_f1
```

禁止为了“重跑干净”覆盖已有正式目录；需要重跑时新建明确 variant 目录。

---

# 33. 正式结果纪律

Run3 的正式数字只能取同一 `train_log.txt` 最后一个完整：

```text
=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

必须同时抄：

```text
Recall
Precision
OA
F1
IoU
Kappa
TOTAL / EFFECTIVE / TRAINABLE params
FLOPs
```

不要用：

```text
best checkpoint 文件名
epoch 行
README 汇总
experiment_metrics.xlsx
```

替代正式结果源。

---

# 34. 最终决策树

```text
Run2 Oracle SYSU +1.48
        │
        ▼
Detail Router Audit
        │
        ├── fused ranking 不改善
        │       │
        │       ├── detail-only 明显改善
        │       │       └── 只做 1 个 SYSU detail-only 80K
        │       │
        │       └── detail-only 也不改善
        │               └── STOP CASAA → 主线二
        │
        └── fused ranking 明显改善
                │
                ▼
          A4 SYSU 80K
                │
                ├── < +0.15
                │       └── STOP → 主线二
                │
                ├── +0.15 ~ +0.30
                │       └── 看 IoU/Recall/Audit；不做 sweep
                │
                └── >= +0.30
                        │
                        ▼
                    A4 LEVIR
                        │
                        ├── 下降 >0.15
                        │       └── 不通过
                        │
                        └── 不降 >0.15
                                │
                                ▼
                           CASAA-v2 通过
                                │
                                ├── Run4 K=32 压力测试
                                ├── 补 CDD/WHU
                                └── 主线二 <3M
```

---

# 35. 立即执行顺序

1. **先修正 A4 的 detail 层定义：`layer3/x4`，不是 layer2。**
2. 扩展 `casaa_router_diagnostic.py`，先做 `V / D / F` 三路无训练 Router Audit。
3. Audit 中加入 zero-vector、tie、Top32 overlap、zero-change image 与 change-count 分桶。
4. 若 fused 通过 SYSU Audit gate，再实现 `router=detail_fused` 正式路径。
5. 新增 T6–T14 smoke；确认 baseline/A1/Oracle 回归不变。
6. 做独立 `_dryrun` 真实数据 20–50 steps，验证 frozen ViT checksum。
7. **只启动 A4-SYSU 80K。**
8. 读取最终完整 TEST block；F1 ≥82.34 且 IoU 同向才启动 LEVIR。
9. LEVIR ≥91.69 后，才进入 K=32 stress test 与 CDD/WHU。
10. 任一步失败，停止 router 迭代，启动 `<3M` 主线二。
11. 主线二首选 `DeiT-Tiny width192 prefix-4`，先做 exact pretrained load + 2K ViT health gate。
12. 最终目标预算优先锁定到约 `2.8M`，不要把 3.00M 用满。

---

# 36. 仍需补充的证据

当前真正缺的不是更多 80K，而是以下低成本证据：

1. `layer3 detail score` 在 Run2 A1 checkpoint 上的 SYSU/LEVIR Router Audit；
2. detail-only 与 fused 的 Top32 overlap / precision / coverage；
3. 训练期 dropout 对 Top32 的稳定性；
4. A4 加入后 fvcore unsupported op 数、batch1 latency 与 peak VRAM；
5. A4 成功后补 `A0_FROZEN`，建立 healthy vanilla→A1→A4 的完整链条；
6. 主线二开始前精确分解当前 11.754M effective params 到 ViT/detail/FI/decoder；
7. 4-block DeiT prefix 的 exact weight-load report 与 2K health report。

---

# 37. 最终意见

**我支持继续一次 A4，但支持的是“先证据、后 80K”的 A4。**

Run2 最有价值的发现不是“Oracle 83.52”这个数字本身，而是它已经把问题从：

> CASAA 机制到底有没有用？

缩小成了：

> 能不能找到一个零新增参数、可部署、比 late-ViT cosine 更可信的 change signal？

A4 正好回答这个问题，而且利用的是 ChangeViT 已经存在的 detail branch，不增加推理模块和参数，学术叙事也比再堆一个 scorer MLP 更干净。

但 A4 还有两个硬条件：

1. **必须用当前实现真正的 1/8 `layer3/x4`；**
2. **必须先用 Router Audit 证明它真的提高了 ranking。**

如果这两个条件满足，SYSU 的 +1.48 Oracle headroom 足以支持再给 CASAA 一次正式机会；如果 Audit 或 A4-SYSU 仍失败，就应该果断结束 router 迭代，把研究资源转到 `<3M` 主线二。

---

## 来源定位

- CASA-CD GitHub `main` HEAD：`1ffe9012d4cd44e8f5e26397ab8e5d2c9c2c4bf1`。
- ChangeViT：正文 §3.1 Detail-Capture 给出 1/2、1/4、1/8 三尺度；Table 5 比较单尺度/多尺度 detail feature；Table 4 比较 ResNet18 与轻量 detail backbone。
- SAT/SAA：正文 §3.3 明确 Full-resolution Query + compressed K/V，attention interaction 从 `O(N²d)` 降为 `O(NKd)`；§3.4 为 token aggregation。
- RSML-3 项目约定：`casacd` 环境、RTX 5090×2、数据/日志/checkpoint/DeiT-Tiny 权重路径均沿项目统一说明执行。
