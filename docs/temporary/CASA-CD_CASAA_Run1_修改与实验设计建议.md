# CASA-CD：CASAA 主线源码修改与 Run1 实验设计建议

> 审查对象：`YuqiWang-code/CASA-CD`  
> GitHub 当前最新提交：`f63ddf17351d0abe40d91b76783026411a1fe19e`（2026-09-27）  
> 重点阅读：`README.md`、`docs/temporary/CASA-CD_研究路线_ChatGPT方案记录.md`、`models/`、`others/SAT/`、`train_scripts/baseline/Run1/`、`analyse/`。  
> 本文只讨论创新主线一 **CASAA（Change-Aware Asymmetric Token Modeling）** 与 `train_scripts/CASAA/Run1`。不在 Run1 同时改轻量 Detail Branch、Decoder、Loss，避免变量混杂。

---

## 0. 结论先行

**Run1 不建议直接把 `others/SAT/saa.py` 整个替换进 ChangeViT。**

首选方案是：

> **Paired Late-Stage CASAA：在 ChangeViT-Tiny 的最后 4 个 ViT Block 中，把单时相 Self-Attention 改为“双时相共同决定 K/V 聚合、两个时相各自保留完整 Q”的非对称注意力。**

具体固定为：

- Baseline：现有 ChangeViT-T，继续加载 `deit_tiny_patch16_224-a1311bcf.pth`；
- CASAA 位置：ViT block **8, 9, 10, 11（0-based，最后 4 层）**；
- 输入 256×256、patch=16，因此每个时相每层 `N=16×16=256` token；
- **Q token 数始终 256，不压缩，输出也始终 256 token**；
- K/V 总预算先固定为 `K=64`，即 `keep_ratio=0.25`；
- 其中 `Kc=32`：按双时相 change score 直接保留；
- `Kb=32`：把剩余稳定背景 token 通过共享聚类聚成 32 个 prototype；
- T1/T2 使用**同一组保留索引 + 同一组背景聚类 assignment**，但 K/V 特征仍分别来自各自时相，不做 T1/T2 Value 混合；
- **不新增 learnable router，不新增 loss，不改 decoder，不改 ResNet18，不改训练协议**；
- 保留原 `attn.qkv` 和 `attn.proj` 参数名及权重形状，使 DeiT-Tiny 的 pretrained Q/K/V 参数继续原位加载；
- Run1 同时做一个 **SAA-style content-only control**，在完全相同的 `K=64` 和层位置下，只做内容聚合、不使用 change-aware token 保留。这样能回答“提升究竟来自压缩，还是来自变化感知压缩”。

Run1 的目标不是 `<3M`，而是**单独证明 CASAA 机制是否成立**。极轻量化必须放到主线二，否则无法归因。

---

# 1. 当前仓库证据表

| 证据 | 当前代码/记录事实 | 对 CASAA 的含义 |
|---|---|---|
| `models/model/encoder.py` | Tiny：`embed_dim=192, depth=12, num_heads=6, patch_size=16` | 256×256 输入时 ViT 每时相 N=256 |
| `encoder.py` | `v_x=self.vit(x); v_y=self.vit(y)`，当前两个时相独立通过共享 ViT | **单纯替换 `attention.py` 无法得到 change-aware 信息**，必须让选中的 block 同时看到 T1/T2 token |
| `layers/attention.py` | 预训练 attention 使用融合参数 `self.qkv = nn.Linear(dim, dim*3)` 和 `self.proj` | CASAA 应保留 `qkv/proj` 参数名和形状，不应照搬 SAT 的独立 `q/k/v` projection |
| `encoder.py` | Tiny 权重通过 `torch.load(...)[“model”]` 加载，`strict=False` | 新模块若不改变既有 qkv/proj shape，可保留 pretrained attention 权重 |
| `decoder.py` | Feature Injector 与 difference modeling 已完整复现 | Run1 不动它，保证唯一变量是 ViT token modeling |
| `others/SAT/saa.py` | SAT SAA：Full Q + `cluster_and_merge` 压 K/V；默认 `M=0.03`、`c_ratio=0.5` | 只复用“非对称 K/V 聚合”思想，不直接照抄参数化方式 |
| `others/SAT/saa.py` | `cluster_and_merge` 内部使用 `torch.randperm` 随机子采样 | 直接照搬会使推理 routing 带随机性；Run1 建议改为 deterministic stratified sampling |
| baseline Run1 汇总 | CDD F1 97.75 / LEVIR 91.95 / SYSU 82.48 / WHU 94.84 | Run1 的工作对照 |
| baseline 复杂度 | Effective Params 11.754M；TOTAL/TRAINABLE 20.661M（含 ResNet 死参数）；FLOPs 26.3246G | CASAA-v1 应基本不增加参数；总 FLOPs下降不会很大，因为主耗时并不只在 ViT Attention |

**证据纪律说明：** GitHub 当前没有原始服务器 `train_log.txt`，仓库只有由 `analyse/models_to_txt.py` 从最后完整 `=== TEST RESULTS ===` 区块抽取出的汇总。上表可作为 Run1 的工作基线；论文正式结果仍应以服务器对应 `train_log.txt` 最后一个完整 TEST RESULTS 区块为准。

---

# 2. 先指出两个必须修正的概念问题

## P0-1：完整 Query 不是“原图逐像素 Query”

ChangeViT-T 在 256×256 输入、patch=16 时，ViT Query 是：

\[
N=(256/16)^2=256
\]

即 **16×16 patch-level spatial queries**，不是 256×256 的 65536 个像素 Query。

因此论文应写：

> **full spatial queries at the current feature stage are retained**

而不要直接写“所有像素 Query 完整保留”。最终像素级输出分辨率由 decoder 恢复。

这是审稿时很容易被抓住的表述问题。

## P0-2：当前“最终 <3M”目标与完整 DeiT-Tiny 存在硬冲突

仓库记录已经表明，ChangeViT-T 的 **ViT 本身约 5.5M 参数**。因此即使把 ResNet18 detail branch 和 decoder 全换成零参数模块，只要 12 层 DeiT-Tiny 全保留，最终也不可能 `<3M`。

所以总路线必须在主线二再补一项：

> **预训练 Tiny ViT 也必须结构轻量化（例如浅层化 / 参数共享 / 更小的预训练主干），仅换 ResNet18 + decoder 不足以达到 <3M。**

这不影响当前 CASAA Run1。Run1 先只验证机制，**不要在同一 Run 同时解决 <3M**。

---

# 3. 三个候选实现，首选哪个

## 候选 A：直接复制 SAT SAA 到 12 个 ViT Block —— 不建议

问题：

1. SAT `SAA` 把 `q/k/v` 改成三个独立 Linear，并且 `c_ratio=0.5`；
2. ChangeViT pretrained 是融合 `qkv: 192 -> 576`；
3. 直接换后无法原位继承原 pretrained Q/K/V；
4. `M=0.03` 对 N=256 只有约 7 个 K/V token，第一轮过激；
5. SAT 当前提取代码 routing 使用随机子采样；
6. SAT 是单图内容密度，不是双时相变化感知。

它适合做**对照**，不适合作为 CASA-CD 主方法。

## 候选 B：先在 Feature Injector 做 CASAA —— 暂缓

优点是这里 token 很多：

- c2：约 128×128=16384；
- c3：64×64=4096；
- c4：32×32=1024；
- c5 Query：16×16=256。

Feature Injector 的 cross-attention 是当前显存/FLOPs的重要来源，确实非常适合压 K/V。

但研究路线已经明确把创新主线一定位在 **pretrained ViT token modeling**。如果 Run1 先改 Feature Injector，会把论文故事变成 decoder/context injection 优化，并且不能验证“保留 pretrained Q/K/V 后做 change-aware asymmetric attention”。

因此把它保留为后续候选，不放 Run1。

## 候选 C：Paired Late-Stage CASAA —— **首选**

核心：

- 前 8 层保持原 ChangeViT；
- 最后 4 层同时接收 `X1^l, X2^l`；
- 双时相只用于计算“哪些 K/V context 值得保留/聚合”；
- 每个时相的 Q/K/V 仍由自己的 pretrained `qkv` 参数生成；
- T1 不直接拿 T2 的 V，T2 也不直接拿 T1 的 V；
- Query token 数完全不变；
- decoder/difference modeling 完全不动。

这个方案对“方法创新、预训练兼容、唯一变量”三点最干净。

---

# 4. CASAA-v1 的推荐数学定义

设第 \(l\) 个 CASAA block 的两个时相输入（已经过 `norm1`）为：

\[
X_1^l,X_2^l\in \mathbb{R}^{B\times N\times C}
\]

当前 Tiny：

\[
N=256,\quad C=192
\]

## 4.1 参数自由的 change relevance

第一版不要加 learnable scorer。使用对称、尺度较稳定的 cosine dissimilarity：

\[
s_i = 1-\cos(x_{1,i},x_{2,i})
\]

它具有：

- T1/T2 交换对称；
- 无新增参数；
- 无新 loss；
- 不需要 GT；
- inference 可直接使用。

总 K/V budget：

\[
K=\lceil rN\rceil,\quad r=0.25
\]

Run1 固定：

\[
K=64
\]

其中 change-preserve quota：

\[
K_c=\lfloor 0.5K\rfloor=32
\]

取：

\[
\mathcal I_c=\operatorname{TopK}(s,K_c)
\]

这些位置的 context token **不聚合，原位保留**。

## 4.2 稳定背景共享聚合

剩余索引：

\[
\mathcal I_b=\{1,\dots,N\}\setminus \mathcal I_c
\]

构造共享背景描述：

\[
z_i=\operatorname{Norm}\left(\frac{x_{1,i}+x_{2,i}}{2}\right),\quad i\in \mathcal I_b
\]

只在背景 token 上做 deterministic density-peak clustering，得到：

\[
K_b=K-K_c=32
\]

个 cluster，并得到一个**共享 assignment**：

\[
a:\mathcal I_b\rightarrow \{1,\dots,K_b\}
\]

然后分别对两个时相聚合：

\[
\tilde x_{t,j}=
\frac{1}{|\mathcal C_j|}
\sum_{i:a(i)=j}x_{t,i},\quad t\in\{1,2\}
\]

最终两个时相 context bank：

\[
C_t=
[X_{t,\mathcal I_c};\tilde X_{t,bg}]
\in\mathbb R^{B\times K\times C}
\]

**关键：索引/聚类拓扑共享，特征不混时相。**

这样不会因为 T1/T2 分别聚类而制造人工 temporal misalignment。

## 4.3 继续使用 pretrained fused QKV

不要新建 `q/k/v` Linear。

保留现有：

```python
self.qkv = nn.Linear(C, 3*C)
```

在 forward 时按权重切片：

\[
W=[W_Q;W_K;W_V]
\]

分别执行：

\[
Q_t=X_tW_Q
\]

\[
K_t=C_tW_K,\quad V_t=C_tW_V
\]

因此：

\[
Q_t\in\mathbb R^{B\times N\times C}
\]

\[
K_t,V_t\in\mathbb R^{B\times K\times C}
\]

最后：

\[
Y_t=
\operatorname{Softmax}
\left(
\frac{Q_tK_t^T}{\sqrt d}
\right)V_t
\]

输出：

\[
Y_t\in\mathbb R^{B\times N\times C}
\]

**N 从头到尾不变。**

---

# 5. 为什么 Run1 不使用 SAT 的 `c_ratio=0.5`

SAT 的 `saa.py` 会把 Q/K channel 降到 `c_ratio*C`。

CASA-CD 第一轮不应该这么做，原因不是“保守”，而是为了实验因果关系：

- 我们要证明的是 **change-aware K/V token allocation**；
- 如果同时把 Q/K channel 从 192 变 96，就多了一个变量；
- 还会破坏原 DeiT pretrained qkv 的原位继承；
- 之后无法回答提升/下降到底来自 token compression 还是 channel compression。

因此 Run1：

\[
C_Q=C_K=C_V=192
\]

只改变 **token 数**，不改变 channel 维度。

---

# 6. 训练图与推理图

## 训练图

```text
T1 image ─ PatchEmbed ─ blocks 0..7 ─ X1
                                  │
T2 image ─ PatchEmbed ─ blocks 0..7 ─ X2
                                  │
                         blocks 8..11
                                  │
                       ┌──────────┴──────────┐
                       │  Change score s_i   │  (无参数、无GT)
                       └──────────┬──────────┘
                                  │
                   top-K change positions + shared bg clusters
                                  │
                     C1(K tokens)     C2(K tokens)
                                  │
             Q1(full N) → CASAA ← Q2(full N)
                      │                │
                     Y1(N)            Y2(N)
                      │                │
                 原 Detail Branch / Feature Injector / Decoder
                                  │
                              BCE + Dice
```

## 推理图

与训练图完全相同，只关闭 dropout；没有 teacher、没有额外辅助头、没有部署删除步骤。

### 梯度路径

- Q 路径：所有 N 个 query 都参与 attention，梯度完整；
- change token：直接作为 context，K/V 有梯度；
- background prototype：通过聚合后的 context 回传到组成 cluster 的背景 token；
- TopK/cluster assignment 是离散 routing，Run1 的 scorer 无参数，因此 routing 本身不需要梯度；
- routing 索引建议在 `torch.no_grad()` 下计算，聚合本身仍用原 feature tensor 完成，从而避免建立无意义的 similarity 反向图。

---

# 7. SAT 代码应该怎么复用，哪些不能照搬

`others/SAT/saa.py` 可直接参考：

- density peak 中心选择；
- token-to-center assignment；
- cluster mean；
- Full Q / compressed K/V 的注意力结构；
- norm preservation 思想。

Run1 需要改掉：

1. **随机子采样 → deterministic stratified sampling**  
   不要让 eval routing 随 RNG 改变。

2. **单流聚类 → 双时相共享聚类**  
   routing 从 `X1,X2` 联合决定。

3. **全 token 都聚类 → change token 直保留 + stable background 聚类**。

4. **独立 q/k/v + c_ratio → 原 fused qkv 权重切片**。

5. **M=0.03 → Run1 先用 K/N=0.25**。  
   N=256 时 M=0.03 只有 7 个 context token，第一轮太激进。

---

# 8. 逐文件修改清单

## 8.1 新建 `models/model/layers/casaa.py`

建议只放 CASAA 核心，不掺训练逻辑。

包含：

- `change_score_cosine(x1, x2)`
- `deterministic_density_assign(z_bg, Kb)`
- `aggregate_with_shared_assignment(x1_bg, x2_bg, assign_idx, Kb)`
- `CASAAAttention`

`CASAAAttention` 的参数命名必须继续是：

```text
qkv.weight
qkv.bias
proj.weight
proj.bias
```

从而使 `blocks.i.attn.qkv.*` 与 pretrained state dict 完全兼容。

建议配置：

```text
keep_ratio=0.25
change_share=0.50
router_mode="change"       # change / content
norm_preserve=True
```

其中 `router_mode="content"` 用作 SAA-style control。

---

## 8.2 修改 `models/model/layers/__init__.py`

导出：

```python
from .casaa import CASAAAttention
```

---

## 8.3 修改 `models/model/layers/block.py`

为 `NestedTensorBlock` 增加：

```text
forward_pair(x1, x2)
```

逻辑：

- `norm1(x1/x2)`；
- 如果 `self.attn` 支持 `forward_pair`，调用 CASAA paired attention；
- 分别 residual add；
- 两个时相分别走原 MLP/residual；
- 输出 `(x1, x2)`。

不要为 CASAA 另建一套 MLP / LayerNorm；继续使用原 block 参数。

---

## 8.4 修改 `models/model/encoder.py`

给 `DinoVisionTransformer` 增加配置：

```text
casaa_enabled
casaa_layers
casaa_keep_ratio
casaa_change_share
casaa_router
```

推荐：

```text
casaa_layers = [8, 9, 10, 11]
```

构造 blocks 时：

- 普通层仍是 `MemEffAttention`；
- 指定层的 `attn` 替换为 `CASAAAttention`；
- 替换后的 `qkv/proj` shape 与原 attention 完全相同；
- 然后再执行原 pretrained `load_state_dict`。

增加：

```text
DinoVisionTransformer.forward_pair(x1, x2)
```

循环：

```text
block 0..7: x1=blk(x1), x2=blk(x2)
block 8..11: x1,x2=blk.forward_pair(x1,x2)
```

`Encoder.forward(x,y)`：

- baseline mode 保持原 `self.vit(x); self.vit(y)`；
- CASAA mode 调 `self.vit.forward_pair(x,y)`；
- 后续 ResNet detail branch 完全不变。

---

## 8.5 修改 `models/model/trainer.py`

增加参数透传：

```text
mode
casaa_layers
casaa_keep_ratio
casaa_change_share
casaa_router
```

Decoder 不改。

---

## 8.6 修改 `models/train.py`

新增 CLI：

```text
--mode baseline|saa|casaa
--casaa_layers 8,9,10,11
--casaa_keep_ratio 0.25
--casaa_change_share 0.50
--casaa_router change|content
```

日志正式写：

```text
[MODE] casaa
[CASAA-LAYERS] 8,9,10,11
[CASAA-KEEP-RATIO] 0.25
[CASAA-CHANGE-SHARE] 0.50
[CASAA-ROUTER] change
```

`=== TEST RESULTS ===` 继续保持现有格式，使 `analyse/extract_metrics_to_excel.py` 不需要改。

建议把当前 header 中硬编码的：

```text
train_scripts/baseline/Run1
```

改为由 `--mode`/实验路径生成，避免 CASAA 日志仍写 baseline。

---

## 8.7 修改 `models/eval.py`

这是 **P0 正确性项**。

CASAA-v1 可以做到零新增参数，因此 CASAA checkpoint 的 state dict 与 baseline 高度兼容。如果 `eval.py` 错误地按 baseline 架构实例化，有可能“权重正常加载，但实际执行的是普通 Attention”。

因此 eval 必须提供和训练一致的：

```text
--mode
--casaa_layers
--casaa_keep_ratio
--casaa_change_share
--casaa_router
```

独立测试时必须从对应 Run 脚本复用同一组参数。

---

## 8.8 修改 `models/smoke_test.py`

不要只测 forward/backward。增加下面 6 个机制测试：

1. **Baseline 等价测试**  
   `casaa_enabled=False` 时，新 `forward_pair` 与旧的 `vit(x), vit(y)` 在 eval 下 max error `<1e-6`。

2. **Pretrained load 测试**  
   CASAA 与 baseline 相比，pretrained missing/unexpected keys 集合不得因为 qkv/proj 改造而新增核心 attention key。

3. **Shape 测试**  
   block 输入 `B×256×192`，输出仍 `B×256×192`；最终 change map 仍 `B×1×256×256`。

4. **Token budget 测试**  
   Run1 必须打印：`N=256, K=64, Kc=32, Kb=32`。

5. **T1/T2 交换对称性**  
   eval 模式下交换 T1/T2，routing 的 change indices 与 background assignment 应一致；两个时相输出应对应交换。

6. **梯度测试**  
   `attn.qkv.weight`、`attn.proj.weight`、上游 token 均有有限非零 gradient，不允许 NaN/Inf。

---

# 9. Run1 的唯一变量设计

## 9.1 不要在 Run1 改这些

保持 baseline 一致：

- DeiT-Tiny pretrained；
- ResNet18 detail branch；
- Feature Injector；
- decoder；
- BCE+Dice；
- Adam；
- lr=2e-4；
- poly + 200 iter warmup；
- max_steps=80000；
- batch=16；
- 256×256；
- seed=16；
- test-as-val；
- 数据增强和 label gray≥128；
- checkpoint / log 规则。

否则不能把结果归因于 CASAA。

---

# 10. Run1 最小实验矩阵

建议不是只跑一个 CASAA，而是包含 **一个必要对照**：

| Variant | Q | K/V token 数 | Change-aware | Layer | 目的 |
|---|---:|---:|---:|---|---|
| A0 Baseline（已有） | N=256 | N=256 | × | 12层原 Attention | 现有对照 |
| A1 SAA-style | N=256 | K=64 | × | blocks 8-11 | 证明“仅压缩上下文”是否有效 |
| **A2 CASAA-v1** | **N=256** | **K=64** | **✓** | **blocks 8-11** | 验证变化感知 allocation 是否额外有效 |

A1 不需要完全复制 SAT 的 `c_ratio=0.5`。为了公平：

- A1 与 A2 都用原 pretrained qkv；
- 都用 K=64；
- 都用相同 layer；
- 区别只在：
  - A1：全部 token 内容聚类成 64 个 context；
  - A2：32 change-sensitive token 直保留 + 背景聚成 32 prototype。

这样如果 A2>A1，就能把收益归因到 **change awareness**。

---

# 11. Run1 首先跑哪两个数据集

### 第一阶段只筛 LEVIR + SYSU

原因：

- **LEVIR：变化像素约 4.1%**，建筑小目标、稀疏变化，最适合验证“保留稀疏变化 token”；
- **SYSU：变化像素约 21.1%**，变化密度高得多，能验证固定 K budget 是否过度压缩真实变化；
- WHU 与 LEVIR场景作用相近，CDD baseline F1 已 97.75，天花板高，不适合作为第一轮机制筛选。

先跑：

```text
A1-SAA / LEVIR
A1-SAA / SYSU
A2-CASAA / LEVIR
A2-CASAA / SYSU
```

如果 A2 通过成功门槛，再补：

```text
A2-CASAA / CDD
A2-CASAA / WHU
```

最终 A2 主方法必须四数据集齐全；A1 作为机制对照可先只保留 LEVIR+SYSU。

---

# 12. Baseline 工作对照

当前仓库汇总：

| Dataset | Recall | Precision | F1 | IoU | OA | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| CDD | 97.72 | 97.78 | **97.75** | **95.60** | 99.44 | 97.43 |
| LEVIR | 91.57 | 92.33 | **91.95** | **85.10** | 99.18 | 91.52 |
| SYSU | 80.77 | 84.27 | **82.48** | **70.19** | 91.91 | 77.23 |
| WHU | 93.62 | 96.09 | **94.84** | **90.18** | 99.60 | 94.63 |

Run1 必须和这些结果同协议比较。

---

# 13. Run1 推荐固定超参数

```text
model_type          = tiny
mode                = casaa
casaa_layers        = 8,9,10,11
casaa_keep_ratio    = 0.25
casaa_change_share  = 0.50
casaa_router        = change
cluster             = deterministic density-peak
norm_preserve       = on
seed                = 16
max_steps           = 80000
batch_size          = 16
input               = 256x256
loss                = BCE + Dice (不改)
```

### 为什么先用 25%，不直接 3%

SAT 的 3% 在 N=256 时只有约 7 个 K/V token。

RSBCD 对小建筑和边界很敏感，第一轮用 7 个 context 一旦失败，很难判断是“CASAA 思想失败”还是“budget 太激进”。

25% = 64 token 能产生明显压缩，同时足够稳健。

ratio sweep 放到 Run2：

```text
0.50 / 0.25 / 0.125 / 0.0625
```

不要 Run1 同时扫。

---

# 14. `train_scripts/CASAA/Run1` 建议目录

```text
train_scripts/
└── CASAA/
    └── Run1/
        ├── README.md
        ├── run_screen.sh
        ├── run_full_casaa.sh
        ├── train_A1_SAA_LEVIR-CD-256.sh
        ├── train_A1_SAA_SYSU-CD-256.sh
        ├── train_A2_CASAA_LEVIR-CD-256.sh
        ├── train_A2_CASAA_SYSU-CD-256.sh
        ├── train_A2_CASAA_CDD-CD-256.sh
        └── train_A2_CASAA_WHU-CD-256.sh
```

路径不要覆盖 baseline。

### Checkpoint

```text
/share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run1/A1_SAA/<dataset>/
/share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run1/A2_CASAA/<dataset>/
```

### Log

```text
/home/yqwang/outputs/CASA-CD/CASAA/Run1/A1_SAA/<dataset>/train_log.txt
/home/yqwang/outputs/CASA-CD/CASAA/Run1/A2_CASAA/<dataset>/train_log.txt
```

这个目录格式正好能被现有 `analyse/extract_metrics_to_excel.py` 解析成：

```text
Tag = CASAA
Run = Run1
Experiment = A1_SAA / A2_CASAA
Dataset = ...
```

---

# 15. GPU 启动顺序

当前 baseline batch16 单任务约 15.7GB，并且仓库已经记录过 GPU 争用导致 SYSU OOM。

因此仍保持：

> **一张 RTX 5090 同时最多一个任务。**

Run1 screening 推荐：

```text
GPU0: A1-SAA-LEVIR -> A2-CASAA-LEVIR
GPU1: A1-SAA-SYSU  -> A2-CASAA-SYSU
```

不要在同一张 GPU 上并行。

screen 通过后：

```text
GPU0: A2-CASAA-CDD
GPU1: A2-CASAA-WHU
```

---

# 16. Run1 成功阈值与失败判据

这是方法有效性探索，不用把 +0.05 当“提升”。

## 建议“通过”

LEVIR + SYSU 两数据集满足：

1. A2-CASAA 相对 baseline **任一数据集 F1 ≥ +0.30 个百分点**；
2. 另一数据集 F1 不得下降超过 **0.15 个百分点**；
3. IoU 与 F1 方向基本一致；
4. A2 相对 A1 在至少一个数据集 F1 ≥ +0.15 个百分点，并且另一数据集不明显更差；
5. 参数量不增加（允许日志四舍五入误差）；
6. Query/output token 数完全不变；
7. profiler 和理论计算不能显示 CASAA 总计算反而明显高于 baseline。

## 强通过

- LEVIR、SYSU 都 ≥ +0.30 F1；
- A2 都优于 A1；
- Recall 尤其在 LEVIR/WHU 类稀疏小目标数据上提升或保持。

## 直接判失败

- A2 在 LEVIR、SYSU 都不优于 baseline；
- A2 在两个数据集都不优于 A1；
- 任一数据集 F1 下降 ≥0.50；
- pretrained qkv 无法正常加载；
- output token 数/空间输出被压缩；
- routing 本身导致总 FLOPs/VRAM显著增加而没有精度收益。

---

# 17. 不同失败现象应该怎么解释

### 情况 A：LEVIR 提升，SYSU 下降

最可能不是概念失败，而是：

> 固定 `Kc=32` 对高变化率 SYSU 不够。

下一轮研究：

- adaptive change quota；
- 或固定总 K，但根据 change-score distribution 动态分配 Kc/Kb。

这正好能发展成有机制意义的 v2。

### 情况 B：SYSU 提升，LEVIR 下降

可能是 patch16 change score 对很小建筑不够敏感。

下一步应研究：

> 用轻量 detail cue 辅助 routing

而不是加一个新 loss。

### 情况 C：A1 和 A2 都下降

首先把 `keep_ratio` 从 0.25 放宽到 0.5，或只改最后 2 个 block。

如果仍下降，说明该 pretrained ViT 在当前分辨率下对完整 context 依赖较强，需重新审视 CASAA 插入位置。

### 情况 D：A1≈A2，都提升

说明：

> “压缩冗余 context”成立，但“变化感知 allocation”的额外价值尚未证明。

这时不能把 Change-Aware 当核心贡献，需要重新设计 change routing。

### 情况 E：A2 指标提高但计算不降

说明 routing 算法本身过重。

应优化 assignment，而不是靠论文只报告 attention matrix FLOPs。

---

# 18. FLOPs 论文表述要特别谨慎

CASAA attention：

\[
O(N^2C)\rightarrow O(NKC)
\]

这个说法对于 attention 矩阵是正确的。

但如果：

\[
K=rN
\]

且 r 是固定比例，那么从严格渐进复杂度看：

\[
O(NK)=O(rN^2)
\]

仍然是二次量级，只是常数显著下降。

因此 Run1 可以写：

> attention interaction cost is reduced from \(N^2\) to \(NK\)

但不要直接写“从 quadratic 变 linear”。

如果后面要做“高分辨率可扩展”的强主张，应改成：

\[
K=\min(K_{\max},\lceil rN\rceil)
\]

使高分辨率下 K 被 cap，才更接近真正的线性扩展。

另外 density clustering 自身也有成本，最终 FLOPs 必须把 routing 算进去，不能只报 `QK^T`。

---

# 19. 预训练权重兼容性验收标准

CASAA-v1 最重要的工程/科研正确性指标不是“能跑”，而是：

> **原 DeiT-Tiny attention 参数是否真的继续被利用。**

验收：

1. `attn.qkv.weight/bias` shape 不变；
2. `attn.proj.weight/bias` shape 不变；
3. CASAA block 的 state_dict key 与 baseline block 对应 key 一致；
4. pretrained load 后，这些 key 不应出现在新增 missing key 中；
5. 与 baseline 相比，新增 missing key 只能来自真正新增参数；而推荐的 parameter-free CASAA 理论上甚至不需要新增 learnable key。

---

# 20. Run1 前的 smoke / dry run 顺序

## Smoke-1：旧 baseline 等价

新代码 `--mode baseline`：

- forward；
- backward；
- 参数量；
- FLOPs；
- 固定输入下与修改前 baseline output 对齐。

不通过就不要训练。

## Smoke-2：CASAA 单 block

随机：

```text
B=2, N=256, C=192
```

检查：

```text
Q N=256
K/V K=64
output N=256
Kc=32
Kb=32
```

交换 T1/T2 检查 symmetry。

## Smoke-3：完整网络

随机 256×256 A/B：

- forward/backward 3 step；
- 无 NaN；
- final output `(2,1,256,256)`；
- qkv/proj gradient 非零。

## Dry run：真实 LEVIR

先跑几十到几百 iteration，不保存正式结果。

检查：

- loss 正常下降；
- 显存；
- 单 step 耗时；
- CASAA token stats；
- checkpoint 能恢复。

通过后再启动 80000 steps。

---

# 21. 我对 CASAA Run1 的最终建议

### 这轮一定做

- Paired late-stage CASAA；
- last 4 ViT blocks；
- Full Q；
- K/V=25%；
- parameter-free cosine change routing；
- shared T1/T2 assignment；
- pretrained fused qkv 原位继承；
- SAA-style same-budget control；
- LEVIR + SYSU screening；
- 通过后 CASAA 补齐四数据集。

### 这轮一定不做

- 新 loss；
- learnable change score head；
- edge/boundary 辅助模块；
- 同时换 ResNet18；
- 同时换 decoder；
- 量化/剪枝；
- 12 层全部 CASAA；
- 3% K/V ratio；
- q/k channel `c_ratio=0.5`；
- T1/T2 cross-value mixing；
- 复杂的 adaptive budget。

否则 Run1 无法回答最基本的问题：

> **“变化感知的非对称 K/V 建模本身是否有效？”**

---

# 22. 立即执行顺序

1. **先建分支 / 保存当前 baseline commit**，不要覆盖已复现代码状态；
2. 新增 `layers/casaa.py`，先实现 parameter-free paired router；
3. 给 `block.py` 加 `forward_pair`；
4. 给 `encoder.py` 加 `vit.forward_pair` 与指定层 CASAA；
5. 保证 CASAA block 仍使用原 `qkv/proj` key；
6. 修改 `trainer.py / train.py / eval.py` 参数透传；
7. 扩充 `smoke_test.py` 的 6 项机制测试；
8. `--mode baseline` 做回归等价；
9. CASAA random smoke；
10. LEVIR 真实 dry run；
11. 建 `train_scripts/CASAA/Run1`；
12. 先启动 A1/A2 × LEVIR/SYSU；
13. 读**最后完整 TEST RESULTS**；
14. 根据成功/失败门槛决定是否补 CDD/WHU；
15. CASAA 机制成立后，再进入“极轻量结构”主线二。

---

# 23. 仍需补充的证据

1. GitHub 当前没有 raw `train_log.txt`，正式比较前需要服务器对应 baseline/CASAA 日志；
2. 当前 `fvcore` 对 CASAA 动态 `topk/gather/scatter/cluster` 的 FLOPs 覆盖率需要 smoke 后实际检查 `unsupported_ops`；
3. 需要实测 CASAA 后 peak VRAM 和 step latency，不能只按理论 attention FLOPs判断；
4. 最终 `<3M` 路线必须另外解决 **DeiT-Tiny 本身约 5.5M** 的硬下限问题；这属于主线二，不应塞进 CASAA Run1。

---

## 一句话总结

**CASAA Run1 应该是“保留 DeiT-Tiny 预训练 QKV 的双时相共享路由实验”，而不是“复制 SAT 模块”。先在最后 4 个 ViT Block 用 `Full Q + 25% change-aware K/V` 单独证明机制，再谈 `<3M` 的极轻量结构。**
