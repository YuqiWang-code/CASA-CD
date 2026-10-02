# CASA-CD 主线重构方案：CASAA 编码器 + STR 时空结构重参数化
## ——面向 ≤5M 参数、四数据集硬目标的可执行研究与实验方案

> 日期：2026-10-02  
> 适用项目：`YuqiWang-code/CASA-CD`  
> 关联项目：`YuqiWang-code/STR-RepNet`  
> 任务限定：CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256，全监督遥感二值变化检测  
> 最终硬目标：**SYSU F1 ≥ 85、LEVIR F1 ≥ 92.5、WHU F1 ≥ 95、CDD F1 ≥ 98；IoU 同方向；有效推理参数 ≤ 5M**  
> 本文定位：研究路线重构 + 文献核验 + 模型方案 + 逐文件修改清单 + 四数据集实验矩阵 + 正确性/部署审计方案

---

# 0. 结论先行

## 0.1 当前主线确实偏离了最初想做的论文

当前 CASA-CD 最近的主线已经变成：

> Frozen ViT4 → TASS 空间 stem → TAR/DCR

这个路线的主要科学问题是“冻结 patch16 token 缺少 sub-patch 空间证据，是否需要从像素域重新学习 spatial residual”。它本身可以研究，但**它把 CASAA 从论文第一创新点降成了历史分析，把 TASS 变成了新的主要创新点**。这与你现在重新明确的论文结构不一致。

你真正想做的论文应该重新收束成**两条明确且互相配合的创新主线**：

1. **创新点一：CASAA——编码器内部的变化感知非对称 Token 建模**  
   完整保留 Query 的逐位置判别能力，只压缩 K/V；疑似变化 token 尽量保留，稳定背景 token 聚合，形成真正服务于双时相变化检测的非对称上下文建模。

2. **创新点二：STR——编码器到解码器的多尺度时空结构重参数化**  
   直接继承 STR-RepNet 中已经实现并验证折叠正确性的 TAR + DCR 思路：训练期多分支增强时相/尺度建模，推理期解析折叠成单路径静态卷积。

因此，**TASS 不应继续作为最终论文主线**；现有 TASS、B4、O-PRE、Depth-Pyramid 等代码全部保留为历史研究资产，但新主模型不再围绕它们扩展。

---

## 0.2 我建议的最终主架构

本文首选的最终方向是：

> **Pretrained SHViT-S1 truncated hierarchical encoder + CASAA@1/16 + TAR + DCR**

内部可暂命名：

**CASA-STRNet**

整体数据流：

```text
                         Shared Siamese pretrained SHViT-S1 trunk
                 ┌────────────────────────────────────────────────┐
A ───────────────► 1/4 F1 ─► 1/8 F2 ─► 1/16 F3 ─► 1/32 F4      │
                 │                         │                       │
B ───────────────► 1/4 F1 ─► 1/8 F2 ─► 1/16 F3 ─► 1/32 F4      │
                 └─────────────────────────┼───────────────────────┘
                                           │
                          CASAA on paired 1/16 features
                  Full Query N=256, compressed K/V K=64
                                           │
                  ┌────────────────────────┴─────────────────────┐
                  │ Four native hierarchical scales per time     │
                  └────────────────────────┬─────────────────────┘
                                           │
                                  Multi-Scale TAR
                         concat + sum + signed-diff
                              train multi-branch
                              deploy single 1×1
                                           │
                                 DCR top-down decoder
                         cross-scale RepPairFuse + RepLocal
                              train multi-branch
                              deploy single-path
                                           │
                                       1×1 head
                                           │
                                     256×256 mask
```

### 为什么不是继续 DeiT-Tiny prefix-4

因为你已经用 Run4–Run9 和 STRFusion 的大量结果证明：

- DeiT patch16 的 token 语义可以很强；
- 但“同一 16×16 token lattice 上取多深度、上采样、重建”不能稳定替代真实层次化空间表征；
- 当前 STRFusion 的 C0 本身很弱，TASS 在 LEVIR 的 +2.37pp 很大程度是在补过度轻量化后失去的空间能力，而不是在一个足够强的最终骨干上验证 CASAA。

新骨干必须同时满足：

- **有 ImageNet 大规模预训练**；
- **模型足够小**；
- **天然输出 1/4、1/8、1/16、1/32 多尺度特征**，能直接接 STR；
- **至少存在 token/attention 接口**，使 CASAA 真正属于编码器，而不是后接一个“伪 Transformer 模块”；
- 总模型能稳定控制在 5M 内。

在 2024–2026 的候选中，SHViT 是当前最干净的主方案。

---

## 0.3 Backbone 不要求“整网必须是纯 Transformer”

**结论：不需要。**

CASAA 真正要求的是：

> 在某个编码阶段存在二维空间 token，并使用 Q/K/V attention 进行上下文建模。

因此骨干可以是：

- 纯 Transformer；
- CNN-Transformer hybrid；
- 前期卷积 + 后期 attention；
- 其它层次化 backbone，只要在 CASAA 所在 stage 有明确的 token/QKV 接口。

反而在 **≤5M + 高分辨率 BCD** 约束下，强行使用纯 ViT 不一定合理，因为：

- 纯 patch16 ViT 天然缺少高分辨率层次特征；
- 你又需要把多尺度 lateral 输入给 TAR/DCR；
- 额外再补 detail branch 会重新变成三条主线。

所以本文推荐：

> **让轻量卷积负责早期局部/多尺度空间表征，让 CASAA 在 1/16 token stage 负责全局变化感知上下文。**

这比“为了有 token 而强行全 Transformer”更符合你的任务和论文故事。

---

# 1. 证据来源与阅读范围

## 1.1 已核对的 CASA-CD 内容

本轮分析以 CASA-CD 当前 `main` 为主，重点核对：

- `README.md`
- `docs/参考文献/文献索引.md`
- `docs/temporary/过去的想法/CASA-CD_研究路线_ChatGPT方案记录.md`
- `models/model/layers/casaa.py`
- `models/model/encoder.py`
- `models/model/decoder.py`
- `models/model/str_reparam.py`
- `models/model/str_tar.py`
- `models/model/str_dcr.py`
- `models/model/str_encoder.py`
- `models/model/str_fusion.py`
- `models/model/tass_stem.py`
- `models/model/str_tass_fusion.py`
- `models/train.py`
- `models/eval.py`
- `models/smoke_test.py`
- `others/SAT/`
- `others/ViT-CoMer/`
- `others/LaViT/`
- `others/LiFT/`
- `others/ResCLIP/`
- `others/EoMT/`
- `others/TokenCropr/`
- `others/FDAM/`

CASA 当前 README 也明确把最终目标更新为：

- SYSU ≥85
- LEVIR ≥92.5
- WHU ≥95
- CDD ≥98
- 有效推理参数 ≤5M

这与本轮用户重新确认的目标一致。

---

## 1.2 已完整读取的 STR-RepNet Run2 快照

本轮上传的：

`models_and_metrics_TAR-DCR_Run2.txt`

包含 **29 个 models 源码文件 + 26 组结果汇总**，已完整读取。重点核对了：

- `STRRepNet.py`
- `tar.py`
- `dcr_decoder.py`
- `reparam.py`
- `Mamba_backbone.py`
- `train.py`
- `smoke_test.py`
- `test_reparam_equivalence.py`
- 数据集与 metrics 代码

因此本文对 TAR/DCR 的判断不是只来自 README，而是来自实际源码数据流与折叠公式。

---

## 1.3 腾讯文档访问说明

用户要求先阅读：

`https://docs.qq.com/doc/DSU14emhGZ29SWVB0?u=bb125ab3a5b2455db1fe9b6b6c5a6029`

当前环境直接抓取腾讯文档页面失败（页面返回 cache miss），因此**本文不把未读取到的腾讯文档正文当作已知证据**。

本报告对该文档的依据仅限：

1. 用户本次消息中明确贴出的“非对称、变化感知 token / Full Query + Compressed K/V / Change-Aware SAA”段落；
2. CASA-CD 仓库中的原始路线文档，该文档与用户本次贴出的核心路线一致。

如果后续把腾讯文档导出为 PDF/Word/Markdown 上传，可以再做一次逐条对齐审查。

---

# 2. 证据表：现在真正知道了什么

| ID | 类型 | 证据 | 对新路线的含义 |
|---|---|---|---|
| E1 | 代码/实验事实 | ChangeViT-T 复现：CDD 97.75 / LEVIR 91.95 / SYSU 82.48 / WHU 94.84；有效参数约 11.754M，FLOPs 26.32G | 精度锚点较强，但明显超 5M |
| E2 | 代码/实验事实 | CASAA content-only K/V=25% 在已做实验中基本不损 F1 | “Full Q + compressed K/V”本身有可行性 |
| E3 | 代码/实验事实 | 原 cosine change router 未形成稳定增益 | 不能继续把简单 cosine TopK 当最终创新 |
| E4 | 代码/实验事实 | Oracle router 在 SYSU 相对 A1 约 +1.48 F1 | change-aware routing 的上限存在，主要瓶颈是可部署 change signal / placement |
| E5 | 代码/实验事实 | detail-only router Top32 precision 明显提高，但 F1 仍未提高 | “ranking 更准”本身不足；CASAA 必须重新放到更合适的层次化 encoder 中验证 |
| E6 | 代码/实验事实 | DeiT 12→4 block 损失不大，说明语义深度有冗余 | 可以用更浅、更小的预训练 encoder |
| E7 | 代码/实验事实 | B4/B12、O-PRE、depth pyramid、STRFusion 等说明 16×16 patch token 很难凭 decoder 重建真实 sub-patch evidence | 新 backbone 应天然提供真实层次化空间特征 |
| E8 | 代码/实验事实 | 当前 TASS：SYSU -0.41pp、LEVIR +2.37pp（已完成部分） | 空间 source 有价值但数据集依赖；更说明应该从 backbone 解决，而不是把 TASS 做成第三创新 |
| E9 | STR 源码事实 | TAR 每尺度 `concat + sum + signed(Q-P)`，训练多分支，部署解析折叠成单 1×1 | 可作为第二创新点的 temporal skip/bridge |
| E10 | STR 源码事实 | DCR 的 RepDW / RepPW / RepPairFuse 都可折叠，top-down decoder 推理为单路径 | 可把训练期表达力与部署轻量化同时保留 |
| E11 | STR 结果事实 | Run2 full_last2：CDD 98.42 / LEVIR 91.44 / SYSU 83.45 / WHU 95.14，但整网 29.57M | STR 结构有一定性能潜力，但原 VMamba backbone 完全不符合 ≤5M |
| E12 | 外部论文事实 | MobileNetV4 2024 已明确研究“高分辨率 Q + 下采样 K/V” | “Full Q + reduced K/V”本身不能单独作为新颖性；必须突出**双时相变化感知选择/背景聚合** |
| E13 | 外部论文事实 | Token Cropr 直接删 token；CASAA 不删 Query | Full-Q dense prediction 是与 pruning 路线的重要实质区别 |
| E14 | 外部论文事实 | LWGANet AAAI 2026 已把遥感“空间冗余 + Top-K salient region”作为核心问题 | CASAA 不能只讲“遥感背景冗余”；必须讲**双时相变化条件下的非对称上下文压缩** |
| H1 | 待验证假设 | 预训练层次化轻量 backbone + CASAA@1/16 能同时保留局部结构和变化感知全局上下文 | 新主线的第一个核心假设 |
| H2 | 待验证假设 | STR 在新轻量 backbone 上能给出跨四数据集一致的训练优化增益，同时 deploy 图不增参数 | 第二创新点必须通过 4 数据集受控消融 |
| H3 | 待验证假设 | 二者组合能达到 85 / 92.5 / 95 / 98 且 ≤5M | 最终系统假设 |

---

# 3. 为什么现在的 TASS 主线不适合作为最终论文主线

## 3.1 它改变了论文的创新结构

原本应当是：

```text
Innovation 1: CASAA in encoder
Innovation 2: STR skip/decoder
```

但当前变成：

```text
Frozen ViT4
+ TASS spatial source
+ TAR/DCR
```

这样论文会出现三个问题：

1. CASAA 没有处于主模型；
2. TASS 反而需要被包装成新的核心贡献；
3. STR 变成普通 scaffold，而不是第二个受控验证的创新。

论文故事会散。

---

## 3.2 当前 C0 token scaffold 本身过弱

现有 TASS 实验中，LEVIR：

- C0 token-only：87.79
- TASS：90.16
- 原 ChangeViT-T：91.95

TASS 虽然提升 2.37pp，但它仍没有恢复到强 baseline 水平。

因此当前问题更像：

> 先把 backbone/detail 能力削弱，再让 TASS 补回一部分。

这不利于写成“两个清晰创新共同形成极轻量 SOTA”。

---

## 3.3 STR 需要真实层次化 skip，而不是 depth-as-scale

STR 原始设计天然消费：

```text
1/4
1/8
1/16
1/32
```

这样的多尺度 feature hierarchy。

当前 CASA STRFusion 把不同 ViT block 的 16×16 token 做上/下采样伪造成四尺度，这与真正的层次化 backbone 不是一回事。

因此更合理的是：

> 找一个已经有 ImageNet 预训练、原生多尺度 feature 的极轻量 hybrid backbone，然后把 CASAA 放在其中的 token stage，再接 STR。

---

# 4. 2024–2026 Backbone 调研与筛选

## 4.1 筛选标准

不是简单找“参数最小”的模型，而是同时看：

1. 2024–2026 高水平论文；
2. 有可核验官方代码；
3. 有 ImageNet-1K 等大规模预训练权重；
4. 能在 5M 总预算内留下 decoder 余量；
5. 能输出 dense multi-scale feature；
6. CASAA 能自然放入 encoder；
7. 不与 CASAA 的 novelty 发生严重冲突。

---

## 4.2 关键候选比较

| Backbone | Venue / 层级 | 官方预训练 | 原模型规模 | CASAA 适配性 | 结论 |
|---|---|---|---|---|---|
| **SHViT-S1** | CVPR 2024，CCF-A | 有，ImageNet-1K | 6.3M / 241M FLOPs @224 | **高**：后段有 QKV attention，前段卷积保留局部结构；可截断 | **首选** |
| **LWGANet-L0** | AAAI 2026，CCF-A | 有，ImageNet-1K 300e | 极轻，RS 专用 | 中高：内部存在 global attention，但其 TGFI 已做空间 Top-K 冗余抑制 | **强备选，但 novelty 碰撞风险高** |
| EfficientViM-M1 | CVPR 2025，CCF-A | 有 | 6.7M / 239M | 低：核心是 HSM-SSD，不是 QKV attention；本体已超总预算 | 否决主线 |
| RepViT-M0.9 | CVPR 2024，CCF-A | 有 | 5.1M / 0.8G | 低：主要是 mobile CNN，没有自然 Full-Q/KV 接口，且 backbone 单体接近上限 | 否决主线 |
| MambaOut-Femto | CVPR 2025，CCF-A | 有 | 7.3M / 1.2G | 低：无自然 QKV，且超预算 | 否决 |
| MobileNetV4 Hybrid-M | ECCV 2024，**CCF-B** | 官方体系有预训练方案 | 10.5M / 1.2G | 理论上高，但远超预算；且 Mobile MQA/SRA 已是高 Q、低分辨率 KV | 否决主线，但属于 CASAA 必须讨论的近邻 |
| StarNet | CVPR 2024，CCF-A | 有 | compact CNN | 低：无原生 QKV，需要额外造 attention | 只适合纯轻量 backbone 对照，不适合作 CASAA 主体 |
| DeiT-Tiny prefix-4 | 原基线资产 | 有 | 当前项目已验证约 2M 级 prefix | QKV 兼容性非常好 | 但只有 16×16 token，真实 multi-scale 弱，已经被前序实验暴露 |

> **层级纠正**：ECCV 按 CCF 目录是 B，不应标成 CCF-A。STR-RepNet 旧索引里有把 ECCV 与 CCF-A 并列的项目内部简化写法；正式论文必须纠正。

---

# 5. 为什么首选 SHViT-S1 truncated

## 5.1 它正好提供你需要的“Hybrid”结构

SHViT 的关键设计是：

- 早期 stage 可以用卷积替代 attention；
- 后期只在部分通道做 Single-Head Self-Attention；
- 同时保持非常低的参数和 FLOPs。

这与 CASA-CD 的目标非常契合：

> 早期局部细节不需要强行 token 化；真正需要全局上下文时再用 attention。

---

## 5.2 官方代码可以自然截断并保留预训练权重

SHViT-S1 官方配置：

```text
embed_dim   = [128, 224, 320]
depth       = [2, 4, 5]
partial_dim = [32, 48, 68]
types       = ["i", "s", "s"]
```

官方 patch embedding 是连续四个 stride=2 卷积，因此在 256×256 输入上天然经过：

```text
1/2
1/4
1/8
1/16
```

随后 stage2 进入 1/32。

我们不需要完整的 6.3M SHViT-S1。

只保留：

- patch embed；
- blocks1；
- blocks2；

删除：

- blocks3；
- ImageNet classifier。

按照官方源码逐层解析计算，**截断 trunk 约 1.861M 参数**。

注意：

> 1.861M 是源码解析估算，不是 RSML-3 机器审计事实。实现后必须用 PyTorch 实例化模型重新统计。

---

## 5.3 四个输出正好对应 STR

推荐输出：

```text
F1: 1/4  × C32
F2: 1/8  × C64
F3: 1/16 × C128
F4: 1/32 × C224
```

这样 TAR 可直接改成：

```python
encoder_dims=(32, 64, 128, 224)
```

不再需要 TASS，不再需要“B1↑4/B2↑2/B4↓2”这种 depth-as-scale 人工重采样。

---

# 6. 创新点一：重新定义 CASAA

## 6.1 论文主张必须比“Full Q + compressed K/V”更具体

仅仅写：

> 保留 Query，压缩 K/V。

已经不够。

因为 2024 MobileNetV4 的 Mobile MQA/SRA 已经明确使用：

> high-resolution Query + spatially downsampled K/V

所以 CASA-CD 的创新性必须放在：

> **bi-temporal change-conditioned selective context construction**

即：

- Query 全保留；
- K/V 不是均匀池化；
- K/V 不是普通内容聚类；
- K/V 的“直保留 vs 聚合”由双时相变化线索决定；
- 变化候选保留为独立 context token；
- 稳定背景强聚合。

---

## 6.2 推荐 CASAA 放在 1/16，而不是最后 1/32

256×256 输入：

```text
1/16 -> 16×16 -> N=256
```

这是非常合适的位置：

- N 足够大，压缩有意义；
- 比 1/8 的 N=1024 轻得多；
- 比 1/32 的 N=64 更有“保留完整空间 query”的实际价值；
- 与旧 ChangeViT 的 N=256 可以直接对照。

固定：

```text
N = 256
K = 64
keep_ratio = 0.25
```

注意力矩阵交互从：

```text
256 × 256
```

降为：

```text
256 × 64
```

但必须诚实：

> 这是 attention interaction 的复杂度下降，不等价于整网 FLOPs 降 75%。Q 投影、backbone、TAR/DCR 仍有计算。

---

## 6.3 变化 score：不再用 ViT 同层 cosine 作为唯一信号

当前仓库已经证明：

- 同层 ViT cosine router 不够；
- 早期 detail cue 的 ranking 更好。

新 backbone 本身就有真实 1/8 feature，因此不再需要 ResNet detail branch。

建议：

\[
S_{loc}(i)=1-\cos(\operatorname{Pool}_{2\times2}F^A_{1/8}(i),
                  \operatorname{Pool}_{2\times2}F^B_{1/8}(i))
\]

得到：

```text
32×32 F2
  ↓ avg pool 2
16×16 score
```

然后每图 rank-normalize：

\[
\hat S = RankNorm(S_{loc})
\]

这有四个优点：

1. 完全参数自由；
2. A/B 对换对称；
3. 来源于 ImageNet 预训练的真实 1/8 local feature；
4. 与 CASAA 的 16×16 token 一一对齐。

**本轮不要继续做 score zoo。**

先把 CASAA 放到正确的 hierarchical backbone 中，验证其机制。

---

## 6.4 Context bank

保持当前已验证机械正确的两部分：

总数：

\[
K=64
\]

其中：

```text
Kc = 32    change candidates
Kb = 32    stable-background prototypes
```

### Change bank

\[
I_c = TopK(\hat S,K_c)
\]

对应 token 原位保留：

\[
C^A_c=X^A[I_c],\qquad C^B_c=X^B[I_c]
\]

不平均、不池化。

### Background bank

排除 `Ic` 后，对稳定 token 做共享 assignment 的密度聚类：

\[
Z_{bg}=Norm\left(\frac{X^A_{bg}+X^B_{bg}}{2}\right)
\]

assignment 在共享 descriptor 上计算，但 A/B 各自聚合：

\[
C^A_b=Aggregate(X^A_{bg}),\qquad
C^B_b=Aggregate(X^B_{bg})
\]

因此：

- cluster id 在两个时相一致；
- feature value 不跨时间直接混合；
- 仍保留双时相可比较性。

第一版继续复用现有 `deterministic_density_assign()` 和 `norm_preserve()`，**不要同时重写 backbone、score、聚类算法三个变量**。

---

## 6.5 Full Query attention

对于时相 \(t\in\{A,B\}\)：

\[
Q_t=X_tW_Q
\]

\[
K_t=C_tW_K,\qquad V_t=C_tW_V
\]

\[
Y_t=Softmax\left(\frac{Q_tK_t^T}{\sqrt d}\right)V_t
\]

其中：

```text
Q: B × H × 256 × d
K: B × H × 64  × d
V: B × H × 64  × d
```

输出：

```text
Y: B × 256 × C
```

所以 **Query 空间位置数从头到尾不变**。

---

## 6.6 新 CASAA block 用 zero-init residual 接入预训练 backbone

SHViT stage1 原本没有 attention，因此 CASAA 是新增模块。

为了不在初始化时破坏预训练 trunk：

\[
F'_3=F_3+\beta \cdot CASAA(F^A_3,F^B_3)
\]

其中：

```text
beta = 0 at init
```

A/B 各有输出，但共享同一个 CASAA 权重。

因此在 eval 模式：

```text
epoch-0 CASAA model
==
truncated SHViT pretrained trunk
```

这是必须做的 smoke 检查。

注意：

- β=0 时第一步 CASAA qkv 梯度可能为 0；
- β 本身必须有非零梯度；
- 做一次 β nudge / 两步优化后，qkv/proj 梯度必须出现；
- 这是正确性测试，不是性能 gate。

---

# 7. 创新点二：STR 作为真正的 skip + decoder

## 7.1 TAR：四尺度二时相桥接

每尺度输入：

\[
P=F^A_i,\qquad Q=F^B_i
\]

训练图包含：

1. concat branch：
   \[
   Conv_c([P,Q])
   \]

2. sum branch：
   \[
   Conv_s(P+Q)
   \]

3. signed-diff branch：
   \[
   Conv_d(Q-P)
   \]

每支独立 BN。

输出：

\[
T_i=BN_c(Conv_c([P,Q]))
   +BN_s(Conv_s(P+Q))
   +BN_d(Conv_d(Q-P))
\]

部署时因为三条路径均为线性变换，可以解析合并为：

\[
T_i=Conv_{eq}([P,Q])
\]

即一个单独的 1×1 conv。

### 为什么保留 signed difference

`abs(P-Q)` 是非线性的，不能无损折叠进单个 1×1。

`Q-P` 是线性的，可以被精确吸收：

\[
W_P=W_{c,P}+W_s-W_d
\]

\[
W_Q=W_{c,Q}+W_s+W_d
\]

因此它是“变化方向基 + 可解析折叠”的核心。

---

## 7.2 DCR：top-down decoder

四尺度：

```text
t1 = 1/4
t2 = 1/8
t3 = 1/16
t4 = 1/32
```

top-down：

```text
d4 = t4

u4 = up(d4 → t3)
d3 = RepLocal(RepPairFuse(t3, u4))

u3 = up(d3 → t2)
d2 = RepLocal(RepPairFuse(t2, u3))

u2 = up(d2 → t1)
d1 = RepLocal(RepPairFuse(t1, u2))

d1 = RepLocal(d1)
```

最后：

```text
Conv1×1(D→1)
bilinear ×4
sigmoid
```

---

## 7.3 DCR 内部的部署折叠

继续使用 STR 已有原语：

### RepDW3

训练：

```text
DW3×3
+ DW1×3
+ DW3×1
+ αI
```

部署：

```text
single DW3×3
```

### RepPW1x1

训练：

```text
dense 1×1
+ low-rank serial 1×1
+ diagonal branch
+ αI
```

部署：

```text
single dense 1×1
```

### RepPairFuse1x1

训练：

```text
concat
+ sum
+ signed diff
+ α·low
```

部署：

```text
single 1×1 on concat(low, high)
```

---

## 7.4 第二创新点必须如何表述

不能写：

> “sum/diff 提供了部署时额外信息。”

因为部署后它仍是一个普通线性卷积，其函数类并没有超出 dense 1×1。

更准确的论文表述是：

> **通过具有变化先验含义的训练期多分支参数化，为时相融合与跨尺度融合提供更有结构的优化路径；部署时解析折叠为与 plain 对照相同的单路径算子，实现 train-rich / deploy-simple。**

因此第二创新点必须有：

```text
plain
vs
full reparameterized
```

的严格同部署函数类对照。

如果 full 在四数据集上没有稳定增益，第二创新点不能仅凭“能折叠”就声称有效。

---

# 8. 最终模型的训练图与推理图

## 8.1 训练图

```text
A/B
 │
 ├── shared pretrained SHViT-S1 truncated
 │       ├── F1 1/4
 │       ├── F2 1/8 ── local change score
 │       ├── F3 1/16 ── paired CASAA
 │       └── F4 1/32
 │
 ├── TAR(full)
 │     concat + sum + signed-diff
 │
 ├── DCR(full)
 │     RepPairFuse + RepDW + RepPW
 │
 └── head → sigmoid mask
```

建议：

- retained SHViT backbone：低学习率；
- CASAA + TAR/DCR + head：基础学习率。

---

## 8.2 推理图

```text
A/B
 │
 ├── truncated SHViT
 ├── CASAA (Full Q + K=64 compressed KV)
 ├── folded TAR (single 1×1 per scale)
 ├── folded DCR (single-path convs)
 └── head
```

推理期不存在：

- teacher；
- cache；
- auxiliary branch；
- deep supervision head；
- TASS；
- edge head；
- 新 loss branch。

---

# 9. 参数与 FLOPs 预算

## 9.1 SHViT trunk

按官方 SHViT-S1 源码解析：

```text
patch_embed + blocks1 + blocks2
≈ 1.861M params
```

这是**解析估算**。

---

## 9.2 CASAA

若 CASAA 在 C=128：

标准 fused QKV + proj 大约：

\[
3C^2+C^2\approx4C^2
\]

即约：

```text
~0.066M
```

再加 LayerNorm / β，仍非常小。

---

## 9.3 TAR + DCR

保持：

```text
D = 160
encoder_dims = (32,64,128,224)
```

按部署算子解析估算：

```text
TAR + DCR + 1ch head
≈ 0.52M
```

---

## 9.4 总参数解析估计

\[
1.861+0.066+0.52\approx2.45M
\]

因此即使考虑实现细节差异，也有非常大的 5M 余量。

**但论文与 README 不允许直接写 2.45M 正式结果。**

实现后必须在 RSML-3 上机器审计：

```text
TOTAL
TRAINABLE
EFFECTIVE DEPLOY
FLOPs
```

正式硬门槛：

```text
DEPLOY EFFECTIVE <= 5.000M
```

---

## 9.5 为什么 D 固定 160，不再扫宽度

D=160 是 STR 已经长期验证过的解码宽度。

新路线不应：

```text
D=96 / 112 / 128 / 160 / 192
```

逐个扫 F1。

否则论文又回到工程调参。

本轮固定 D=160，参数本身已经远低于 5M。

---

# 10. 预训练权重兼容方案

## 10.1 官方权重

首选：

**SHViT-S1 ImageNet-1K official pretrained checkpoint**

官方仓库：

`https://github.com/ysj9909/SHViT`

---

## 10.2 加载规则

新 truncated class 只构造：

```text
patch_embed
blocks1
blocks2
```

从官方 checkpoint 加载同名 retained keys。

允许 missing：

```text
blocks3.*
head.*
head_dist.*
```

不允许 retained key：

```text
shape mismatch
missing
unexpected remap
```

---

## 10.3 必须写入日志的预训练审计

```text
[PRETRAIN-AUDIT]
source = shvit_s1 official ImageNet-1K
retained_keys = ...
exact_loaded_keys = ...
shape_mismatch = 0
unexpected_retained = 0
omitted_stage3_keys = ...
omitted_head_keys = ...
retained_checksum_before = ...
```

不能只打印：

```text
strict=False
```

然后默认“加载成功”。

---

# 11. 优化器与训练协议

## 11.1 固定协议

所有正式实验统一：

```text
Loss           = BCE + Dice
Optimizer      = Adam
base lr        = 2e-4
betas          = (0.9, 0.99)
weight_decay   = 1e-4
LR             = poly, power=0.9
warmup         = 200 iter
max_steps      = 80000
batch          = 16
input          = 256×256
seed           = 16
best criterion = test F1
threshold      = 0.5
label          = gray >= 128
```

所有 A/B/label 几何增强同步。

---

## 11.2 Backbone LR

考虑到 CASA 已经真实观察到 DeiT 在统一 2e-4 下发生严重 collapse，新预训练 backbone 不建议直接与随机初始化 decoder 使用相同学习率。

固定、预注册：

```text
backbone lr = 2e-5
new modules lr = 2e-4
```

即：

```text
backbone_lr_ratio = 0.1
```

这不是调参贡献。

所有实验组固定一致。

**禁止根据四数据集结果再修改 ratio。**

---

# 12. 代码修改方案

## 12.1 不删除历史代码

以下全部保留：

```text
casaa.py
TASS
B4
O-PRE
depth pyramid
旧 ChangeViT baseline
STRFusion
```

原因：

- 保留完整科研证据链；
- 后续论文补历史分析时可复现；
- 避免覆盖已经完成的结果。

新路线全部加性实现。

---

## 12.2 新增 `models/model/shvit_trunc.py`

职责：

1. 复刻 SHViT-S1 retained trunk；
2. exact load official pretrained weights；
3. 逐尺度返回：
   - 1/4 C32
   - 1/8 C64
   - 1/16 C128
   - 1/32 C224
4. 提供 paired forward，允许在 1/16 插入 CASAA；
5. 提供 retained checksum。

推荐接口：

```python
class SHViTS1Truncated(nn.Module):
    def __init__(self, pretrained_path, attn_mode="none", ...):
        ...

    def forward_pair(self, a, b):
        # return feats_a, feats_b
        # each: [f1, f2, f3, f4]
        ...
```

---

## 12.3 新增 `models/model/layers/casaa_hier.py`

不要直接重写旧 `casaa.py`。

复用旧代码：

- deterministic density assignment；
- shared assignment；
- norm preserve；
- Full Q / compressed KV；
- diagnostics。

改动：

1. 输入从 `(B,N,C)` 支持 `(B,C,H,W)`；
2. score 来自 external 1/8 paired feature；
3. CASAA 位于 C=128、N=256；
4. zero-init residual β；
5. `attention_mode` 支持：
   - `none`
   - `full`
   - `saa`
   - `casaa`

这样所有 CASAA 消融使用同一个类，不会出现实现漂移。

---

## 12.4 新增 `models/model/casa_str_net.py`

顶层：

```python
class CASASTRNet(nn.Module):
    def __init__(
        self,
        pretrained_path,
        attn_mode="casaa",
        rep_mode="full",
        str_dim=160,
        keep_ratio=0.25,
        change_share=0.5,
    ):
        self.encoder = SHViTS1Truncated(...)
        self.tar = MultiScaleTAR(
            encoder_dims=(32,64,128,224),
            dim=160,
            ...
        )
        self.decoder = DCRDecoder(...)
        self.head = Conv2d(160,1,1)
```

forward：

```text
A,B
→ encoder.forward_pair()
→ TAR
→ DCR
→ 1ch logits
→ resize 256
→ sigmoid
```

deploy：

```python
switch_to_deploy():
    tar.switch_to_deploy()
    decoder.switch_to_deploy()
```

CASAA 不做部署删除。

---

## 12.5 复用现有 STR 文件

继续复用并只做泛化：

```text
models/model/str_reparam.py
models/model/str_tar.py
models/model/str_dcr.py
```

要求：

- `encoder_dims` 可传 `(32,64,128,224)`；
- plain/full 两个 rep mode；
- 不加入 TASS 逻辑；
- 不加入新 loss；
- 不加入 edge head。

---

## 12.6 修改 `models/train.py`

新增：

```text
--arch casa_str
--backbone shvit_s1_trunc
--attn_mode none|full|saa|casaa
--rep_mode plain|full
--str_dim 160
--casaa_keep_ratio 0.25
--casaa_change_share 0.50
--backbone_lr_ratio 0.1
```

optimizer：

```python
backbone_params -> lr=2e-5
new_params      -> lr=2e-4
```

其它协议不改。

---

## 12.7 修改 `models/eval.py`

必须：

1. 从 `arch.json` 读取并严格对齐结构；
2. 加载 best；
3. TAR/DCR `switch_to_deploy()`；
4. 在 deploy 图跑正式 test；
5. 输出最后完整：

```text
=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

---

## 12.8 新增 `analyse/audit_casa_str.py`

只做正确性与复杂度审计，不做“raw F1 gate”。

检查：

```text
pretrained exact load
feature shapes
full Q N unchanged
K=64
A/B routing swap symmetry
gradient paths
params
FLOPs
deploy folding
argmax disagreement
```

**不要再用 raw feature PR-AUC 来决定是否允许 80K。**

原因：

TASS-D0 已经给出过直接反例：

> 零训练 raw gate 与端到端训练后的真实价值可以方向不一致。

用户现在又明确要求所有正式实验完整跑四数据集，因此 accuracy gate 应取消。

---

# 13. Smoke / P0 正确性测试

正式 80K 前必须全过，但这些不是“性能实验”。

## T0：shape

256×256：

```text
F1 = B×32 ×64×64
F2 = B×64 ×32×32
F3 = B×128×16×16
F4 = B×224×8×8
```

CASAA：

```text
N=256
K=64
output N=256
```

---

## T1：pretrained exact load

所有 retained SHViT keys：

```text
checkpoint tensor == model tensor
```

逐位检查。

---

## T2：CASAA epoch-0 identity

eval：

```text
attn_mode=none
vs
attn_mode=casaa, beta=0
```

在共享 trunk 输出处：

```text
max_abs_diff = 0
```

---

## T3：CASAA 梯度

step0：

```text
beta.grad != 0
```

β 轻微 nudge 或完成一步 optimizer 后：

```text
qkv.grad != 0
proj.grad != 0
background aggregated tokens upstream grad != 0
```

---

## T4：routing 对换一致性

```text
score(A,B) == score(B,A)
Ic(A,B) == Ic(B,A)
assignment(A,B) == assignment(B,A)
```

---

## T5：TAR/DCR block fold

建议：

```text
FP64 algebra < 1e-10
FP32 block max_abs < 2e-5
```

---

## T6：整网 fold

```text
train graph vs deploy graph
max_abs < 2e-4
```

随机输入与真实数据 batch 都测试。

---

## T7：二值决策

正式 checkpoint：

```text
threshold=0.5
train graph mask == deploy graph mask
```

优先要求 disagreement=0。

---

## T8：预算

```text
effective deploy params <= 5M
```

不通过则不能进入正式训练。

---

# 14. 实验原则：每个实验必须四数据集完整训练

这一点按用户本轮要求重新固定。

以后禁止：

```text
只跑 SYSU
→ 看结果
→ 决定是否跑 LEVIR
```

也禁止：

```text
只跑 LEVIR+WHU 消融
```

一个“正式实验变体”的定义是：

```text
同一结构
× CDD
× LEVIR
× SYSU
× WHU
```

**四个 80K 全部完成后，才允许给该变体下结论。**

smoke / 参数审计 / deploy 等价性不属于准确率实验，可以先跑。

---

# 15. 最小但完整的论文实验矩阵

不建议几十个模块 sweep。

最终论文只需要 5 个结构变体。

## 15.1 A0：Base + Plain STR

```text
SHViT truncated
no extra attention
TAR/DCR plain
```

用途：

> 新轻量 backbone + deploy topology 基线。

四数据集 ×1。

---

## 15.2 A1：Full Attention + Plain

```text
SHViT truncated
full self-attention @ 1/16
plain TAR/DCR
```

用途：

> 回答“直接加完整全局 attention 是否就够了”。

与 CASAA 使用相同 QKV/proj 通道。

四数据集 ×1。

---

## 15.3 A2：SAA-style + Plain

```text
Full Query
content-only compressed K/V
K=64
plain TAR/DCR
```

用途：

> 证明 asymmetric K/V compression 是否能保持 full-attention 精度同时降低 attention interaction。

四数据集 ×1。

---

## 15.4 A3：CASAA + Plain

```text
Full Query
change-aware Kc + background aggregation
plain TAR/DCR
```

用途：

> 创新点一主消融。

比较：

```text
A3 vs A2 -> change awareness
A3 vs A1 -> selective compressed context vs full context
A3 vs A0 -> CASAA system contribution
```

四数据集 ×1。

---

## 15.5 M1：CASAA + Full STR

```text
SHViT truncated
CASAA
TAR full
DCR full
```

用途：

> 最终方法。

比较：

```text
M1 vs A3
```

唯一变量就是：

```text
plain parameterization
vs
train-time structural reparameterization
```

部署拓扑与参数量应完全相同。

四数据集 ×1。

---

# 16. 总训练预算

完整论文核心矩阵：

```text
5 variants × 4 datasets = 20 个 80K
```

两张 5090 可以并行，但一个实验组的结论必须等四数据集全部结束。

推荐执行顺序：

```text
Phase 0  correctness audit

Phase 1
A0_BASE_PLAIN × 4
M1_CASAA_STR  × 4

Phase 2（M1 四数据集完成后）
A3_CASAA_PLAIN × 4

Phase 3
A2_SAA_PLAIN × 4
A1_FULLATTN_PLAIN × 4
```

为什么先 A0 + M1：

> 先判断新系统是否值得成为最终模型，而不是先花大量 GPU 做机制消融。

但无论哪一组，一旦启动，必须把四数据集完整跑完。

---

# 17. 预注册成功 / 失败判据

## 17.1 最终系统硬门槛

M1 必须同时：

| Dataset | F1 |
|---|---:|
| SYSU | ≥ 85.00 |
| LEVIR | ≥ 92.50 |
| WHU | ≥ 95.00 |
| CDD | ≥ 98.00 |

并且：

```text
IoU 与 F1 同方向
deploy effective params <= 5M
deploy fold 合格
```

任意一项不过：

> 不能说最终硬目标达成。

---

## 17.2 CASAA 成功判据

不要用某一个数据集 +0.1 就说成功。

建议：

### Strong

```text
A3 - A2 Macro F1 >= +0.20pp
至少 3/4 数据集 A3 > A2
任何数据集下降不超过 0.10pp
Macro IoU 同方向
```

### Weak

```text
Macro F1 > 0
但只在 1–2 个数据集有效
```

只能写 dataset-dependent observation。

### Fail

```text
Macro F1 <= 0
或主要表现为 Recall↓ / Precision↑ 的 confidence sharpening
```

不能把 CASAA 写成稳定精度创新。

---

## 17.3 SAA 压缩判据

A2 对 A1：

```text
平均 F1 损失 <= 0.15pp
attention FLOPs 明显下降
```

才可以写：

> 压缩冗余 context 基本无损。

---

## 17.4 STR 成功判据

M1 vs A3：

### Strong

```text
Macro F1 >= +0.15pp
至少 3/4 数据集非负
IoU 同方向
deploy params/FLOPs 与 A3 对应 plain deploy 图一致
```

### Fail

若再次出现：

```text
Recall 明显下降
Precision 上升
F1 中性
```

则应认定：

> structural branch 主要产生 confidence sharpening，而非稳定判别增益。

这点在 STR 历史实验中已经多次出现，不能重复包装。

---

# 18. 日志与正式结果纪律

任何结果只认：

```text
同一 train_log.txt
最后一个完整

=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

禁止用：

- best epoch 行；
- checkpoint 文件名；
- README 手填数字；
- validation 最佳行；
- raw gate；
- dry run。

正式 TEST block 必须包含：

```text
Recall
Precision
OA
F1
IoU
Kappa
total params
trainable params
effective deploy params
FLOPs
reparam max abs error
argmax / binary disagreement
backbone pretrained audit id
```

---

# 19. 训练目录建议

```text
train_scripts/
└── CASA-STR/
    └── Run1/
        ├── README.md
        ├── audit.sh
        ├── A0_BASE_PLAIN/
        │   ├── train_CDD-CD-256.sh
        │   ├── train_LEVIR-CD-256.sh
        │   ├── train_SYSU-CD-256.sh
        │   └── train_WHU-CD-256.sh
        ├── A1_FULLATTN_PLAIN/
        ├── A2_SAA_PLAIN/
        ├── A3_CASAA_PLAIN/
        └── M1_CASAA_STR/
```

---

# 20. 服务器路径

代码：

```text
/home/yqwang/projects/CASA-CD
```

数据：

```text
/share_datasets/CD/<dataset>
```

建议新权重：

```text
/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth
```

checkpoint：

```text
/share_datasets/yqwang/checkpoints/CASA-CD/CASA-STR/Run1/<group>/<dataset>/
```

日志：

```text
/home/yqwang/outputs/CASA-CD/CASA-STR/Run1/<group>/<dataset>/train_log.txt
```

诊断：

```text
/home/yqwang/outputs/CASA-CD/diagnostics/CASA-STR/Run1/
```

---

# 21. 两卡启动方式

不建议同一数据集同一组跨卡 DDP，保持现有单卡可复现方式。

示例：

```text
GPU0:
CDD → LEVIR

GPU1:
SYSU → WHU
```

每个 group 两条 queue。

例如：

```text
run_gpu0_queue.sh
run_gpu1_queue.sh
```

只有：

- 自己的 `last.pth` 崩溃恢复；
- 不允许从其他组 checkpoint 初始化；
- 不允许 A0 → M1 fine-tune。

---

# 22. 为什么不推荐其它主 backbone

## 22.1 LWGANet-L0

它非常有吸引力：

- AAAI 2026 Oral；
- Remote Sensing 专用；
- ImageNet 预训练；
- 已有 CD 代码；
- 本身很轻。

但它的核心 TGFI 已经是：

> 用 Top-K salient regions 处理遥感 spatial redundancy。

如果再把 CASAA 放进去，审稿人非常容易问：

> CASAA 与 TGFI 的 spatial redundancy / Top-K 到底有什么本质区别？

所以它适合作为：

> SHViT 主线失败后的预注册备选 backbone。

不建议现在直接作为论文第一版本主干。

---

## 22.2 MobileNetV4

MobileNetV4 很重要，但它更像**近邻证据**而不是主 backbone。

因为它明确使用：

> high-resolution Q + downsampled K/V

所以它提醒我们：

> Full-Q/Reduced-KV 本身不是 CASA-CD 的 novelty。

而 Hybrid-M 参数量约 10.5M，明显不满足总模型 ≤5M。

---

## 22.3 EfficientViM / VMamba / MambaVision

问题不是它们“不先进”。

问题是：

> CASAA 的核心是 Q/K/V asymmetric token attention。

若主 backbone 主要是 SSM：

- 要么 CASAA 只能作为外挂；
- 要么论文会变成 Mamba + Transformer + Rep decoder 三种机制堆叠；
- 第一创新的因果链反而不干净。

此外 EfficientViM-M1 已 6.7M，MambaVision 更大。

因此不推荐。

---

## 22.4 RepViT / StarNet

都非常适合做轻量 backbone。

但它们没有自然的 Q/K/V token self-attention 主干。

为了 CASAA 再额外加 Transformer，论文就会被问：

> 为什么不是直接用已有 backbone，而是为了 CASAA 强行插 attention？

所以不如 SHViT 自然。

---

# 23. 与关键近邻工作的实质区别

## 23.1 vs SAT

SAT：

```text
single-image SR
content clustering
Full Q
compressed K/V
```

CASAA：

```text
bi-temporal CD
change-conditioned selection
change candidate direct retention
stable background aggregation
Full Q
compressed K/V
```

CASAA 不应只说“借鉴 SAT”。

真正区别是：

> **K/V bank 的构造目标从“内容代表性”变成“变化判别需要的上下文分配”。**

---

## 23.2 vs MobileNetV4 SRA

MobileNetV4：

```text
spatially downsample K/V
```

属于规则性空间压缩。

CASAA：

```text
change token 不被均匀池化
background 才被强聚合
```

所以压缩预算由**变化语义**决定，而不是空间下采样规则。

---

## 23.3 vs Token Cropr

Token Cropr：

```text
直接移除 token
```

会减少后续 Query positions。

CASAA：

```text
Query positions 从 N 到输出一直保持 N
```

只减少提供 context 的 K/V。

对于逐像素变化检测，这一差异很重要。

---

## 23.4 vs LWGANet TGFI

LWGANet：

> 从单幅遥感图像的空间冗余出发，强调 Top-K salient feature interaction。

CASAA：

> 从双时相变化稀疏性出发，区分 change candidate 与 stable background；并且对 Query 与 K/V 采用不同策略。

不能把两者混成“都是 top-k”。

---

# 24. 与 ChangeViT 的实质区别

ChangeViT：

```text
DeiT plain ViT semantic
+
ResNet18 detail
+
Feature Injector
+
decoder
```

新 CASA-STR：

```text
single pretrained lightweight hierarchical hybrid encoder
+
CASAA inside encoder
+
TAR
+
DCR
```

所以它不再是：

> “给 ChangeViT 换一个更小 detail branch。”

而是：

> **重新设计适合极轻量 BCD 的 encoder–bridge–decoder 分工。**

这样论文独立性更强。

---

# 25. 与 STR-RepNet 的实质区别

STR-RepNet 原路线：

```text
VMamba-Tiny
+
TAR
+
DCR
```

其 deploy 参数约 29M 量级，远超你的最终预算。

新方法：

```text
~2M-class pretrained hybrid encoder
+
CASAA
+
TAR
+
DCR
```

STR 贡献被保留，但 backbone 与核心 token 建模机制完全不同。

此外：

> 新论文不能把 STR-RepNet 的所有历史 Run3–Run10 机制都搬来。

只拿最干净的：

```text
TAR
DCR
BN-FR / fold primitives
```

否则又会模块堆叠。

---

# 26. 当前 STR 结果应该怎样解释

上传快照中的 `full_last2`：

| Dataset | F1 | IoU |
|---|---:|---:|
| CDD | 98.42 | 96.89 |
| LEVIR | 91.44 | 84.24 |
| SYSU | 83.45 | 71.60 |
| WHU | 95.14 | 90.74 |

说明：

- CDD 已过 98；
- WHU 已过 95；
- SYSU 距 85 约 1.55pp；
- LEVIR 距 92.5 约 1.06pp。

因此 STR 并不是“完全失败”。

更准确的判断是：

> STR 的 bridge/decoder 已经能够达到接近目标的性能，但原 VMamba-Tiny 整体参数约 29.57M，不符合轻量目标；而且 LEVIR/SYSU 仍缺表征能力。

这正好支持：

> 把 STR 的 bridge/decoder 留下，换一个更适合 CASAA 且极轻量的 pretrained encoder。

---

# 27. 论文故事建议

最终论文不要写成：

> 我们把几个轻量模块拼起来。

建议写成一条统一逻辑：

### 问题一：高分辨率 BCD 的上下文计算不均匀

变化区域稀疏，而逐像素 Query 不能丢。

因此：

> **CASAA 把“判别位置”和“上下文提供者”分开处理：完整 Query，选择性 K/V。**

### 问题二：轻量模型训练期表达力与推理期复杂度矛盾

简单把 decoder 变小容易损伤跨时相/跨尺度建模。

因此：

> **STR 在训练期显式展开 temporal/scale basis，推理时解析折叠为单路径。**

最终统一成：

> **非对称计算负责“哪里不能省、哪里可以聚合”；结构重参数化负责“训练时怎样丰富建模、推理时怎样不增加成本”。**

这两点是可以放在同一篇论文里的。

---

# 28. 可证伪假设

## H-CASAA

> 在保持全部 Query 的条件下，变化感知 K/V bank 比 content-only compressed K/V 更适合 BCD。

反证：

```text
A3 <= A2
```

跨四数据集不成立。

---

## H-Compression

> K/V 25% context 足以维持 full attention 所需的上下文。

反证：

```text
A2 相对 A1 在多个数据集明显下降
```

---

## H-STR

> 训练期 temporal/scale reparameterization 在相同 deploy function class 下提供稳定优化收益。

反证：

```text
M1 <= A3
```

或只表现为 P↑R↓。

---

## H-System

> 两个机制组合能在 ≤5M 达到四数据集硬目标。

反证：

任一绝对 F1 门槛不过。

---

# 29. 最重要的 P0/P1/P2 风险

## P0：正确性

1. SHViT retained weights 没有真正加载；
2. CASAA 输出 Query 数变了；
3. A/B routing 不对称；
4. background aggregation 把两个时相 feature 直接平均成同一个 value；
5. TAR/DCR fold 后模型输出不等价；
6. `arch.json` 不严格检查导致 eval 跑错结构；
7. label 阈值不是 gray≥128；
8. A/B/label 几何增强不同步。

这些任何一个都比 F1 更优先。

---

## P1：方法瓶颈

1. SHViT 截断 stage2 的语义能力可能不足；
2. 1/8 cosine score 仍可能不能逼近 Oracle；
3. STR 在新 backbone 上仍可能只做 confidence sharpening；
4. 1/4 early feature 可能太浅。

这些只能由四数据集正式实验回答。

---

## P2：实验工程

1. 20 个 80K 的双卡排程；
2. 崩溃恢复；
3. 日志完整性；
4. checkpoint 路径隔离；
5. 训练后自动提取最后 TEST block。

这些不能改变方法结论。

---

# 30. 如果首选路线失败，备用方向

只允许一个 fallback：

> **LWGANet-L0 pretrained backbone + 替换/关闭与 CASAA 冲突的 TGFI，再接 CASAA + TAR/DCR。**

但必须重新预注册。

不建议失败后立刻做：

- 新 loss；
- EdgeGate；
- 多个 attention；
- teacher distillation；
- threshold tuning；
- width sweep；
- 多 seed 救单一结果；
- TASS + CASAA + STR 三主线堆叠。

---

# 31. 立即执行顺序

## Step 1：停止继续扩 TASS

当前正在跑的任务可以让已经启动的四数据集队列完成，作为历史证据，不需要删除。

但：

> 不再为 TASS 设计 Run12/Run13。

---

## Step 2：下载并登记 SHViT-S1 官方 ImageNet 权重

放：

```text
/home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth
```

记录 SHA256。

---

## Step 3：实现 truncated SHViT encoder

先不加 CASAA。

验证：

```text
1/4,1/8,1/16,1/32
exact pretrained load
params
FLOPs
```

---

## Step 4：实现 generic CASAA@1/16

只复用当前已验证机制：

```text
Full Q
K=64
Kc=32
Kb=32
deterministic background clustering
1/8 local change score
zero-init residual
```

---

## Step 5：接现有 TAR/DCR

固定：

```text
encoder_dims=(32,64,128,224)
D=160
```

---

## Step 6：完成 P0 smoke

全过后再正式训练。

---

## Step 7：A0 四数据集

得到新 backbone + plain deploy topology 的强度锚点。

---

## Step 8：M1 四数据集

这是最重要的一组。

只有四个结果全部回来后再裁决。

---

## Step 9：若 M1 有最终价值，再补完整消融

```text
A3
A2
A1
```

每组四数据集。

---

# 32. 仍需补充的证据

1. 腾讯文档完整导出正文；
2. SHViT-S1 official checkpoint 在 RSML-3 的 SHA256；
3. truncated SHViT 机器参数量/FLOPs；
4. CASAA@C128 的真实 FLOPs（包含 routing，不只算 attention matrix）；
5. CASA-STR deploy effective params；
6. full/plain deploy graph equality；
7. A0/M1 四数据集最终 TEST RESULTS；
8. 若要正式写“SOTA”，还需要统一评估协议下的 2024–2026 轻量 SOTA 对照表，不能混用不同 crop、dataset split、训练预算。

---

# 33. 文献调研：本轮建议保留的核心引用

下面只列本路线真正需要的文献，不把所有“轻量网络”都堆进 Related Work。

## 33.1 CASAA / efficient token modeling

### SAT
**SAT: Selective Aggregation Transformer for Image Super-Resolution**  
arXiv:2604.07994  
官方代码：  
https://github.com/PhuTran1005/SAT

用途：

> Full Query + compressed K/V 的直接机制来源。

状态写作需谨慎：当前项目材料/官方仓库标注“CVPR 2026 Findings”，正式投稿时应以可核验 proceedings 状态为准，不写成 CVPR main-track。

---

### MobileNetV4
**MobileNetV4: Universal Models for the Mobile Ecosystem**  
ECCV 2024，CCF-B  
官方论文：  
https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5647_ECCV_2024_paper.php

关键近邻：

> Mobile MQA + Spatial Reduction Attention 使用高分辨率 Query 与下采样 K/V。

它直接告诉我们：

> Full-Q/low-resolution-KV 不是 CASAA 的独立 novelty。

---

### Token Cropr
**Token Cropr: Faster ViTs for Quite a Few Tasks**  
CVPR 2025，CCF-A  
https://openaccess.thecvf.com/content/CVPR2025/html/Bergner_Token_Cropr_Faster_ViTs_for_Quite_a_Few_Tasks_CVPR_2025_paper.html  
官方代码：  
https://github.com/benbergner/cropr

用途：

> 与“直接 prune Query token”的路线做对照。

---

### LaViT
**You Only Need Less Attention at Each Stage in Vision Transformers**  
CVPR 2024，CCF-A  
https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_You_Only_Need_Less_Attention_at_Each_Stage_in_Vision_CVPR_2024_paper.html

用途：

> attention redundancy / saturation 的外部依据。

---

## 33.2 Backbone

### SHViT
**SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design**  
CVPR 2024，CCF-A  
论文：  
https://openaccess.thecvf.com/content/CVPR2024/html/Yun_SHViT_Single-Head_Vision_Transformer_with_Memory_Efficient_Macro_Design_CVPR_2024_paper.html  
官方代码：  
https://github.com/ysj9909/SHViT

官方模型：

```text
S1: 6.3M / 241M / ImageNet-1K Top1 72.8
```

本文使用其官方 pretrained checkpoint 的 retained subset。

---

### LWGANet
**LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention**  
AAAI 2026 Oral，CCF-A  
官方代码：  
https://github.com/AeroVILab-AHU/LWGANet

用途：

- RS 专用 lightweight backbone 强近邻；
- 空间冗余 / Top-K prior art；
- 备选 backbone。

---

### EfficientViM
**EfficientViM: Efficient Vision Mamba with Hidden State Mixer-based State Space Duality**  
CVPR 2025，CCF-A  
论文：  
https://openaccess.thecvf.com/content/CVPR2025/html/Lee_EfficientViM_Efficient_Vision_Mamba_with_Hidden_State_Mixer_based_State_CVPR_2025_paper.html  
官方代码：  
https://github.com/mlvlab/EfficientViM

M1：

```text
6.7M / 239M
```

由于本体已超过 5M 且不是 QKV attention 主体，不作为主 backbone。

---

### RepViT
**RepViT: Revisiting Mobile CNN From ViT Perspective**  
CVPR 2024，CCF-A  
https://github.com/THU-MIG/RepViT

M0.9：

```text
5.1M / 0.8G
```

单 backbone 已接近总参数上限，且无自然 CASAA QKV 接口。

---

### MambaOut
**MambaOut: Do We Really Need Mamba for Vision?**  
CVPR 2025，CCF-A  
官方代码：  
https://github.com/yuweihao/MambaOut

Femto：

```text
7.3M / 1.2G
```

超预算，不选。

---

## 33.3 Dense prediction / local detail

### ViT-CoMer
**ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions**  
CVPR 2024，CCF-A  
https://openaccess.thecvf.com/content/CVPR2024/html/Xia_ViT-CoMer_Vision_Transformer_with_Convolutional_Multi-scale_Feature_Interaction_for_Dense_CVPR_2024_paper.html

用途：

> plain ViT inner-patch interaction 与 feature-scale diversity 问题。

---

### EoMT
**Your ViT is Secretly an Image Segmentation Model**  
CVPR 2025，CCF-A

用途：

> plain ViT 有 dense prediction 潜力，但大模型/预训练规模与本课题轻量约束不同。

---

## 33.4 结构重参数化

2024–2026 主证据：

- **UniRepLKNet**, CVPR 2024，CCF-A
- **RepAn**, CVPR 2024，CCF-A
- **RepViT**, CVPR 2024，CCF-A
- **ASR**, ECCV 2024，CCF-B
- **CD-RLKNet**, IJAEO 2024，SCI
- **MixCDNet**, IEEE TGRS 2024，SCI
- **RFANet**, ISPRS JPRS 2024，SCI
- **ChangeMamba**, IEEE TGRS 2024，SCI
- **M2M-LINet**, IEEE TGRS 2025，SCI
- **DMFANet**, IEEE TGRS 2025，SCI
- **ST-Mamba**, IEEE TGRS 2025，SCI
- **DEIF-Mamba**, IEEE TGRS 2025，SCI

经典 RepVGG / DBB / OREPA 仅作为历史数学锚点，不算本轮 2024–2026 主调研。

---

# 34. 最终建议

现在最重要的不是继续在 TASS 上做下一轮小修补，而是**恢复论文原始的两个创新中心**。

我建议正式冻结为：

> **Innovation 1 — CASAA**  
> 在轻量层次化 hybrid encoder 的 1/16 token stage 中，用完整 Query + 变化感知 compressed K/V 进行非对称上下文建模。

> **Innovation 2 — STR**  
> 在四尺度 Siamese skip 与 decoder 中使用可解析折叠的 Temporal Algebraic Reparameterization + Decoder Compositional Reparameterization，训练多分支、部署单路径。

骨干首选：

> **ImageNet-pretrained SHViT-S1 truncated (through stage2)**

最终主模型：

```text
SHViT-S1 truncated
+ CASAA@1/16
+ TAR(full)
+ DCR(full)
```

预计参数：

```text
~2.5M 级
```

但正式数值以 RSML-3 机器审计为准。

实验纪律：

> **每个正式变体必须 CDD / LEVIR / SYSU / WHU 四数据集完整 80K；消融也一样；不再用单数据集 gate 决定方法有效性。**

最终只有当：

```text
SYSU >= 85
LEVIR >= 92.5
WHU >= 95
CDD >= 98
deploy effective params <= 5M
```

全部同时满足，并且 CASAA/STR 的受控消融在四数据集上有可解释的稳定贡献，才能把这条线作为最终论文主方法。

---

# 35. 建议下一步交付物

下一步不要直接启动训练。

先完成一个独立实现批次：

```text
1. shvit_trunc.py
2. casaa_hier.py
3. casa_str_net.py
4. train.py / eval.py 接入
5. audit_casa_str.py
6. smoke tests
7. 5 groups × 4 datasets 的脚本
```

完成后只做：

```text
py_compile
pretrain audit
shape
gradient
fold
budget
60-step dry run
```

全部通过后，再进入正式四数据集 80K。

这比继续沿 TASS Run12 迭代，更符合你最初想做的论文，也更有机会形成：

> **“变化感知的非对称 token 建模 + 推理零额外分支的时空结构重参数化”**

这样一条清晰、可投稿、可做严格消融的主线。
