# CASA-CD 下一步方案：Run11 · TASS（Task-Adaptive Spatial Stem）设计与预注册方案

> **日期**：2026-10-02  
> **适用仓库**：`YuqiWang-code/CASA-CD`（以当前 `main` 为准）  
> **任务**：CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256，全监督二值变化检测  
> **硬目标**：SYSU F1 ≥ 85 / LEVIR ≥ 92.5 / WHU ≥ 95 / CDD ≥ 98；**有效推理参数 ≤ 5M**  
> **固定协议**：BCE+Dice；Adam(lr=2e-4, betas=0.9/0.99, wd=1e-4)；poly 0.9 + 200 iter warmup；80000 steps；batch 16；256×256；seed 16；test-as-val；GPU1 单卡  
> **结果纪律**：正式结果只读对应 `train_log.txt` **最后一个完整 `=== TEST RESULTS ===` → `=== END TEST RESULTS ===` 区块**。任何 best epoch 行、checkpoint 文件名、零训练 gate、dry run 都不能替代正式 TEST RESULTS。  
> **从头训练纪律**：所有正式 C0/M1 都必须从同一 `deit_tiny_patch16_224-a1311bcf.pth` + seed 16 构建后完整 80K；**禁止加载任何已有 CD checkpoint 作为初始化或续训**。  
> **本方案状态**：预注册稿。任何阈值一旦开始执行，不允许根据结果事后修改。

---

## 0. 结论先行

**下一步选 (a)，不选 (b)，暂不进入 (c)。**

但我不建议把 (a) 做成“再换一个轻量 CNN detail branch”。Run4/Run5 已经足够说明，单纯换 LightDetail / MobileNet prefix 很容易重复失败。下一轮只值得做一个更严格、可证伪的结构假设：

> **TASS：Task-Adaptive Spatial Stem（任务适配空间 Stem）**  
> 保留已经验证健康的 **Frozen ViT4** 作为语义锚点；保留已经机器验证可折叠的 **TAR+DCR** 作为统一时空建模/解码骨架；新增一条极小、共享权重的**可训练局部空间 stem**，在 `1/4、1/8、1/16` 三个尺度产生真正来自像素网格的 task-adaptive spatial residual，并以 **zero-init residual injection** 的方式加到原有 B1/B2/B3 token 金字塔上。  
> 它不是替代 token，也不是从 token“重建”细节，而是承认前八轮证据：**sub-patch evidence 必须从像素域重新产生，并且必须经过任务训练。**

核心数据流：

```text
A/B ─────────────── Frozen DeiT-Tiny ViT4 ── B1/B2/B3/B4
 │                                             │
 │                                             ├─ B1 ↑4 ───────────────┐
 │                                             ├─ B2 ↑2 ───────────────┤
 │                                             ├─ B3 native ───────────┤
 │                                             └─ B4 ↓2 ───────────────┤
 │                                                                        │
 └─ shared TASS stem ─ S1(1/4)/S2(1/8)/S3(1/16) ─ 1×1→192 ─ α·residual ┘
                                               α1=α2=α3=0 at init
                                                        │
                                    [t1,t2,t3,t4] per time
                                                        │
                                      TAR(full rep) → DCR(full rep)
                                                        │
                                               1×1 head → 256×256
```

其中：

\[
t_i^t = T_i^t + \alpha_i\,P_i(S_i^t),\qquad i\in\{1,2,3\},\ t\in\{A,B\}
\]

\[
t_4^t = \operatorname{AvgPool}_{2\times2}(B4^t)
\]

`T1/T2/T3` 分别是原 STRFusion 的 `B1↑4 / B2↑2 / B3`；`P_i` 把 stem 通道投影到 192；`α_i` 为可学习标量，**初始化为 0**。因此 M1 在 epoch-0、eval 模式下应与 C0 **逐位一致**，新分支不会靠随机初始化先污染已知健康路径。

### 为什么现在值得花这一次 80K 预算

当前 deploy STRFusion 已经只有 **2.596M / 2.2136G**；预算从 `<3M` 放宽到 `≤5M` 后，多出的约 **2.404M 参数额度不应再花在 token 重采样、router 或更复杂 decoder 上**，因为这些方向已经被八轮证据系统否决。最合理的预算用途只有一个：**任务适配的高分辨率空间表征源**。

本方案的 TASS 预计只增加约 **0.858M deploy 参数**，总有效推理参数约 **3.454M**；双时相 stem + 三个投影预计使总 FLOPs 从 2.2136G 增至约 **3.53G**。仍显著低于 ChangeViT-T 复现的 26.32G，同时保留约 **1.55M 参数余量**。这些是**解析估算，不是机器事实**；正式执行前必须由 `fvcore + effective-param audit` 重新测量，超过 5M 直接 STOP，不通过“缩宽度临时救场”。

---

# 1. 决策：为什么选 (a)，而不是 (b)/(c)

| 候选 | 与现有证据是否匹配 | 预算 | 主要风险 | 决策 |
|---|---|---:|---|---|
| **(a) tiny 可训练 spatial stem** | **最匹配**。R8/R9/STRFusion 的共同结论是 token ranking 有效，但 16×16 token 网格无法提供 sub-patch dense evidence；需要“训练过的、任务适配的空间表征” | 可控制在约 3.45M deploy | 可能重复 Run4 的轻量 detail 失败；必须在结构上正面修复 Run4 诊断 | **选** |
| (b) 解冻 ViT block0-1，低 lr | 只能改变 token 内容，**不能增加 patch16 后的空间位置数**；SF-D0 已表明多深度 token 仍受 inner-patch 限制。且项目已有 ViT 在统一 2e-4 下进入精确零权重吸收态的直接事实 | 参数不增 | 训练稳定性风险最高；即使健康也未解决空间采样格本身 | **不选，除非 (a) 被证伪后另开预注册** |
| (c) 分析型论文收尾 | 科学上完全可行，八轮负结果链已具备较强分析价值 | 无新增模型预算 | 当前硬目标和“≤5M 可用预算”尚有一个非常明确、未被直接证伪的空间：task-adaptive spatial evidence | **作为 (a) 的唯一失败后 fallback** |

## 1.1 为什么 (b) 不是当前优先级

受限解冻 block0-1 能让浅层 token 更贴合 CD，但它仍然工作在 `16×16` token lattice 上。对于 256×256 输入，一个 token 对应 16×16 像素区域。解冻 block0-1 可以改变“每个 patch token 表示什么”，但不会直接产生 `64×64/32×32` 的真实空间自由度。

这与当前最强负证据完全同向：

- R8：B4 token ranking 优势不能转化为 dense boundary reconstruction；
- R9：把 patch sampling lattice 变密但仍不形成任务训练过的空间表征，G0/G1 全失败；
- STRFusion SF-D0：B1–B4 多深度自由融合在边界带上仍是 0/4 通过；
- 因而当前瓶颈不是“ViT 再多适配一点”，而是**缺一个从像素域产生的、可学习的局部空间自由度**。

此外，项目已经直接观察到统一 `lr=2e-4` 训练 ViT 会进入精确零权重吸收态。即使给 block0-1 单独低 lr，也必须另设 2K-step health gate，并且仍不解决 lattice 问题。它应是后手，不应先于 (a)。

## 1.2 为什么暂不直接选 (c)

如果参数硬约束仍是 `<3M`，我会更倾向直接收尾；但现在上限已放宽到 `≤5M`。当前 2.596M deploy 模型留下约 2.404M 可支配额度，而前八轮已经很明确地告诉我们“这笔预算应该花在哪”：**不是再做 token-side trick，而是补一个 task-adaptive local spatial source**。

因此，Run11 应被定义为**最后一次机制性冲击**：

- 不 sweep width；
- 不换 loss；
- 不改增强；
- 不做 threshold tuning；
- 不补第二个 stem；
- 不再救 router；
- 不在失败后接 adapter / edge head / attention；
- gate 失败或 80K 判据失败，直接转 (c)。

---

# 2. 证据表：哪些是事实，哪些仍是假设

| ID | 类型 | 证据 | 对 Run11 的约束 |
|---|---|---|---|
| E1 | **代码/机器事实** | STRFusion deploy `2.596353M / 2.2136G`；T0/T1/T2/T2b 折叠全过；活分支整网误差约 `5e-7`，二值 disagreement=0 | TAR/DCR 不再改机制；只作为固定 scaffold |
| E2 | **代码/机器事实** | SF-D0：fuse4 相对 B4 边界 PR-AUC lift：CDD +0.0100 / SYSU +0.0089 / LEVIR −0.0061 / WHU −0.0090；G1=0/4 | 禁止继续在 frozen token depth/fusion 上救 |
| E3 | **代码/机器事实** | Run8：B4 token ranking 优势到真实边界 ±4px 基本消失 | 新结构必须引入 sub-patch spatial DOF，而不是仅上采样 token |
| E4 | **代码/机器事实** | Run4-D0：LightDetail 1/2 浅层 raw 不差，但 1/4、1/8 深层 PR 崩到约 0.32；复盘指向“随机初始化 + 无 residual + DW 先降采样后混合” | 新 stem 的下采样必须先做 dense 3×3 mixing；stage 内必须有 residual local mixing |
| E5 | **代码/机器事实** | Run5 MobileNetV3 prefix 4 项 gate 过 3 项，唯 1/4 PR=0.4831<0.50，按预注册停止 | “预训练轻量 CNN prefix”不是答案；Run11 不再引入第二套 ImageNet 预训练 |
| E6 | **代码/机器事实** | R4-0 healthy full12 frozen SYSU 83.14；R4-1 ViT4+ResNet+旧头 82.77；R4-1 是当前最强已验证轻量结构，但 8.195M 超新硬预算 | 4-block ViT 的语义能力够用，预算应转给空间分支 |
| E7 | **论文事实** | ChangeViT 明确指出 plain ViT 对 fine-grained/local changes 较弱，并用 1/2、1/4、1/8 CNN detail features 补足；其 detail-capture 约 2.7M | “局部空间支路”有直接任务依据，但 Run11 必须比原 ResNet detail 更轻、更干净 |
| E8 | **论文事实** | ViT-CoMer (CVPR 2024) 将 plain ViT dense prediction 的关键问题归因于 inner-patch interaction 不足与 feature-scale diversity 不足，并引入 CNN multi-scale spatial features | 与项目 R8/SF-D0 的独立外部证据一致 |
| E9 | **证据支持推断** | 现在最值得投入的新增参数，不是扩大 ViT/decoder，而是训练一个真正从像素网格出发的 spatial residual source | Run11 的 budget-allocation 核心 |
| H11 | **待验证假设** | 在 frozen ViT4 + 固定 TAR/DCR 下，以 zero-init 方式加入 task-adaptive 1/4–1/16 spatial residual，可把 token 语义锚点与 sub-patch spatial evidence 互补起来，并显著提升 dense boundary + 最终 F1 | **本轮唯一主假设** |
| M1 | **缺失信息** | TASS 在四数据集的 raw source complementarity 尚未测；参数/FLOPs 尚未由仓库代码机器审计 | 先跑 TASS-D0 + budget gate，过后才允许正式 80K |

---

# 3. 创新主张与实质区别

## 3.1 若实验通过，允许主张的创新

建议内部名称：

**TASS — Task-Adaptive Spatial Stem**  
中文：**任务适配空间残差 Stem**

论文层面的主张应写成：

> **Evidence-allocated hybrid encoder under a strict lightweight budget**：  
> 冻结浅层 plain ViT 只承担粗粒度全局语义；新增 tiny trainable spatial stem 只承担 token lattice 无法表达的局部空间残差；二者不做复杂双向交互，而是在固定的多尺度接口通过 **zero-init residual injection** 合并，随后统一进入可折叠的 TAR/DCR。参数预算不是均匀分配，而是由八轮负证据明确地转向“task-adaptive dense evidence”。

这里真正可写成创新的不是：

- “用了卷积”；
- “用了 RepViT block”；
- “加了 skip connection”；
- “参数更少”；

而是这三个连成一个机制闭环：

1. **Scale-separated responsibility**：ViT4 保语义，TASS 只补 `1/4–1/16` 局部空间自由度；
2. **Residual—not replacement**：空间分支只学习 token 路径缺失的 correction，`α=0` 使 M1 epoch-0 与 C0 完全等价；
3. **Budget reallocation by falsified evidence**：新增推理预算只投到已被实验定位为缺口的 spatial source，TAR/DCR/ViT 不继续加容量。

## 3.2 与 ChangeViT 的实质区别

ChangeViT 的结构是：

```text
plain ViT semantic
+ ResNet18 C2-C4 detail branch
+ cross-attention Feature Injector
+ cascade decoder
```

本方案是：

```text
Frozen shallow ViT4 token pyramid (semantic anchor)
+ tiny task-trained spatial residual stem
+ zero-init per-scale residual injection
+ TAR/DCR temporal + decoder reparameterization
```

差异不在“CNN + ViT”这个表面组合，而在：

- ChangeViT 的 detail branch 是独立多尺度 feature producer，再用 cross-attention 把 detail 注入 ViT；
- TASS **不修改 ViT token 内部表示、不使用 Feature Injector、不做三路 cross-attention**；
- TASS 只在 decoder interface 处提供 residual spatial evidence；
- TASS 的 residual 初始严格为 0，因此可建立 C0/M1 epoch-0 exact-equivalence；
- 目标是在约 3.45M deploy 参数下保留细节，而不是复刻 11.68M ChangeViT-T 的 detail+FI。

## 3.3 与 ViT-CoMer 的实质区别

ViT-CoMer 的主要思想是 CNN 和 ViT **多阶段双向交互**，CNN spatial pyramid 不断增强 ViT backbone。

TASS 不做双向融合：

- ViT4 全程冻结；
- stem 不反馈到 ViT blocks；
- 只在 decoder 输入接口做一次多尺度 residual merge；
- 训练预算集中在空间 correction + CD decoder。

因此如果实验成立，可把它表述为一种**任务预算下的单向 spatial correction**，而不是通用 dense ViT backbone 重设计。

## 3.4 与 Run4/Run5 的实质区别

这是 Run11 是否值得做的关键。

Run4/Run5 已经排除了“随便换个小 CNN”的路线，因此 TASS 必须满足以下不同点：

- **Run4 LightDetail**：随机轻量 detail 直接承担 detail source，深层 1/4、1/8 raw representation 崩；TASS 不是替换，而是 residual correction；
- **Run4 诊断指出 DW-first downsampling 是瓶颈之一**：TASS 每级下采样先用普通 `3×3 stride=2` 做跨通道/空间 mixing，再做轻量局部 block；
- **Run4 无 residual**：TASS 的每个 LocalMixBlock 都是 residual；
- **Run5 Mobile prefix**：依赖第二套 ImageNet 预训练且 gate 仍失败；TASS 不引入第二个预训练 checkpoint，全部新参数在 CD 任务内学习；
- **旧 head/FI**：Run4/5 的 detail 必须经过 ChangeViT legacy injector/head；TASS 直接进入已经验证可折叠、低开销的 TAR/DCR scaffold。

如果不坚持这些差异，Run11 就没有必要跑。

---

# 4. TASS 逐层结构定义

## 4.1 Frozen semantic path（不改）

输入 `x ∈ R^{3×256×256}`：

```text
DeiT-Tiny PatchEmbed, patch=16
→ 16×16 × 192
→ block0
→ block1
→ block2
→ block3
```

保持 corrected DeiT loader：

- `patch_embed` 从 `deit_tiny_patch16_224-a1311bcf.pth` 原位加载；
- pos embed `14×14 → 16×16` bicubic；
- blocks 4–11 不构造；
- ViT4 全参数 `requires_grad=False`；
- 每次训练前后 checksum 必须逐位一致。

C0 继续使用：

```text
T1 = B1 bilinear ×4 → 64×64×192
T2 = B2 bilinear ×2 → 32×32×192
T3 = B3              → 16×16×192
T4 = AvgPool(B4,2)   →  8×8×192
```

## 4.2 TASS local stem（M1 唯一新增结构）

共享 Siamese 权重，同一个 stem 分别处理 A/B。

### Stem0

```text
Conv3×3, stride=2, 3→32, bias=False
BN
SiLU
# 128×128×32
```

`128×128` 只作为内部状态，不直接送 decoder，避免把 DCR 最细尺度升到 128×128 导致 FLOPs 失控。

### Stage-1：输出 1/4 spatial feature

```text
Conv3×3, stride=2, 32→64, bias=False
BN + SiLU
LocalMixBlock(64) × 2
→ S1: 64×64×64
```

### Stage-2：输出 1/8 spatial feature

```text
Conv3×3, stride=2, 64→128, bias=False
BN + SiLU
LocalMixBlock(128) × 2
→ S2: 32×32×128
```

### Stage-3：输出 1/16 spatial feature

```text
Conv3×3, stride=2, 128→192, bias=False
BN + SiLU
LocalMixBlock(192) × 2
→ S3: 16×16×192
```

### LocalMixBlock(C)

采用 RepViT 轻量设计原则，但**不把 block 本身包装成创新**：

```text
u = x + BN(DWConv3×3(x))
y = u + BN(PW2(SiLU(PW1(u))))
```

其中：

```text
PW1: C → 2C
PW2: 2C → C
```

关键不是名字，而是两条约束：

1. 下采样由 **dense 3×3 stride-2** 完成，不能再次使用 Run4 已暴露问题的“DW 先降采样、再做 mixing”；
2. stage 内局部建模是 residual，避免 Run4 “无 residual 深层 representation 快速塌缩”的已知失败模式。

## 4.3 192-channel residual projectors

```text
P1: 1×1 Conv 64→192 + BN
P2: 1×1 Conv 128→192 + BN
P3: 1×1 Conv 192→192 + BN
```

每个尺度一个可学习标量：

```text
alpha1 = alpha2 = alpha3 = 0.0
```

M1：

```text
t1 = B1↑4 + alpha1 * P1(S1)
t2 = B2↑2 + alpha2 * P2(S2)
t3 = B3   + alpha3 * P3(S3)
t4 = B4↓2
```

C0：

```text
t1 = B1↑4
t2 = B2↑2
t3 = B3
t4 = B4↓2
```

### 为什么用 scalar alpha，而不是 attention/gate network

因为当前需要验证的是“**task-adaptive spatial residual 是否有用**”，而不是再引入一个 gating 机制。

`alpha_i` 的作用只有两个：

- epoch-0 精确退化为 C0；
- 给每个尺度一个最小的学习幅度自由度。

不做：

- spatial attention；
- channel attention；
- dynamic routing；
- edge supervision；
- learned threshold；
- extra loss。

## 4.4 TAR / DCR / head（固定，不再改）

两组都固定：

```text
rep_mode = full
dim = 160
encoder_dims = (192,192,192,192)
```

即：

```text
[t1_A,t2_A,t3_A,t4_A], [t1_B,t2_B,t3_B,t4_B]
→ MultiScaleTAR(full)
→ DCRDecoder(full)
→ Conv1×1(160→1)
→ bilinear 256×256
→ sigmoid
```

TAR/DCR 的 C0/M1 分支结构和参数形状完全相同，避免把“spatial stem”和“rep_mode”两个变量混在一起。

---

# 5. 训练图 / 推理图 / 预训练兼容

## 5.1 C0 训练图

```text
ImageNet DeiT-Tiny pth
→ Frozen ViT4
→ B1/B2/B3/B4 token pyramid
→ TAR(full, train graph)
→ DCR(full, train graph)
→ head
```

- 不加载任何 CD checkpoint；
- TAR/DCR 从固定 seed 重新初始化；
- 完整 80K。

## 5.2 M1 训练图

```text
ImageNet DeiT-Tiny pth
→ Frozen ViT4 ─────────────────────────────┐
                                          ├→ token + alpha*TASS residual
A/B → shared trainable TASS stem ─────────┘
→ TAR(full, train graph)
→ DCR(full, train graph)
→ head
```

- ViT4 与 C0 完全相同；
- TASS 随机初始化，**不加载 MobileNet/RepViT/ResNet 第二套预训练**；
- `alpha=0`；
- 完整 80K。

## 5.3 推理图

正式 TEST 前：

```text
TAR.switch_to_deploy()
DCR.switch_to_deploy()
```

TASS 是单路径轻量卷积 stem；BN 可在导出审计中折到相邻 conv，但这只是部署等价变换，不作为创新。

最终 M1 deploy：

```text
Frozen ViT4
+ TASS stem
+ 3× projector + alpha
+ folded TAR
+ folded DCR
+ head
```

不含：

- teacher；
- cache；
- training-only auxiliary output；
- edge head；
- extra loss branch。

## 5.4 预训练兼容验证

必须打印：

```text
[VIT-PRETRAIN-AUDIT]
patch_embed exact loaded: True
pos_embed resized: 14x14 -> 16x16
blocks present: 0..3
blocks absent: 4..11
checksum before train == checksum after train: True
```

C0/M1 都只能使用：

`/home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth`

---

# 6. 参数量 / FLOPs 预算

以下为**按当前代码和所定义 TASS 的解析估算**；机器审计结果优先。

## 6.1 已有机器事实

| 模块/模型 | Deploy effective params | FLOPs |
|---|---:|---:|
| 当前 STRFusion | **2.596353M** | **2.2136G** |
| ChangeViT-T baseline | 11.754M effective | 26.32G |

当前 STRFusion 中 frozen ViT4 约 1.977M，因此 deploy 的 TAR+DCR+head 约 0.619M。

## 6.2 TASS 解析参数量

固定宽度：

```text
Stem0 = 32
Stage1 = 64
Stage2 = 128
Stage3 = 192
blocks = (2,2,2)
expansion = 2
project_to = 192
```

估算：

| 部分 | Deploy params（约） |
|---|---:|
| Stem0 + 3×DownConv + 6×LocalMixBlock | 0.783M |
| P1/P2/P3 + folded bias | 0.074M |
| α1/α2/α3 | <0.001M |
| **TASS 新增合计** | **≈0.858M** |

因此：

\[
2.596353M + 0.858M \approx \mathbf{3.454M}
\]

预计 headroom：

\[
5.000M - 3.454M \approx \mathbf{1.546M}
\]

**这 1.55M 不是允许继续堆模块的邀请。Run11 width 固定，不做容量 sweep。**

## 6.3 FLOPs 解析估算

按 fvcore 常见 MAC 口径：

- TASS stem 双时相约 **1.147G**；
- 3 个 192-channel projector 双时相约 **0.170G**；
- 原 STRFusion 2.2136G；

预计：

\[
2.2136 + 1.147 + 0.170 \approx \mathbf{3.53G}
\]

因此 Run11 预期复杂度：

| 模型 | Effective Params | FLOPs | 备注 |
|---|---:|---:|---|
| C0 token-only STRFusion | 2.596M | 2.214G | 机器已测 |
| M1 TASS | **≈3.454M** | **≈3.53G** | 待机器审计 |
| 硬门槛 | **≤5.000M** | 必须报告 | Params 超限直接停止 |

## 6.4 训练图参数

当前 M1 full-rep STRFusion 训练图机器记录：

- total ≈ 3.123M；
- trainable ≈ 1.146M。

TASS 训练参数约 0.860M，因此预计：

- training graph total ≈ 3.98M；
- trainable ≈ 2.01M；
- deploy effective ≈ 3.45M。

正式文档只用机器 audit 值替换上述估算。

---

# 7. TASS-D0：正式训练前零训练 raw source gate

## 7.1 为什么不能直接拿“随机初始化 TASS feature”做 raw gate

Run11 与 R8/SF-D0 不同：TASS 本来就是一个需要任务训练的 feature producer。随机卷积 feature 的 PR-AUC 高低没有稳定科研含义。

因此零训练 gate 必须明确叫：

> **Source Feasibility Gate**：验证 TASS 将要接触的原始像素网格，在 `1/4–1/16` 上是否真的含有相对 B4 互补的 boundary evidence。

它是**必要条件，不是充分条件**；通过它只代表“值得花 80K”，不能提前宣称 TASS 有效。

## 7.2 参数自由 spatial proxy

对原始 `[0,1] RGB` A/B（几何严格对齐，不用 label 参与构造）：

```text
R64 = mean_c |AvgPool4(A)  - AvgPool4(B)|   # 64×64
R32 = mean_c |AvgPool8(A)  - AvgPool8(B)|   # 32×32
R16 = mean_c |AvgPool16(A) - AvgPool16(B)|  # 16×16
```

分别：

1. bilinear 到 256×256；
2. dataset 内 rank-normalize；
3. `Rspatial = mean(rank(R64), rank(R32), rank(R16))`；
4. `Rfuse = 0.5*rank(B4_up) + 0.5*Rspatial`。

**0.5/0.5 在运行前固定，不 sweep。**

同时计算：

- full-pixel PR-AUC；
- boundary-band PR-AUC（沿用 R8/SF-D0 的 GT 边界 ±4 px）；
- Top32 patch precision 仅作诊断，不作为主 gate。

## 7.3 Gate

### G0：审计有效性

必须全部满足：

- 数据集 sample count 与 `list/test.txt` 一致；
- label 统一 `gray>=128`；
- B4-only boundary PR-AUC 复现 SF-D0：
  - CDD 0.5216；
  - LEVIR 0.5369；
  - SYSU 0.6113；
  - WHU 0.5718；
- 容忍误差 `±0.01`；
- frozen ViT4 checksum 与 corrected loader 一致。

不满足 → **AUDIT INVALID**，只修审计代码，不判机制。

### G1：fine-scale boundary complementarity

主门槛：

```text
PRbnd(Rfuse) - PRbnd(B4) >= +0.020
```

要求：

- **SYSU 必过**；
- 四数据集至少 **2/4** 通过。

这里继续沿用上一轮 +0.02 量级，不能因为上一轮失败而事后降低。

### G2：不能靠全局伪变化换边界 lift

要求：

```text
PRpixel(Rfuse) >= PRpixel(B4) - 0.010
```

至少 **3/4 数据集**通过，且 SYSU 必须通过。

这防止 raw RGB difference 只是在边缘附近响应更高、同时把大量 unchanged 区域也推高。

### G3：预算

机器构建 M1 deploy 后：

```text
EFFECTIVE_PARAMS <= 5,000,000
```

且必须报告 FLOPs；不设 FLOPs 新硬阈值，但若实际值 > 5G，应在启动 80K 前重新评估是否仍符合“极轻量”叙事，**不能直接训练后再解释**。

## 7.4 TASS-D0 裁决

```text
G0 invalid  -> 修 audit 后重跑，不算机制失败
G1 FAIL     -> Run11 STOP
G2 FAIL     -> Run11 STOP
G3 FAIL     -> Run11 STOP
全部 PASS   -> 才允许 smoke / dry run / 80K
```

**Run11-D0 FAIL 后不允许：**

- 改 0.5/0.5 权重；
- 换 Sobel/Laplacian proxy 刷 gate；
- 换 width；
- 增加 edge branch；
- 改 boundary band；
- 先训练看看。

失败即转方向 (c)。

---

# 8. Smoke / dry run：P0 正确性门槛

TASS-D0 PASS 后才实现/运行正式训练入口。

## 8.1 T-S11-1：shape

对 batch=2：

```text
S1 = [2, 64, 64, 64]
S2 = [2,128, 32, 32]
S3 = [2,192, 16, 16]

P1(S1) = [2,192,64,64]
P2(S2) = [2,192,32,32]
P3(S3) = [2,192,16,16]
t4      = [2,192, 8, 8]
pred    = [2,1,256,256]
```

## 8.2 T-S11-2：C0/M1 epoch-0 exact equivalence

使用相同 frozen ViT、相同 TAR/DCR/head shared weights：

```text
alpha1=alpha2=alpha3=0
model.eval()
max_abs(pred_C0 - pred_M1) == 0.0
```

若不是 0：

- 先查 RNG stream；
- stem 必须在 shared modules 构造完成后创建，或使用 local Generator；
- 不允许用 `allclose` 放宽。

## 8.3 T-S11-3：梯度路径

第一个 backward：

```text
grad(alpha1/2/3) finite
至少 2/3 alpha grad != 0
ViT grad is None
```

因为 `alpha=0`，stem conv 第一步梯度为 0 是**预期行为**。

随后做一次 alpha nudge 或真实 optimizer step，再第二次 backward：

```text
stem0 / stage1 / stage2 / stage3 至少各一个 conv grad != 0
projector grad != 0
```

若第二步仍为 0 → dead branch，P0 FAIL。

## 8.4 T-S11-4：冻结审计

训练前、3-step 后：

```text
vit checksum exactly unchanged
```

## 8.5 T-S11-5：rep deploy 等价

复用现有 T0/T1/T2/T2b：

- TAR/DCR fold max error 仍满足原门槛；
- deploy 后不残留 aux BN/branch；
- 正式模型最终 `[REPARAM-ARGMAX-DISAGREE] = 0`。

TASS 本身不做训练多分支 reparameterization，避免把第二个创新混进来。

## 8.6 T-S11-6：预算

打印：

```text
[TOTAL-PARAMS]
[EFFECTIVE-PARAMS]
[TRAINABLE-PARAMS]
[DEPLOY-PARAMS]
[FLOPS-G]
[TASS-PARAMS]
```

`DEPLOY/EFFECTIVE > 5M` → STOP。

## 8.7 60-step dry run

SYSU，scratch 路径：

- 从 ImageNet pth + seed16 构建；
- `--resume` 必须为空；
- loss finite；
- alpha 离开 0；
- stem grad 非零；
- ViT checksum 不变；
- log TEST 区块格式正常；
- dry-run 目录与正式目录完全分开。

dry run 只检查实现，不根据 60-step F1 决定方法成败。

---

# 9. 唯一变量正式实验：C0 vs M1

## 9.1 两组定义

### C0 — TOKEN-ONLY

```text
Frozen ViT4
B1↑4 / B2↑2 / B3 / B4↓2
TAR(full)
DCR(full)
head
```

### M1 — TASS

```text
Frozen ViT4
B1↑4 + alpha1*TASS1
B2↑2 + alpha2*TASS2
B3    + alpha3*TASS3
B4↓2
TAR(full)
DCR(full)
head
```

**唯一变量：是否存在 task-adaptive spatial residual。**

以下全部固定：

- ImageNet pth；
- seed16；
- optimizer；
- lr；
- warmup；
- poly；
- 80K；
- batch16；
- augmentation；
- loss；
- test-as-val；
- TAR/DCR `full`；
- dim=160；
- threshold=0.5；
- GPU1；
- DataLoader list；
- label gray≥128。

## 9.2 RNG 纪律

C0/M1 shared modules 必须逐位同初始化。

推荐构造顺序：

```text
1. ViT4
2. head
3. TAR
4. DCR
5. M1 only: TASS stem + projectors + alpha
```

TASS 使用独立 local generator 初始化，禁止消耗 shared global RNG 后再构造共同模块。

smoke 保存并比较 shared state dict：

```text
0 / N shared keys differ
```

## 9.3 不允许用历史 checkpoint 当 C0

虽然当前已有 STRFusion 实现和各种历史 checkpoint，但 Run11 的 C0/M1 必须都按新纪律重新完整训练。

历史 R4-0/R4-1/ChangeViT baseline 只作上下文，不作为 C0 替代。

---

# 10. 预注册训练判据

## 10.1 Phase-1：SYSU 决策实验

先后顺序固定：

```text
SYSU C0 80K
→ 完整 TEST RESULTS
→ SYSU M1 80K
→ 完整 TEST RESULTS
→ 一次性裁决
```

不并行，避免 GPU/RNG/资源问题。

### PASS

同时满足：

```text
M1 F1 >= 85.00
M1 - C0 F1 >= +0.30 pp
M1 IoU > C0 IoU
deploy effective params <= 5M
REPARAM-ARGMAX-DISAGREE == 0
```

解释：既达到最难数据集硬目标，又有超过历史 ±0.15pp 噪声量级的模块增益。

### WEAK

满足以下任一：

```text
A. 84.50 <= M1 F1 < 85.00 且 M1-C0 >= +0.30 pp
B. M1 F1 >= 85.00 但 0 <= M1-C0 < +0.30 pp
```

WEAK 的含义：

- 不能宣称 Run11 主假设已被充分支持；
- **不自动扩展** LEVIR/WHU/CDD；
- 不调结构救；
- 归档后进入论文收尾决策。

### FAIL

包括：

```text
M1 F1 < 84.50 且未达到上面的 WEAK-A
或 M1-C0 < 0
或 deploy params > 5M
或 ViT checksum 改变
或正式 deploy disagreement != 0
```

另外记录一个**失败签名**：

```text
Recall 明显下降、Precision 上升、F1 仅微增
```

若复现 STR 历史“confidence sharpening”特征，即使未触发硬 FAIL，也必须在结论中标为 **mechanism-not-clean**，不得包装成稳定提升。

## 10.2 Phase-2：四数据集扩展

**只有 SYSU PASS 才允许。**

之后每个数据集都重新训练 **C0 + M1 两个完整 80K**，不能只跑 M1。

顺序：

```text
LEVIR C0 → LEVIR M1
WHU   C0 → WHU   M1
CDD   C0 → CDD   M1
```

阶段 gate：

| 数据集 | M1 硬目标 | 模块增益要求 | 失败动作 |
|---|---:|---:|---|
| SYSU | **≥85.00** | **ΔF1 ≥ +0.30pp** | 停 |
| LEVIR | **≥92.50** | ΔF1 ≥ +0.20pp | 停 |
| WHU | **≥95.00** | ΔF1 ≥ +0.15pp | 停 |
| CDD | **≥98.00** | ΔF1 ≥ +0.15pp | 停 |

这里 WHU/CDD 的增益门槛略低，是因为其硬目标距离历史 baseline 本来就只有约 0.16/0.25pp；但仍要求正向、且必须真正达到硬目标。

### 最终“Run11 成功”定义

只有同时满足：

```text
SYSU  >= 85
LEVIR >= 92.5
WHU   >= 95
CDD   >= 98
effective deploy params <= 5M
四数据集 M1-C0 均不为负
SYSU ΔF1 >= +0.30pp
LEVIR/WHU/CDD 达到各自预注册 ΔF1 门槛
```

才允许写：

> TASS 在统一协议下实现了 ≤5M 的四数据集硬目标，并得到受控 C0/M1 证据支持。

单一数据集 PASS 不得写“普适提升”。

---

# 11. 失败模式预案：失败后做什么，不做什么

## F1：TASS-D0 raw source gate 失败

解释：

> 原始 fine-scale pixel evidence 在当前数据协议下并不比 B4 提供稳定互补信息，或者伪变化噪声抵消了 boundary lift。

动作：

```text
STOP Run11
不实现 80K
转方向 (c)
```

不允许换 proxy 再刷一次。

## F2：raw gate PASS，但 M1≈C0

解释：

> spatial information 存在，但当前 task-adaptive stem 没能把它转成判别收益；H11 被端到端证伪。

动作：

- 保存 alpha/grad/feature audit；
- 不加 attention / edge head / deeper stem；
- 转 (c)。

## F3：M1 比 C0 差

优先检查：

- alpha 是否异常放大；
- LEVIR/WHU 是否出现 registration/radiometric pseudo-change；
- Recall/Precision 是否呈现 confidence sharpening；
- shallow residual 是否压过 token semantic。

这些只用于**解释**，不用于救模型。

## F4：SYSU 有明显收益但未到 85

若落入 WEAK-A：

- 可以在论文中说“task-adaptive spatial residual 得到局部支持”；
- 不能说达成硬目标；
- 不继续扫宽度；
- 是否转分析型论文由下一轮单独决定。

## F5：SYSU PASS，LEVIR/WHU 失败

说明机制可能偏向 SYSU 的高变化像素比例，对极不平衡建筑小目标/伪变化场景不稳。

按预注册：

```text
停止后续扩展
不调 alpha
不加边缘 loss
不改 threshold
```

## F6：参数/FLOPs 审计与估算不一致

若 deploy >5M：

- 属于硬约束失败；
- 不允许训练；
- 重新设计需另开预注册文档。

---

# 12. 文献依据（仅 2024–2026 高水平来源）

## 12.1 ChangeViT — Pattern Recognition 2026（DOI 2025）

**Duowang Zhu et al., “ChangeViT: Unleashing plain vision transformers for change detection in remote sensing images.” Pattern Recognition, Vol.172, 112539.**  
DOI: https://doi.org/10.1016/j.patcog.2025.112539  
官方代码: https://github.com/zhuduowang/ChangeViT

与本方案最直接的两点：

- plain ViT 对大尺度变化有优势，但 fine-grained/local changes 较弱；
- 原文 detail-capture 使用 ResNet18 的多尺度局部特征，并明确输出 1/2、1/4、1/8 features，再与 ViT semantic feature 协同。

本方案不是否定 ChangeViT，而是把其“global semantic + local detail”原则在 **≤5M** 预算下重新分配：ViT 深度截到 4，detail 不再用 2.7M ResNet + FI，而用约 0.86M residual stem。

## 12.2 ViT-CoMer — CVPR 2024，CCF-A

**Chunlong Xia et al., “ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions.” CVPR 2024.**  
CVF: https://openaccess.thecvf.com/content/CVPR2024/html/Xia_ViT-CoMer_Vision_Transformer_with_Convolutional_Multi-scale_Feature_Interaction_for_Dense_CVPR_2024_paper.html  
官方代码: https://github.com/Traffic-X/ViT-CoMer

其核心观察与 CASA-CD 的 R8/SF-D0 高度一致：

- plain ViT 缺少 inner-patch information interaction；
- feature scale diversity 不足；
- CNN spatial pyramid 可补局部空间细节。

区别：ViT-CoMer 是多阶段双向 CNN–Transformer interaction；TASS 是冻结 ViT + 单向 residual correction，目标是极低预算 CD，而非通用 dense backbone。

## 12.3 RepViT — CVPR 2024，CCF-A

**Ao Wang et al., “RepViT: Revisiting Mobile CNN From ViT Perspective.” CVPR 2024.**  
CVF: https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html  
官方代码: https://github.com/THU-MIG/RepViT

它支持的不是“RepViT 就能解决 CD”，而是：

- mobile CNN 仍可以用极低开销提供强局部建模；
- token mixer / channel mixer 分离与 residual 设计可在移动端保持较好效率。

TASS 只借其**轻量局部 block 设计原则**；不把 RepViT block 本身当新贡献，也不加载 RepViT checkpoint。

## 12.4 RFANet — ISPRS JPRS 2024，权威 SCI

**Zhi-Hui You et al., “Robust feature aggregation network for lightweight and effective remote sensing image change detection.” ISPRS Journal of Photogrammetry and Remote Sensing, 215:31–43, 2024.**  
DOI: https://doi.org/10.1016/j.isprsjprs.2024.06.013  
官方代码: https://github.com/Youzhihui/RFANet

与本课题有关的核心点：

- 轻量 backbone 的原始 feature 若缺失 fine-grained detail，后续 decoder 很难凭空恢复；
- 多层 feature complementarity 对 lightweight CD 重要。

这与 SF-D0 “decoder 无法从 16×16 token 网格凭空造出 sub-patch evidence”形成独立支撑。

## 12.5 MixCDNet — IEEE TGRS 2024，权威 SCI

**Linlin Wang et al., “MixCDNet: A Lightweight Change Detection Network Mixing Features Across CNN and Transformer.” IEEE TGRS 62, 2024.**  
DOI: https://doi.org/10.1109/TGRS.2024.3438228

论文报告约 **0.32M / 1.59G**，因此再次说明：

> “≤5M”或“3.45M”本身绝对不能当创新点。

TASS 必须靠**为什么这样分配有限预算、为什么 residual spatial source 能解决已有证伪瓶颈**来成立。

## 12.6 EdgeRefNet — IEEE TGRS 2026，权威 SCI

**“EdgeRefNet: An Edge-Guided Refinement Network for Building Change Detection in Remote Sensing Images.” IEEE TGRS, 2026.**  
DOI: https://doi.org/10.1109/TGRS.2026.3689397

其 dual-path context/edge 设计说明高分辨率 BCD 的边界细节仍是现实瓶颈。

但本方案**不照搬**：

- 不加 edge label；
- 不加 edge loss；
- 不加独立 edge head；
- 不以 loss 作为创新。

TASS 的 fine-scale information 只通过主 CD loss 学习。

## 12.7 SAT — CVPR 2026 Findings

**Dinh Phu Tran et al., “SAT: Selective Aggregation Transformer for Image Super-Resolution.” CVPR 2026 Findings / arXiv:2604.07994.**  
arXiv: https://arxiv.org/abs/2604.07994  
官方代码: https://github.com/PhuTran1005/SAT

SAT 仍是 CASAA 的机制来源：Full Query + compressed K/V。但 Run3 已经证实 deployable routing ranking 提升不等于 F1 提升，因此 Run11 **不复活 CASAA router**。它在论文中应降级为 token-efficiency analysis/ablation 资产。

---

# 13. 逐文件修改清单

## 13.1 新增：`models/model/tass_stem.py`

实现：

```text
TASSStem
LocalMixBlock
SpatialProjector
```

要求：

- shared Siamese；
- widths 固定 `(32,64,128,192)`；
- stage blocks 固定 `(2,2,2)`；
- expansion 固定 `2`；
- `alpha = nn.Parameter(torch.zeros(3))` 或三个独立 scalar；
- stem 使用 local Generator 初始化；
- 提供 `param_breakdown()`；
- 不含 attention；
- 不含 loss；
- 不含 label。

## 13.2 新增：`models/model/str_tass_fusion.py`

建议新文件，不破坏已归档 `str_fusion.py`。

```python
class STRTASSNet(nn.Module):
    spatial_mode in {"token", "tass"}
```

- `token` = C0；
- `tass` = M1；
- `rep_mode` 固定 full，不开放实验旋钮；
- 构造 shared modules 在前，TASS 在后；
- `forward(pre, post, label=None)`；
- `switch_to_deploy()` 只折 TAR/DCR；
- `train()` 强制 frozen ViT eval；
- 提供 `shared_state_dict()` 用于 C0/M1 初始化逐位对拍。

## 13.3 复用：`models/model/str_encoder.py`

不改变历史 `forward` 语义。

最多新增一个**纯加性 helper**：

```text
forward_depth_features(x)
```

返回 B1/B2/B3/B4，供新模型组合；旧 `STRFusionNet` 行为必须不变。

## 13.4 不改机制：`str_tar.py / str_dcr.py / str_reparam.py`

除非出现 P0 bug，否则 Run11 禁止改结构。

这是实验纪律：不要让“新 stem”与“新版 decoder”同时变化。

## 13.5 修改：`models/train.py`

新增：

```text
--arch str_tass
--spatial_mode token|tass
```

但 Run11 脚本固定：

```text
--str_rep_mode full
--str_dim 160
```

训练启动时必须断言：

```text
resume is None
pretrained path == DeiT Tiny official local pth
seed == 16
```

正式 test 时记录：

```text
[TASS-MODE]
[ALPHA]
[TASS-PARAMS]
[DEPLOY-PARAMS]
[DEPLOY-FLOPS]
[VIT-CHECKSUM]
[REPARAM-MAX-ABS-ERROR]
[REPARAM-ARGMAX-DISAGREE]
```

## 13.6 修改：`models/eval.py`

- 加 `arch=str_tass`；
- 必须读取并检查 `arch.json`；
- mismatch 直接拒绝；
- deploy 后再跑正式 test。

## 13.7 修改：`models/smoke_test.py`

新增 `--mode tass`，实现 §8 的 T-S11-1 ~ T-S11-6。

旧 smoke 全部保留。

## 13.8 新增：`analyse/run11_tass_source_gate.py`

复用：

- `boundary_band`
- `pixel_pr_auc`
- `rank_normalize_np`
- B4 audit code

输出固定：

```text
[TASS-D0-G0]
[TASS-D0-G1]
[TASS-D0-G2]
[TASS-D0-GATE] PASS|FAIL|AUDIT-INVALID
```

并保存四数据集 CSV/JSON 明细。

## 13.9 新增：`analyse/run11_tass_budget.py`

输出：

```text
C0 train/deploy total/effective/trainable
M1 train/deploy total/effective/trainable
TASS-only params
FLOPs
headroom_to_5M
```

## 13.10 新增训练目录

```text
train_scripts/STR-Fusion/Run2_TASS/
  README.md
  dryrun_SYSU.sh
  C0_TOKEN/
    SYSU/train.sh
    LEVIR/train.sh
    WHU/train.sh
    CDD/train.sh
  M1_TASS/
    SYSU/train.sh
    LEVIR/train.sh
    WHU/train.sh
    CDD/train.sh
```

每个正式目录都是全新 checkpoint/log 路径，禁止自动发现 Run1/Run4 checkpoint。

## 13.11 README

在 D0 或正式实验结束之前，不预写“实验结果”。

只可先加“Run11 预注册中”的链接；最终数字必须从对应 `train_log.txt` 最后完整 TEST 区块回填。

---

# 14. 路径与恢复纪律

服务器：

```text
project:
  /home/yqwang/projects/CASA-CD

pretrain:
  /home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth

checkpoint:
  /share_datasets/yqwang/checkpoints/CASA-CD/STR-Fusion/Run2_TASS/<C0_TOKEN|M1_TASS>/<dataset>/

log:
  /home/yqwang/outputs/CASA-CD/STR-Fusion/Run2_TASS/<C0_TOKEN|M1_TASS>/<dataset>/train_log.txt

diagnostic:
  /home/yqwang/outputs/CASA-CD/diagnostics/STR-Fusion/Run2_TASS/
```

### 正式实验不允许恢复历史 CD checkpoint

允许的“恢复”只有：

> 同一个 Run11 80K 任务因机器中断后，从该任务自己的 `last.pth` 继续，以保持总步数/优化器状态一致。

不允许：

- C0 best → M1 微调；
- R4-1 → TASS 微调；
- STRFusion Run1 → Run11 续训；
- baseline → TASS 微调。

训练脚本应在首次启动时写 `run_manifest.json`：

```text
arch
spatial_mode
seed
pretrain sha256
git commit
dataset
max_steps
optimizer
lr
loss
checkpoint_dir
resume_source
```

`resume_source` 首次必须为空。

---

# 15. 立即执行顺序

```text
[0] 冻结本文档：阈值不再改

[1] 只写 analyse/run11_tass_source_gate.py
    不实现 TASS 模型
    → GPU1 跑四数据集 TASS-D0

[2] TASS-D0 FAIL
    → 写 Run11_D0终止记录
    → 不实现 80K
    → 转方向 (c)

[3] TASS-D0 PASS
    → 实现 tass_stem.py + str_tass_fusion.py
    → train/eval/smoke 加性接入

[4] py_compile
    → smoke T-S11-1..6
    → rep T0/T1/T2/T2b 回归
    → run11_tass_budget.py
    → deploy <=5M 确认

[5] SYSU 60-step dry run
    → alpha / grad / checksum / log format 全绿

[6] 正式 SYSU C0 80K
    → 只读最后完整 TEST RESULTS
    → 归档 log

[7] 正式 SYSU M1 80K
    → 只读最后完整 TEST RESULTS
    → PASS / WEAK / FAIL 一次性裁决

[8] 非 PASS
    → 停，不救

[9] PASS
    → LEVIR C0 80K → M1 80K → gate
    → WHU   C0 80K → M1 80K → gate
    → CDD   C0 80K → M1 80K → gate

[10] 四数据集全部通过
     → 汇总 Recall/Precision/OA/F1/IoU/Kappa
     → deploy params/trainable params/FLOPs
     → README
     → experiment_metrics.xlsx
     → 论文表格
```

---

# 16. 如果 Run11 失败，方向 (c) 如何接管

本轮不选择 (c)，因此不在这里重新展开完整论文大纲；但**预先锁死 fallback**：

> TASS-D0 或 SYSU 80K 非 PASS 后，不再开 Run12 “换 stem / 加 edge / 调宽度”的救援线。下一轮直接进入分析型论文收尾设计。

届时核心材料已经非常清晰：

```text
CASAA: ranking improvement ≠ F1
depth truncation: B4 robust, B12 worse
B4-only: token ranking ≠ dense boundary reconstruction
O-PRE: denser sampling lattice ≠ complementary evidence
multi-depth STRFusion: depth pyramid ≠ sub-patch evidence
TASS: task-adaptive spatial residual 是否能突破上述瓶颈
```

如果连 TASS 都失败，那么“**在极低预算下，空间表征预算应该投在哪里，以及哪些 seemingly-reasonable token/dense recovery 路径不可行**”就会形成一条完整的 negative-evidence / budget-allocation 研究链，比继续模块堆叠更有论文价值。

---

# 17. 仍需补充的证据

在真正写代码前，还缺以下机器证据；这些不是让方案继续发散，而是 Run11 的执行前置项：

1. **TASS-D0 四数据集 raw source gate**：当前完全缺失，是第一优先级；
2. **TASS exact deploy params / FLOPs**：本文 3.454M / 3.53G 为解析估算；
3. **C0/M1 common initialization bitwise audit**：必须在新构造顺序下重新验证；
4. **alpha 两步梯度链**：确认 zero-init residual 不形成永久 dead branch；
5. **正式 C0 的真实 80K 性能**：当前 STRFusion 因 SF-D0 被 gate 否决，从未有同协议 C0 80K；不能用 R4-1 或其它 checkpoint 代替；
6. **轻量 SOTA 同协议重跑**：只有 Run11 主方法通过后再做。优先从仓库已有文献索引中的 MixCDNet / RFANet 等选 1–2 个可复现、全监督、256×256、公开代码方法，严格用本项目 data split / gray≥128 / 80K / seed16 重新训练；不能直接拿论文数字做主表“公平胜负”。

---

# 18. 最终科研判断

当前证据已经把问题定位得很窄：

> **CASA-CD 现在不是“缺一个更聪明的 token trick”，而是缺一个在 ≤5M 预算内真正学到 sub-patch spatial evidence 的路径。**

所以 Run11 不应该继续围绕 CASAA、B1-B4 融合或 decoder 花样做文章。  
最值得做的一次实验，是把预算从已被证明冗余/无效的 frozen-token recovery 重新分配给一个**zero-init、task-adaptive、极小的 spatial residual stem**，并用 C0/M1 从头 80K 直接检验它是否能跨过 SYSU 85 这一最难门槛。

**主假设 H11：**

> 在保持 Frozen ViT4 coarse semantics、TAR/DCR 和训练协议不变时，加入 `1/4–1/16` 的 task-adaptive spatial residual，会提供 frozen patch16 token 无法表达的 sub-patch change evidence，从而使 M1 相对 C0 获得可测的 dense boundary / F1 增益，并在 ≤5M deploy 参数内达到四数据集硬目标。

**可证伪条件：**

- raw source gate 不显示 B4 的边界互补性；或
- SYSU M1 无法达到 F1 85；或
- M1-C0 < +0.30pp；或
- 任何 deploy 参数越过 5M。

任一成立，就停止，不把失败改写成“需要更多调参”。

---

## 附：本方案直接依据的仓库事实

本方案已对照当前仓库以下内容：

- `README.md` 的「研究定位与约定」「实验结果」；
- `docs/temporary/CASA-CD_STR融合_Run1_SF-D0结果与融合主线终止.md`；
- `docs/temporary/CASA-CD_STR融合_Run1_设计与预注册方案.md`；
- `models/model/str_fusion.py`；
- `models/model/str_encoder.py`；
- `models/model/str_tar.py`；
- `models/model/str_dcr.py`；
- `models/model/str_reparam.py`；
- `docs/参考文献/文献索引.md`；
- ChangeViT 与 SAT 原论文。

**执行时若仓库代码与本文档冲突：服务器实际文件 > 当前 GitHub main > 本文档中的解析估算。**
