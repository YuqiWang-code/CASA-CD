# CASA-CD Run6：ViT 语义预算重分配——无独立 Detail 的 Patch-Token 金字塔重建 + SGDP 可执行预注册方案

> **项目**：CASA-CD — 极轻量遥感二值变化检测  
> **审查基准**：GitHub `YuqiWang-code/CASA-CD` `main`  
> **本次核对 HEAD**：`12901b3aa370089c05e127b4afecb85ced70663c`（2026-10-01）  
> **事实权威读取顺序**：README → 2026-09-30 交接文档 → Run5 方案 → Run5 R5-D0 停止记录 → Run4 终局 → R4-D0 audit → Run4 原始方案 → Run5 README → 当前 `encoder.py / mobile_detail.py / sgdp_head.py / trainer.py / run5_postmortem_and_mobile_audit.py / run4_detail_interface_audit.py`。  
> **固定任务**：CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256；全监督二值变化检测；A/B/label；`gray >= 128`。  
> **固定训练协议**：BCE+Dice；Adam(`lr=2e-4, betas=(0.9,0.99), wd=1e-4`)；poly(power=0.9)+200 iter warmup；80000 steps；batch16；256×256；seed16；test-as-val；GPU1；ViT frozen。  
> **正式结果纪律**：只认单个 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===`。  
> **复杂度硬约束**：effective inference params `<3M`，工程目标 `<=2.20M`；Run6 hard FLOPs `<=2.00G`。  
> **禁止项**：不改 loss；不做多 seed；不做 sweep；不解冻 ViT；不回 detail branch 搜索；不做量化/剪枝/蒸馏核心贡献。

---

# 0. 结论先行

Run6 不应继续搜索新的独立 detail branch，也不建议第一步把 DeiT-Tiny width 从 192 改成 128/96。

**唯一主推方案**：

> ## **ViT4-192 frozen + PTPR + 原样 SGDP**
>
> - 保留已经被 R4-0→R4-1 验证的 `TinyViT4-192`；
> - **删除独立 raw-image detail path**（ResNet / Light / PSD / Mobile 全部不进入推理图）；
> - 从同一 ViT 的 **pre-position PatchEmbed token** 中重建 1/8、1/4、1/2 的轻量空间金字塔；
> - 重建器命名为 **PTPR：Patch-Token Pyramid Reconstruction**；
> - `SGDPHead` **不改结构、不改通道、不改 gate**，直接复用 Run5 已 smoke 的 115,267 参数实现。

总体：

```text
T1 ─► shared Frozen ViT4-192 ─┬─► P0_1: 192×16×16（PatchEmbed raw token，pos 之前）
                              └─► V1:   192×16×16（block4 + final LN）

T2 ─► same Frozen ViT4-192 ───┬─► P0_2
                              └─► V2

P0_t ─► shared PTPR ─► R2_t:16×128×128
                      R4_t:16×64×64
                      R8_t:24×32×32

(R2_1,R4_1,R8_1,V1) + (R2_2,R4_2,R8_2,V2)
                    │
                    ▼
                  SGDP
     |V1-V2| + semantic-gated local differences
                    │
                    ▼
              256×256 change map
```

设计参数预算：

```text
TinyViT4-192     1,976,832
PTPR                 7,648
SGDP               115,267
--------------------------------
TOTAL            2,099,747  ≈ 2.100M
TRAINABLE          122,915  ≈ 0.123M
预算余量           100,253  ≈ 0.100M
```

设计阶段预期总 FLOPs：约 `1.6–1.8G`；正式以服务器 `fvcore` 为准，hard gate `<=2.00G`。

**核心决策**：Run6 的预算分配从“ViT semantic + 独立 CNN detail”改成：

```text
~94.1% 参数：冻结 pretrained ViT4 semantic / patch representation
~0.36% 参数：PTPR（从已有 patch token 恢复空间相位）
~5.49% 参数：SGDP change-evidence head
0% 参数：独立 raw-image detail encoder
```

---

# 1. 证据判读：Run5-D0 后为什么应该取消独立 detail path

## 1.1 四个 R5-D0 结果的联合含义

### 事实 A：Mobile raw ranking 已接近可用，但仍没过 gate

```text
Mobile D4 PR-AUC   0.4831 < 0.50
Mobile D8 PR-AUC   0.5317 >= 0.52
Top32 precision    0.5006 / 0.5105，均过门槛
```

它说明成熟 ImageNet prefix 确实显著优于随机超轻 detail，但在严格预注册下仍不够稳定。

### 事实 B：PSD 失败主因之一确实是 pretrained stem 被重新写坏

```text
PSD conv rel_L2 0.639 vs R4-1 ResNet 0.211
activation cos 0.838 vs 0.916
```

这说明“超轻 detail + 少量 pretrained stem”不是稳妥方案：当后续容量太弱时，stem 会承担过高的适配压力。

### 事实 C：adapter 不是关键瓶颈

TileAdapter inference-only：

```text
ΔF1 = -0.013
```

因此不能再把研究预算花在 adapter、归一化、通道 remap 等接口小修。

### 事实 D：旧 FI 自身存在尺度零权重吸收

R4-1 F1=82.77 的模型里，1/2、1/4 的 FI 路径被训练成近/精确零，实际上主要依赖 1/8 注入。

这意味着：

> “必须保留三尺度独立 detail + 三路 FI 才能保小目标”并没有被当前端到端证据支持。

## 1.2 Run6 的预算重新分配

因此 Run6 不再问：

> “还能不能找到第四个 0.01–0.20M 的 detail branch？”

而改问：

> **“既然 ViT4 已被证明是最值钱的 1.98M 参数，能否直接从它已经计算出的 PatchEmbed token 恢复局部空间证据，从而彻底删除第二个图像编码器？”**

这比继续搜 Mobile/Efficient/Shuffle prefix 更符合当前证据链，也更适合论文机制叙事。

---

# 2. 明确回答：还要不要独立 detail path？

## 2.1 决策

> **不要。Run6 主方案中独立 detail path 参数 = 0。**

不再注册：

```text
encoder.resnet.*
encoder.detail (Light / PSD / Mobile)
encoder.detail_adapters.*
legacy FeatureInjector
legacy Decoder
```

## 2.2 小目标和边界靠什么保住？

不是靠一个新的 raw-image CNN，而是靠 **PatchEmbed 内部的“隐式子 patch 空间编码”**。

DeiT patch embedding 是：

```text
Conv2d(3→192, kernel=16, stride=16)
```

一个 16×16 patch 虽然最终变成一个 192-D token，但每个输出通道来自 16×16 内不同位置权重的线性组合，因此通道中仍包含**位置相关的 patch 内部响应**。Run6 不只使用经过 4 个全局 block 的最终 token，而是额外保留：

```text
P0 = PatchEmbed(image)       # 加 pos_embed 之前，尚未被全局 mixing 改写
```

PTPR 用三次 phase-aware PixelShuffle 把这些通道逐步变成：

```text
16×16 token grid
→ 32×32 (1/8)
→ 64×64 (1/4)
→ 128×128 (1/2)
```

所以“小目标/边界保真”的机制不是“额外抽取细节”，而是：

> **从同一 pretrained patch token 中解码 latent intra-patch spatial phase。**

之后 SGDP 用最终 ViT semantic difference 去 gate 这些 reconstructed local differences，抑制光照、纹理、错配产生的伪变化。

这只是**待验证机制假设**，不是既成结论；它必须先过 Run6-D0 和 SYSU/LEVIR gate。

---

# 3. 为什么不主推 width 192→128/96

Run6 当前不授权 width 缩减，原因不是“width 一定不能缩”，而是**证据和预训练兼容性不够干净**。

## 3.1 width 改变会同时破坏大量 exact inheritance

`192→128/96` 后，以下全部 shape 变化：

```text
patch_embed output channels
pos_embed channels
LayerNorm channels
qkv/proj matrices
MLP fc1/fc2
head partition
```

要继续继承 DeiT-Tiny，只能引入：

```text
channel slicing / structured pruning
SVD projection
learned projection distillation
random re-init
```

这些都会让“width 变量”与“pretrain transfer 方式”纠缠，不再是一个干净的单变量实验；其中部分还会靠近项目明确不作为核心贡献的剪枝/压缩路线。

## 3.2 当前没有预算必要性

ViT4 + PTPR + SGDP 已经约：

```text
2.100M
```

满足 `<=2.20M`，还保留约 `0.100M` 余量。

所以**没有必要为了省参数先破坏唯一已被验证的 semantic asset**。

## 3.3 depth3 的定位

`depth=3, width=192` 仍能 exact 继承 DeiT prefix，因此 Run6-D0 会顺手测 `B3` raw ranking，作为未来证据储备；但**本 Run6 不自动授权 ViT3 80K**。

如果主方案失败，不用“再砍一个 block”救火；下一轮只允许进入新的 semantic source 方案并重新预注册。

---

# 4. Run6-D0：零训练 Semantic / Patch-Token Audit

## 4.1 目的

在任何新 80K 前回答两个问题：

1. `P0`（PatchEmbed raw token）有没有足够的局部 change ranking，值得承担 reconstructed detail source？
2. `V4`（4-block final token）是否仍保留足够 semantic change ranking，且与 `P0` 融合后达到“可替代外部 detail”的最低水平？

## 4.2 数据与权重

**数据**：完整 SYSU test，4000 pairs，顺序固定、不 shuffle。

**必须加载**：

```text
pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth
R4-1 VIT4_OLDHEAD / SYSU best checkpoint
```

**诊断参考**：

```text
R4-0 A0_FULL12_FROZEN / SYSU best checkpoint
```

R4-1 的 ViT 是 frozen，因此 D0 开始先做 checksum 断言：

```text
R4-1 encoder.vit.patch_embed == corrected DeiT init
R4-1 blocks.0-3           == corrected DeiT init
R4-1 pos_embed            == corrected 14×14→16×16 interpolation
```

否则 D0 直接无效，不进入指标计算。

## 4.3 抽取位置

同一 forward 一次性记录：

```text
P0  : PatchEmbed 输出，pos_embed 之前，16×16×192
B1  : block0 后
B2  : block1 后
B3  : block2 后
B4  : block3 后（Run6 semantic source）
B12 : R4-0 full12 最终 token（只作诊断，不作 gate）
```

对 B1-B4 使用同一个 final LN 做 audit copy，不改变正式模型前向。

## 4.4 指标口径

完全复用 R4-D0 / R5-D0 的口径：

```text
score_i = 1 - cosine(f1_i, f2_i)
GT      = AvgPool16(label) → occupancy
positive patch = occupancy > 0
```

每个 source 统计：

```text
PR-AUC
Spearman(score, occupancy)
Top32 precision
Top32 changed-pixel coverage
```

另外构造一个**只用于 audit 的参数自由融合**：

```text
S_fuse = 0.5 * RankNorm(S_P0) + 0.5 * RankNorm(S_B4)
```

它不是 Run6 网络的一部分，不会进入训练/推理，只是检验“浅 patch evidence + 深 semantic evidence”是否互补。

## 4.5 audit 自检门槛

先复现 R4-1 ResNet 1/8 旧结果：

```text
PR-AUC        = 0.6535 ± 0.005
Top32 prec    = 0.5948 ± 0.005
```

任一超出容差：

> **[AUDIT-INVALID]，停止。**

说明数据/归一化/score 口径和历史 audit 不一致，不能用新数字作 gate。

## 4.6 Run6-D0 预注册 gate

### G1：Patch-token local floor（P0）

```text
P0 PR-AUC      >= 0.44
P0 Top32 prec  >= 0.40
```

解释：约为 R4-1 ResNet 1/4 raw ranking 的 70% 下限；低于此值，不值得让 P0 承担 detail reconstruction source。

### G2：ViT4 semantic floor（B4）

```text
B4 PR-AUC      >= 0.40
B4 Top32 prec  >= 0.36
B4 Spearman    >= 0.18
```

这不是要求 ViT4 像 ResNet detail 一样强；它只要求 semantic source 明显高于无信息水平，并与既有 ViT cosine audit 的有效区间一致。

### G3：双源融合 floor（P0 + B4）

```text
Fuse PR-AUC      >= 0.50
Fuse Top32 prec  >= 0.46
Fuse Spearman    >= 0.25
```

这里沿用 Run5 对关键 local interface 的实际强度要求：组合证据至少达到 Mobile gate 的 D4 等级，才值得花 80K。

### PASS 条件

```text
G1 全部通过
AND G2 全部通过
AND G3 全部通过
```

**不设“差 0.01 也算过”的宽限。**

### FAIL

任一 gate FAIL：

> **不实现/不训练 R6-1；永久停止“ViT4 token-only + token reconstruction”这条 Run6 路线。**

后续唯一允许方向：

> **新的 semantic source 重构（候选 3），另开下一轮重新预注册。**

禁止：

```text
再找 detail branch
放宽 D0 threshold
改 score 函数后重跑直到 PASS
width 128/96 sweep
depth 3/2 sweep
```

## 4.7 D0 附加诊断（不参与 gate）

记录：

```text
B1 / B2 / B3 / B12 的全部 ranking 指标
按 changed-patch 数量分组：1-16 / 17-64 / >64
每组 Top8 hit / Top32 coverage
```

用途：论文 analysis 与后续失败归因；不授权额外训练。

---

# 5. Run6 唯一主方案：PTPR + SGDP

## 5.1 ViT semantic path

完全沿用 R4-1：

```text
PatchEmbed: 3→192, kernel16, stride16
Token grid: 16×16 = 256
Embed dim: 192
Heads: 6
MLP ratio: 4
Blocks: 0,1,2,3
Final LN
```

权重：

```text
DeiT-Tiny ImageNet-1K
patch_embed exact load
blocks0-3 exact load
norm exact load
pos_embed: 去 extra token，14×14 bicubic → 16×16
```

训练：

```text
requires_grad=False
optimizer 中虽可保留 vit param group，但必须无 grad、checksum 全程不变
```

## 5.2 PTPR：Patch-Token Pyramid Reconstruction

输入：

```text
P0_t: B×192×16×16
```

注意：`P0_t` 是 **PatchEmbed 后、pos_embed 前** 的 token map；每个时相只计算一次 PatchEmbed，不允许为了 PTPR 重复跑 patch embedding。

### Stage R8：16×16 → 32×32

```text
Conv1×1 192→16, bias=False
BN(16)
ReLU
Conv1×1 16→96, bias=False
BN(96)
ReLU
PixelShuffle(2)
```

输出：

```text
R8_t: 24×32×32
```

参数：

```text
192*16 + 2*16 + 16*96 + 2*96
= 4,832
```

设计理由：用 16ch bottleneck 控制参数；96 channel 经 PixelShuffle2 恢复为 24×2×2 spatial phase。

### Stage R4：32×32 → 64×64

```text
Conv1×1 24→64, bias=False
BN(64)
ReLU
PixelShuffle(2)
```

输出：

```text
R4_t: 16×64×64
```

参数：

```text
24*64 + 2*64 = 1,664
```

### Stage R2：64×64 → 128×128

```text
Conv1×1 16→64, bias=False
BN(64)
ReLU
PixelShuffle(2)
```

输出：

```text
R2_t: 16×128×128
```

参数：

```text
16*64 + 2*64 = 1,152
```

### PTPR 总参数

```text
4,832 + 1,664 + 1,152 = 7,648
```

### PTPR 结构约束

- 无 raw-image shortcut；
- 无 3×3/5×5 独立 image encoder；
- 无 attention；
- 无 residual branch；
- T1/T2 **同一套权重共享**；
- 不加 aux loss；
- 不单独 pretrain；
- 初始化沿用仓库默认 Conv/BN 初始化，seed=16，不引入新的初始化技巧。

## 5.3 SGDP

`models/model/sgdp_head.py` **结构完全不变**：

```text
输入：
R2: 16×128×128
R4: 16×64×64
R8: 24×32×32
V : 192×16×16
```

与 Run5 MobileDetail native channel 完全一致，所以：

```text
SGDP 115,267 params
无需 adapter
无需改 detail projection
无需改 gate
无需改 channel width
```

仍执行：

```text
ΔV  = |V1-V2|
ΔR8 = |R8_1-R8_2|
ΔR4 = |R4_1-R4_2|
ΔR2 = |R2_1-R2_2|
```

并按：

```text
semantic up → spatial semantic gate → gated local difference → SepBlock fusion
```

逐级 16→32→64→128→256。

## 5.4 时间交换对称

因为：

```text
PTPR(T1), PTPR(T2) 使用共享函数
SGDP 只使用绝对差值
```

理论上：

```text
f(T1,T2) == f(T2,T1)
```

smoke 必须用数值断言：

```text
max_abs(pred(T1,T2)-pred(T2,T1)) <= 1e-6
```

---

# 6. 参数预算与 FLOPs

## 6.1 参数预算

| Component | Params | Trainable | Inference |
|---|---:|---:|---:|
| TinyViT4-192 | 1,976,832 | 0 | ✓ |
| PTPR | 7,648 | 7,648 | ✓ |
| SGDP | 115,267 | 115,267 | ✓ |
| **TOTAL** | **2,099,747** | **122,915** | **2,099,747** |
| Budget reserve to 2.20M | **100,253** | — | — |

该预算按保守口径包含 ViT 内当前注册参数；最终 `effective_params` 仍以实际 forward graph 审计为准。

## 6.2 FLOPs 粗估

沿 Run5 已验证组合：

```text
ViT4 pair                  ≈ 1.18G
PTPR pair                  ≈ 0.014G MAC 量级
SGDP                       ≈ 0.40G
framework / counting diff  ≈ 余量
```

预期服务器 `fvcore`：

```text
~1.6–1.8G
```

hard gate：

```text
FLOPs <= 2.00G
unsupported ops 数量必须记录
```

`PixelShuffle` 本身是重排，不应被当成大量 arithmetic FLOPs；但必须保留同一 `fvcore` 口径与 Run5 比较。

---

# 7. 与 Run6 三个候选的关系

| 决策叉 | Run6 处理 |
|---|---|
| 1. width 192→128/96 / depth3 | **不主推**。width 破坏 exact pretrain；depth3 只做 D0 诊断，不授权 80K |
| 2. 去掉独立 detail | **采用，作为唯一主线** |
| 3. 换 semantic 源 | **只作为 Run6 彻底失败后的下一轮唯一出口** |

本方案不是“ViT-only + bilinear decoder”，而是：

> **ViT semantic + same-backbone patch-token spatial reconstruction**。

所以它仍然有 fine-scale reconstruction path，但不再有第二个 image encoder。

---

# 8. 预注册正式实验 gate

# 8.1 R6-1：SYSU / VIT4_TOKENRECON_SGDP

## 唯一变量

相对 R4-1，Run6 把外围表示方式整体从：

```text
independent ResNet detail + legacy FI/decoder
```

替换为一个预注册的统一“**token reconstruction downstream package**”：

```text
PTPR + SGDP
```

ViT4、DeiT 预训练、冻结策略、loss、optimizer、schedule、steps、batch、seed、数据协议全部不变。

这是 Run6 唯一结构候选；不与其它 head/width 并行比较。

## Reference

直接 reference：

```text
R4-1 VIT4_OLDHEAD
F1  = 82.77
IoU = 70.61
```

健康 ceiling 仅作背景：

```text
R4-0 full12 frozen
F1  = 83.14
IoU = 71.14
```

## 终 gate

```text
F1             >= 82.30
IoU            >= 69.92
ΔF1 vs R4-1    >= -0.47
ΔIoU vs R4-1   >= -0.69
effective param <= 2.20M
FLOPs           <= 2.00G
```

其中前四项本质对应同一个 accuracy floor；全部必须满足。

### PASS

→ 进入 R6-2 LEVIR。

### FAIL

> **Run6 主方案永久停止。**

不允许：

```text
PTPR bottleneck 16→24/32 sweep
增加 PixelShuffle stage
SGDP width 调整
SGDP gate 调整
depth4→3 rescue
width192→128 rescue
回 Mobile/PSD/ResNet detail
改 loss/LR/steps
```

下一轮唯一允许：

> **semantic source replacement，重新预注册。**

---

# 8.2 R6-2：LEVIR 跨场景 gate

只在 R6-1 PASS 后启动；模型结构和所有训练超参完全冻结，只换 dataset。

```text
F1              >= 91.50
IoU             >= 84.33
effective params <= 2.20M
FLOPs            <= 2.00G
```

参考背景：

```text
baseline trained 91.95
frozen A1       91.84
```

### PASS

→ 架构正式冻结，补 CDD / WHU。

### FAIL

> **停止四数据集铺开。**

不补 CDD/WHU，不做针对 LEVIR 的小目标专用修补，不改 PTPR/SGDP。

这一步是“无独立 detail 是否能跨到稀疏建筑变化”的关键可证伪 gate。

---

# 8.3 R6-3：CDD 最终覆盖

只在 LEVIR PASS 后运行；**不再开发模型**。

内部 retention floor：

```text
baseline CDD F1  = 97.75
Run6 CDD F1      >= 97.25   （drop <= 0.50）
Run6 CDD IoU     >= 94.65
params/FLOPs     同上
```

若失败：记录局限，不改结构。

---

# 8.4 R6-4：WHU 最终覆盖

只在 LEVIR PASS 后运行；可与 CDD 串行，仍只用 GPU1。

内部 retention floor：

```text
baseline WHU F1  = 94.84
Run6 WHU F1      >= 94.34   （drop <= 0.50）
Run6 WHU IoU     >= 89.29
params/FLOPs     同上
```

若失败：记录局限，不改结构。

> CDD/WHU 这两个 floor 只是本项目内部“相对健康 baseline 的保真门槛”，不是轻量 SOTA 结论。是否超过 RFANet / SeCoR / Lighter / CGLNet，必须后续按相同评估协议核对其公开结果或重跑。

---

# 9. 停止规则总表

| Gate | 失败后唯一动作 | 永久停止内容 |
|---|---|---|
| R6-D0 | Run6 不花 80K；下一轮只准 semantic source replacement | token-only / PTPR；任何新 detail；width/depth sweep |
| R6-1 SYSU | Run6 主方案终止；下一轮重新预注册 semantic source | PTPR bottleneck/PixelShuffle/SGDP 调参；ViT3 rescue；detail 回退 |
| R6-2 LEVIR | 不补 CDD/WHU；记录跨场景失败 | LEVIR-specific 小目标模块；loss/LR 特调；detail 回退 |
| R6-3/4 | 只记录限制，不再改模型 | 任何针对单数据集的结构修补 |

这保证 Run6 不会再次变成 backbone/head 搜索。

---

# 10. 逐文件修改清单

## 10.1 新增 `models/model/token_reconstructor.py`

实现：

```python
class PatchTokenPyramidReconstructor(nn.Module):
    # params exact = 7,648
    # input  B,192,16,16
    # output d2(16,128,128), d4(16,64,64), d8(24,32,32)
```

要求：

- 三个 stage 与 §5.2 完全一致；
- 不额外加 residual / SE / attention；
- 不添加 dropout；
- T1/T2 共享同一个实例。

## 10.2 修改 `models/model/encoder.py`

新增：

```text
detail_mode = token_recon
```

ViT 增加一个不重复计算 PatchEmbed 的接口，例如：

```python
forward_with_patch_tokens(x)
    p0 = patch_embed(x)          # retain raw token
    x  = p0 + pos_embed
    x  = blocks0-3(x)
    v  = norm(x)
    return p0, v
```

Encoder 在 `token_recon` 模式：

```text
不注册 resnet
不注册 Light/PSD/Mobile
注册 token_reconstructor
P0→reshape 192×16×16→PTPR
V→reshape 192×16×16
return [R2,R4,R8,V]
```

**严禁**为 PTPR 再调用一次 `patch_embed(x)`。

## 10.3 `models/model/sgdp_head.py`

**不改网络结构。**

只允许：

- 更新 docstring，注明 Run6 的 detail 输入可来自 PTPR；
- 不改任何 channel / gate / SepBlock / classifier。

必须保持：

```text
sum(params) == 115,267
```

## 10.4 修改 `models/model/trainer.py`

保留：

```text
head_mode='sgdp'
```

让 `detail_mode='token_recon'` 与 SGDP 组合合法。

不新建第二套 head，不注册 legacy Decoder。

## 10.5 修改 `models/train.py`

- CLI/日志支持 `detail_mode=token_recon`；
- `arch.json` 保存该模式；
- 日志新增：

```text
[PATCH-TOKEN-RECON] 1
[PTPR-PARAMS] 7648
[SGDP-PARAMS] 115267
```

- `measure_effective_params` 对 Run6 应等于 total（无死 ResNet）；
- 保持最终 TEST RESULTS 格式完全不变。

## 10.6 修改 `models/eval.py`

支持从 `arch.json` 严格恢复：

```text
vit_depth=4
detail_mode=token_recon
head_mode=sgdp
```

并 strict load best checkpoint。

## 10.7 修改 `models/smoke_test.py`

新增 T-R6 测试（见下一节）。

## 10.8 新增 `analyse/run6_semantic_token_audit.py`

复用：

```text
pr_auc
spearman
top32_stats
loader / Normalize / test protocol
```

输出固定：

```text
[P0]
[B1]
[B2]
[B3]
[B4]
[B12]
[FUSE-P0-B4]
[R6-D0-GATE] PASS/FAIL
```

返回码：

```text
PASS -> 0
FAIL -> 2
AUDIT_INVALID -> 3
```

## 10.9 新增 `train_scripts/UltraLight/Run6/`

至少：

```text
README.md
audit_R6_D0_SEMANTIC_SYSU.sh
dryrun_R6_1_SYSU.sh
train_R6_1_VIT4_TOKENRECON_SGDP_SYSU.sh
train_R6_2_VIT4_TOKENRECON_SGDP_LEVIR.sh
train_R6_3_VIT4_TOKENRECON_SGDP_CDD.sh
train_R6_4_VIT4_TOKENRECON_SGDP_WHU.sh
```

**不做自动全队列**；每步人工读 gate 后再启动下一步。

---

# 11. Smoke / dry run 预注册清单

## T-R6-0：DeiT exact inheritance

必须复用 `vit_pretrain_audit.py` 逻辑：

```text
patch_embed exact == source
blocks0-3 exact == source
norm exact == source
pos 14×14→16×16 规则一致
blocks4-11 不存在
```

## T-R6-1：无独立 detail

state_dict 中不得出现：

```text
encoder.resnet
mobile_detail
psd_detail
light_detail
detail_adapters
legacy structure_enhance
legacy decoder up_c*
```

## T-R6-2：PTPR shape + params

```text
input  : B×192×16×16
R8     : B×24×32×32
R4     : B×16×64×64
R2     : B×16×128×128
params : 7,648 exact
```

## T-R6-3：SGDP unchanged

```text
params == 115,267
```

最好保存当前 Run5 SGDP state_dict key 列表，Run6 assert keys/shape 完全一致。

## T-R6-4：总参数

```text
TOTAL/EFFECTIVE == 2,099,747  （允许仅因明确记录的无效 mask token 口径出现 <=192 差异）
TRAINABLE        == 122,915   （freeze_vit 后）
```

若不一致，先打印逐组件 breakdown，禁止直接训练。

## T-R6-5：时间交换对称

```text
max_abs(pred(A,B)-pred(B,A)) <= 1e-6
```

Eval 模式、固定输入。

## T-R6-6：梯度路径

训练 3 step：

```text
PTPR grad finite & non-zero
SGDP grad finite & non-zero
ViT grad is None
loss finite
```

## T-R6-7：Frozen ViT checksum

记录训练前：

```text
patch_embed
pos_embed
block0 qkv
block3 qkv
norm
```

3-step 后 `max_abs_diff == 0`。

## T-R6-8：FLOPs

```text
fvcore total <= 2.00G
unsupported_ops 明确打印
```

## T-R6-9：checkpoint/eval restore

60-step dry run 后：

```text
last.pth 可 resume
arch.json 完整
eval.py strict restore
best/state dict 不含 dead module
```

---

# 12. 真实数据 dry run

R6-D0 PASS 后才实现/部署正式模型，然后：

```text
SYSU train split
60 steps
batch 16
GPU1 only
临时 ckpt/log 目录
```

必须检查：

```text
exit 0
loss finite，无 nan/inf
无 OOM
TOTAL/EFFECTIVE/TRAINABLE 参数符合预注册
FLOPs <= 2.0G
ViT checksum 完全不变
PTPR + SGDP 参数确实更新
A/B exchange symmetry smoke 已过
日志头完整
```

Dry run 失败：只修 P0 correctness / 工程 bug；修复后重新 smoke + dry run，**不改变架构定义**。

---

# 13. 正式执行顺序与 80K 预算

## Step 0：同步与版本锁定

```bash
cd /home/yqwang/projects/CASA-CD
git status
git rev-parse HEAD
```

记录 Run6 启动 SHA；正式训练期间不在同一目录热改模型文件。

## Step 1：R6-D0（0 个 80K）

```text
完整 SYSU test 4000 pairs
只 forward
读 [R6-D0-GATE]
```

FAIL → Run6 结束。

## Step 2：smoke + SYSU 60-step dry run

全部通过才进入正式训练。

## Step 3：R6-1 SYSU 80K

```text
formal run #1
```

只认最后完整 TEST RESULTS。

FAIL → Run6 结束。

## Step 4：R6-2 LEVIR 80K

```text
formal run #2
```

FAIL → 不补 CDD/WHU。

## Step 5：CDD / WHU 定稿覆盖

```text
formal run #3 = CDD
formal run #4 = WHU
```

只在 LEVIR PASS 后执行；两者完成后**不再改结构**。

### 正式 80K 总预算

```text
最坏定稿路径 = 4 个
SYSU + LEVIR + CDD + WHU
```

完全符合本轮 `<=3–4` 个正式 run 到定稿的预算。

---

# 14. Checkpoint / log 路径

建议：

```text
/checkpoint root
/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run6/
  R6_1_VIT4_TOKENRECON_SGDP/<DATASET>/

/log root
/home/yqwang/outputs/CASA-CD/UltraLight/Run6/
  R6_1_VIT4_TOKENRECON_SGDP/<DATASET>/train_log.txt

/audit
/home/yqwang/outputs/CASA-CD/UltraLight/Run6/
  R6_D0_SEMANTIC_TOKEN_AUDIT/audit_report.txt
```

CDD/LEVIR/SYSU/WHU 可以共用同一 variant 名，只换 `<DATASET>`，避免产生“R6-2/3/4 是不同模型”的误解。

---

# 15. 论文机制故事（仅在 gates 成立后使用）

## 15.1 第一层：Plain ViT 的深度冗余可以被大幅压缩

已有直接证据：

```text
healthy full12 frozen 83.14
ViT4 frozen           82.77
ΔF1                   -0.37
减少参数              ~3.56M
```

因此在当前 CD 协议中，DeiT-Tiny 后 8 个 block 存在明显 deployment redundancy。

## 15.2 第二层：极低预算下，“独立 detail encoder”并不是稳妥的预算去向

现有可证伪链：

```text
Light32      FAIL
Light48      FAIL
PSD          FAIL
Mobile raw   FAIL
```

同时旧 FI 出现尺度零权重吸收。

因此论文不能再写“我们只是设计了更好的轻量 CNN detail”。更有证据的论点是：

> **在约 2M 总预算下，第二个图像编码器的边际收益不足以覆盖其表示/训练不稳定性。**

## 15.3 第三层：同一 pretrained patch token 可以兼顾 semantic 与 detail source

Run6 的机制创新是：

> **不重新编码原图，而是重用 plain ViT 的 PatchEmbed latent，进行轻量 patch-to-pyramid spatial reconstruction。**

PTPR 通过：

```text
channel bottleneck
→ phase-aware PixelShuffle ×3
```

把 16×16 patch tokens 解码到 1/8、1/4、1/2 局部表征。

这与“简单 bilinear 上采样 final token”有实质区别：

- 取的是 **pre-position shallow patch token**，不是只取 block4 semantic；
- PixelShuffle 让 channel 被显式分配到更细的 spatial phase；
- 三尺度 reconstructed feature 仍由同一 ViT patch representation 产生，不存在第二 backbone。

## 15.4 第四层：SGDP 只在“变化证据域”融合

SGDP 保持 Run5 设计：

```text
先 |T1-T2|
再 semantic gate local difference
再 coarse-to-fine reconstruction
```

而不是：

```text
先对 T1/T2 各自做重型 FI
再进行差分
```

论文中可表述为：

> **change-evidence-first fusion avoids spending parameters on reconstructing two complete per-temporal feature hierarchies that are discarded immediately after differencing.**

## 15.5 如果最终四数据集成立，可形成的贡献表述

建议最终贡献控制为两项主贡献 + 一项系统性分析：

1. **Patch-Token Pyramid Reconstruction**：提出无独立 CNN detail encoder 的细粒度重建机制，从 pretrained ViT PatchEmbed latent 中恢复多尺度局部空间表征；
2. **Semantic-Guided Difference Pyramid**：以差分先行 + coarse semantic gate 的方式，在极低参数下融合 reconstructed local evidence；
3. **Budget allocation study**：通过 full12→ViT4、三类自定义 detail、成熟 Mobile prefix、FI 零权重吸收和 D0 raw audit，系统证明约 2M 参数预算下“语义保留 + token reconstruction”比继续堆独立 detail 更值得验证。

**不能提前写的结论**：

```text
“PTPR 一定保边界”
“无 detail 普适优于 CNN detail”
“超过所有轻量 SOTA”
```

这些只在 SYSU + LEVIR + CDD + WHU 正式结果和同协议 SOTA 比较完成后才能写。

---

# 16. 可证伪假设

## H6-A：PatchEmbed token 仍含可利用的局部 change evidence

预测：

```text
R6-D0 P0 PR-AUC >= 0.44
P0 Top32 >= 0.40
```

失败：PTPR 没有足够信息源，方案在训练前结束。

## H6-B：浅 patch evidence 与深 semantic evidence 互补

预测：

```text
Fuse(P0,B4) PR-AUC >= 0.50
Top32 >= 0.46
Spearman >= 0.25
```

失败：无独立 detail 的双源基础不成立。

## H6-C：PTPR+SGDP 可以在 2.20M 内保住 SYSU 性能

预测：

```text
F1 >= 82.30
IoU >= 69.92
params <= 2.20M
FLOPs <= 2.0G
```

失败：token reconstruction 不能替代独立 detail，Run6 主方案终止。

## H6-D：该机制能跨到稀疏小建筑变化

预测：

```text
LEVIR F1 >= 91.50
IoU >= 84.33
```

失败：说明 SYSU 成功主要来自高变化密度，不能称为通用轻量 BCD 架构。

---

# 17. 最小消融（不额外消耗 Run6 正式 80K）

当前资源约束下，不再为论文单独开 2–3 个 80K ablation。优先利用已有链条：

```text
R4-0 full12 frozen              semantic depth reference
R4-1 ViT4 + ResNet + legacy     depth-reduced strong reference
R4-2/2b/2d                      custom detail negative evidence
R5-D0 Mobile raw                mature micro-detail negative gate
R6-D0 P0/B4/fused               zero-training source evidence
R6-1                             final token-reconstruction model
```

如果最终论文审稿前必须补一个最小消融，优先级只有一个：

> **PTPR → parameter-matched bilinear token pyramid**（仅在最终模型四数据集成立且还有训练预算时另行预注册）。

本 Run6 当前**不授权**该 80K。

---

# 18. 立即执行顺序

1. **只先写 `analyse/run6_semantic_token_audit.py`**，不要先改最终模型；
2. 用现成 R4-1/R4-0 + corrected DeiT 跑完整 SYSU R6-D0；
3. 人工核对 control、P0、B4、Fuse gate；
4. 只有 `[R6-D0-GATE] PASS` 才实现 `token_reconstructor.py` 与 `detail_mode=token_recon`；
5. 跑 `py_compile`、T-R6 smoke、参数/FLOPs、对称性、pretrain exact-load；
6. GPU1 做 SYSU 60-step dry run；
7. 启动 R6-1 SYSU 80K；
8. 收最后完整 TEST RESULTS，过 `82.30 / 69.92 / 2.20M / 2.00G` 才启动 LEVIR；
9. LEVIR 过 `91.50 / 84.33` 后架构冻结，补 CDD、WHU；
10. 四数据集结束后才更新论文 SOTA 表与最终方法命名。

---

# 19. 仍需补充的证据

在开始 R6-1 之前，只缺一项真正关键的新证据：

> **R6-D0 的 P0 / B4 / Fuse raw ranking。**

其它内容已经足够：

- ViT4 depth redundancy 已有正式 80K；
- 三个自定义 detail 失败已有正式结果；
- Mobile 成熟 prefix 已有零训练 gate；
- SGDP 结构、参数、对称性、FLOPs 已 smoke；
- corrected DeiT loader 已有 exact-load audit；
- Run6 参数预算已经闭合到约 2.100M。

因此下一步不需要再调研新的 CNN detail，也不需要先开任何 80K。

---

# 20. Run6 一句话预注册摘要

> **先用完整 SYSU test 对 DeiT ViT4 的 PatchEmbed token 与 block4 token 做零训练 raw ranking gate；只有浅 patch evidence、深 semantic evidence 及两者参数自由融合全部达标，才训练唯一主模型：冻结 TinyViT4-192，删除独立 detail encoder，用 7.648K PTPR 从 PatchEmbed token 逐级 PixelShuffle 恢复 1/8–1/2 局部金字塔，复用 115.267K SGDP 做 difference-first + semantic-gated coarse-to-fine change reconstruction；总参数约 2.100M、FLOPs hard <=2.0G。SYSU 必须 F1>=82.30/IoU>=69.92，随后 LEVIR 必须 F1>=91.50/IoU>=84.33；任一 gate 失败立即终止，不回 detail branch、不做 width/depth/head sweep。**
