# CASA-CD Run4：R4-2 / R4-2b 失败后的下一步决策与可执行方案

> **审查基准**：GitHub `YuqiWang-code/CASA-CD` `main`  
> **本次核对 HEAD**：`b6502f675d0784299fd9b6f2919557053a58c916`（2026-09-30）  
> **重点已读**：`README.md`、`docs/temporary/CASA-CD_Run4_极轻量结构主线_可执行方案.md`、`docs/temporary/CASA-CD_CASAA-v2_A4_Run3审查与主线二启动方案.md`、`models/model/encoder.py`、`models/model/light_detail.py`、`models/model/decoder.py`、`models/model/trainer.py`、`models/model/resnet.py`、`models/train.py`、`models/eval.py`、`models/smoke_test.py`、`analyse/param_breakdown.py`、`analyse/vit_pretrain_audit.py`、`train_scripts/UltraLight/Run4/`。  
> **固定协议**：BCE+Dice；Adam `2e-4`；poly + 200 warmup；80000 steps；batch 16；256×256；seed 16；test-as-val；GPU1；ViT frozen。  
> **最终约束**：工程目标 `≤2.20M effective params`；论文硬口径 `<3M`；SYSU F1 `≥82.30`；LEVIR F1 `≥91.50`；不改 loss、不做多 seed、不把训练技巧当贡献。

---

# 0. 结论先行

## 0.1 现在**不要直接进入 R4-3**

原 Run4 的逐级 gate 是：

```text
R4-1
  ↓
R4-2 detail replacement
  ↓ 只有 detail gate 通过
R4-3 SABI
  ↓
R4-4 DFPD
```

而现在：

```text
R4-2  : ΔF1 = -0.47
R4-2b : ΔF1 = -0.45
```

均没有达到预注册：

```text
ΔF1 >= -0.30
ΔIoU >= -0.50
```

因此旧的“32/64/128 DWConv → SABI → DFPD”顺序应当正式标记为：

> **R4-A 原始 LightDetail 分支未通过 detail gate。**

如果现在无条件进入 SABI，相当于在前一个假设已经失败后继续叠下一个变量，后面即使结果变化，也很难判断是：

- SABI 本身；
- LightDetail；
- adapter；
- 原 FI；
- 原 decoder；

中的哪一个导致。

所以本轮明确建议：

> **不选 a。**

---

## 0.2 当前最可能的问题不是“通道不够”

本次源码审查发现一个比宽度更关键的事实：

### R4-1 的 ResNet detail 是 ImageNet 预训练的

`models/train.py` 默认：

```python
--resnet_pretrained 1
```

`Encoder` 在 `detail_mode='resnet'` 时：

```python
self.resnet = resnet18(pretrained=resnet_pretrained)
```

所以 R4-0 / R4-1 的 detail branch 实际使用：

```text
ImageNet-pretrained ResNet18 C2-C4
```

### R4-2 / R4-2b 的 LightDetail 是随机初始化的

`detail_mode in ('light', 'light48')` 时直接：

```python
self.detail = LightDetail(widths=...)
```

没有任何 ImageNet pretrain。

因此 R4-1 → R4-2 并不是严格意义上的：

```text
ResNet topology → DSConv topology
```

单因素比较，而是同时发生了：

```text
2.7M pretrained residual CNN
          ↓
0.037M random-initialized non-residual DSConv
```

这不是结果无效——它仍然真实回答了“这个 LightDetail 能否直接替代 ResNet”——但对于**失败原因诊断**非常重要。

---

## 0.3 我的原因排序

### 第一位：④“其它”——**预训练先验被一起拿掉 + detail 拓扑发生根本改变**

置信度：**高**

它包含两个相互关联的问题：

1. ResNet C2-C4 有 ImageNet 低层/中层视觉先验；
2. 当前 LightDetail 从随机初始化开始；
3. 当前 LightDetail 每级是普通 DSConv，没有 residual identity；
4. 当前 stride-2 下采样是：

```text
DWConv stride=2
→ PWConv
```

即先对每个输入 channel 独立降采样，再做跨 channel 混合。

在高分辨率 detail 路径中，这可能过早丢失“跨通道组合后的局部结构”。

而 R4-2 → R4-2b：

```text
32/64/128 → 48/96/160
```

只增加宽度，拓扑、初始化、下采样顺序都没有改变，所以 `+0.02 F1` 很符合：

> **瓶颈不是 width capacity。**

---

### 第二位：① LightDetail 的**结构表达方式不足**，不是纯“容量不足”

置信度：**中高**

当前 `LightDetail`：

```text
Stem
→ DSConv
→ DSConv(stride2)
→ DSConv
→ DSConv(stride2)
→ DSConv
```

没有残差。

而原 ResNet detail 的 `BasicBlock`：

```text
3×3
→ BN/ReLU
→ 3×3
→ BN
+ identity/downsample
→ ReLU
```

对局部边缘、纹理和弱变化来说，“identity preservation + repeated spatial mixing”与简单增加通道数不是一回事。

所以：

```text
Light48 ≈ Light32
```

更支持：

> **需要换局部建模方式，而不是继续加 width。**

---

### 第三位：② adapter / 旧 FI / decoder 接口不够自然

置信度：**中等偏低，但值得无训练诊断**

当前 adapter 是：

```python
nn.Conv2d(c_in, c_out, 1, bias=False)
```

没有：

```text
BN
ReLU
```

而原 ResNet 的三个输出全部位于 BasicBlock 的 final ReLU 之后。

因此：

```text
ResNet feature：post-BN/residual/ReLU，非负
Light raw：BN/ReLU，非负
Light adapter output：裸 1×1 linear，允许正负值、无显式归一化
```

确实存在接口统计差异。

但是这个问题我**不排第一**，因为：

- FeatureInjector 内部会对 detail token 做 `LayerNorm(dim2)`；
- 原 decoder 的 difference MLP 第一层后有 BN；
- FI/decoder 每个 run 都是重新初始化、联合训练，并不是把一个“只见过 ResNet feature 的已训练 decoder”直接套到 LightDetail 上。

因此 adapter mismatch 可能贡献部分损失，但它不像“唯一主因”。

---

### 第四位：③ 训练协议

置信度：**低**

理由：

- R4-1 / R4-2 / R4-2b 都是同一个 80K；
- width bump 后没有明显改善；
- 没有证据表明 LightDetail 在 80K 时仍明显未收敛；
- 为 LightDetail 单独延长 steps、换 optimizer、改 LR，会把研究带回训练技巧搜索。

除非最终日志额外显示：

```text
R4-2 和 R4-2b 的 best 都出现在最后 5% steps
且 F1 仍持续单调上升
```

否则本轮**不改训练协议**。

---

# 1. 下一步正式选择

我的推荐不是 a / b / c 中直接盲选一个，而是：

> ## **选 d：先做一次零训练成本的 Detail Interface Audit，然后按结果在 b 与 c 中二选一。**

但基于当前代码事实，我预期最终大概率会走：

> ## **c：重新设计 detail 分支，而不是再做 width sweep。**

推荐流程：

```text
R4-D0：无训练 feature audit
          │
          ├── 明确显示“raw LightDetail 不差，但 adapter 显著破坏”
          │       ↓
          │   R4-2c AdapterAlign（仅 1 个 80K）
          │
          │       ├── 过 gate → R4-3
          │       └── 不过 → PSD-Detail
          │
          └── raw LightDetail 本身已经弱 / 结果模糊
                  ↓
              直接 PSD-Detail
```

**不允许**：

```text
Adapter + BN
Adapter + LN
Adapter + GN
不同 alpha
更多 width
更多 depth
```

连续扫。

---

# 2. R4-D0：先做一个“不训练”的 Detail Interface Audit

建议新增：

```text
analyse/run4_detail_interface_audit.py
```

使用已经训练完的：

```text
R4-1 best checkpoint
R4-2 best checkpoint
R4-2b best checkpoint
```

只在 GPU1 前向。

建议直接跑完整 SYSU test 4000 对，避免再引入抽样随机性。

---

# 3. Audit 具体测什么

对每个尺度：

```text
1/2
1/4
1/8
```

统一先：

```text
AdaptiveAvgPool2d(16×16)
```

变成与 ViT 相同的 256 个位置。

然后对两个时相 feature 计算：

\[
s_i = 1-\cos(f^1_i,f^2_i)
\]

GT：

```text
label → AvgPool16×16 → occupancy
```

统计：

### R4-1

```text
ResNet direct feature
```

### R4-2 / R4-2b

同时统计：

```text
Light raw feature
Light after 1×1 adapter
```

---

## 3.1 Primary 指标

每尺度：

```text
PR-AUC(score, GT_patch_changed)
Spearman(score, GT_occupancy)
Top32 precision
Top32 changed-pixel coverage
```

---

## 3.2 Feature-stat 指标

同时记录：

```text
mean
std
L2 norm
zero fraction
negative fraction
T1/T2 cosine mean
mean |F1-F2| / mean |F|
```

尤其关注：

```text
ResNet direct
vs
Light raw
vs
Light adapted
```

---

# 4. R4-D0 预注册决策规则

## 情况 A：支持 adapter mismatch

至少 **2/3 个尺度**同时满足：

```text
PR-AUC(Light raw) >= PR-AUC(ResNet) - 0.02
```

且：

```text
PR-AUC(Light adapted)
<=
PR-AUC(Light raw) - 0.03
```

并且 Spearman 同方向下降。

### 决策

只做一次：

> `R4-2c ADAPTER_ALIGN`

---

## 情况 B：支持 representation / pretraining / topology 问题

至少 **2/3 个尺度**满足：

```text
PR-AUC(Light raw)
<=
PR-AUC(ResNet) - 0.04
```

且：

```text
Light48 raw 对 Light32 raw 改善 < 0.02
```

### 决策

跳过 adapter 训练。

直接进入：

> `R4-2d PSD_DETAIL`

---

## 情况 C：Audit 模糊

不满足 A/B。

### 决策

仍然优先：

> `PSD_DETAIL`

原因：

- 已有 2 个 width run 都失败；
- 源码已确认存在 pretrained-vs-random confound；
- adapter-only 再做 80K 的信息增益低于结构重设计。

---

# 5. 如果 Audit 明确支持 adapter mismatch：唯一允许的 R4-2c

只修改 adapter：

当前：

```python
Conv1x1
```

改为：

```text
Conv1×1
→ BatchNorm2d
→ ReLU
```

三个尺度全部相同。

其余：

```text
LightDetail = 32/64/128
ViT4 frozen
原 FI
原 decoder
80000 steps
seed16
```

全部不变。

---

## 5.1 参数增量

BN learnable params：

```text
2 × (64+128+256)
= 896
```

即：

```text
< 0.001M
```

可忽略。

---

## 5.2 R4-2c Gate

仍使用原 gate，**不降低标准**。

R4-1：

```text
F1  = 82.77
IoU = 70.61
```

所以 R4-2c 必须：

```text
F1  >= 82.47
IoU >= 70.11
```

才叫通过。

---

## 5.3 R4-2c 停止规则

若没有达到上述两个条件：

> **adapter 分支永久停止。**

不做：

```text
LN
GN
SiLU
LeakyReLU
更多 adapter 层
```

直接进入新 detail 架构。

---

# 6. 我更推荐的新 detail：PSD-Detail

如果 Audit 不明确支持 adapter，首选：

> # **PSD-Detail：Pretrained Stem + Residual Depthwise Detail Pyramid**

核心思想不是“再加一点通道”，而是：

> **把有限的 detail 预算集中在最浅层的视觉先验与残差细节保真上，而不是继续扩大深层 DWConv 宽度。**

---

# 7. PSD-Detail 为什么比当前 LightDetail 更匹配问题

当前 LightDetail 的两个关键弱点：

### 弱点 1：完全随机的输入 stem

```text
3×3 s2 3→32
```

在 R4-2 中从零学习。

而 R4-1 的 ResNet：

```text
7×7 s2 3→64
```

是 ImageNet pretrained。

---

### 弱点 2：stride-2 时先 DW，再 PW

当前：

```text
DWConv stride2
→ PWConv
```

每个输入 channel 被独立降采样以后，才发生跨 channel mixing。

PSD 改为：

```text
PW channel mixing
→ DW spatial downsampling
```

即：

> **先形成跨通道局部组合，再进行空间降采样。**

这与 detail preservation 的目标更一致。

---

### 弱点 3：无 residual identity

PSD 在 1/2、1/4 两级使用 residual DS block：

```text
x + DS(x)
```

防止 detail branch 在连续低成本卷积中把原始边缘/纹理完全重写。

---

# 8. PSD-Detail 定稿结构

输入：

```text
B×3×256×256
```

---

## Stage 0：Pretrained Detail Stem

**直接复用 ImageNet ResNet18 的：**

```text
conv1 7×7, stride2, 3→64
bn1
ReLU
```

输出：

```text
64×128×128
```

只保留这部分 ResNet 权重。

没有：

```text
maxpool
layer1
layer2
layer3
layer4
fc
```

所以不会重新引入重型 ResNet。

---

## Stage D2：ResidualDS-64

```text
DWConv3×3 64→64, s1
BN
ReLU
PWConv1×1 64→64
BN
+ identity
ReLU
```

输出：

```text
D2 = 64×128×128
```

---

## Stage D4：MixDown 64→128

不要用当前 LightDetail 的：

```text
DW stride2 → PW
```

改为：

```text
PWConv1×1 64→128
BN
ReLU
DWConv3×3 128→128, stride2, groups=128
BN
ReLU
```

输出：

```text
128×64×64
```

然后：

### ResidualDS-128

```text
DW3×3 128
BN/ReLU
PW1×1 128→128
BN
+ identity
ReLU
```

得到：

```text
D4 = 128×64×64
```

---

## Stage D8：MixDown 128→256

```text
PW1×1 128→256
BN
ReLU
DW3×3 256→256, stride2
BN
ReLU
```

输出：

```text
D8 = 256×32×32
```

这里**不再加 residual block**。

理由：

- 控制参数；
- 控制 FLOPs；
- D8 后马上进入 16×16 semantic alignment；
- 深层 detail capacity 已经被 R4-2b 证明不是主要瓶颈。

---

# 9. PSD-Detail 参数预算

| 部分 | Params |
|---|---:|
| Pretrained Conv7×7 3→64 + BN | **9,536** |
| ResidualDS 64→64 | **4,928** |
| MixDown 64→128 | **9,856** |
| ResidualDS 128→128 | **18,048** |
| MixDown 128→256 | **36,096** |
| **PSD-Detail 总计** | **78,464 ≈ 0.078M** |

所以：

```text
0.078M << 0.25M
```

---

# 10. 为什么这个结构特别适合接 16×16 ViT

ViT4 已经承担：

```text
16×16
global / semantic
```

detail branch 不应该再复制另一个深层 semantic backbone。

PSD 的容量安排是：

```text
高分辨率 1/2：
  pretrained 64ch + residual preservation

1/4：
  128ch，局部结构进一步聚合

1/8：
  256ch，只做一次廉价下采样
  ↓
接 1/16 semantic fusion
```

因此它的逻辑是：

> **越接近原图，越强调 pretrained local-detail preservation；越接近 ViT，越减少重复建模。**

这比：

```text
32 →64→128
```

单纯“越来越宽”更符合双路径分工。

---

# 11. PSD-Detail 计算量

设计阶段 MAC 粗估：

```text
单时相 ≈ 0.58G MAC
双时相 ≈ 1.16G MAC
```

正式仍必须用实际代码的 `fvcore` 报 FLOPs。

相比原 ResNet C2-C4，依然属于大幅降低。

---

# 12. PSD-Detail 的另一个优势：**不需要 adapter**

输出直接就是：

```text
64 / 128 / 256
```

与当前原版：

```text
FeatureInjector(dim2=[64,128,256])
Decoder(in_dim=[64,128,256,192])
```

完全一致。

所以 `R4-2d` 可以做到真正干净：

```text
唯一变量 = detail branch
```

不会再混入：

```text
32→64 adapter
64→128 adapter
128→256 adapter
```

这比继续修 adapter 更适合作为论文实验。

---

# 13. PSD-Detail 的预训练方式

不要把整个 ResNet18 注册进最终模型。

只在初始化阶段读取 ImageNet ResNet18：

```python
ref = resnet18(pretrained=True)
```

把：

```text
ref.conv1
ref.bn1
```

权重原位复制进：

```text
PSDDetail.stem_conv
PSDDetail.stem_bn
```

然后释放临时模型。

最终 `state_dict()` 中不得出现：

```text
layer1
layer2
layer3
layer4
fc
```

---

## 13.1 Smoke 要求

新增：

```text
T-PSD0
```

断言：

```text
stem conv max_abs_diff == 0
stem BN weight/bias/running_mean/running_var == source
```

并断言：

```text
PSD detail params == 78,464
```

---

# 14. PSD-Detail 训练方式

和 R4-1/R4-2 完全一致：

```text
ViT4 frozen
PSD stem trainable
PSD random blocks trainable
old FI trainable
old decoder trainable
BCE+Dice
lr2e-4
80K
seed16
GPU1
```

不对 pretrained stem 单独设 LR。

理由：

> 原 ResNet detail 本身就是按同一训练协议更新的；本轮不引入新的训练技巧变量。

---

# 15. R4-2d Gate

仍然坚持原 R4-2 gate。

相对：

```text
R4-1 = 82.77 / 70.61
```

要求：

```text
F1  >= 82.47
IoU >= 70.11
```

---

## 强通过

如果达到：

```text
F1  >= 82.57
IoU >= 70.25
```

说明在：

```text
~0.078M detail
```

下已经只损失 ≤0.20 F1，后续 SABI/DFPD 有足够 headroom。

---

## 失败

任一：

```text
F1 < 82.47
或
IoU < 70.11
```

即：

> **停止当前“轻量 detail + ViT4 双路径”路线。**

不要再做：

```text
PSD 96/192
PSD deeper
更多 pretrained block
更多 adapter
width sweep
```

此时直接结束 Run4 detail rescue，下一阶段应重新分配架构预算，而不是继续救 detail branch。

---

# 16. 为什么我不建议直接“接受 −0.45 再赌 SABI”

R4-2 已经：

```text
F1 = 82.30
```

恰好等于你设定的**最终论文候选门槛**。

但它此时还保留：

```text
原重型 FeatureInjector
原重型 decoder
```

有效参数仍：

```text
5.492M
```

接下来还必须从：

```text
5.49M
→
~2.1M
```

再砍掉约 3.3M。

如果现在就把 `−0.45` 当成“已经接受的成本”，意味着：

> 后面两个最激进的轻量替换必须总体零损失甚至涨点。

这个风险太高。

所以必须先把 detail replacement 至少修回原：

```text
≤0.30 F1 loss
```

gate。

---

# 17. R4-3 更新方案：只有 detail gate 通过才启动

如果 R4-2c 或 R4-2d 通过，才进入：

> **R4-3 SABI**

若 PSD-Detail 通过，则 SABI 输入还是：

```text
64/128/256
```

因此原定 SABI 基本无需改机制。

---

## R4-3 新判据

旧判据：

```text
relative drop <=0.30
```

现在建议收紧，因为已经知道最终 SYSU 目标是 82.30。

预注册：

```text
R4-3 vs detail-pass model:
F1 drop <= 0.20
IoU drop <= 0.35
```

同时必须：

```text
absolute F1 >= 82.40
```

才允许进入 R4-4。

---

# 18. R4-4 新判据

R4-4：

```text
SABI
→ DFPD
```

最终必须同时满足：

```text
effective params <= 2.20M
SYSU F1 >= 82.30
SYSU IoU >= 69.92
```

三者缺一不可。

旧的：

```text
Gate-A >=81.50
```

现在只保留为“模型没有崩”的 sanity threshold，不再作为继续论文路线的研究门槛。

原因：

> R4-0 已经给出了 83.14 的健康参考，81.5 的相对损失过大，不足以支撑“极轻量高精度”定位。

---

# 19. 如果最终用 PSD-Detail，预计参数预算

### TinyViT4

```text
1,976,832
```

### PSD-Detail

```text
78,464
```

### SABI（输入 64/128/256，48d）

估算：

```text
45,504
```

### DFPD（detail 64/128/256）

估算：

```text
82,184
```

### 合计

```text
1,976,832
+ 78,464
+ 45,504
+ 82,184
= 2,182,984
≈ 2.183M
```

因此仍满足：

```text
≤2.20M
```

并保留约：

```text
17K
```

工程余量。

正式值必须以最终代码参数统计为准。

---

# 20. R4-0 = 83.14 后，原 baseline 82.48 应该怎么用

这个结果会**明确改变**原 baseline 的地位。

---

## 20.1 不能再用 82.48 作为 Run4 的主内部 baseline

原：

```text
baseline Run1 SYSU = 82.48
```

来自：

```text
官方统一 lr=2e-4
ViT 可训练
且后续已证实存在 ViT collapse / partial collapse 风险
```

而：

```text
R4-0 = 83.14
```

是：

```text
corrected DeiT loader
full12
vanilla attention
frozen healthy ViT
```

所以所有 Run4 结构结论都应该相对：

> **R4-0 Healthy Full12 Reference = 83.14**

---

## 20.2 R4-1 的正确表述

可以说事实：

```text
R4-1 82.77 > 原 Run1 baseline 82.48
```

但不能用它证明：

> “4-block 比 12-block 更好。”

公平结构比较是：

```text
R4-0 full12 healthy = 83.14
R4-1 prefix4 healthy = 82.77
Δ = -0.37
```

所以论文中的真实结论应是：

> **将健康的 12-block pretrained ViT 截断为前 4 blocks，只损失 0.37 F1。**

这是一个很强的 lightweight 证据。

---

# 21. 建议以后把“baseline”拆成两个名字

## CVT-T Official-Protocol Reproduction

```text
SYSU 82.48
```

用途：

- 说明最初复现；
- 对齐原 ChangeViT 训练协议；
- 同时报告后续发现的 ViT instability。

---

## HR-12：Healthy Reference

```text
R4-0
SYSU 83.14
```

用途：

- Run4 所有内部消融；
- 参数裁剪保真率；
- shallow-ViT 实验；
- 最终 ultra-light accuracy-retention 计算。

---

# 22. 最终论文中对 ChangeViT 的建议表述

不要写：

> “我们的 4-block 模型优于 baseline 82.48，因此裁剪 ViT 提升性能。”

更严谨：

> “Under a corrected and frozen pretrained ViT reference, reducing the transformer depth from 12 to 4 blocks decreases SYSU F1 by only 0.37 points while removing approximately 3.56M parameters, indicating substantial depth redundancy under the dual-path change-detection framework.”

这才是 R4-0 / R4-1 真正支持的结论。

---

# 23. 更新后的 Run4 实验矩阵

| ID | 结构 | 是否训练 | 唯一问题 | Gate |
|---|---|---:|---|---|
| 已有 R4-0 | full12 + ResNet + old FI/decoder | 已完成 | healthy reference | 83.14 |
| 已有 R4-1 | ViT4 + ResNet + old FI/decoder | 已完成 | depth redundancy | **PASS** |
| 已有 R4-2 | ViT4 + Light32 + adapters + old FI/decoder | 已完成 | original light detail | FAIL |
| 已有 R4-2b | ViT4 + Light48 + adapters + old FI/decoder | 已完成 | width fallback | FAIL |
| **R4-D0** | feature interface audit | **否** | raw detail vs adapter vs ResNet | 按 §4 |
| R4-2c | Light32 + Conv-BN-ReLU adapters | 条件运行 | adapter mismatch | F1≥82.47 & IoU≥70.11 |
| **R4-2d** | **PSD-Detail 0.078M** + old FI/decoder | 主推荐 | pretrained/local topology | F1≥82.47 & IoU≥70.11 |
| R4-3 | detail-pass + SABI + old decoder | 条件运行 | lightweight injector | rel drop≤0.20 + abs F1≥82.40 |
| R4-4 | ViT4 + detail-pass + SABI + DFPD | 条件运行 | final `<2.20M` | **F1≥82.30 / IoU≥69.92 / ≤2.20M** |
| R4-4 LEVIR | final | 条件运行 | sparse-change transfer | **F1≥91.50** |
| R4-4 CDD/WHU | final | 条件运行 | final generalization | LEVIR pass 后启动 |

---

# 24. 完整停止规则

## Stop-1：R4-D0

如果明确：

```text
raw LightDetail 已明显弱于 ResNet
```

则：

> 不做 adapter 80K，直接 PSD。

---

## Stop-2：R4-2c

如果 adapter 诊断 run：

```text
F1 <82.47
或 IoU <70.11
```

则：

> 永久停止 adapter 修补。

---

## Stop-3：R4-2d PSD

如果：

```text
F1 <82.47
或 IoU <70.11
```

则：

> **停止当前轻量 detail 路线。**

不启动 SABI。

也不启动 DFPD。

原因：

> 前端 detail 已经没有足够性能 margin，继续连续砍 FI/decoder 不具有合理成功概率。

---

## Stop-4：R4-3

若：

```text
relative F1 drop >0.20
或
relative IoU drop >0.35
或
absolute F1 <82.40
```

则：

> SABI 路线停止，不进入 DFPD。

---

## Stop-5：R4-4

若最终：

```text
Params >2.20M
或
F1 <82.30
或
IoU <69.92
```

则不能称为 Run4 paper candidate。

---

## Stop-6：LEVIR

若：

```text
LEVIR F1 <91.50
```

则不直接宣称“跨数据集轻量 SOTA”。

优先做 failure analysis，而不是立刻补 CDD/WHU 包装平均分。

---

# 25. 如果 PSD 也失败，下一步不是再救 DWConv

此时证据会变成：

```text
Light32 FAIL
Light48 FAIL
Adapter（若运行）FAIL
PSD FAIL
```

已经足以说明：

> 该双路径中 `<0.25M` custom detail branch 无法在当前 old FI/decoder 接口下保留足够精度。

这时应该结束：

```text
自定义 DW / residual-DW detail 搜索
```

进入新的 Run5 方向，例如：

```text
重新分配 semantic/detail 参数预算
或
使用有 ImageNet 预训练的成熟超轻 backbone 子层
或
去掉独立 detail branch、让 lightweight decoder 从浅层 token/image feature 重建细节
```

但不要在 Run4 中继续堆第 4、第 5 个 detail variant。

---

# 26. 训练协议是否需要改变

本轮答案：

> **不改变。**

所有新增正式 run：

```text
BCE+Dice
Adam 2e-4
poly
warmup200
80000 steps
batch16
seed16
ViT frozen
GPU1
```

---

## 唯一要补查的训练证据

从每个 `train_log.txt` 记录：

```text
best_epoch
last_epoch F1
best-last gap
最后 20% steps 的 F1 trend
```

如果：

```text
R4-2 与 R4-2b 都在最后 5% steps 刷新 best
且末段仍持续上升
```

可把“80K 可能不足”记为 missing evidence。

但**本轮仍不延长 steps**。

---

# 27. 需要新增的代码

## 新增

```text
analyse/run4_detail_interface_audit.py
models/model/psd_detail.py
```

---

## 修改

```text
models/model/encoder.py
models/model/trainer.py
models/train.py
models/eval.py
models/smoke_test.py
```

新增：

```text
--detail_mode psd
```

---

# 28. `psd_detail.py` 推荐结构伪代码

```python
class ResidualDS(nn.Module):
    def __init__(self, c):
        self.dw = Conv3x3(c, c, groups=c, bias=False)
        self.bn1 = BN(c)
        self.pw = Conv1x1(c, c, bias=False)
        self.bn2 = BN(c)

    def forward(self, x):
        y = relu(self.bn1(self.dw(x)))
        y = self.bn2(self.pw(y))
        return relu(x + y)


class MixDown(nn.Module):
    def __init__(self, cin, cout):
        self.pw = Conv1x1(cin, cout, bias=False)
        self.bn1 = BN(cout)
        self.dw = Conv3x3(cout, cout, stride=2,
                         groups=cout, bias=False)
        self.bn2 = BN(cout)

    def forward(self, x):
        x = relu(self.bn1(self.pw(x)))
        x = relu(self.bn2(self.dw(x)))
        return x


class PSDDetail(nn.Module):
    def __init__(self, pretrained_resnet_state):
        # exact pretrained stem
        self.conv1 = Conv7x7(3,64,stride=2)
        self.bn1 = BN(64)

        self.d2_refine = ResidualDS(64)
        self.down4 = MixDown(64,128)
        self.d4_refine = ResidualDS(128)
        self.down8 = MixDown(128,256)

        load_exact_resnet_stem(...)

    def forward(self, x):
        x = relu(self.bn1(self.conv1(x)))
        d2 = self.d2_refine(x)
        d4 = self.d4_refine(self.down4(d2))
        d8 = self.down8(d4)
        return d2, d4, d8
```

---

# 29. PSD smoke

至少加：

### T-PSD0：pretrained stem

```text
conv1 max_abs_diff == 0
bn1 affine == source
bn1 running stats == source
```

### T-PSD1：shape

```text
D2 = B×64×128×128
D4 = B×128×64×64
D8 = B×256×32×32
```

### T-PSD2：params

```text
PSD detail == 78,464
```

### T-PSD3：no hidden ResNet

state dict 中不存在：

```text
layer1
layer2
layer3
layer4
fc
```

### T-PSD4：gradient

```text
stem grad finite
RDS grad finite
MixDown grad finite
decoder grad finite
ViT grad None
```

### T-PSD5：frozen ViT checksum

训练 dry run 前后：

```text
max_abs_change == 0
```

---

# 30. Dry run

正式 R4-2d 前：

```text
SYSU
200–500 iter
GPU1
```

路径：

```text
/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/UltraLight/R4_PSD/SYSU-CD-256/

/home/yqwang/outputs/CASA-CD/_dryrun/UltraLight/R4_PSD/SYSU-CD-256/
```

验收：

```text
loss finite
shape correct
no OOM
ViT checksum stable
PSD stem correctly loaded
checkpoint resume
eval architecture identical
```

---

# 31. 正式脚本建议

新增：

```text
train_scripts/UltraLight/Run4/
├── audit_R4_D0_detail_interface_SYSU.sh
├── train_R4_2c_ADAPTER_ALIGN_SYSU.sh      # 仅 audit A 条件触发
└── train_R4_2d_PSD_DETAIL_SYSU.sh         # 主推荐
```

不要建立：

```text
run_all_rescue.sh
```

因为每一步必须人工读 gate 后决定。

---

# 32. R4-2d 路径

建议：

```text
checkpoint:
/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_2d_PSD_DETAIL/SYSU-CD-256/

log:
/home/yqwang/outputs/CASA-CD/UltraLight/Run4/R4_2d_PSD_DETAIL/SYSU-CD-256/train_log.txt
```

---

# 33. 结果读取纪律

继续只认：

```text
同一个 train_log.txt
最后一个完整
=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

R4-D0 的 audit 不是正式 F1 结果。

---

# 34. 需要补充的证据

当前仓库 README 已有：

```text
F1
IoU
Params
FLOPs
```

但为了诊断 R4-2 的具体错误类型，还建议从真实最终 TEST block 补：

```text
Recall
Precision
OA
Kappa
```

尤其要看：

```text
R4-1 → R4-2
```

到底主要是：

```text
Recall 掉
还是 Precision 掉
```

解释不同：

- Recall 明显掉：更支持细节/弱变化表达不足；
- Precision 明显掉：更支持 feature fusion / background discrimination 不稳。

---

# 35. R4-0 健康参考对最终论文还有一个影响

如果最终 R4-4 成功并准备正式写论文，建议补：

```text
R4-0 Healthy Full12
```

在：

```text
LEVIR
CDD
WHU
```

的结果。

原因：

如果最终论文要写：

> “11.754M → 2.18M，精度仅下降 X”

那么 X 应该相对**健康、同 loader、同 frozen protocol**的 full12 reference，而不是相对已经发现 ViT collapse 的历史 Run1。

现在不需要立即补。

只有最终 `<3M` 模型在 SYSU+LEVIR 通过后再补。

---

# 36. 对外 SOTA 定位不变

最终仍然优先统一协议对标：

```text
RFANet     2.86M
SeCoR      2.50M
Lighter    1.10M
CGLNet     0.99M
```

但现在有一个更清楚的目标：

### 如果 PSD 最终方案成立

预计：

```text
~2.183M
```

正好处于：

```text
Lighter 1.10M
       ↓
CASA-CD ~2.18M
       ↓
SeCoR 2.50M
RFANet 2.86M
```

之间。

因此论文不是追求“绝对最小参数”，而是争取：

> **2M 级参数下更高的 F1/IoU Pareto 点。**

---

# 37. 更新后的论文机制故事

如果 PSD → SABI → DFPD 成功，主线可以更统一。

## 第一贡献：Shallow Pretrained Semantic Path

R4-0 → R4-1 已直接证明：

```text
12 blocks
→ 4 blocks
仅 -0.37 F1
```

说明双路径 CD 中存在明显 transformer depth redundancy。

---

## 第二贡献：Detail-Prior Budget Reallocation

R4-2/2b 的失败反而提供了一个很有价值的机制观察：

> **把 detail path 单纯改成随机 DWConv、甚至增加宽度，并不能保持性能。**

因此 PSD 不是“又加一个 CNN”，而是：

> **在极低预算下只保留最浅层 pretrained visual prior，并将后续多尺度构建交给 residual depthwise operations。**

如果 PSD 成功，这个故事比“32/64/128 DSConv”更有论文价值。

---

## 第三贡献：SABI + Difference-First Reconstruction

只有其消融 gate 通过后再保留。

---

# 38. A1 / CASAA 在论文里的位置不变

CASAA change-aware 不再回主方法。

A1 继续作为：

```text
analysis / ablation
```

可用于说明：

> late-stage context 具有冗余，Full Query 并不要求完整 K/V。

但最终模型若不用 A1，就不要把它列为 Contribution 1。

---

# 39. 最终推荐决策树

```text
已有：
R4-0 83.14
  ↓
R4-1 82.77 PASS
  ↓
R4-2 82.30 FAIL
R4-2b 82.32 FAIL
  ↓
R4-D0：无训练 detail-interface audit
  │
  ├── raw Light≈ResNet，但 adapter 明显破坏
  │      ↓
  │   R4-2c AdapterAlign
  │      │
  │      ├── F1>=82.47 & IoU>=70.11
  │      │       ↓
  │      │     R4-3
  │      │
  │      └── FAIL
  │              ↓
  │          PSD-Detail
  │
  └── raw Light 本身弱 / audit 模糊
         ↓
      R4-2d PSD-Detail
         │
         ├── F1>=82.47 & IoU>=70.11
         │       ↓
         │    R4-3 SABI
         │       │
         │       ├── rel F1 drop<=0.20
         │       │   且 abs F1>=82.40
         │       │       ↓
         │       │    R4-4 DFPD
         │       │       │
         │       │       ├── <=2.20M
         │       │       ├── SYSU>=82.30
         │       │       └── IoU>=69.92
         │       │               ↓
         │       │          LEVIR >=91.50
         │       │               ↓
         │       │          CDD + WHU
         │       │
         │       └── FAIL → stop SABI path
         │
         └── FAIL
                 ↓
           STOP lightweight-detail Run4
           → 另开 Run5 架构预算重分配
```

---

# 40. 立即执行顺序

1. **不要启动 R4-3。**
2. 先新增 `run4_detail_interface_audit.py`。
3. 对 R4-1 / R4-2 / R4-2b 同一 SYSU test 做完整 feature audit。
4. 按 §4 的预注册条件判断是否值得做 adapter-align。
5. 若不是明显 adapter mismatch，直接实现 **PSD-Detail**。
6. PSD 做 exact pretrained stem smoke。
7. 统计：
   ```text
   Params = 78,464
   ```
8. 做 200–500 iter dry run。
9. 只启动一个 `R4-2d PSD_DETAIL / SYSU / 80K`。
10. 从最后完整 TEST RESULTS 读六项指标。
11. 必须：
    ```text
    F1>=82.47
    IoU>=70.11
    ```
    才能进入 SABI。
12. PSD 未过，停止当前 custom light-detail 路线；不再宽度/深度 sweep。
13. PSD 通过后才实现 R4-3。
14. 最终 R4-4 必须：
    ```text
    <=2.20M
    SYSU>=82.30
    IoU>=69.92
    ```
15. SYSU 通过后跑 LEVIR，正式目标：
    ```text
    >=91.50
    ```
16. LEVIR 通过后才补 CDD/WHU 与外部轻量 SOTA 统一协议对照。

---

# 41. 最终判断

当前两次失败**不意味着“轻量 detail 不可行”**，但已经足以否定：

> **“随机初始化的普通 DSConv 三尺度，只靠增加通道宽度，就能无损替代 pretrained ResNet detail。”**

源码审查进一步发现，R4-1 与 R4-2 之间还同时移除了 ResNet 的 ImageNet detail prior，这使得继续做 `48/96/160 → 更宽` 的意义更低。

因此最合理的下一步不是：

```text
继续 R4-3
```

也不是：

```text
再调训练 LR
```

而是先用无训练 audit 排除 adapter 这一最便宜的解释；如果没有明确证据，就把 detail 设计改成：

> **PSD-Detail：保留极少量 pretrained shallow visual prior + residual depthwise multi-scale pyramid。**

它只有约：

```text
0.078M detail params
```

最终模型预计仍能控制到约：

```text
2.183M
```

同时不再依赖额外 adapter，且直接解决当前 LightDetail 最值得怀疑的三个问题：

```text
随机 stem
无 residual
先 DW 下采样、后 channel mixing
```

这条路线的**信息增益、方法动机、参数预算和论文可解释性**都明显高于继续在当前 LightDetail 上做宽度或训练超参搜索。
