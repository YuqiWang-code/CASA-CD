# CASA-CD Run5：成熟超轻预训练 Detail + 语义门控差分金字塔方案

> **项目**：CASA-CD — 极轻量遥感二值变化检测  
> **审查基准**：GitHub `YuqiWang-code/CASA-CD` `main`  
> **本次核对 HEAD**：`c5d1b78fae33311fc6c0640fde08e0e796e1843c`（2026-10-01）  
> **事实权威读取顺序**：README → 交接文档 → R4-2失败后方案 → R4-D0审计 → R4-2d失败终止 → Run4原始设计 → Run4 README → 当前 `encoder.py / psd_detail.py / light_detail.py / decoder.py / trainer.py`。  
> **任务**：CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256，全监督二值变化检测，A/B/label，`gray >= 128`。  
> **固定协议**：BCE+Dice；Adam(`lr=2e-4, betas=(0.9,0.99), wd=1e-4`)；poly(power=0.9)+200 iter warmup；80000 steps；batch16；256×256；seed16；test-as-val；GPU1；ViT frozen。  
> **正式结果纪律**：只认单个 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===`。  
> **最终硬约束**：effective inference params `<3M`；工程目标 `≤2.20M`；Run5 目标 **≈2.10M**；SYSU F1 `≥82.30`；LEVIR F1 `≥91.50`。  
> **禁止项**：不改 loss；不做多 seed；不做超参 sweep；不把训练技巧、量化、剪枝、蒸馏包装成核心创新；每个正式实验只有一个结构变量，并带预注册 gate。

---

# 0. 结论先行

Run4 应按已有 Stop-3 **彻底结束**，不再救 LightDetail / PSDDetail，也不再启动旧 SABI / DFPD。

Run5 唯一主推方向定为：

> ## **候选方向 2：成熟超轻 ImageNet 预训练子层作为 Detail**
>
> 保留已经被 R4-0→R4-1 证明有效的 **TinyViT4-192**，  
> 把失败的自定义 LightDetail / PSDDetail 替换为 **MobileNetV3-Small 的极浅预训练 prefix（features 0–3）**，  
> 再把原 3.44M 的 FeatureInjector+Decoder 整体替换成一个统一的  
> **SGDP：Semantic-Guided Difference Pyramid（语义门控差分金字塔）**。

最终设计：

```text
TinyViT4-192 frozen                ≈ 1.977M
MobileNetV3-Small pretrained P3   ≈ 0.0105M
SGDP head                         ≈ 0.1153M
------------------------------------------------
TOTAL                             ≈ 2.103M
```

设计阶段双时相 256×256 粗估：

```text
~1.6–1.8G MAC/FLOPs 量级
```

正式值以服务器 `fvcore` + unsupported ops + batch1 latency 为准。

---

# 1. 为什么 Run5 **不再继续压缩 ViT4**

这是本轮最重要的预算判断。

当前结构事实：

```text
TinyViT4 = 1.976832M
工程预算 = 2.20M
外围预算 ≈ 0.223M
```

看上去外围预算极紧。

但 Run4 已经证明两件事：

### 事实 A：ViT4 是当前唯一已经通过 gate 的轻量组件

```text
R4-0 full12 healthy : 83.14
R4-1 prefix4        : 82.77
ΔF1                 : -0.37
参数减少            : ~3.56M
```

这说明：

> **4-block pretrained semantic path 是有直接正证据的。**

---

### 事实 B：外围失败的不是“0.22M 数学上装不下”，而是过去的 detail 表征质量不够

Run4 三次 detail 失败：

```text
Light32  82.30
Light48  82.32
PSD      82.02
```

R4-D0 又直接显示：

```text
Light 1/4, 1/8 raw PR-AUC ≈ 0.32
ResNet 1/4, 1/8        ≈ 0.63 / 0.65
```

所以当前证据更像：

> **自定义极小 detail 的深层变化表征坏了。**

而不是：

> **只要外围 <0.22M 就一定做不好。**

---

### Run5 决策

因此本轮**不同时再砍 ViT**。

理由：

1. 继续把 ViT4→ViT2，会破坏目前唯一已经过 gate 的结构资产；
2. 这会额外消耗至少一个 SYSU 80K，只回答“更浅 ViT 行不行”；
3. 成熟 MobileNetV3-Small 的 1/2–1/8 prefix 实际只有约 **10.5K 参数**；
4. 新 head 可以控制在约 **115K**；
5. 整体已经可以在 **2.103M** 内完成，不需要为了预算强行再动 ViT。

所以：

> **Run5 先验证“2M 中绝大多数预算给已验证 ViT4，外围用成熟预训练 micro-detail + 极简 change head”这一分配。**

若 Run5 失败，才把“进一步改变 ViT 参数预算”升级为 **Run6**；Run5 内不再并行第二条路线。

---

# 2. 三个 Run5 候选方向的最终取舍

| 候选 | 本轮决策 | 原因 |
|---|---|---|
| ① 重新分配 semantic/detail 参数预算（继续砍 ViT） | **暂不选** | ViT4 已过 gate；当前无需再砍也能做到 ~2.10M |
| ② 成熟超轻 pretrained 子层做 detail | **唯一主推** | 直接针对 Run4“自定义深层 detail 表征崩坏”的证据；参数极低 |
| ③ 去掉独立 detail branch | **不选** | 当前证据没有证明 plain token path 能保小目标/边界；ChangeViT 与 R4-D0 都支持 detail 有价值 |

如果 ② 在预注册 gate 中失败：

> **永久停止“保留 ViT4 + 独立 detail”这条 Run5 路线；下一轮只允许进入候选 ① 的预算重分配，不回头做新的 detail branch。**

---

# 3. PSD 比 LightDetail 低 0.28 F1：4 个候选原因排序

先区分两件事：

- `PSD < LightDetail` 的 **0.28 F1 差距**；
- `PSD < R4-1` 的 **0.75 F1 总损失**。

0.28 可能受到单 seed 波动影响，但 0.75 的路线级失败不能靠噪声解释。

---

## 3.1 排名 1：① LightDetail 的可学习 adapters 提供了有效接口自由度

### 当前支持证据

R4-2 中：

```text
LightDetail 本体     ≈ 37K
3 个 1×1 adapters   ≈ 43K
```

即 Light32 实际上大约有一半“detail 预算”花在：

> **每尺度的 learnable channel-basis remapping**

而 PSD：

```text
≈78K
```

几乎全部花在 stem / spatial / residual 建模上，直接裸输出 64/128/256。

这意味着两者总参数接近，但**参数位置完全不同**。

R4-D0 中 raw→adapted 的 cosine PR-AUC 变化不大，只能说明：

> adapter 没有明显提高 `1-cos` 的单一 ranking 指标。

它不能证明：

> adapter 对下游 FI / difference MLP 的 channel basis 没作用。

因此在四条候选里，**①目前最值得怀疑**。

---

## 3.2 排名 2：③ PSD 深层 channel geometry / downstream interface 不匹配

### 支持点

PSD 的 1/4、1/8 虽然尺寸和通道数与 ResNet 一致：

```text
128 / 256
```

但“通道数相同”并不等于“channel basis 相同”。

原 FI：

```text
LayerNorm(detail)
→ K/V linear
```

原 decoder 又直接对：

```text
[x1, x2, |x1-x2|]
```

做卷积。

所以 downstream 同时依赖：

- normalized token geometry；
- raw spatial channel organization。

PSD 没有显式接口映射，确实可能在这两处吃亏。

不过：

- FI 有 LayerNorm；
- difference MLP 首层后有 BN；
- FI/decoder 本身也是每个 run 重训；

所以我把 ③ 放在第二，不放第一。

---

## 3.3 排名 3：④ 单 seed 噪声

对“PSD 82.02 vs Light 82.30”这 **0.28** 差距，噪声可能解释一部分。

但它解释不了：

```text
PSD vs R4-1 = -0.75
```

也解释不了三条轻量 detail 路线都没有过原 gate。

所以：

> 噪声是残余不确定性，不是路线失败的首因。

由于你明确不做多 seed，本项目后续也不为它增加训练。

---

## 3.4 排名 4：② pretrained stem 在统一 lr 下被重写

这个假设最容易检查，但我的先验排序最低。

原因：

R4-1 的 ResNet detail：

```text
也是 ImageNet pretrained
也是同一个 Adam 2e-4 + wd=1e-4
也是可训练
```

却得到：

```text
82.77
```

所以“统一 lr 会把 pretrained stem 本身必然毁掉”并没有被现有结果支持。

除非 PSD 因为整体结构不同导致 stem 梯度明显更大，否则它不是最优先解释。

---

# 4. 一个更重要的总体归因：R4-D0 的 proxy 与端到端 F1 并非等价

这是四条候选之外、我认为必须写入研究记录的一点。

R4-D0 用：

```text
1 - cosine(f1,f2)
vs
GT patch occupancy
```

评价 detail feature。

它适合回答：

> “这个 feature 自身是否能形成 change ranking？”

但原 ChangeViT downstream 不是直接拿该 ranking 做预测。

它实际经过：

```text
FeatureInjector
→ per-scale [x1,x2,|x1-x2|] difference MLP
→ cascade decoder
```

所以：

> **单独 PR-AUC 高，不保证 feature 在现有 downstream 中可被利用。**

PSD 的失败说明：

> R4-D0 是很好的筛查工具，但不能再把它当成“最终 F1 的充分条件”。

Run5-D0 仍使用无训练 audit，但只把它用于**淘汰明显不合格候选**，不再用它预测一定涨点。

---

# 5. Run5-D0：零训练成本的“一次性 postmortem + 新候选筛查”

新增：

```text
analyse/run5_postmortem_and_mobile_audit.py
```

使用：

```text
R4-1 best checkpoint
R4-2 best checkpoint
R4-2d best checkpoint
ImageNet ResNet18 source weight
ImageNet MobileNetV3-Small source weight
完整 SYSU test 4000 pairs
```

全部 GPU1 forward，不训练、不改 checkpoint。

---

# 6. Run5-D0-A：检查 PSD pretrained stem 是否漂移

分别对：

```text
R4-1 ResNet stem
R4-2d PSD stem
```

和同一个 ImageNet ResNet18 初始 stem 比较。

对：

```text
conv1.weight
bn1.weight
bn1.bias
bn1.running_mean
bn1.running_var
```

统计。

---

## 6.1 权重指标

对卷积：

\[
D_{rel}(W)=\frac{\|W_{trained}-W_{init}\|_2}
{\|W_{init}\|_2+\epsilon}
\]

以及：

\[
C(W)=\cos(W_{trained},W_{init})
\]

---

## 6.2 激活漂移

对固定前 256 个 SYSU test pair：

```text
ImageNet-init stem
trained R4-1 stem
trained PSD stem
```

计算 stem 输出的：

```text
mean cosine activation
RMS ratio
```

---

## 6.3 H2 判据

### 强支持“PSD stem 被重写”

同时满足：

```text
PSD conv rel_L2 >= 0.15
且
PSD rel_L2 >= 1.5 × R4-1 ResNet stem rel_L2
```

或者：

```text
PSD activation cosine(init, trained) < 0.90
同时 R4-1 >= 0.95
```

### 弱支持 / 基本排除

```text
PSD conv rel_L2 < 0.10
且
PSD / R4-1 drift ratio <= 1.2
且
activation cosine >= 0.95
```

中间区间：

```text
inconclusive
```

这项只做归因，不授权任何“stem 小 LR”训练。

---

# 7. Run5-D0-B：adapter 是否真的承担了关键接口映射

由于 Light raw 通道：

```text
32 / 64 / 128
```

而旧 downstream 需要：

```text
64 / 128 / 256
```

无法直接完全 bypass。

因此做一个**零参数、确定性、variance-preserving 的 TileAdapter**作为 inference-only ablation：

```text
32  →64  : channel repeat ×2 / sqrt(2)
64  →128 : channel repeat ×2 / sqrt(2)
128 →256 : channel repeat ×2 / sqrt(2)
```

加载 R4-2 best checkpoint 后：

```text
只把三个 learned 1×1 adapter
临时替换为 TileAdapter
其它权重完全不动
```

完整 SYSU test 一次。

---

## 7.1 H1 判据

以原 R4-2：

```text
F1 82.30
IoU 69.92
```

为 reference。

### 强支持 adapter 功能重要

```text
ΔF1 <= -0.30
且
ΔIoU <= -0.45
```

### adapter 功能较弱

```text
|ΔF1| < 0.15
且
|ΔIoU| < 0.25
```

其它：

```text
inconclusive
```

注意：

> 大幅下降只能算“支持”，不能算严格因果证明，因为 downstream 是在 learned adapter 输出分布上训练的。

但如果换成 TileAdapter 几乎不掉点，就足以显著削弱假设 ①。

---

# 8. Run5-D0-C：检查 PSD 与旧 FI 的真正接口，而不是只看 raw mean/std

对 R4-1 / R4-2 / R4-2d：

给 FeatureInjector 三个 block 加 hook：

```text
c2_c5
c3_c5
c4_c5
```

记录每个尺度：

```text
detail raw token
norm2(detail)
attn.kv(norm2(detail))
attention entropy
||attn_output|| / ||query||
```

两个时相都统计，最终取全 test 均值。

---

## 8.1 指标

### Post-LN RMS

```text
r_ln = RMS_variant / RMS_R4-1
```

### KV RMS

```text
r_kv = RMS(KV_variant) / RMS(KV_R4-1)
```

### Attention residual ratio

\[
r_a =
\frac{\|\mathrm{AttnOut}\|_2/\|Q\|_2}
{(\|\mathrm{AttnOut}\|_2/\|Q\|_2)_{R4-1}}
\]

### Normalized entropy

\[
H_n = -\sum p\log p / \log N_K
\]

比较：

```text
|H_variant - H_R4-1|
```

---

## 8.2 H3 判据

### 强支持 interface geometry mismatch

至少 2/3 scale 满足下列三项中的至少两项：

```text
r_kv 不在 [0.70, 1.30]
r_a  不在 [0.70, 1.30]
|Δ entropy| >= 0.08
```

### 基本削弱 H3

全部 3 scale：

```text
r_kv ∈ [0.80,1.25]
r_a  ∈ [0.80,1.25]
|Δ entropy| < 0.05
```

中间：

```text
inconclusive
```

---

# 9. H4 单 seed 噪声：零训练诊断不能真正分辨

这一点必须明确：

> **不做新 seed，就不可能用零训练 audit 真正估计训练随机方差。**

所以 Run5-D0 能一次性增强/削弱：

```text
H1 / H2 / H3
```

但 H4 只能作为残余不确定性保留。

不会因为 H4 再开任何 run。

---

# 10. Run5-D0-D：筛查新的 MobileNetV3-Small pretrained detail

这是 Run5-D0 最有资源价值的一部分。

使用官方 ImageNet-1K MobileNetV3-Small 权重，**不做 CD 训练**。

建议本地固定文件：

```text
/home/yqwang/projects/CASA-CD/pretrained_weight/
mobilenet_v3_small-047dcff4.pth
```

来源：

```text
torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1
```

正式训练时不联网下载。

---

# 11. MobileDetail-P3 具体截取位置

使用 torchvision MobileNetV3-Small：

## Feature 0：1/2

```text
Conv 3×3, stride2, 3→16
BN
Hardswish
```

输出：

```text
D2 = 16 × 128 × 128
```

参数：

```text
464
```

---

## Feature 1：1/4

MobileNetV3 inverted residual：

```text
DW 3×3 stride2, 16ch
BN + ReLU
SE
PW 1×1 16→16
BN
```

输出：

```text
D4 = 16 × 64 × 64
```

参数：

```text
744
```

---

## Feature 2：1/8

```text
PW 1×1 16→72
BN + ReLU
DW 3×3 stride2, 72ch
BN + ReLU
PW 1×1 72→24
BN
```

输出：

```text
24 × 32 × 32
```

参数：

```text
3,864
```

---

## Feature 3：1/8 refinement

```text
PW 1×1 24→88
BN + ReLU
DW 3×3 stride1, 88ch
BN + ReLU
PW 1×1 88→24
BN
residual add
```

最终：

```text
D8 = 24 × 32 × 32
```

参数：

```text
5,416
```

---

## MobileDetail-P3 总参数

```text
464 + 744 + 3,864 + 5,416
= 10,488
≈ 0.0105M
```

这是**完整成熟预训练链条**，不是：

```text
只复制一个 pretrained stem
后面全随机
```

这正是它与 PSD 的核心区别。

---

# 12. MobileDetail-P3 的无训练 gate

仍使用 R4-D0 完全相同的：

```text
pool to 16×16
s = 1 - cos(f1,f2)
PR-AUC
Spearman
Top32 precision
Top32 coverage
```

只要求 1/4、1/8 两个关键深层尺度。

R4-1 ResNet：

```text
D4 PR-AUC = 0.6263
D8 PR-AUC = 0.6535
D4 Top32 precision = 0.5710
D8 Top32 precision = 0.5948
```

预注册要求 MobileDetail 至少保留大约 80% 的这一 raw ranking 能力：

```text
D4 PR-AUC >= 0.50
D8 PR-AUC >= 0.52
D4 Top32 precision >= 0.46
D8 Top32 precision >= 0.48
```

### PASS

四项全部通过。

→ 才允许 R5-1 80K。

### FAIL

任意一项未通过：

> **不训练 MobileDetail。Run5 候选 ② 立即永久停止。**

后续另开 Run6，只讨论：

```text
ViT 预算重分配
```

不再找第二个、第三个 MobileNet/EfficientNet/ShuffleNet prefix。

这是为了避免新的 backbone sweep。

---

# 13. Run5 唯一主方法：MobileDetail-P3 + SGDP

总体：

```text
T1 ─┬─ TinyViT4-192 ─────────────► V1: 192×16×16
    └─ MobileDetail-P3 ─► D2_1:16×128×128
                         D4_1:16×64×64
                         D8_1:24×32×32

T2 ─┬─ same TinyViT4 ─────────────► V2
    └─ same MobileDetail-P3 ──────► D2_2,D4_2,D8_2

                  │
                  ▼
              temporal
            difference first
                  │
                  ▼
      SGDP: Semantic-Guided Difference Pyramid
                  │
                  ▼
           256×256 probability map
```

T1/T2：

```text
ViT 权重共享
MobileDetail 权重共享
SGDP 只处理双时相差异，不复制两份 head
```

---

# 14. Semantic Path：保持 R4-1，不再修改

```text
PatchEmbed:
3→192, kernel16, stride16

Tokens:
16×16 = 256

ViT:
DeiT-Tiny
width=192
heads=6
MLP ratio=4
blocks 0–3
final LN
```

参数：

```text
1,976,832
```

权重：

```text
DeiT-Tiny ImageNet-1K
patch_embed exact load
blocks0-3 exact load
pos 14×14 →16×16 bicubic
```

训练：

```text
全部冻结
```

---

# 15. Run5 第二核心：SGDP（Semantic-Guided Difference Pyramid）

原 ChangeViT：

```text
对 T1/T2 每个时相先做三路 FeatureInjector
→ 再做 4 个 heavy difference MLP
→ 再 deconv cascade
```

Run5 改成：

> **先形成变化证据，再融合。**

理由：

二值变化检测最终关注：

```text
where / whether changed
```

不需要先用 1.4M FI 为 T1 和 T2 分别构造两套增强语义，再由 2.0M decoder 比较。

---

# 16. SGDP 输入

```text
V1,V2   : B×192×16×16

D8_1,D8_2 : B×24×32×32
D4_1,D4_2 : B×16×64×64
D2_1,D2_2 : B×16×128×128
```

先计算：

```text
ΔV  = |V1 - V2|
ΔD8 = |D8_1 - D8_2|
ΔD4 = |D4_1 - D4_2|
ΔD2 = |D2_1 - D2_2|
```

时间交换完全对称。

---

# 17. SGDP 基础算子

## ProjBlock(cin, cout)

```text
Conv1×1 cin→cout, bias=False
BN(cout)
ReLU
```

---

## SepBlock(cin, cout)

```text
DWConv3×3 cin, stride1, groups=cin, bias=False
BN(cin)
ReLU
PWConv1×1 cin→cout, bias=False
BN(cout)
ReLU
```

---

# 18. SGDP Stage 16：语义变化底座

输入：

```text
ΔV : 192×16×16
```

处理：

```text
ProjBlock 192→160
```

得到：

```text
H16 = 160×16×16
```

参数：

```text
31,040
```

---

# 19. SGDP Stage 8：语义引导 1/8 detail

### Semantic up

```text
bilinear H16 ×2
SepBlock 160→128
```

得到：

```text
S8 = 128×32×32
```

参数：

```text
22,496
```

### Detail projection

```text
ProjBlock 24→128
```

得到：

```text
L8 = 128×32×32
```

参数：

```text
3,328
```

### Spatial semantic gate

```text
g8 = sigmoid(Conv1×1(S8, 128→1, bias=True))
```

参数：

```text
129
```

### Fusion

\[
H_8 = SepBlock(S_8 + g_8 \odot L_8)
\]

```text
SepBlock 128→128
```

参数：

```text
18,048
```

---

# 20. SGDP Stage 4

### Semantic state

```text
bilinear H8 ×2
SepBlock 128→96
```

输出：

```text
S4 = 96×64×64
```

参数：

```text
13,888
```

### Detail

```text
ProjBlock 16→96
```

参数：

```text
1,728
```

### Gate

```text
Conv1×1 96→1 + sigmoid
```

参数：

```text
97
```

### Fusion

```text
SepBlock 96→96
```

参数：

```text
10,464
```

---

# 21. SGDP Stage 2

### Semantic state

```text
bilinear H4 ×2
SepBlock 96→64
```

输出：

```text
S2 = 64×128×128
```

参数：

```text
7,328
```

### Detail

```text
ProjBlock 16→64
```

参数：

```text
1,152
```

### Gate

```text
Conv1×1 64→1 + sigmoid
```

参数：

```text
65
```

### Fusion

```text
SepBlock 64→64
```

参数：

```text
4,928
```

---

# 22. Final prediction

```text
bilinear H2: 128→256
Conv3×3 64→1, padding1, bias=False
sigmoid
```

参数：

```text
576
```

不在 full-resolution 再加 SepConv。

原因：

> 256×256 上增加 channel-heavy block 会显著提高 FLOPs，但对参数贡献很小；Run5 的目标是把高分辨率计算保持最低。

---

# 23. SGDP 参数表

| 子模块 | Params |
|---|---:|
| semantic 192→160 | 31,040 |
| up16→8 Sep 160→128 | 22,496 |
| detail8 24→128 | 3,328 |
| gate8 | 129 |
| fuse8 Sep 128→128 | 18,048 |
| up8→4 Sep 128→96 | 13,888 |
| detail4 16→96 | 1,728 |
| gate4 | 97 |
| fuse4 Sep 96→96 | 10,464 |
| up4→2 Sep 96→64 | 7,328 |
| detail2 16→64 | 1,152 |
| gate2 | 65 |
| fuse2 Sep 64→64 | 4,928 |
| classifier 64→1 | 576 |
| **SGDP total** | **115,267 ≈ 0.1153M** |

---

# 24. 最终参数预算

| Component | Effective Params |
|---|---:|
| TinyViT4-192 | **1,976,832** |
| MobileDetail-P3 | **10,488** |
| SGDP | **115,267** |
| **TOTAL** | **2,102,587 ≈ 2.103M** |

与工程目标：

```text
2.103M <= 2.20M
```

还保留：

```text
~97K
```

余量。

这次不再为了“参数更小”继续压缩。

---

# 25. FLOPs 粗估

设计阶段 MAC 粗估：

```text
TinyViT4 pair      ≈ 1.18G
MobileDetail pair  ≈ 0.04G
SGDP               ≈ 0.40G
--------------------------------
TOTAL              ≈ 1.62G
```

考虑 fvcore 定义、BN/激活与 unsupported op：

> 预期正式日志约在 **1.6–1.8G** 区间。

Run5 最终 hard gate：

```text
FLOPs <= 2.0G
```

输入口径：

```text
2 × 3 × 256 × 256
```

---

# 26. 为什么 SGDP 的 semantic gate 不是“又堆一个 attention”

每个尺度的 gate 只有：

```text
c_i + 1
```

个参数：

```text
128→1
96→1
64→1
```

总共只有：

```text
291 params
```

它做的事情明确：

> 让 coarse semantic change evidence 决定本尺度 local detail difference 是否应被注入。

因此：

```text
大面积真实变化 → semantic gate 打开
纹理/配准/光照引起的局部差异，但 coarse semantic change 弱 → detail 被压制
```

这是面向 binary change 的机制，而不是一般性的 feature fusion 堆叠。

---

# 27. Run5 正式实验总数

在架构定稿前最多：

```text
2 个 SYSU 80K
+ 1 个 LEVIR 80K
= 3 个正式 run
```

即：

```text
R5-1 SYSU
R5-2 SYSU
R5-3 LEVIR
```

只有全部通过后，才补：

```text
CDD
WHU
```

这两次是**定稿模型最终覆盖测试**，不再做结构修改。

所以：

> **结构定稿只需要最多 3 个正式 80K。**

---

# 28. R5-1：先单独验证成熟 pretrained micro-detail

## 唯一变量

对照：

```text
R4-1 VIT4_OLDHEAD
```

R4-1：

```text
ViT4
+ ResNet C2-C4
+ original FeatureInjector
+ original decoder
```

R5-1：

```text
ViT4
+ MobileDetail-P3
+ legacy compatibility adapters
+ original FeatureInjector
+ original decoder
```

只有：

> **detail branch package**

改变。

---

# 29. R5-1 legacy compatibility adapters

为了让 Mobile native：

```text
16 / 16 / 24
```

喂进旧 FI/decoder：

```text
64 / 128 / 256
```

只在 R5-1 使用：

### 1/2

```text
Conv1×1 16→64
BN
ReLU
```

参数：

```text
1,152
```

### 1/4

```text
Conv1×1 16→128
BN
ReLU
```

参数：

```text
2,304
```

### 1/8

```text
Conv1×1 24→256
BN
ReLU
```

参数：

```text
6,656
```

总：

```text
10,112
```

这些 adapters：

> **仅为 R5-1 legacy-head 单变量实验服务，不进入最终 R5-2。**

---

# 30. R5-1 参数 / FLOPs 预期

```text
ViT4               1.9768M
MobileDetail       0.0105M
legacy adapters    0.0101M
old FI+decoder     3.4354M
--------------------------------
≈ 5.433M
```

FLOPs 预期：

```text
≈ 10.4–10.9G
```

---

# 31. R5-1 预注册 gate

Reference：

```text
R4-1:
F1  = 82.77
IoU = 70.61
```

继续沿用之前 detail replacement 的严格标准：

```text
F1  >= 82.47
IoU >= 70.11
```

复杂度硬门槛：

```text
effective params <= 5.45M
FLOPs <= 11.0G
```

---

## R5-1 PASS

四项全部满足。

→ 才允许 R5-2。

---

## R5-1 FAIL

任一 accuracy gate 未过：

> **永久停止“成熟超轻 pretrained detail + ViT4”路线。**

不做：

```text
MobileNet 再多一个 block
MobileNet width 调整
EfficientNet prefix
ShuffleNet prefix
换 activation
冻结 MobileNet
MobileNet 小 LR
```

Run5 到此结束。

下一轮只能进入：

> **Run6：重新分配 ViT semantic 参数预算。**

---

# 32. R5-2：最终 2.10M 模型

## 唯一变量

Reference：

```text
R5-1
```

保持完全相同：

```text
ViT4
MobileDetail-P3
训练协议
```

只把：

```text
old FeatureInjector + old Decoder
```

整体替换为：

> **SGDP unified change head**

为什么这仍然是一个变量：

> 在 Run5 方法定义里，“fusion + difference modeling + reconstruction”不再拆成两个独立模块，而被重新定义为一个统一 downstream change head。

所以 R5-2 检验的是：

```text
legacy ChangeViT head
→
SGDP change-evidence head
```

单一 head 替换。

---

# 33. R5-2 预注册 gate

### Accuracy absolute gate

```text
SYSU F1  >= 82.30
SYSU IoU >= 69.92
```

---

### Relative head gate

相对 R5-1：

```text
ΔF1  >= -0.25
ΔIoU >= -0.40
```

---

### Complexity hard gate

```text
effective params <= 2.11M
FLOPs <= 2.00G
```

---

## R5-2 PASS

全部通过：

> **Run5 架构正式定稿。**

不再对 SGDP 做 width / gate / channel sweep。

直接去 LEVIR。

---

## R5-2 FAIL

任何一项失败：

> **SGDP 路线永久停止，Run5 结束。**

尤其不能做：

```text
160→192
128→144
多 gate
attention 替换 gate
concat 替换 add
extra decoder block
aux loss
```

下一轮只允许重新审查预算分配，不在 SGDP 上继续调。

---

# 34. R5-3：LEVIR 跨场景 gate

只在 R5-2 SYSU 全部 PASS 后运行。

模型结构完全冻结。

---

## LEVIR Gate

```text
F1  >= 91.50
IoU >= 84.33
effective params <= 2.11M
FLOPs <= 2.00G
```

其中：

```text
IoU 84.33
```

与 F1 91.50 的二类集合关系基本对应。

---

## LEVIR PASS

→ 模型正式成为论文候选。

然后补：

```text
CDD
WHU
```

不再改模型。

---

## LEVIR FAIL

> **不补 CDD / WHU，不宣称跨数据集轻量 SOTA。**

Run5 模型停止，不做 LEVIR 专项调参。

---

# 35. CDD / WHU 补全规则

LEVIR PASS 后：

```text
同一最终 R5-2 架构
同一超参
同一 seed16
```

分别跑：

```text
CDD
WHU
```

这些 run 只是最终论文 coverage。

无论结果如何：

> **不再改模型结构。**

如果其中一个明显掉点，则在论文中诚实限定方法适用性，不启动针对性救援。

---

# 36. Run5-D0 与 R5-1/2/3 的完整决策树

```text
Run4 Stop-3
   │
   ▼
Run5-D0：0-training
postmortem + Mobile pretrained audit
   │
   ├── Mobile D4/D8 raw gate FAIL
   │      └── STOP Run5
   │          → Run6 only: semantic/detail budget reallocation
   │
   └── PASS
          │
          ▼
      R5-1 SYSU
      MobileDetail + old head
          │
          ├── F1<82.47 或 IoU<70.11
          │      └── STOP Mobile pretrained detail permanently
          │
          └── PASS
                 │
                 ▼
             R5-2 SYSU
             MobileDetail + SGDP
                 │
                 ├── F1<82.30
                 ├── IoU<69.92
                 ├── Params>2.11M
                 └── FLOPs>2.0G
                        │
                        └── STOP SGDP / STOP Run5
                 │
                 └── ALL PASS
                        │
                        ▼
                    R5-3 LEVIR
                        │
                        ├── F1<91.50 或 IoU<84.33
                        │      └── STOP，不补 CDD/WHU
                        │
                        └── PASS
                               │
                               ▼
                          CDD + WHU
                          final reporting only
```

---

# 37. 代码修改清单

新增：

```text
models/model/mobile_detail.py
models/model/sgdp_head.py
analyse/run5_postmortem_and_mobile_audit.py
analyse/run5_param_breakdown.py

train_scripts/UltraLight/Run5/
  README.md
  audit_R5_D0_SYSU.sh
  dryrun_R5_1_SYSU.sh
  dryrun_R5_2_SYSU.sh
  train_R5_1_MOBILEDETAIL_OLDHEAD_SYSU.sh
  train_R5_2_MOBILEDETAIL_SGDP_SYSU.sh
  train_R5_3_MOBILEDETAIL_SGDP_LEVIR.sh
```

修改：

```text
models/model/encoder.py
models/model/trainer.py
models/train.py
models/eval.py
models/smoke_test.py
```

---

# 38. 推荐 CLI

新增：

```text
--detail_mode mobile_p3
--head_mode legacy|sgdp
--mobile_pretrained_weight_path /home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v3_small-047dcff4.pth
```

R5-1：

```text
--mode baseline
--vit_depth 4
--freeze_vit 1
--detail_mode mobile_p3
--head_mode legacy
```

R5-2：

```text
--mode baseline
--vit_depth 4
--freeze_vit 1
--detail_mode mobile_p3
--head_mode sgdp
```

---

# 39. `mobile_detail.py` 实现原则

推荐不要在最终 model 里注册完整 MobileNetV3。

初始化流程：

```python
ref = mobilenet_v3_small(weights=None)
sd = torch.load(local_path, map_location='cpu')
ref.load_state_dict(sd, strict=True)

self.f0 = deepcopy(ref.features[0])
self.f1 = deepcopy(ref.features[1])
self.f2 = deepcopy(ref.features[2])
self.f3 = deepcopy(ref.features[3])

del ref
```

最终 state dict 中只允许：

```text
f0
f1
f2
f3
```

严禁出现：

```text
features.4+
classifier
avgpool
```

否则 effective params 统计会被污染。

---

# 40. `MobileDetail.forward`

```python
def forward(self, x):
    d2 = self.f0(x)      # 16, 128,128
    d4 = self.f1(d2)     # 16, 64,64
    x8 = self.f2(d4)     # 24, 32,32
    d8 = self.f3(x8)     # 24, 32,32
    return d2, d4, d8
```

T1/T2 调用同一个对象。

---

# 41. legacy adapter 只在 R5-1 注册

`head_mode='legacy'` 时：

```text
16→64
16→128
24→256
```

Conv-BN-ReLU。

`head_mode='sgdp'` 时：

> adapters 必须**完全不存在**于 state_dict。

不是：

```text
requires_grad=False
```

而是根本不注册。

---

# 42. `SGDPHead.forward`

伪代码：

```python
def forward(self, fx, fy):
    d2x, d4x, d8x, vx = fx
    d2y, d4y, d8y, vy = fy

    dv  = torch.abs(vx  - vy)
    dd8 = torch.abs(d8x - d8y)
    dd4 = torch.abs(d4x - d4y)
    dd2 = torch.abs(d2x - d2y)

    h16 = self.sem16(dv)

    s8 = F.interpolate(h16, scale_factor=2, mode='bilinear',
                       align_corners=False)
    s8 = self.up8(s8)
    l8 = self.detail8(dd8)
    g8 = torch.sigmoid(self.gate8(s8))
    h8 = self.fuse8(s8 + g8 * l8)

    s4 = F.interpolate(h8, scale_factor=2, mode='bilinear',
                       align_corners=False)
    s4 = self.up4(s4)
    l4 = self.detail4(dd4)
    g4 = torch.sigmoid(self.gate4(s4))
    h4 = self.fuse4(s4 + g4 * l4)

    s2 = F.interpolate(h4, scale_factor=2, mode='bilinear',
                       align_corners=False)
    s2 = self.up2(s2)
    l2 = self.detail2(dd2)
    g2 = torch.sigmoid(self.gate2(s2))
    h2 = self.fuse2(s2 + g2 * l2)

    out = F.interpolate(h2, scale_factor=2, mode='bilinear',
                        align_corners=False)
    out = self.classifier(out)

    return torch.sigmoid(out)
```

---

# 43. Smoke Test：必须全部通过

## T-R5-0：DeiT exact load

继续沿用 Run4：

```text
patch_embed exact
blocks0-3 exact
pos 14×14→16×16
blocks4-11 不存在
```

---

## T-R5-1：Mobile pretrained exact inheritance

对 features0-3 每一个 parameter：

```text
max_abs_diff == 0
```

与 source MobileNetV3 checkpoint 对齐。

---

## T-R5-2：Mobile dead params 不得注册

state dict 中：

```text
features4+
classifier
```

必须为 0 个 key。

---

## T-R5-3：shape

```text
D2 = B×16×128×128
D4 = B×16×64×64
D8 = B×24×32×32
V  = B×192×16×16

R5-2 pred = B×1×256×256
```

---

## T-R5-4：参数量

R5-2：

```text
ViT         = 1,976,832
Mobile      = 10,488
SGDP        = 115,267
effective   = 2,102,587
```

允许实现误差后：

```text
assert effective <= 2.11M
```

---

## T-R5-5：legacy adapters 只存在于 R5-1

R5-1：

```text
adapters params = 10,112
```

R5-2：

```text
adapter key count == 0
```

---

## T-R5-6：冻结 ViT checksum

正式 dry run 前后：

```text
all encoder.vit tensors
max_abs_change == 0
```

---

## T-R5-7：gradient

R5-2：

```text
ViT grad == None
Mobile f0-f3 grad finite & nonzero
SGDP grad finite & nonzero
```

---

## T-R5-8：time swap

因为：

```text
shared encoder
abs temporal difference
```

eval：

```text
pred(A,B)
pred(B,A)
```

要求：

```text
max_abs_diff < 1e-5
```

如果不满足，先排查，不得训练。

---

## T-R5-9：FLOPs

```text
fvcore
input 2×3×256×256
```

必须打印：

```text
FLOPs
unsupported_ops
```

R5-2 hard gate：

```text
<= 2.0G
```

---

## T-R5-10：train/eval architecture parity

checkpoint 中额外存：

```text
vit_depth
detail_mode
head_mode
mobile weight id
```

独立 `eval.py` 构建必须完全一致。

---

# 44. Dry Run

每个新正式结构先：

```text
60–200 steps
SYSU
GPU1 only
```

建议路径：

```text
/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/UltraLight/Run5/<VARIANT>/SYSU-CD-256/

/home/yqwang/outputs/CASA-CD/_dryrun/UltraLight/Run5/<VARIANT>/SYSU-CD-256/
```

检查：

```text
exit 0
loss finite
pred 非全0/全1
ViT checksum
Mobile exact-load header
params
FLOPs
unsupported_ops
resume
independent eval
```

---

# 45. 正式路径

Checkpoint：

```text
/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run5/<VARIANT>/<DATASET>/
```

Log：

```text
/home/yqwang/outputs/CASA-CD/UltraLight/Run5/<VARIANT>/<DATASET>/train_log.txt
```

Variant：

```text
R5_1_MOBILEDETAIL_OLDHEAD
R5_2_MOBILEDETAIL_SGDP
```

---

# 46. 正式脚本启动顺序

## Step 0：D0

```bash
CUDA_VISIBLE_DEVICES=1 \
python analyse/run5_postmortem_and_mobile_audit.py ...
```

读：

```text
[MOBILE-GATE] PASS/FAIL
```

FAIL：

```text
STOP
```

---

## Step 1：R5-1 SYSU

```bash
nohup bash \
train_scripts/UltraLight/Run5/train_R5_1_MOBILEDETAIL_OLDHEAD_SYSU.sh \
> /dev/null 2>&1 &
```

完成后只读最后 TEST RESULTS。

PASS 才进入下一步。

---

## Step 2：R5-2 SYSU

```bash
nohup bash \
train_scripts/UltraLight/Run5/train_R5_2_MOBILEDETAIL_SGDP_SYSU.sh \
> /dev/null 2>&1 &
```

全部 final gate 通过才定稿。

---

## Step 3：R5-3 LEVIR

```bash
nohup bash \
train_scripts/UltraLight/Run5/train_R5_3_MOBILEDETAIL_SGDP_LEVIR.sh \
> /dev/null 2>&1 &
```

LEVIR ≥91.50 后：

```text
CDD
WHU
```

---

# 47. Run5 绝不做的事情

如果任何 gate 不过，不允许：

```text
Mobile features0-4 再试
Mobile features0-5 再试
换 MobileNetV2
换 EfficientNet
换 ShuffleNet
Mobile branch freeze
Mobile branch 小 LR
SGDP 96/64/48
SGDP 192/144/96
gate 改 channel gate
gate 改 attention
concat/add sweep
extra loss
延长 steps
换 optimizer
```

这次 Run5 必须是：

> **一个 pretrained detail 候选 + 一个 final head。**

---

# 48. R5-1 / R5-2 失败后的明确停止动作

## R5-D0 FAIL

永久停止：

```text
mature ultra-light pretrained detail candidate search
```

Run6 只能做：

> **semantic/detail 参数预算重新分配。**

---

## R5-1 FAIL

即使 D0 raw audit 通过，只要端到端：

```text
F1 <82.47
或
IoU <70.11
```

就证明：

> pretrained Mobile micro-detail 无法在 ChangeViT legacy downstream 中替代 ResNet detail。

永久停止该候选。

---

## R5-2 FAIL

证明：

> 在保留 ViT4 的 0.22M 外围预算内，SGDP 不能同时满足精度和效率。

永久停止：

```text
ViT4 + micro-detail + unified lightweight head
```

下一轮才允许动 ViT 预算。

---

## LEVIR FAIL

证明：

> SYSU 上可行但没有达到预注册跨数据集水平。

不补 CDD/WHU。

不做 LEVIR 专项修复。

---

# 49. 对 R4-0=83.14 与历史 baseline=82.48 的继续使用

这一点保持上一轮结论，不再变化。

---

## 历史 baseline 82.48

命名建议：

> **ChangeViT-T Official-Protocol Reproduction**

用途：

- 说明论文 baseline 成功复现；
- 记录官方训练协议；
- 同时注明后来发现的 ViT instability。

不再用它作为 Run5 组件消融的主 reference。

---

## R4-0 83.14

命名建议：

> **Healthy Full12 Reference**

所有轻量结构机制结论的主参考：

```text
full12 healthy
corrected loader
frozen pretrained ViT
```

---

## R4-1 82.77

用于：

> **ViT4 之后的外围单变量比较。**

所以：

```text
R5-1 vs R4-1
```

是正确的 detail comparison。

---

# 50. Run5 的论文机制故事

如果 Run5 成立，论文不再讲：

```text
“我们做了一个超小 DWConv detail branch”
```

因为 Run4 已经证明那条故事不成立。

---

# 51. 第一贡献：Shallow Pretrained Semantic Redundancy

这一条已经有直接实验证据。

建议论文贡献表述：

> **We identify substantial depth redundancy in the pretrained plain-ViT semantic path of binary change detection. Under a healthy frozen-pretraining protocol, truncating DeiT-Tiny from 12 to 4 Transformer blocks reduces the semantic backbone by about 3.56M parameters while degrading SYSU F1 by only 0.37 points.**

中文机制：

> 在 detail 分支补充局部结构的前提下，ChangeViT 的后 8 个 Transformer block 存在明显参数冗余；低分辨率全局语义不需要完整 12 层 DeiT 容量。

注意：

`12→4` 本身不是靠一次结果就叫“普适定理”，论文中应限定：

```text
在当前 ChangeViT / 四数据集 / 预训练设置下
```

最终四数据集补全后再扩大表述。

---

# 52. 第二贡献：Pretrained Micro-Detail + Semantic-Guided Change Reconstruction

第二贡献建议把 MobileDetail 和 SGDP 作为一个统一机制，而不是两项堆模块。

核心观点：

> **在极端参数预算下，高分辨率路径的关键不是重新从零学习一个极小 CNN，而是保留一个成熟预训练网络的最浅层局部视觉先验；随后直接在 change-evidence domain 中，由低分辨率语义变化逐级筛选高分辨率局部差异。**

对应结构：

```text
ImageNet MobileNetV3 prefix:
只花 ~10.5K 参数保留 1/2,1/4,1/8 local prior

SGDP:
先 temporal difference
再 semantic gate detail
再 coarse-to-fine reconstruction
```

---

# 53. 与 ChangeViT 的实质区别

### ChangeViT

```text
ResNet C2-C4      ~2.7M
FeatureInjector   ~1.395M
Decoder           ~2.040M

每个时相先做 cross-attention detail injection
再做 heavy temporal difference
```

### Run5

```text
Mobile pretrained P3 ~0.0105M
SGDP                 ~0.115M

先形成 temporal change evidence
再由 semantic difference gate local detail difference
```

机制变化不是：

> “把卷积换成 depthwise。”

而是：

> **从“先增强两个时相的 representation，再比较”改为“先形成变化证据，再按语义可信度融合”。**

这更贴合 binary CD 本身。

---

# 54. CASAA 在最终论文的位置

不回主贡献。

保留为：

```text
analysis / ablation / negative-result-guided motivation
```

可支持：

> late-stage context 有冗余；Full-Q 并不要求完整 K/V。

但：

```text
change-aware router
```

已经按预注册规则失败，不再写成最终方法主线。

---

# 55. 最终论文定位

结果出来前：

> **A sub-3M fully supervised binary change detector that reallocates capacity between a shallow pretrained semantic path and a pretrained micro-detail path, with semantic-guided change-evidence reconstruction.**

结果全部通过后再根据统一协议比较决定是否写：

```text
state-of-the-art
```

否则使用：

> **favorable accuracy–efficiency trade-off in the ~2M parameter regime**

---

# 56. 参数对标定位

最终预计：

```text
CASA-CD Run5 ~2.10M
```

参数区间：

```text
CGLNet      ~0.99M
Lighter     ~1.10M
CASA-CD     ~2.10M
SeCoR       ~2.50M
RFANet      ~2.86M
```

论文目标不是“比所有人参数最小”，而是：

> **在 2M 左右形成更高 F1/IoU 的 Pareto 点。**

---

# 57. 最终实验表建议

最终论文主表至少区分：

### Internal controlled experiments

```text
R4-0 Healthy Full12
R4-1 ViT4
R5-1 ViT4+MobileDetail+legacy head
R5-2 final
```

用于机制因果。

---

### External lightweight comparison

同协议优先：

```text
RFANet
SeCoR
Lighter
CGLNet
```

自己的：

```text
Recall
Precision
OA
F1
IoU
Kappa
Params
FLOPs
```

完整报告。

---

# 58. 立即执行顺序

1. 建 `train_scripts/UltraLight/Run5/`。
2. 准备官方 ImageNet MobileNetV3-Small 本地 `.pth`。
3. 写 `run5_postmortem_and_mobile_audit.py`。
4. 先跑 H1/H2/H3 postmortem；只用于研究记录。
5. 同脚本跑 MobileDetail-P3 raw gate。
6. **Mobile raw gate FAIL → 立即停止 Run5，不开任何 80K。**
7. gate PASS → 实现 `mobile_detail.py`。
8. smoke exact pretrained features0-3。
9. 实现 legacy compatibility adapters。
10. R5-1 60–200 step dry run。
11. 启动 R5-1 SYSU 80K。
12. 只读最后 TEST RESULTS。
13. **R5-1 gate FAIL → Run5 结束。**
14. PASS → 实现 `sgdp_head.py`。
15. smoke：shape / params / swap / grad / ViT checksum / FLOPs。
16. R5-2 dry run。
17. 启动 R5-2 SYSU 80K。
18. **R5-2 final gate FAIL → Run5 结束。**
19. PASS → 架构冻结。
20. 跑 R5-3 LEVIR。
21. **LEVIR <91.50 → 不补 CDD/WHU。**
22. LEVIR PASS → CDD + WHU 最终补全。
23. 再进入轻量 SOTA 统一协议重跑。

---

# 59. Run5 最重要的可证伪假设

## H5-1：Mature Micro-Detail Hypothesis

> Run4 的主要问题不是高分辨率 detail 必须拥有大量参数，而是随机/局部预训练的超小自定义分支难以形成稳定深层局部表征；一个完整继承 ImageNet 预训练链条的 MobileNetV3 极浅 prefix 可以用极少参数保留有效 1/2–1/8 detail。

证伪：

```text
D0 raw gate FAIL
或
R5-1 F1 <82.47 / IoU<70.11
```

---

## H5-2：Change-Evidence-First Hypothesis

> Binary CD 不需要先分别构造两套昂贵的增强时相表示；先形成多尺度对称变化证据，并用 coarse semantic change 逐级 gate local detail difference，可以在极低 head 参数下保留精度。

证伪：

```text
R5-2 F1 <82.30
或
IoU <69.92
```

---

## H5-3：Extreme-Budget Allocation Hypothesis

> 在约 2.1M 总参数下，把绝大多数参数保留给已经被验证的 shallow pretrained ViT4，而把 detail+head 压到约 0.126M，能够形成更优的 accuracy–parameter trade-off。

证伪：

```text
R5-2 accuracy gate 或 complexity gate 任意失败
```

---

# 60. 最终建议

Run4 的负结果已经足够明确：

> **不要再从零设计第四个“几十 K 参数的 DWConv detail branch”。**

下一步最有信息量、最节省训练预算的不是继续修改 PSD，而是：

1. **保留已经证实有效的 ViT4；**
2. 用完整继承 ImageNet 的 MobileNetV3-Small 极浅 prefix，把 detail 重新变成“成熟视觉先验”而不是“随机小网络”；
3. 不再保留 ChangeViT 3.44M 的 legacy FI+decoder，把 binary CD 的 downstream 重写成一个统一的 **semantic-guided difference pyramid**；
4. 用两次 SYSU 正式 run 就决定这条路是否成立；
5. 成立后只用一次 LEVIR 做跨场景 gate，再决定是否铺 CDD/WHU。

这条路线正面回应了当前最矛盾的结构事实：

```text
ViT4 已经占 1.977M
```

但结论不是“因此必须再砍 ViT”，而是：

> **先证明外围能否用成熟预训练 micro-detail + change-evidence head 在约 0.126M 内完成。**

如果连这条路线都不能做到 SYSU ≥82.30，那么证据才足够支持下一轮真正动 ViT 预算。

换句话说：

> **Run5 是“保住已验证语义容量，极限重构外围”的最后一次实验。**
>
> **若失败，Run6 才进入“牺牲部分 ViT 容量，把参数重新分给 detail/head”的预算重分配。**
