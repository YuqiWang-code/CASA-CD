# CASA-CD Run8：B4-SPE 最终路线与可执行预注册方案

> 日期：2026-10-02  
> 事实权威：仓库 `YuqiWang-code/CASA-CD` main；按顺序审阅 README、Run7 方案与终局、Run6/5/4 终局、文献索引及当前 `encoder/trainer/decoder/sgdp_head/depth_pyramid_head` 与 R4–R7 audit 代码。  
> 固定协议：BCE+Dice、Adam 2e-4、poly+200 warmup、80000 steps、batch 16、256×256、seed 16、test-as-val、GPU1、ViT 冻结、corrected DeiT loader。  
> 结果纪律：正式结果只认同一个 `train_log.txt` 最后一个完整 `=== TEST RESULTS ===` 至 `=== END TEST RESULTS ===` 区块。

---

## 0. 最终裁决

**Run8 只保留一个主方向：`ViT4 (B4) + B4-SPE`。**

这里接受“候选 1”的核心判断——**保留 B4、彻底删除独立 detail branch 与旧 FI/decoder**——但**不直接使用当前 `SGDPHead`**。当前 `SGDPHead` 的前向明确要求：

```text
(d2, d4, d8, v)
```

且其 `detail8/detail4/detail2 + gate8/gate4/gate2` 是结构主体；“B4 + SGDP（无 detail）”在当前代码里并不是一个已就绪组合。若把 detail 输入删掉，SGDP 的机制已经改变，不能把 Run5 的 smoke 当作“B4-only SGDP 已验证”。

因此 Run8 的终模型采用：

> **B4-SPE = Change-Sensitive B4 Semantic Source + Symmetric Pair Descriptor + Sub-patch Expansion**

只复用 Run7 `DepthPyramidHead` 中已经 smoke 过的 **Sub-patch Expansion** 部分；不复活 Run7 的 B1+B2/CSDP 假设，也不把 B2 改成 B3/B4“救场”。

### 为什么不是候选 2（depth-adaptive B2–B4）

R7-D0 只证明“不同数据集的最优深度不同”，**没有证明 B2/B3/B4 在同一图像或同一 token 上存在可利用的互补性**。直接上 learnable multi-depth fusion，会同时引入：
- 新的 fusion/gating；
- 多 depth pair descriptor；
- 新 head；
- 新的归因困难。

而且 RFL-CDNet 已经把 intermediate feature / multi-stage prediction / learnable fusion 做成主机制。即便我们的 B2–B4 是“同尺度不同深度”，创新边界仍需要额外实验才能立住。最关键的是：**multi-depth fusion 仍没有解决 16×16 token 到 256×256 的 inner-patch 重建风险**。

所以在只剩 3–4 个 80K 的情况下，候选 2 的信息增益不如先直接裁决 B4-only。

### 为什么候选 3 不能作为最终方案

R4-1（ViT4+ResNet+旧 head）82.77/70.61 是最稳妥的已验证资产，但有效参数 8.195M，不满足课题硬目标 `<3M`。它只能作为：
- Run8 参考上界；
- 如果 Run8 最终失败后的论文/硕士课题“分析型收尾”基线。

不能把它包装成已经完成“极轻量最终模型”。

---

## 1. 证据表

| 类型 | 证据 | 对 Run8 的含义 |
|---|---|---|
| 代码/日志直接事实 | R4-0 full12 frozen：83.14/71.14；R4-1 ViT4：82.77/70.61 | 12→4 仅 −0.37 F1，ViT4 已有端到端可用性 |
| 代码/日志直接事实 | R7 四数据集：B4 在 CDD/SYSU 第一、LEVIR/WHU 第二；B12 全面更差 | B4 是当前唯一跨四数据集稳健的 ViT semantic source |
| 代码/日志直接事实 | SYSU：B4 PR-AUC 0.6127，ResNet 1/8 0.6535 | B4 change evidence 已接近重型 detail 深层特征 |
| 代码/日志直接事实 | Light32/48/PSD 连续失败；Mobile raw gate 失败 | 不再搜索独立 detail branch 有充分负证据 |
| 代码/日志直接事实 | R4-1 旧 FI 的 1/2、1/4 路径出现零权重吸收，仅 1/8 真正活跃 | 原 ChangeViT 三尺度 detail 注入并没有被当前训练实际完整使用 |
| 代码直接事实 | `SGDPHead.forward()` 必须消费 d2/d4/d8/v | “B4-only SGDP”不是现成模型，需要重新定义 |
| 代码直接事实 | `depth_pyramid_capture()` 目前把 `vit.norm(block1)` 的结果送进 block2 | **P0：它不等价于 R6/R7 audit 的真实 B2 顺序**；Run7 未训练所以正式结果未受影响，但 Run8 禁止复用这条 capture 数据流 |
| 文献事实 | FDAM：深层自注意力存在低通与 frequency vanishing | 支持“B4 后继续加深可能损伤 dense change evidence”的解释 |
| 文献事实 | LaViT：deep ViT 存在 attention saturation | 支持减少不必要深层 attention |
| 文献事实 | ResCLIP：非最终层 cross-correlation 保留 localization | 支持 intermediate/B4 空间定位价值 |
| 文献事实 | ViT-CoMer：plain ViT 存在 inner-patch interaction 不足 | Run8 最大风险正是 B4-only 的像素级重建 |
| 文献事实 | LiFT：小型 feature transform 能低成本 densify ViT descriptor | 支持极轻量 densifier，而非另加完整 detail backbone |
| 文献事实 | EoMT：plain ViT 可承担更大的 dense prediction 责任，但效果依赖规模与预训练 | 支持“少 task-specific 组件”，但 Tiny 场景不能直接照搬结论 |
| 2026 外部竞争 | SeCoR 2.50M/2.66G；CGLNet 0.99M/0.62G 都已采用 evidence/semantic-guided lightweight decoding | **不能把“semantic-guided lightweight decoder”本身当主创新**；Run8 主贡献必须落在“change-sensitive depth evidence + detail-free budget reallocation” |

---

## 2. P0 / P1 / P2 审查

### P0 正确性

1. **禁止把现有 `SGDPHead` 直接标成“无 detail 可用”**。它当前结构不是 semantic-only。
2. **禁止复用 `depth_pyramid_capture()` 的现有 B1→B2 数据流**：
   ```python
   b1 = vit.norm(vit.blocks[0](tok))
   b2 = vit.norm(vit.blocks[1](b1))
   ```
   这会让 block2 输入 final-LN 后的 B1，而真实 DeiT / R7 audit 是：
   ```text
   x1_raw = block1(tok)
   B1 = final_norm(x1_raw)       # 只用于观测
   x2_raw = block2(x1_raw)       # block2 继续吃 raw state
   B2 = final_norm(x2_raw)
   ```
   Run8 B4 必须走标准 `vit.forward` 或等价 raw-state 顺序。
3. ViT4 必须逐位继承 corrected DeiT：
   - patch_embed；
   - blocks 0–3；
   - final norm；
   - 14×14 pos_embed → 16×16 bicubic；
   - 全部冻结。

### P1 方法瓶颈

唯一未裁决的机制问题是：

> **B4 的 16×16 semantic change evidence 是否足以在没有独立 detail encoder 的情况下，被一个 <0.06M 的 head 重建成 256×256 边界？**

这正对应 ViT-CoMer 提醒的 inner-patch limitation。Run8-D0 必须专门测这个问题，而不是再重复“B4 token PR-AUC 很高”。

### P2 实验工程

只新增：
- 一个零训练 dense-recoverability audit；
- 一个 B4-only head；
- 一套 Run8 脚本。

不再扩搜索空间。

---

## 3. 文献判读与创新边界

### 3.1 FDAM / LaViT / ResCLIP：只作为“为什么 B4 比 B12 更合理”的外部解释

它们不能替代 CASA-CD 自己的四数据集证据。论文表述应是：

> 外部工作分别报告了 frequency vanishing、attention saturation、intermediate localization；CASA-CD 进一步在四个 BCD 数据集上观察到 B12 的 token change ranking 一致劣于 B2–B4，并据此选择 B4 作为稳健终止语义源。

不要写成“FDAM 已证明我们 B4 的因果机制”。

### 3.2 ViT-CoMer：决定 Run8 必须有 dense-recoverability gate

ViT-CoMer 指出 plain ViT 缺 inner-patch interaction 与多尺度 diversity。  
所以“B4 PR-AUC 高”不等价于“B4 能恢复建筑边界”。

因此 Run8-D0 不再看纯 16×16 token 指标是否好，而要看：
- B4 raw score 直接投到 256×256 后的 pixel AUPRC；
- GT boundary band 内的 boundary AUPRC。

### 3.3 LiFT：支持“极小 densifier”，但不照搬自监督 loss

Run8 不增加自监督目标，不改变 BCE+Dice。只借鉴“低成本 feature densification 是可行范式”这一点。

### 3.4 RFL-CDNet：Run8 与它的实质区别

RFL-CDNet：
- backbone 多阶段 intermediate feature；
- deep multiple supervision；
- coarse-to-fine side prediction；
- learnable fusion 多阶段输出。

Run8：
- 不融合 B1/B2/B3；
- 不做 deep supervision；
- 不做 side outputs；
- 只选择**单一 B4 作为 terminal semantic source**；
- B4 后深层全部删除；
- 用一个 <0.06M symmetric expansion head 做 dense reconstruction。

所以 Run8 不是“intermediate feature 全利用”，而是：
> **evidence-based early semantic termination + detail-branch removal**。

### 3.5 SeCoR / CGLNet 对 Run8 的约束

2026 已经有大量“evidence-guided / high-level semantic-guided / guided decoder”轻量 CD 方法。  
因此 B4-SPE 的论文主张不能写成：

> “我们提出一个 semantic-guided lightweight decoder。”

应写成：

> “我们通过跨四数据集的 change-sensitive depth audit 发现 plain DeiT 的最深层并非最适合 BCD；B4 是稳健 semantic source。基于这一任务特定深度证据，我们在 B4 截断并删除独立 detail encoder，只保留一个最小的 symmetric sub-patch reconstruction head。”

head 是实现机制，不是主创新标题。

---

## 4. Run8-D0：B4 Dense Recoverability Audit（零训练）

### 4.1 目的

不是再次验证 `B4 > B12`，而是回答：

> **16×16 B4 change evidence 在无训练、无 detail branch 情况下投影到像素域后，是否仍保留足够的整体定位与边界排序能力？**

### 4.2 数据与 source

四数据集完整 test：
- CDD-CD-256
- LEVIR-CD-256
- SYSU-CD-256
- WHU-CD-256

同一个 frozen corrected depth-12 DeiT，一次前向取得：
- B4；
- B12。

score：
```text
S4  = 1 - cos(B4_A,  B4_B)      # B×256
S12 = 1 - cos(B12_A, B12_B)
```

每图 ordinal rank normalize 到 [0,1]，reshape 为 16×16，bilinear 到 256×256：
```text
U4  = bilinear(rank(S4))
U12 = bilinear(rank(S12))
```

**不得**使用 GT 调 score、不得学参数。

### 4.3 指标

#### M1：Pixel PR-AUC
完整 256×256 像素上计算 PR-AUC。

#### M2：Boundary-band PR-AUC
GT：
1. `edge = mask XOR erode(mask, 3×3)`；
2. `band = dilate(edge, 9×9)`（约 ±4 px）；
3. 只在 `band` 内计算 score vs GT 的 PR-AUC。

它直接衡量“粗 token score 在真实边界邻域的排序能力”。

#### M3：稀疏变化分桶（诊断，不参与主 gate）
复用 R6：
- 1–16 changed patches；
- 17–64；
- >64。
输出 Top8 hit / Top32 coverage。

#### M4：Fixed patch-unembedding（仅诊断）
可选，不参与 gate：
- 把 `|B4_A-B4_B|` reshape 成 `192×16×16`；
- 用 DeiT `patch_embed.proj.weight` 的 transpose-conv 固定反投影；
- 通道 L2 得到像素 score。
仅看它是否比 B12 更保留 sub-patch phase；**不因这项失败否掉 Run8**，因为 Transformer 后 channel basis 已可能旋转。

### 4.4 预注册 gate

四数据集分别计算：

```text
G1_pixel(ds):
    PRpix(B4) >= PRpix(B12) + 0.03

G2_boundary(ds):
    PRbnd(B4) >= PRbnd(B12) + 0.02
```

全局规则：

```text
C1: G1_pixel 至少 3/4 数据集通过
C2: G2_boundary 至少 3/4 数据集通过
C3: 任何数据集都不允许 B4 比 B12 下降 >0.01（pixel 或 boundary）
```

**C1 & C2 & C3 全满足 → PASS。**

为什么用相对阈值而不是固定 PR-AUC 地板：CDD/LEVIR/SYSU/WHU 的变化像素基率差异很大，绝对 AUPRC 不可直接统一；但 B4 与 B12 来自同一图、同一 backbone、同一投影，relative lift 是严格同口径比较。

### 4.5 D0 失败动作

若 FAIL：

> **永久停止“B4-only / no-detail dense reconstruction”路线。**
>
> 不改上采样方式、不调 head width、不加 edge/frequency/detail 模块、不回 Mobile/ResNet。
>
> 直接进入“论文分析型收尾”：R4-1 作为最强已验证轻量化结构，Run1–7 作为 budget-allocation / negative-evidence study；明确 `<3M` 终模型目标未被验证达成。

不再开候选 2 的 learnable depth fusion 作为临场救援。

---

## 5. Run8 唯一主模型：ViT4 + B4-SPE

### 5.1 Encoder

共享 Siamese：

```text
RGB 256×256
  ↓ PatchEmbed 16×16, C=192
256 tokens
  ↓ Block0
  ↓ Block1
  ↓ Block2
  ↓ Block3
  ↓ final LN
B4: B×256×192
```

两个时相共享全部参数；ViT4 全冻结。

### 5.2 Symmetric Pair Descriptor

```text
D = |T_A - T_B|
M = (T_A + T_B) / 2
Z = Linear([D, M], 384 -> 96)
```

`[|Δ|, mean]` 在 A/B 交换下严格不变：

```text
pred(A,B) == pred(B,A)
```

不使用 cross-attention，不使用 teacher，不使用额外 loss。

### 5.3 Sub-patch Expansion

**原样复用 Run7 `DepthPyramidHead` 的 expansion 部分，不复用 p1/p2 双深度融合。**

```text
96×16×16
 -> Conv1×1 96→128 + BN + ReLU
 -> PixelShuffle×2
 -> 32×32×32
 -> DW3×3 + BN + ReLU

32×32×32
 -> Conv1×1 32→64
 -> PixelShuffle×2
 -> 16×64×64
 -> DW3×3

16×64×64
 -> Conv1×1 16→64
 -> PixelShuffle×2
 -> 16×128×128
 -> DW3×3

16×128×128
 -> Conv1×1 16→16
 -> PixelShuffle×2
 -> 4×256×256
 -> DW3×3
 -> Conv1×1 4→1
 -> sigmoid
```

### 5.4 参数预算

已知：

```text
ViT4                         = 1,976,832
Pair Linear(384→96,bias)     =    36,960
Expansion（Run7 已计数）      =    16,912
------------------------------------------------
设计总计                     = 2,030,704 ≈ 2.031M
```

ViT 冻结，因此设计 trainable params：

```text
≈ 53,872
```

正式以 smoke / train header 打印为准。

### 5.5 FLOPs 粗估

已知：
- R4-0 full12 与 R4-1 ViT4 的差为约 2.22G；
- 相差 8 个 block，折算 dual-temporal 约 0.2775G/block；
- Run7 ViT2+CSDP 设计测得 0.673G。

ViT2→ViT4 增加约：

```text
2 × 0.2775 ≈ 0.555G
```

同时 B4-SPE 比 CSDP 少一个 384→96 projector，故 Run8 预估：

```text
约 1.15–1.25G
```

硬 gate 仍设：

```text
FLOPs <= 2.00G
```

实际以 fvcore 同口径测量为准。

---

## 6. 与 SGDP / CSDP 的改动边界

### SGDP

只复用它的研究思想：
- difference-first；
- coarse-to-fine；
- strict temporal symmetry。

**不直接复用 `SGDPHead` 类**，因为删除 detail 后其 `detail{8,4,2}` 和 `gate{8,4,2}` 都失去输入与意义。

### CSDP

只复用：
- `DepthPyramidHead` 的四级 PixelShuffle expansion 拓扑；
- 已有 parameter/FLOPs/symmetry smoke 思路。

不复用：
- B1+B2；
- `p1/p2` 双深度加和；
- 当前 `depth_pyramid_capture()`。

所以这不是“R7 把 B2 偷换成 B4救场”，而是新假设：

> R7 已证 B4 是跨数据集稳健 source；Run8 单独检验“B4 + minimal densifier 是否足以替代独立 detail branch + legacy decoder”。

---

## 7. 逐文件修改清单

### 新增

`models/model/b4_spe_head.py`
- `B4SPEHead`
- 单一 pair descriptor；
- 复用 expansion；
- 无 attention / 无 gate / 无 auxiliary head。

`analyse/run8_b4_dense_recoverability_audit.py`
- B4/B12 pixel PR；
- boundary-band PR；
- bucket diagnostics；
- `[R8-D0-GATE] PASS/FAIL`。

`train_scripts/UltraLight/Run8/`
- `README.md`
- `audit_R8_D0_B4_DENSE.sh`
- `dryrun_R8_1_SYSU.sh`
- `train_R8_1_B4_SPE_SYSU.sh`
- `train_R8_2_B4_SPE_LEVIR.sh`
- `train_R8_3_B4_SPE_WHU.sh`
- `train_R8_4_B4_SPE_CDD.sh`

### 修改

`models/model/encoder.py`
- 新增 `detail_mode='none_b4'`；
- 此模式不注册 ResNet/Light/PSD/Mobile；
- 正常 `vit_depth=4` 标准前向；
- 返回 final-LN B4 token；
- **不走 `depth_pyramid_capture()`**。

`models/model/trainer.py`
- 新增 `head_mode='b4_spe'`；
- decoder = `B4SPEHead()`；
- forward 接 B4 pair。

`models/train.py` / `models/eval.py`
- CLI 加 `none_b4` / `b4_spe`；
- arch sidecar 必须记录；
- `vit_depth != 4` 时直接 assert；
- train/eval 架构不一致直接拒绝。

`models/smoke_test.py`
新增 T-R8：
1. output `(B,1,256,256)`；
2. A/B 交换 max abs err `<1e-6`；
3. total/effective params `<=2.10M`；
4. trainable params `<=0.06M`；
5. FLOPs `<=2.0G`；
6. `encoder.resnet is None`，且 state_dict 不含任何 detail/FI/legacy decoder；
7. patch_embed + block0–3 + norm checksum 与 corrected DeiT 一致；
8. backward 后 ViT grad 全 None/0，head grad finite；
9. `eval.py` arch sidecar mismatch 能被拒绝。

---

## 8. 正式 80K 预注册

### R8-1：SYSU Final Gate

**唯一变量**相对 Run8 架构定义：训练 B4-SPE head；ViT4 frozen。

参考：
```text
R4-1 ViT4+ResNet+legacy:
F1 82.77 / IoU 70.61
```

硬 gate：

```text
F1  >= 82.30
IoU >= 69.92
effective params <= 2.10M
FLOPs <= 2.00G
```

等价于相对 R4-1：
```text
ΔF1  >= -0.47
ΔIoU >= -0.69
```

**PASS** → R8-2 LEVIR。  
**FAIL** → 永久停止 B4-only final model；不调 head、不加 detail、不做 adaptive depth rescue；进入论文分析型收尾。

### R8-2：LEVIR Cross-scene Gate

结构/超参完全不改。

```text
F1  >= 91.50
IoU >= 84.33
params/FLOPs 同 R8-1
```

这里不把 Run1 91.95 当健康架构 reference（LEVIR Run1 ViT 已知死权重）；91.50/84.33 仅作为项目预注册的绝对性能地板。

PASS → R8-3 WHU。  
FAIL → 不补 WHU/CDD；不做 LEVIR 特调；记录“SYSU 有效但跨稀疏建筑场景失败”。

### R8-3：WHU 高风险跨场景 Gate

为什么 WHU 先于 CDD：
- R7 中 WHU 的 B4 raw PR 绝对值最低之一；
- B3 还略高于 B4；
- 它最能检验“固定 B4 + no-detail”是否真能处理稀疏小建筑。

```text
F1  >= 94.34
IoU >= 89.29
```

PASS → R8-4 CDD。  
FAIL → 不再改结构；可停止第四个 80K，论文明确 WHU 局限。

### R8-4：CDD 定稿覆盖

仅在 R8-1/2/3 全通过后执行。

```text
F1  >= 97.25
IoU >= 94.65
```

这是最终覆盖，不允许单数据集修补。

---

## 9. 正式训练预算

最少：
```text
D0（0 个 80K）
+ SYSU（1）
```

若成功：
```text
+ LEVIR（2）
+ WHU（3）
+ CDD（4）
```

最多严格控制为 **4 个正式 80K**。

顺序：
```text
R8-D0 四数据集零训练
  ↓ PASS
smoke
  ↓
SYSU dry run
  ↓
R8-1 SYSU 80K
  ↓ PASS
R8-2 LEVIR 80K
  ↓ PASS
R8-3 WHU 80K
  ↓ PASS
R8-4 CDD 80K
```

绝不并行铺开。

---

## 10. 停止规则总表

| Gate | FAIL 后唯一动作 | 永久禁止 |
|---|---|---|
| R8-D0 dense recoverability | 结束 B4-only 搜索，论文分析型收尾 | 改 D0 阈值、换上采样、加 edge/frequency/detail、adaptive depth 救场 |
| R8-1 SYSU | 停止 `<3M B4-only final model` | head width/gate sweep、ResNet/Mobile 回流、loss/LR/steps 修改 |
| R8-2 LEVIR | 停止跨数据集 claim | LEVIR 特调、补 WHU/CDD |
| R8-3 WHU | 记录固定 B4 在稀疏建筑的局限 | B4→B3 单数据集切换 |
| R8-4 CDD | 记录局限，结构冻结定稿 | CDD 专用结构 |

---

## 11. 论文机制故事（仅在 Run8 成功后成立）

### 主贡献 1：Change-sensitive depth evidence

四数据集 depth audit 不是为了挑一个“漂亮 block”，而是说明：

```text
B2–B4 > B12（所有数据集）
B4 是跨数据集稳健工作点
最优 depth 仍有 dataset dependence
```

因此不是“越深越好”，也不是“统一 B2 最好”。

论文主张：

> **Remote-sensing BCD needs a task-appropriate semantic depth rather than the deepest ViT representation.**

### 主贡献 2：Evidence-driven budget reallocation

Run4–Run7 的负结果形成一个完整的 budget allocation study：
- 自定义 light detail：失败；
- pretrained Mobile prefix：raw gate 失败；
- P0 token reconstruction：失败；
- B2 early truncation：跨数据集失败；
- B4：唯一跨四数据集稳健 semantic source。

因此最终 2.03M 不是“随便砍小”，而是：

> 把预算从已经被证伪的独立 detail / deep ViT / legacy FI-decoder 中撤出，只保留被四数据集 raw evidence 支持的 B4。

### 主贡献 3：Minimal symmetric sub-patch reconstruction

只用：
```text
|B4_A-B4_B| + mean(B4_A,B4_B)
```
构造 change/stable pair evidence，再以 <0.06M expansion 恢复全分辨率。

这里的 head 不宜包装成“全新强大 decoder”；真正贡献是：
- 它足够小；
- 严格时间对称；
- 它验证 B4 是否可独立承担 dense BCD。

---

## 12. CASAA / A1 的最终位置

### 不进入 Run8 final architecture

原因：
1. A1 的“late-stage blocks 8–11 K/V 压缩”不能直接外推到 ViT4 blocks 0–3；
2. 再把 CASAA 塞进 Run8 会增加新的变量；
3. change-aware router 已按 Run3 正式停止。

### 论文中保留为 analysis / secondary evidence

建议最终统一叙事写成：

> **CASA-CD 系统地研究了 BCD 中两类冗余：context-token redundancy 与 representation-depth redundancy。**
>
> - token 维度：Full-Q + K/V=25% 的 content compression 基本无损，但 deployable change-aware routing 未带来收益；
> - depth 维度：B12 在四数据集均劣于 B2–B4，B4 是稳健 semantic source；因此最终模型优先利用 depth redundancy，而不强行保留失败的 change router。

这比“把所有做过的模块都塞进最终网络”更可信。

**不要**把 CASAA 写成最终模型精度提升来源。  
它应该是机制分析资产和 ablation，不是 Run8 主干。

---

## 13. 与 2026 轻量 SOTA 的论文定位

需要明确：MixCDNet 已经到 0.32M，CGLNet 0.99M，SeCoR 2.50M，所以：

> “2.03M 参数”本身不是贡献。

如果 Run8 成立，真正定位应是：

1. 基于 pretrained plain DeiT 的**任务特定 depth 选择证据**；
2. 无独立 CNN detail branch；
3. 以极小 head 完成 dense BCD；
4. 用五轮预注册负结果证明预算为什么这样分配，而不是模块堆叠。

与 SeCoR 的区别：
- SeCoR：opposite-temporal reliable support correction + prototype-prior repair；
- Run8：不做 cross-temporal correction/prototype repair，研究的是 **ViT depth redundancy + terminal semantic source**。

与 CGLNet 的区别：
- CGLNet：高层 semantic guidance + dual-difference + guided multi-scale decoder；
- Run8：没有多尺度 backbone feature；只有单尺度 B4，重点是“为什么在 B4 截断并删除更深层与 detail branch”。

---

## 14. 成功 / 失败后的论文结论边界

### 若四个 80K 全过

可以主张：
- B4 是四数据集稳健 change-sensitive semantic source；
- ViT 12→4 大幅删深度仍保性能；
- detail-free B4-SPE 在约 2.03M / <2G 下跨四数据集达到项目地板；
- depth evidence 与极轻量化存在直接机制联系。

不能主张：
- B4 对所有数据集都是“理论最优 depth”；
- CASAA change router 提升了最终精度；
- 单 seed 证明统计显著性。

### 若 SYSU 过、LEVIR/WHU 失败

只能主张：
- B4-only 在部分场景可行；
- fixed semantic depth 的跨数据集局限仍存在；
- `<3M universal final model` 未被证明。

### 若 SYSU 即失败

结论应非常明确：

> B4 raw ranking 强，但 **ranking quality ≠ detail-free dense reconstruction ability**。

到此停止结构搜索。这个负结论本身与 Run3“router ranking 提高未转化成 F1”形成一致的研究纪律：**raw ranking 只做筛选，不当作端到端性能替代品。**

---

## 15. 立即执行顺序

1. **先只新增** `analyse/run8_b4_dense_recoverability_audit.py`。
2. GPU1 跑 CDD/LEVIR/SYSU/WHU 完整 test，保存 `audit_report.txt`。
3. 人工读取：
   ```text
   [R8-D0-GATE] PASS/FAIL
   ```
4. PASS 后才新建 `b4_spe_head.py` 和模型入口。
5. `smoke_test.py` 跑完 T-R8 全套，特别检查：
   - no detail params；
   - DeiT checksum；
   - temporal symmetry；
   - trainable params；
   - FLOPs。
6. 真实 SYSU dry run 50–100 steps：
   - loss finite；
   - ViT checksum before/after 一致；
   - head grad finite；
   - output 非常数；
   - checkpoint/eval sidecar 正确。
7. R8-1 SYSU 80K。
8. 只根据最后完整 TEST RESULTS 判 gate。
9. PASS 才 LEVIR → WHU → CDD。
10. 每一 gate 后更新 README / experiment_metrics.xlsx / Run8 结果文档，不追着结果修改阈值。

---

## 16. 仍需补充的唯一证据

在花最后一个 SYSU 80K 前，只缺一项：

> **B4 在像素域与边界带上的零训练 dense recoverability 是否相对 B12 保持优势。**

除此之外，不再需要新的 backbone 文献搜索、新 detail 候选、新 loss、新 router。

---

## 17. 最终一句话

**Run8 不再寻找“更聪明的模块”，而是做最后一次最小因果检验：四数据集已经告诉我们 B4 是稳健 semantic source；现在只问——删除独立 detail 与重型 decoder 后，B4 能否用一个约 0.054M 的对称 sub-patch head 独立完成 dense BCD。**

如果能，CASA-CD 得到约 **2.03M / ~1.2G** 的最终极轻量模型，并且轻量化由完整证据链驱动；如果不能，就停止，不再救。

---

## 18. 本轮外部文献核验（2024–2026）

主证据：
- FDAM, ICCV 2025: Frequency-Dynamic Attention Modulation for Dense Prediction  
  https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Frequency-Dynamic_Attention_Modulation_For_Dense_Prediction_ICCV_2025_paper.html
- LaViT, CVPR 2024: You Only Need Less Attention at Each Stage in Vision Transformers  
  https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_You_Only_Need_Less_Attention_at_Each_Stage_in_Vision_CVPR_2024_paper.html
- ResCLIP, CVPR 2025  
  https://openaccess.thecvf.com/content/CVPR2025/html/Yang_ResCLIP_Residual_Attention_for_Training-free_Dense_Vision-language_Inference_CVPR_2025_paper.html
- ViT-CoMer, CVPR 2024  
  https://openaccess.thecvf.com/content/CVPR2024/html/Xia_ViT-CoMer_Vision_Transformer_with_Convolutional_Multi-scale_Feature_Interaction_for_Dense_CVPR_2024_paper.html
- LiFT, ECCV 2024  
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1086_ECCV_2024_paper.php
- EoMT, CVPR 2025  
  https://openaccess.thecvf.com/content/CVPR2025/html/Kerssies_Your_ViT_is_Secretly_an_Image_Segmentation_Model_CVPR_2025_paper.html
- RFL-CDNet, Pattern Recognition 2024  
  https://www.sciencedirect.com/science/article/pii/S0031320324002668

2026 竞争位置补充：
- SeCoR, IEEE JSTARS 2026, 2.50M / 2.66G  
  https://ieeexplore.ieee.org/document/11646457/
- CGLNet, IEEE GRSL 2026, 0.99M / 0.62G  
  https://ieeexplore.ieee.org/document/11664103/
