# CASA-CD Run9：B4-OPRE 可执行预注册方案
## ——冻结 B4 语义 + 共享重叠 Patch Re-Embedding（Overlap Patch Re-Embedding）

> **文档状态**：Run9 预注册草案，可直接存入 `docs/temporary/`
>
> **审查基准**：GitHub `YuqiWang-code/CASA-CD` `main`，本轮读取时 HEAD =
> `aa43ed37aa0305939e0603d2d3e4899f08f08842`（2026-10-01，`update code`）。
>
> **事实权威顺序**：README / Project Settings / Run8 终局文档 / 历史 Run4–7 结果 /
> 当前源码 / 服务器统一说明。正式模型结果只认对应 `train_log.txt` 最后一个完整
> `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块；本文所有 Run4–8 raw
> audit 数字均为**零训练诊断证据，不是正式模型结果**。
>
> **路径勘误**：用户给出的 Run6/Run7 “过去的想法”路径与当前 `main` 不一致；
> 当前仓库中 `CASA-CD_Run6_R6-D0结果与Run6停止.md` 和
> `CASA-CD_Run7_R7-D0结果与CSDP停止.md` 位于 `docs/temporary/` 根目录。
> 本文以仓库实际路径为准。

---

# 0. 结论先行

**Run9 不再继续“换一个轻量 CNN detail / B4-only 解码 / B2 截断 / CASAA router 调参”。**
本轮只预注册一个新主方案：

> **B4-OPRE：Frozen ViT4 B4 Semantic + Overlap Patch Re-Embedding**
>
> 保留经过四数据集 raw audit 支持的 **B4 语义源**；同时不再依赖独立 CNN detail
> branch，而是**原位复用同一个冻结 DeiT-Tiny `patch_embed.proj` 权重**，把 patch
> 投影的采样步长从主干中的 `16` 改为辅助重嵌入中的 `8`，得到 `32×32` 的重叠 patch
> evidence。该 evidence 只作为局部变化补充，由 B4 semantic feature 在 `1/8`
> 尺度门控注入，再做极轻量 PixelShuffle 重建。
>
> **主干的标准 ViT 路径完全不改**：仍然 `patch16/stride16 → Block1…Block4 → final LN`。
> O-PRE 是一条**权重共享、零新增 encoder 参数**的辅助采样路径，不复制 patch
> projection，不新增 backbone，不改变 DeiT 预训练权重的数值。

本轮机制假设不是“P0 本身能做变化检测”，而是：

> **H9：Run8 暴露的主要缺口是固定 16×16 非重叠 patch 网格丢失的 sub-patch
> spatial phase / boundary ordering。单网格 P0 的语义 ranking 很弱（Run6 已证伪），
> 但同一预训练 patch kernel 在更密的重叠采样格上，可能提供与 B4 强语义互补的局部
> 几何 evidence；只有当 B4 认为该位置语义上可疑时才注入它。**

这与 Run6 的 P0 reconstruction、Run8 的 B4-only reconstruction 都是**不同的可证伪命题**。

**预估模型预算**（实现后必须由 smoke 实测，而不是以此估算替代）：

| 组件 | 参数量 | 备注 |
|---|---:|---|
| Frozen DeiT-Tiny depth-4 | 1,976,832 | 已由 Run8 验证 |
| O-PRE encoder 新参数 | **0** | 直接共享 `patch_embed.proj.weight/bias` |
| B4-SPE 原 head | 53,872 | Run8 已 smoke |
| O-PRE `192→32` projector + BN | 6,208 | 新增 |
| semantic gate `32→1` | 33 | 新增 |
| **预计总有效参数** | **2,036,945 ≈ 2.037M** | `<3M` 且 `<2.20M` |
| **预计可训练参数** | **60,113 ≈ 0.060M** | ViT4 全冻结 |

Run8 已实测 B4-SPE 为 **1.2186G FLOPs**。O-PRE 的共享 `16×16, stride=8`
双时相重嵌入理论 MAC 约 **0.302G**，加上 projector/gate 后，预计总量约
**1.53–1.60G**。**正式硬门槛仍设为 ≤2.00G**，以 `fvcore + 新增算子手工核算`
共同验收。

**重要：这只是结构可行性，不代表指标可行性已经成立。** 新目标中 SYSU 需要相对
健康 full12 **+1.86 F1 / +2.77 IoU**，属于真实的方法增益要求。Run9 只有先通过
四数据集零训练 O-PRE complementarity gate，才允许花第一个 80K。

---

# 1. 证据表：现有结果、新目标与逐数据集 gap

## 1.1 新硬目标换算

二分类正类 F1 与 IoU 满足：

\[
IoU=\frac{F1}{2-F1}.
\]

因此本课题新目标对应的最小 IoU 为：

| 数据集 | F1 硬目标 | 对应 IoU 硬目标 |
|---|---:|---:|
| SYSU | **85.00** | **73.91** |
| LEVIR | **92.50** | **86.05** |
| WHU | **95.00** | **90.48** |
| CDD | **98.00** | **96.08** |

## 1.2 与当前最好结果的 gap

| 数据集 | 当前仓库最好可引用结果 | 当前 F1 / IoU | 新目标 F1 / IoU | 需要补的 gap | 证据解释 |
|---|---|---:|---:|---:|---|
| SYSU | **R4-0 healthy full12 frozen** | **83.14 / 71.14** | **85.00 / 73.91** | **+1.86 / +2.77** | 最关键、最困难；健康参考可信 |
| LEVIR | Run1 baseline（ViT 后证实部分死权重） | 91.95 / 85.10 | 92.50 / 86.05 | +0.55 / +0.95 | 该模型存在 ViT collapse，不能作为健康机制参考 |
| LEVIR | **A1 frozen healthy** | **91.84 / 84.91** | **92.50 / 86.05** | **+0.66 / +1.14** | 更应以此估计真实难度 |
| WHU | Run1 baseline（ViT 死权重） | 94.84 / 90.18 | 95.00 / 90.48 | +0.16 / +0.30 | **健康 frozen full12 同协议结果缺失** |
| CDD | Run1 baseline（ViT 死权重） | 97.75 / 95.60 | 98.00 / 96.08 | +0.25 / +0.48 | **健康 frozen full12 同协议结果缺失** |

**结论**：

1. **SYSU 不是“轻量保分”问题。** 从健康 full12 的 83.14 提到 85.00，同时从
   11.754M 降到约 2.04M，需要 +1.86 F1 的结构增益。
2. **LEVIR 也不是只需保留 Run1 91.95。** 健康 frozen A1 是 91.84，因此真实硬 gap
   至少 +0.66 F1。
3. WHU/CDD 的表面 gap 很小，但当前最好数字来自后来被发现 ViT 权重异常的 Run1；
   **本轮不能假设健康模型同样已有 94.84/97.75**。这两项的真实健康 gap 是缺失信息。
4. 因而 Run9 的合理策略必须直接击中一个**已被证据定位、且现有 2.04M 路线没有解决
   的精度瓶颈**，而不是再压参数。

## 1.3 六轮负结果对 Run9 的直接约束

| 负结果 | 已证伪的命题 | 对 Run9 的硬约束 |
|---|---|---|
| Run3 CASAA deployable router | ranking 变好不等于 F1 变好；Top32 precision 提升未转化 | 不做 router-only 精度主线；raw ranking 只做准入，不替代端到端 |
| Run4 Light32/48/PSD | 自定义多层轻量 detail 在 1/4、1/8 表达崩坏 | 不再搜索 LightDetail width/depth，不再构造新 CNN pyramid |
| Run5 Mobile prefix | 成熟预训练轻量 CNN 仍未过严格四项 gate；PSD stem 会漂移 | 不再找 Mobile/Efficient/Shuffle 前缀；不把“预训练 stem”本身包装成创新 |
| Run6 P0 | 单个 canonical P0：SYSU PR-AUC 0.3711；P0+B4 rank fuse 反而稀释 | O-PRE 不得宣称“P0 语义强”；必须证明**改变采样 lattice 后的 complementarity** |
| Run7 B2 CSDP | B2≈B4 不是跨数据集规律 | 固定保留 B4，不再 B2/B3 截断 rescue |
| Run8 B4-only | B4 token ranking 强，但 dense boundary lift 不够 | 不再训练任何 B4-only/no-detail head；新路线必须引入**新的空间信息源** |

---

# 2. 现阶段证据分级

## 2.1 代码/日志直接事实

- corrected DeiT loader 已能原位继承 `patch_embed.proj`、指定 block、final norm，
  pos embedding 14×14→16×16 bicubic。
- ViT4 为 1,976,832 参数。
- Run8 B4-SPE 为 2,030,704 参数、1.2186G，时间交换严格对称，但因 D0 FAIL **未训练**。
- 四数据集 token audit 中，B4 是最稳妥 depth：CDD/SYSU 第一，LEVIR/WHU 第二；
  B12 四数据集均劣于更浅中层。
- Run8 dense audit 中 B4 相对 B12 的 boundary lift：
  CDD −0.0006、LEVIR +0.0164、SYSU +0.0171、WHU +0.0377，仅 1/4 过预注册阈值。
- Run4-D0 的 SYSU 证据显示：轻量 local feature **1/2 尺度并不弱**，Light32 raw
  PR-AUC 0.5546，甚至略高于 ResNet 0.5467；真正崩坏出现在自己堆出的 1/4、1/8。
- ChangeViT 原论文的诊断实验表明 plain ViT 对细节不足，detail-capture 对 LEVIR/WHU
  有大幅增益，说明“global semantic + local detail”这一任务分工本身有依据；但原论文
  的大 ResNet detail 不满足本课题参数目标。

## 2.2 证据支持的推断

- **剩余瓶颈更像 spatial sampling / sub-patch geometry，而不是缺 B4 semantic ranking。**
  B4 已有较强 token change ranking，但 B4-only 的 dense boundary 恢复失败。
- Run6 的 canonical P0 失败并不自动等价于“所有 patch projection 的局部信息都无用”；
  它只证明**固定非重叠 16×16 lattice 上的单网格 P0**不足以担当语义/重建源。
- 若只改变 patch projection 的**采样几何**而严格共享已有预训练 kernel，可在几乎不增
  参数的情况下增加局部相位覆盖，并避免 Run4/5 的独立 CNN detail 参数预算与训练漂移。

## 2.3 Run9 待验证假设

- **H9-A**：共享 patch kernel 的 `stride=8` overlap re-embedding，相比 canonical
  `stride=16` P0，在边界带上提供显著更好的变化排序。
- **H9-B**：O-PRE evidence 与 B4 semantic **互补而非稀释**；经过固定的
  semantic-conditioned proxy 融合后，四数据集 pixel/boundary PR-AUC 有可重复的正 lift。
- **H9-C**：训练后的单尺度 semantic-gated injection 能把这种互补转化成最终 F1/IoU，
  尤其 SYSU 达到 85.00/73.91。
- **H9-D**：共享重嵌入在不复制权重的条件下仍保持完整 DeiT 继承，整体有效参数约
  2.04M、FLOPs <2G。

## 2.4 缺失信息

- WHU/CDD 没有健康 frozen full12 同协议正式结果，不能精确知道其真正的目标 gap。
- 四数据集上尚无 O-PRE raw complementarity 数字。
- 本仓库没有 B4-SPE 训练结果；它被 Run8 gate 正确拦截，不能拿一个未训练模型作为基线。
- 单 seed=16 无法证明普适性；即使四数据集均过目标，也应把主张限定为固定协议结果。

---

# 3. P0 / P1 / P2 问题清单

## P0 正确性：不解决就不能跑 80K

### P0-1：O-PRE 必须**共享** patch_embed 参数，不得复制

正确实现必须类似：

```python
w = self.vit.patch_embed.proj.weight
b = self.vit.patch_embed.proj.bias
x_pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
ore = F.conv2d(x_pad, w, b, stride=8, padding=0)  # B×192×32×32
```

禁止：

```python
self.ore = nn.Conv2d(3, 192, 16, stride=8, padding=4)
self.ore.load_state_dict(...)
```

后者会注册一份新的 147k 左右参数，并把“零新增 encoder 参数、严格共享预训练 kernel”
这一机制改成另一个模型。

### P0-2：主 ViT 的 canonical patchification 不得改变

主干仍是 `kernel=16 / stride=16 / no overlap`。O-PRE 只是旁路。不能为了方便把
`PatchEmbed.stride` 全局改成 8，否则 token 数从 256 改变，位置编码、Block 计算量、
预训练语义和全部 Run4–8 证据都失效。

### P0-3：O-PRE 不加 32×32 positional embedding

O-PRE 是保留 2D 网格的局部 evidence，不进入 Transformer。不得新建 32×32
learnable pos embedding，也不得把 16×16 pos embedding粗暴插值后相加；否则新增了
另一个未经证据支持的语义源。

### P0-4：时间交换严格对称

最终 head 只能使用：

\[
|A-B|,\quad \frac{A+B}{2},
\]

以及由它们派生的 gate。必须满足：

\[
\max |f(A,B)-f(B,A)| < 10^{-6}
\]

（eval / FP32 smoke）。

### P0-5：冻结必须覆盖 ViT4 和共享 O-PRE kernel

`--freeze_vit 1` 后 `encoder.vit.patch_embed.proj.*` 的 `requires_grad=False`。
由于 O-PRE 引用的是同一 Parameter，它也自然冻结。dry run 前后 checksum 必须逐位一致。

### P0-6：训练/eval 架构侧写必须一致

新增：

- `detail_mode = opre`
- `head_mode = opre_spe`

并同时修改 `train.py` / `eval.py` choices、`Trainer`、`arch.json` 验证路径。
禁止 checkpoint 用 `opre_spe` 训练、eval 却静默实例化 `b4_spe/legacy`。

---

## P1 方法瓶颈

### P1-1：SYSU 需要的是 +1.86 F1，不是 +0.2 的边界修补

因此 raw gate 对 SYSU 单独设置更严格要求。若 O-PRE 只能产生轻微 boundary lift，
不值得用一个 80K 去赌 85.0。

### P1-2：O-PRE 可能对辐射差/配准误差敏感

overlap patch evidence 本质上更局部，容易把照明、阴影、微错位当成变化。
解决方式不是新 loss，而是**B4 semantic gate**：local evidence 只作为 residual，
不能独立决定 foreground。

### P1-3：32×32 仍不是像素级

O-PRE 把采样步长从 16 降到 8，但最终仍要从 1/8 恢复到像素域。若 D0 boundary
互补不足，继续把 stride 改成 4/6/12 会变成 sweep，**预注册禁止**。

### P1-4：冻结 B4 可能形成语义天花板

这是有意接受的风险。现有训练过的 ViT 会 collapse，且 B4 raw semantic 已强。
Run9 不解冻 ViT。若 SYSU 正式 80K 仍明显低于目标，不能用“解冻一点/小 LR”
在同一轮 rescue；另开下一轮重新论证。

---

## P2 实验工程

### P2-1：FLOPs 需要双口径

`fvcore` 继续作为统一报告口径；同时对 O-PRE 共享 conv 手算并记录：
`32×32×192×3×16×16×2 ≈ 0.302G MAC`。若 fvcore 报 unsupported op，
不能把漏算当作低 FLOPs。

### P2-2：Run8 的 200 次自动 retry 不适合 Run9

Run9 脚本最多自动 retry **3 次**。同一确定性错误连续失败后停止，避免 200 次反复
写日志/覆盖运行状态。

### P2-3：best 与 last 的角色不混淆

- `last.pth`：resume，包含 optimizer / epoch / arch；
- `best_F1=*.pth`：裸 state_dict；
- 正式结果：只取 `train_log.txt` 最后一个完整 TEST RESULTS 区块。

---

# 4. 候选机制比较：只选一个主方案

| 候选 | 能否解释 Run8 失败 | 参数/FLOPs | 与负结果冲突 | 创新边界 | 决策 |
|---|---|---|---|---|---|
| **A. B4 + O-PRE（共享重叠 Patch Re-Embedding）** | **直接针对固定 patch lattice 的 sub-patch 丢失** | ~2.04M / 预计1.5G | 不重复 Run6：改的是 sampling geometry，且只做 gated local evidence | overlap embedding 本身非新；“冻结预训练 patch kernel 权重共享重采样 + B4 gated bi-temporal residual”是本课题具体机制 | **主方案** |
| B. B4 + frozen ResNet stem phase | 能提供 1/2 local cue | ~2.04M，但额外 CNN | 接近“回 detail branch”；而 Edge-CVT / LHICD 已使 edge/high-frequency 叙事拥挤 | 新意弱，容易被审稿人视为 tiny CNN detail | 否决 |
| C. B3/B4 adaptive depth fusion | 解决 dataset-dependent depth | 很低 | Run7 已说明 B4 是稳健点；Run8 表明 dense boundary 仍缺 | 接近 RFL-CDNet intermediate fusion；没有新空间信息 | 否决 |
| D. CASAA-v3 / 新 router | 解决 token 冗余 | 参数低 | Run3 已两次证明 ranking→F1 不成立；oracle SYSU 也只有 83.52 | 效率资产可保留，不能承担新 85 目标 | 否决 |

---

# 5. 文献核验与创新边界

本轮先检查仓库 `docs/参考文献/文献索引.md` 的 11 篇主证据，已有材料已经覆盖
“plain ViT inner-patch 不足 / 中层定位 / 深层 frequency vanishing / 轻量 multi-scale
融合拥挤”等关键问题，因此**不需要重新做大范围文献扫库**。为了避免把
“edge / high-frequency CNN branch”误包装成新意，本轮额外补查了两篇 2025 权威 SCI。

## 5.1 已有主证据

1. **ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions**
   - CVPR 2024，CCF-A。
   - 官方：https://openaccess.thecvf.com/content/CVPR2024/html/Xia_ViT-CoMer_Vision_Transformer_with_Convolutional_Multi-scale_Feature_Interaction_for_Dense_CVPR_2024_paper.html
   - 关系：支持 plain ViT 的 patch/grid 对 dense prediction 不充分；但它通过卷积多尺度
     交互重构 ViT，B4-OPRE 不改 Transformer block，只复用现有 patch projection。

2. **LiFT: A Surprisingly Simple Lightweight Feature Transform for Dense ViT Descriptors**
   - ECCV 2024；**CCF 官方目录为 B**，不是 CCF-A。
   - 官方：https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1086_ECCV_2024_paper.php
   - 关系：证明轻量 transform 可以把 pretrained ViT descriptor 变 dense；
     B4-OPRE 不引入 LiFT 的自监督 feature-matching loss，也不复制其 transform。

3. **You Only Need Less Attention at Each Stage in Vision Transformers（LaViT）**
   - CVPR 2024，CCF-A。
   - 官方：https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_You_Only_Need_Less_Attention_at_Each_Stage_in_Vision_CVPR_2024_paper.html
   - 关系：为“更深 attention 并非必需”提供外部解释，与本仓库 B4>B12 相符；
     它不解决 sub-patch dense recovery。

4. **Frequency-Dynamic Attention Modulation for Dense Prediction（FDAM）**
   - ICCV 2025，CCF-A。
   - 官方：https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Frequency-Dynamic_Attention_Modulation_For_Dense_Prediction_ICCV_2025_paper.html
   - 关系：解释 ViT 深层 spatial detail/frequency 衰减；本方案不做 attention
     frequency modulation，而是在 B4 外引入权重共享的局部采样 lattice。

5. **RFL-CDNet: Towards Accurate Change Detection via Richer Feature Learning**
   - Pattern Recognition 2024，权威 SCI。
   - 官方：https://www.sciencedirect.com/science/article/pii/S0031320324002668
   - 关系：它强调 intermediate multi-level feature 全利用/深监督；本方案反而固定
     单一 B4 semantic source，不融合 B1/B2/B3。

6. **ChangeViT**
   - Pattern Recognition 172 (2026) 112539，权威 SCI。
   - DOI：https://doi.org/10.1016/j.patcog.2025.112539
   - 官方代码：https://github.com/zhuduowang/ChangeViT
   - 关系：其核心是 plain ViT global semantic + CNN detail + cross-attention。
     B4-OPRE 保留“semantic/detail 互补”的任务事实，但把独立 2.7M ResNet detail
     改成**零新增 encoder 参数的共享 patch re-embedding**。

7. **SAT: Selective Aggregation Transformer for Image Super-Resolution**
   - arXiv 2604.07994；官方仓库标注 CVPR 2026 Findings，**不写 main-track**。
   - arXiv：https://arxiv.org/abs/2604.07994
   - 官方代码：https://github.com/PhuTran1005/SAT
   - 关系：CASAA 的来源；Run3 后保留为 token redundancy/效率背景，不进入 Run9 精度主线。

## 5.2 本轮新增的“创新性防撞”证据

8. **Edge-CVT: Edge-informed CNN and vision transformer for building change detection in satellite imagery**
   - ISPRS Journal of Photogrammetry and Remote Sensing, 227 (2025) 48–68，权威 SCI。
   - DOI：https://doi.org/10.1016/j.isprsjprs.2025.05.021
   - 出版社：https://www.sciencedirect.com/science/article/pii/S0924271625002084
   - 关系：说明“CNN + ViT + edge information”已经有直接高水平先例。因此 Run9
     **不采用 Edge CNN / edge loss / edge supervision 作为贡献点**。

9. **Learning From Human Insights for Remote Sensing Change Detection**
   - IEEE TGRS 63 (2025)，权威 SCI；DOI `10.1109/TGRS.2025.3642171`。
   - IEEE：https://ieeexplore.ieee.org/document/11289560/
   - 关系：该文已有 high-frequency residual / property-difference 机制。因此
     “加一个高频残差分支”本身也不是足够创新的 Run9 主线。

## 5.3 本文可主张与不可主张的创新边界

**不能主张**：
- overlapping patch embedding 是首创；
- edge/high-frequency detail 是首创；
- global-local fusion 是首创；
- semantic gate 是首创。

**可作为论文方法贡献去验证的具体命题**：

> 对于冻结的 plain ViT change detector，固定 non-overlap patchification 造成的
> **sampling-lattice bottleneck** 可以通过**同一预训练 patch projection 的权重共享
> overlap re-embedding**补偿；局部重嵌入只改变采样密度，不复制 encoder 参数，
> 并由经 depth audit 选出的 B4 semantic evidence 条件化注入，从而在约 2.04M 参数内
> 同时保留语义判别与 sub-patch spatial evidence。

截至本轮检查的项目 11 篇文献 + 上述两篇补查，**未发现完全同构的
“frozen pretrained ViT patch kernel 共享、以更密 stride 重采样作为 BCD 辅助局部 evidence，
再由审计选出的 B4 语义门控注入”高水平 2024–2026 RSBCD 方法**。这只能作为当前
检索范围内的创新空白，不能写成“全领域首次”绝对表述。

---

# 6. 首选机制：B4-OPRE 数学定义与逐层数据流

## 6.1 主语义路径：不改的 Frozen ViT4

输入：

\[
I_A,I_B\in\mathbb{R}^{3\times256\times256}.
\]

canonical patch embedding：

\[
X_t^0=PE_{16,16}(I_t)+P,\quad t\in\{A,B\},
\]

其中 `PE` 为 DeiT-Tiny 预训练 `Conv2d(3,192,k=16,s=16)`，`P` 为由 14×14
bicubic 到 16×16 的预训练位置编码。

只执行前 4 个 DeiT block：

\[
X_t^l=B_l(X_t^{l-1}),\quad l=1,\ldots,4
\]

并用当前 repo 的 final LN：

\[
V_t=LN(X_t^4)\in\mathbb{R}^{256\times192}.
\]

reshape：

\[
V_t\rightarrow\mathbb{R}^{192\times16\times16}.
\]

**ViT4 全冻结，数值继承必须与 corrected DeiT loader 逐位一致。**

## 6.2 O-PRE：只改变采样 lattice，不增加 encoder 参数

取同一个：

\[
W_p,b_p = \texttt{vit.patch\_embed.proj.weight/bias}.
\]

先 reflection pad 4：

\[
\bar I_t=\operatorname{ReflectPad}_{4}(I_t),
\]

再用**完全相同的 kernel**，但 stride=8：

\[
O_t=\operatorname{Conv2D}(\bar I_t;W_p,b_p,k=16,s=8)
\in \mathbb{R}^{192\times32\times32}.
\]

这里没有：
- 新卷积权重；
- 新 positional embedding；
- 新 Transformer；
- 新 CNN detail backbone。

因此 O-PRE 新 encoder 参数量为 **0**。

## 6.3 B4 对称 semantic pair descriptor

\[
D_s=
\left[
|V_A-V_B|,
\frac{V_A+V_B}{2}
\right]
\in\mathbb{R}^{384\times16\times16}.
\]

逐位置 `Linear/Conv1×1 384→96`：

\[
Z_s=P_s(D_s)\in\mathbb{R}^{96\times16\times16}.
\]

参数：
`384×96 + 96 = 36,960`。

## 6.4 16→32 semantic expansion

保持 Run8 已 smoke 的第一段：

1. `Conv1×1 96→128`, bias=False
2. `BatchNorm2d(128)`
3. `ReLU(inplace=True)`
4. `PixelShuffle(2)` → `32×32×32`
5. `DWConv3×3, 32→32, groups=32, s=1, p=1`, bias=False
6. `BatchNorm2d(32)`
7. `ReLU(inplace=True)`

得到：

\[
S_{32}\in\mathbb{R}^{32\times32\times32}.
\]

## 6.5 O-PRE 局部双时相 evidence

严格对称局部 difference：

\[
D_o=|O_A-O_B|
\in\mathbb{R}^{192\times32\times32}.
\]

仅一个 projector：

1. `Conv1×1 192→32`, bias=False
2. `BatchNorm2d(32)`
3. `ReLU(inplace=True)`

得到：

\[
L_{32}\in\mathbb{R}^{32\times32\times32}.
\]

**不加入 local mean。** 原因是 O-PRE 的职责仅是提供局部变化 evidence；
稳定场景语义由 B4 提供，避免把局部外观本身再次塞进 head。

## 6.6 Semantic-gated residual injection

gate 只看 semantic：

\[
G_{32}=\sigma(\operatorname{Conv}_{1\times1}(S_{32}))
\in[0,1]^{1\times32\times32}.
\]

融合：

\[
F_{32}=S_{32}+G_{32}\odot L_{32}.
\]

解释：

- B4 负责“这里像不像真实变化”；
- O-PRE 负责“这个变化在固定 16×16 patch 内部的局部相位/边界证据是什么”；
- O-PRE 永远以 residual 形式加入，避免自己成为一个容易被照明/错位骗过的变化判别器。

## 6.7 32→256 重建

继续复用 Run8 B4-SPE 已 smoke 的剩余三段：

### Stage 32→64

- `Conv1×1 32→64`, bias=False
- BN64 + ReLU
- PixelShuffle2 → `16×64×64`
- DW3×3 16 groups + BN16 + ReLU

### Stage 64→128

- `Conv1×1 16→64`, bias=False
- BN64 + ReLU
- PixelShuffle2 → `16×128×128`
- DW3×3 16 groups + BN16 + ReLU

### Stage 128→256

- `Conv1×1 16→16`, bias=False
- BN16 + ReLU
- PixelShuffle2 → `4×256×256`
- DW3×3 4 groups + BN4 + ReLU

### Classifier

- `Conv1×1 4→1`, bias=False
- sigmoid

最终：

\[
\hat M\in[0,1]^{1\times256\times256}.
\]

## 6.8 归一化 / 激活 / 残差 / 共享权重汇总

| 模块 | kernel/stride | Norm | Act | Residual | 权重 |
|---|---|---|---|---|---|
| ViT patch embed | 16/16 | 无 | 无 | — | DeiT frozen |
| ViT blocks 1–4 | 原 DeiT | LN | GELU | Transformer residual | DeiT frozen |
| O-PRE | 16/8，reflect pad4 | 无 | 无 | — | **与 patch embed 完全共享** |
| semantic pair proj | 1×1 / 1 | 无 | 无 | — | trainable |
| semantic expansion | 1×1 + PS2 + DW3 | BN | ReLU | — | trainable |
| local projector | 1×1 / 1 | BN | ReLU | — | trainable |
| semantic gate | 1×1 / 1 | 无 | sigmoid | — | trainable |
| fusion | — | — | — | `S + G⊙L` | 无参数 |
| later expansion | 1×1 + PS2 + DW3 | BN | ReLU | — | trainable |

## 6.9 训练图与推理图

### 训练图

```text
A/B
 ├─ Frozen canonical DeiT patch16 → Frozen Block1..4 → B4 semantic
 └─ same frozen patch kernel, stride8 overlap re-embedding → local pair evidence
                         ↓
          symmetric pair descriptor / semantic gate
                         ↓
                ultra-light PixelShuffle head
                         ↓
                    BCE + Dice
```

只有约 0.060M head 参数更新。

### 推理图

与训练图完全相同，仅无 backward：
- 不需要 teacher；
- 不需要 cache；
- 不需要训练期辅助 head；
- 不需要剪枝/量化；
- 无部署删除步骤。

## 6.10 预训练继承

- 主 ViT：沿用 `Encoder._load_pretrained_deit()` corrected loader。
- O-PRE：**不 load 第二份权重**，forward 时直接引用现有
  `self.vit.patch_embed.proj.weight/bias`。
- smoke 必须验证：
  1. `stride=16, pad=0` 的 functional conv 与 `patch_embed.proj(x)` max error `<1e-6`；
  2. O-PRE 不产生独立 `state_dict` 参数 key；
  3. dry run 后 patch_embed checksum 仍 exact equal。

---

# 7. 参数量 / FLOPs 与新目标可行性

## 7.1 参数量

预计：

\[
1,976,832 + 53,872 + 6,208 + 33
=2,036,945.
\]

因此：

- `<3M`：有约 0.963M margin；
- 工程目标 `≤2.20M`：有约 0.163M margin；
- 相对健康 ChangeViT full12 11.754M：有效参数下降约 82.7%。

实现后若 **effective >2.10M**，说明代码注册了不该出现的参数，先查重复
patch embedding / dead modules，不允许直接把门槛放宽。

## 7.2 FLOPs

已知 Run8 B4-SPE：1.2186G。

新增 O-PRE 主项：

\[
32\cdot32\cdot192\cdot3\cdot16\cdot16\cdot2
\approx0.302G.
\]

其余 1×1 projector / gate 约为千万级以下 MAC。因此预计约 1.53–1.60G。

预注册：
- **FLOPs hard gate：≤2.00G**
- `unsupported_ops` 必须列出；
- 共享 O-PRE conv 手工 0.302G 必须附到 smoke 报告，防止 fvcore 漏算。

## 7.3 为什么它有资格挑战新目标

### SYSU：83.14 → 85.00

gap 最大。Run8 已说明 B4 semantic ranking 本身不是主要问题，但 B4-only dense
boundary 不足；Run4 又说明低层 1/2 local evidence 本身可以很强，坏的是后续轻量层。
O-PRE 试图在**不学习一个新的深层 detail encoder**前提下补回更密的低层采样。
这是目前证据链中少数正面针对“最后缺口”的方案。

但必须强调：**没有任何现有结果保证 +1.86 F1。** 因此 SYSU 被放在正式训练第一位，
而且 D0 对 SYSU 单独设强 gate。

### LEVIR：91.84 healthy → 92.50

需要 +0.66 F1 / +1.14 IoU。建筑小目标、变化像素稀少，对边界排序更敏感；
若 O-PRE 只增加 false positive，LEVIR 会第一时间暴露。因此它排正式训练第二。

### WHU：目标 95.00

观测最好 94.84，但健康 frozen full12 缺失。WHU 的 B4 token PR-AUC 只有 0.3449，
比 SYSU 低很多；O-PRE 必须改善小建筑局部定位且不能被阴影骗。若 D0 WHU 直接
严重 pixel PR 下降，Run9 不应进入训练。

### CDD：目标 98.00

目标只比观测最好高 0.25 F1，但 CDD 需要抗伪变化。semantic gate 的意义在这里最
重要：O-PRE 不能独立决策，只对 B4 semantic evidence 做 residual correction。

---

# 8. Run9-D0：零训练 O-PRE Complementarity Audit

**任何模型代码 80K 之前，先只写 audit。**

## 8.1 数据与固定来源

- 四数据集完整 test：
  `CDD-CD-256, LEVIR-CD-256, SYSU-CD-256, WHU-CD-256`
- frozen corrected DeiT depth-12，一次前向抽 B4；
- canonical P0：现有 `stride=16` patch projection；
- O-PRE：同权重 `stride=8, reflect pad4`；
- SYSU 继续跑 R4-1 ResNet 1/8 positive control：
  PR-AUC 0.6535 / Top32 precision 0.5948，容差 ±0.005。

## 8.2 三张零训练 score map

### Canonical P0

\[
s_{P0}=1-\cos(P^A_{16},P^B_{16})
\]

16×16 → per-image ordinal rank → bilinear 256×256。

### B4

\[
s_{B4}=1-\cos(B4_A,B4_B)
\]

16×16 → rank → bilinear 256×256。

### O-PRE

\[
s_O=1-\cos(O_A,O_B)
\]

32×32 → rank → bilinear 256×256。

## 8.3 固定 proxy fusion

**不调 λ。固定 λ=0.5。**

\[
U_{proxy}=U_{B4}\cdot(1+0.5U_O).
\]

含义：O-PRE 只能重排/增强已有 B4 semantic suspicion，不能在 B4 低响应区完全独立
制造 change response。该 proxy 只用于**验证互补性**，不是训练 head 的替代。

## 8.4 指标

对 P0 / O-PRE / B4 / proxy 均报告：

- 全像素 PR-AUC；
- Run8 同定义 boundary-band PR-AUC；
- changed patch 分桶 1–16 / 17–64 / >64 的 Top8 hit、Top32 coverage；
- 仅诊断：O-PRE 与 B4 score Spearman（过高说明只是冗余，过低不自动代表互补）。

## 8.5 预注册 gate

### G0：改变 sampling lattice 必须真的比 canonical P0 更有边界价值

至少 **3/4** 数据集：

\[
PR^{bnd}_{OPRE}\ge PR^{bnd}_{P0}+0.03.
\]

### G1：与 B4 的 fixed proxy 必须形成实质 boundary lift

至少 **3/4** 数据集：

\[
PR^{bnd}_{proxy}\ge PR^{bnd}_{B4}+0.02.
\]

### G2：不能用边界提升换全局 pixel 崩坏

全部 **4/4**：

\[
PR^{pix}_{proxy}\ge PR^{pix}_{B4}-0.005,
\]

且至少 **2/4**：

\[
PR^{pix}_{proxy}\ge PR^{pix}_{B4}+0.01.
\]

### G3：SYSU 单独强 gate

SYSU 必须同时：

\[
\Delta PR^{bnd}\ge +0.03,
\qquad
\Delta PR^{pix}\ge +0.015.
\]

### 总判定

```text
[R9-D0-GATE] PASS  iff  G0 & G1 & G2 & G3
```

任何一项失败：

> **Run9 B4-OPRE 停止；不实现训练模型，不启动 80K。**

## 8.6 D0 失败后禁止 rescue

- 不把 stride 8 改成 4 / 6 / 12；
- 不改 reflect pad；
- 不换 cosine→L1/L2 后重新刷 gate；
- 不加多 phase ensemble；
- 不把 P0/B1/B2 拼回；
- 不重新加 ResNet/Mobile detail；
- 不放宽 G0–G3；
- 不训练一次“看看端到端会不会救回来”。

---

# 9. 逐文件修改清单

只有 **R9-D0 PASS 后** 才允许执行模型改动。

## 9.1 新增

### `models/model/opre_head.py`

新增 `OPREHead`：
- B4 symmetric pair `384→96`;
- semantic 16→32 expansion；
- O-PRE abs diff `192→32`;
- semantic gate；
- 32→64→128→256 PixelShuffle；
- sigmoid output。

### `analyse/run9_overlap_reembedding_audit.py`

由 Run8 audit 改：
- 加 canonical P0；
- 加 functional shared O-PRE；
- 加 proxy；
- 实现 G0–G3；
- 返回码：
  - PASS=0
  - FAIL=2
  - AUDIT_INVALID=3。

### `train_scripts/UltraLight/Run9/`

新增：

```text
README.md
audit_R9_D0_OPRE.sh
dryrun_R9_1_SYSU.sh
train_R9_1_B4_OPRE_SYSU.sh
train_R9_2_B4_OPRE_LEVIR.sh
train_R9_3_B4_OPRE_WHU.sh
train_R9_4_B4_OPRE_CDD.sh
train_R9_A1_NOGATE_SYSU.sh      # 仅四数据集全部达标后允许
```

## 9.2 修改

### `models/model/encoder.py`

新增 `detail_mode='opre'`：

```text
canonical: vit(x) → B4
aux:       overlap_patch_capture(x) → 192×32×32
return:    [ore, b4]
```

新增 `overlap_patch_capture()`，必须用 functional conv 引用现有 patch projection。

### `models/model/trainer.py`

新增：

```python
elif head_mode == "opre_spe":
    self.decoder = OPREHead()
```

并约束：

```text
detail_mode='opre' requires vit_depth=4
head_mode='opre_spe' requires detail_mode='opre'
```

### `models/train.py`

- choices 增加 `opre` / `opre_spe`;
- 不改 optimizer / loss / LR / warmup / max_steps；
- `arch.json` 仍记录 `{vit_depth, detail_mode, head_mode, mode}`，足以唯一标识 Run9；
- 建议新增日志：
  `[OPRE-KERNEL] 16`
  `[OPRE-STRIDE] 8`
  `[OPRE-PADDING] reflect4`
  `[OPRE-WEIGHT-SHARED] 1`。

### `models/eval.py`

同步 choices，保持 `arch.json` mismatch 拒绝。

### `models/smoke_test.py`

加 T-R9 系列，见下一节。

---

# 10. Smoke / dry run 预注册清单

## 10.1 无真实数据 smoke

### T-R9-0：functional conv 等价性

在 `stride=16 / no pad`：

```text
patch_embed.proj(x)
vs
F.conv2d(x, same_weight, same_bias, stride=16)
```

`max_abs_err < 1e-6`。

### T-R9-1：O-PRE shape

`B×3×256×256 → B×192×32×32`。

### T-R9-2：无 duplicate params

state_dict 中不得出现：

```text
encoder.ore.weight
encoder.opre.weight
```

之类独立 patch kernel。

### T-R9-3：预训练逐位继承

- patch_embed；
- blocks 0–3；
- final norm；
- interpolated pos_embed；

与 corrected DeiT reference exact equal。

### T-R9-4：完整输出

`pred.shape == (B,1,256,256)`，finite，范围 [0,1]。

### T-R9-5：时间交换对称

eval FP32：

```text
max_abs(pred(A,B)-pred(B,A)) < 1e-6
```

### T-R9-6：梯度图

- `encoder.vit.*.grad is None`；
- OPRE 因共享 frozen patch weight，无独立 grad；
- `decoder.pair / local_proj / gate / classifier` grad finite & nonzero。

### T-R9-7：参数硬门槛

```text
TOTAL == EFFECTIVE
EFFECTIVE <= 2.10M
TRAINABLE <= 0.065M
```

预计约：

```text
effective = 2.036945M
trainable = 0.060113M
```

允许实现细节带来的极小计数差，但**不允许超过门槛后改门槛**。

### T-R9-8：FLOPs

```text
fvcore <= 2.00G
```

附：
- unsupported op 列表；
- O-PRE shared conv 手工约 0.302G；
- 若 fvcore 与手算明显冲突，先解决口径，不训练。

### T-R9-9：arch sidecar

训练写出：

```json
{
  "vit_depth": 4,
  "detail_mode": "opre",
  "head_mode": "opre_spe",
  "mode": "baseline"
}
```

故意用错误 `detail_mode/head_mode` 跑 eval，必须非 0 退出并显示
`[ARCH-MISMATCH]`。

## 10.2 真实 SYSU dry run

仅 D0 PASS + smoke 全绿后：

- `max_steps=60`；
- batch16；
- GPU1；
- 独立 `_dryrun` checkpoint/log；
- 验收：
  - exit=0；
  - loss finite；
  - output 非 NaN/Inf；
  - ViT4 checksum dryrun 前后 exact；
  - `patch_embed.proj` exact；
  - no ResNet/Mobile/Light/PSD module；
  - `arch.json` 正确；
  - `last.pth` 能 resume；
  - eval 正确架构能加载；
  - 错误架构被拒绝。

dry run **不看 F1 高低，不用它调结构**。

---

# 11. 正式实验设计与 80K 总预算

固定协议全部不变：

```text
Loss        = BCE + Dice
Optimizer   = Adam(lr=2e-4, betas=(0.9,0.99), wd=1e-4)
LR          = poly^0.9 + 200 iter warmup
max_steps   = 80000
batch       = 16
input       = 256×256
seed        = 16
validation  = test-as-val
best        = test F1
GPU         = GPU1 单卡
ViT4        = frozen
```

## 11.1 正式 Run 顺序

### R9-1 — SYSU：硬目标先行

唯一变量：相对已存在的 Run8 B4-SPE 架构定义，加入 O-PRE local evidence +
semantic-gated residual；训练协议不变。

Gate：

```text
F1  >= 85.00
IoU >= 73.91
effective <= 2.10M
FLOPs <= 2.00G
```

**不过即停止 Run9。** 不跑 LEVIR/WHU/CDD，不做 no-gate ablation。

原因：SYSU 需要最大方法增益。如果主方案连 SYSU 都不能达到终局门槛，
训练另外三个数据集只会消耗预算而不能让“全部达标”的课题目标成立。

### R9-2 — LEVIR

仅 R9-1 PASS：

```text
F1  >= 92.50
IoU >= 86.05
```

不过 → Run9 失败，停止后续正式数据集，不改结构救。

### R9-3 — WHU

仅 R9-1/2 PASS：

```text
F1  >= 95.00
IoU >= 90.48
```

不过 → Run9 失败，停止。

### R9-4 — CDD

仅 R9-1/2/3 PASS：

```text
F1  >= 98.00
IoU >= 96.08
```

四项全部通过，才可以把 B4-OPRE 称为**满足本课题当前终局目标的候选主模型**。

## 11.2 条件式最小消融：R9-A1 NOGATE / SYSU

**只有四数据集全部达到硬目标后才跑。**

唯一变量：

\[
F_{32}=S_{32}+G_{32}\odot L_{32}
\]

改为：

\[
F_{32}=S_{32}+L_{32}.
\]

其它完全一致。

目的：验证 O-PRE 的 local evidence 是否需要 B4 semantic conditioning，
而不是证明“gate 本身新颖”。

判读：

- 若主模型比 NOGATE **F1 ≥ +0.30 或 IoU ≥ +0.45**：支持 semantic conditioning
  对抑制局部伪变化有实际作用；
- 若差异低于该阈值：论文中不把 gate 单独作为贡献，只把它视为实现细节；
- 不因消融结果再训练第二种 gate。

## 11.3 80K 总预算

**最大正式 80K 数量 = 5 个：**

1. SYSU main
2. LEVIR main
3. WHU main
4. CDD main
5. SYSU NOGATE conditional ablation

但采用逐 gate 启动：

- D0 FAIL：**0 个 80K**
- SYSU FAIL：**1 个**
- LEVIR FAIL：**2 个**
- WHU FAIL：**3 个**
- CDD FAIL：**4 个**
- 四数据集全 PASS：**4 个**
- 全 PASS 后做最小机制消融：**5 个**

已知 SYSU 单个 80K 约 3–3.5h，因此本轮不会预先一次性占用 5×80K；
严格串行 gate 执行。

## 11.4 正式结果记录

每个 run 都必须在最后完整 TEST RESULTS 中记录：

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
BEST-F1 / epoch
```

README 更新不得只抄 `best_F1=*.pth` 文件名。

---

# 12. 失败解释与停止规则

## Stop-0：D0 raw gate FAIL

结论：

> overlap sampling 没有形成足够的局部 complementarity。

动作：
- 不实现模型；
- 不调 stride / padding / score；
- 回到文献和错误画像，开 Run10。

## Stop-1：参数或 FLOPs 超门槛

若共享权重正确实现仍：

```text
effective > 2.10M
or FLOPs > 2.00G
```

先审计重复 module / FLOPs 漏算；若确实超门槛，Run9 不训练。
不靠“<3M 其实还够”临时放宽本轮工程预注册门槛。

## Stop-2：SYSU <85.00 或 IoU <73.91

即使得到 84.8，也判 FAIL。

不能：
- stride8→4；
- proj32→48/64；
- 多加一个 refinement block；
- 解冻 patch embed；
- 修改 loss；
- 继续 120K；
- 换 seed；
- 加 edge loss。

## Stop-3：后续任一数据集不达标

新硬目标要求四数据集全部达到；单个数据集 failure 即说明 Run9 不是最终解。

下一轮必须先做该数据集 error audit，再重新提出可证伪假设。

## Stop-4：指标过线但精度来源不可解释

若最终过线，但：
- O-PRE gate 激活与边界/小目标没有对应关系；
- no-gate 与 main 无差；
- raw D0 与端到端完全相反；

则论文中可以报告模型结果，但**不能把未经支持的“sub-patch recovery”当成既成事实**。
需要降低机制主张强度。

---

# 13. 启动顺序、路径与精确恢复

## 13.1 D0

脚本：

```text
train_scripts/UltraLight/Run9/audit_R9_D0_OPRE.sh
```

输出：

```text
/home/yqwang/outputs/CASA-CD/UltraLight/Run9/
  R9_D0_OPRE_AUDIT/audit_report.txt
```

命令风格：

```bash
export CUDA_VISIBLE_DEVICES=1
cd /home/yqwang/projects/CASA-CD
bash train_scripts/UltraLight/Run9/audit_R9_D0_OPRE.sh
```

脚本内部 `--gpu_id 0`，因为 GPU1 被映射成唯一可见的逻辑 0。

## 13.2 dry run

Checkpoint：

```text
/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/UltraLight/Run9/
  R9_1_B4_OPRE/SYSU-CD-256
```

Log：

```text
/home/yqwang/outputs/CASA-CD/_dryrun/UltraLight/Run9/
  R9_1_B4_OPRE/SYSU-CD-256/train_log.txt
```

## 13.3 正式路径

### SYSU

```text
CKPT:
/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run9/
  R9_1_B4_OPRE/SYSU-CD-256

LOG:
/home/yqwang/outputs/CASA-CD/UltraLight/Run9/
  R9_1_B4_OPRE/SYSU-CD-256/train_log.txt
```

### LEVIR

```text
.../Run9/R9_2_B4_OPRE/LEVIR-CD-256
```

### WHU

```text
.../Run9/R9_3_B4_OPRE/WHU-CD-256
```

### CDD

```text
.../Run9/R9_4_B4_OPRE/CDD-CD-256
```

### Conditional ablation

```text
.../Run9/R9_A1_NOGATE/SYSU-CD-256
```

## 13.4 精确 resume

当前 `train.py` 会自动检查：

```text
<ckpt_dir>/last.pth
```

其中含：
- state_dict；
- optimizer；
- epoch；
- best_f1；
- best_epoch；
- iters_per_epoch；
- arch。

因此断点恢复时：

1. **不要删除/改名原 `last.pth`**；
2. 不改 script 的 architecture args；
3. 先检查 `arch.json`；
4. 重新运行同一脚本即可 auto-resume；
5. 若手动恢复，则显式
   `--resume <same_ckpt_dir>/last.pth`；
6. 重启后确认日志出现 `[RESUME] loading .../last.pth`；
7. checksum 再查 frozen ViT4。

自动 retry 上限改成 **3**；第三次仍失败就人工审计。

---

# 14. 论文机制故事与可解释性

## 14.1 论文不应再讲“我们不断轻量化 ChangeViT”

更强的故事是：

### 事实 1：semantic depth 有冗余

12→4 block 在健康 SYSU 只损失约 0.37 F1；
四数据集 raw audit 中 B4 是最稳健深度，B12 反而全面变差。

### 事实 2：但“有语义”不等于“能做 dense boundary”

Run8 直接证明：B4 token ranking 的优势投影到 pixel/boundary 后无法在四数据集稳定成立。
因此真正缺的是 **dense spatial evidence**。

### 事实 3：直接从 canonical P0 重建也不行

Run6 P0 PR-AUC 仅 0.3711，P0+B4 rank fuse 反而弱于 B4。
所以不是“把浅层 token 加回来”这么简单。

### 新机制：改变采样几何，而不是再堆一个 detail encoder

DeiT 的 patch projection 本身已经由 ImageNet 预训练，但标准 ViT 只在互不重叠的
16×16 lattice 上调用一次。B4-OPRE **不学习第二套局部 backbone**，而是复用同一
patch kernel，在 stride8 overlap lattice 上再观察一次图像，从而增加 sub-patch
phase coverage；B4 再决定哪些 local difference 值得进入 dense reconstruction。

这样“为什么 <3M”与“为什么可能提精度”来自同一个设计：

- **轻量**：encoder 权重共享，O-PRE 0 new params；
- **精度**：采样 lattice 变密，补 B4 缺失的局部空间 evidence；
- **可解释**：B4 semantic gate 显式控制局部 residual；
- **创新边界清晰**：不把 overlap embedding、gate、edge 当作单独首创。

## 14.2 可解释性证据

正式论文至少保留四组图/表：

1. **Depth curve**：P0/B1/B2/B3/B4/B12 四数据集 PR-AUC；
2. **Dense recoverability**：Run8 B4 vs B12 pixel/boundary；
3. **O-PRE raw audit**：
   - canonical P0 vs O-PRE；
   - B4 vs B4×O-PRE proxy；
4. **Learned gate visualization**：
   - B4 semantic response；
   - O-PRE raw local difference；
   - gate；
   - final map；
   - GT；
   - FP/FN overlay。

同时按 change-size bucket 报告，检查提升到底来自：
- 1–16 patch 小变化；
- 17–64 中变化；
- >64 大变化。

## 14.3 不能过度表述

即使四数据集都达到新目标，也只可写：

> 在统一 80K / seed16 / test-as-val 协议下，B4-OPRE 在四个公开二值变化检测数据集
> 达到目标结果，并把有效参数控制在约 2.04M。

不能写：
- “普适优于所有轻量模型”——除非完成同协议公平比较；
- “任何数据集都 B4 最优”——Run7 已证明最优 depth dataset-dependent；
- “raw ranking 可以预测最终 F1”——Run3/Run8 已反证；
- “O-PRE 天然恢复真实边界”——必须以 D0 + gate visualization + endpoint ablation 支持。

---

# 15. 与已有停止规则的关系：为什么 Run9 没有违规重启旧路线

## 15.1 不是回 detail branch 搜索

Run4/5 停止的是：
- 自定义多层 CNN detail；
- Mobile/Efficient/Shuffle 等 backbone prefix 搜索；
- width/depth/adapter rescue。

O-PRE：
- 没有独立 CNN；
- 不新增 encoder 参数；
- 不搜索 backbone；
- 只复用主 ViT 自己已经存在的 pretrained patch kernel。

## 15.2 不是 P0 reconstruction 重启

Run6 用的是**canonical non-overlap P0**作为浅层 source，且 P0+B4 fixed rank fusion
失败。

Run9 的唯一新变量是：

```text
same patch kernel
stride 16, non-overlap  →  stride 8, overlap
```

并把它限定为 gated local residual，不担当主 semantic source。

因此必须先用 G0 明确证明“sampling lattice 改变”本身带来新信息；
G0 不过就视为 Run6 的负结论仍然覆盖本路线。

## 15.3 不是 B2/B3 rescue

主 semantic source 固定 B4，不做 depth sweep。

## 15.4 不是 B4-only rescue

Run8 停止的是“无新 spatial source 的 B4-only dense reconstruction”。
Run9 明确引入一个新的、可审计的 32×32 overlap observation。
若 O-PRE D0 不提供 complementarity，就不训练。

---

# 16. 立即执行顺序

1. **先提交/保存本预注册，不改 gate。**
2. 新建 `analyse/run9_overlap_reembedding_audit.py`，只做零训练四数据集 audit。
3. 跑 SYSU R4-1 control；不在 ±0.005 → `AUDIT_INVALID`，先修口径。
4. 跑四数据集 P0 / B4 / O-PRE / proxy。
5. 打印 `[R9-D0-GATE] PASS/FAIL`。
6. **FAIL：本轮结束，0 个 80K。**
7. PASS：实现 `opre_head.py` + encoder/trainer/train/eval 接口。
8. 跑 T-R9-0…T-R9-9 smoke。
9. 参数 >2.10M / FLOPs >2.00G / symmetry/checksum 任一失败 → 不训练。
10. 跑 SYSU 60-step dry run。
11. dry run 全绿 → 启动 R9-1 SYSU 80K。
12. 读取最后完整 TEST RESULTS：
    - `<85.00` 或 IoU `<73.91` → Run9 停止；
    - 达标 → R9-2 LEVIR。
13. LEVIR 达标 → WHU；WHU 达标 → CDD。
14. 四数据集全部达标后，再跑唯一一个 80K 机制消融 `R9-A1_NOGATE_SYSU`。
15. 更新 README 结果表、Run9 结果归档文档、`文献索引.md`（若最终采用新增两篇）。
16. Git 提交前：
    - `git status`
    - `git diff --cached`
    - 确认不含 checkpoint / log / dataset / cache / pretrained weights / secret。

---

# 17. 仍需补充证据

1. **最优先：四数据集 O-PRE D0 raw complementarity。**
   没有它，本方案仍只是合理假设。
2. WHU/CDD 的健康 frozen full12 正式参考仍缺失；如未来论文需要“相对健康
   ChangeViT 的精确增益”，应补同协议参考，但不应在 Run9 D0 之前占用训练预算。
3. 当前新文献核验已经证明 edge/high-frequency 本身拥挤，但“共享 pretrained
   patch kernel + stride-dense re-embedding”还应在写论文 Related Work 前再做一次
   **只针对 exact mechanism 的 2024–2026 高水平检索**，防止遗漏同构工作。
4. 若 D0 PASS 但 SYSU endpoint FAIL，需要保存：
   - per-size bucket F1/IoU；
   - boundary/non-boundary FP/FN；
   - gate activation distribution；
   - O-PRE local score 与错误区域；
   再决定 Run10，不在 Run9 内救。
5. 单 seed16 只能支撑固定协议结果；最终投稿若时间允许，应对最终模型补多 seed
   稳健性，但这属于终稿验证，不改变当前预注册 gate。

---

# 18. Run9 一句话判据

> **只有当同一个冻结 DeiT patch kernel 在 stride8 overlap lattice 上，相对 canonical
> P0 明确增加边界信息，并且该信息与 B4 semantic 在四数据集形成可测的互补时，
> 才允许用约 2.04M 参数的 B4-OPRE 花第一个 80K；随后每个数据集必须直接达到
> 85/92.5/95/98 的终局门槛，任何一步不过即停止，不做本轮 rescue。**
