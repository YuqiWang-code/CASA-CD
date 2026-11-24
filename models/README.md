# CASA-CD 工作模型源码

共享 MobileNetV2 四尺度特征（24/32/96/320 通道，64²/32²/16²/8²）→ 四级 TAR（64 通道）→ 跨尺度检索/上下文 → DCR → 共享 1×1 预测头。检索始终保留**完整 Q（5,440 个位置）**，只把 KV 压缩成**224 个类条件原型**（128/32/32/32）。

**本树对应 CASA-MinorGoal / Run10 的精简主干**（协议 `casa_minor_run10_lean_v1`）。相对 Run7 只改三处、不改拓扑：

- **DCR**：主输出从 64² 提到 **128²**——`FineRefine` 在 128² 上用编码器 stem（32 通道）融合 `1×1(1+32→64) → BN → SiLU → DW3×3(replicate) → BN → SiLU → 1×1(零初始化)`，再 ×2 到 256²；64² 的 `DCRDecoder.refine` 随之删除。深监督四级权重改 `1 / .5 / .25 / .125`（`--ds-weights`）。
- **TAR（`cg`）**：门从"增强"改成"竞争"——`g=σ(1×1(positive−negative))`、`fused=block(g·n(content)+(1−g)·n(delta))`，`n` 是逐像素跨通道归一，门零初始化（开局 0.5 对 0.5）；局部对比核随尺度 `(3,5,7,7)`。
- **HRA（`manifold`）**：`−mean` 变成可学习 `mean_gate`（零初始化），检索可用 `--retrieval-scales` 只在粗两级跑，原型可用 `--compact-prototypes` 压到 32（每尺度 2×2 格 ×2 类），support 先验可换 `--support confidence`（`|2p−1|`）。

更早的 `spp` / `difference`（Run1–Run3）、H/E 几何对照（CASA-Pyramid Run12–Run16）与 `qkv` 遗留臂都还在这棵树里、可复现。Run8 的尺寸无损分解（CD²-TAR）、Run9 的换骨干、以及 Run10 之前的双曲层次原型（`--prototype-tree`）**不在本树**，只留在 `train_scripts/`、`docs/` 与 git 历史。

## 组件与开关

### TAR：时相融合与通道投影（`model/str_tar.py`）

| `--tar-variant` | 每尺度结构 | 训练/部署参数* | 用途 |
|---|---|---|---|
| `reparam`（默认） | `TemporalRep1x1`（concat / P+Q / Q−P 三支 1×1+BN，可折叠）→ SiLU → `RepLocalBlock` | 2.047125 / 1.935810 M | 主线基线 |
| `plain` | 单 1×1+BN → SiLU → `PlainLocalBlock` | 1.974157 / 1.935810 M | 去可折叠冗余的对照 |
| `diff` | `(P−Q).abs()` → 1×1+BN → SiLU → `PlainLocalBlock` | 1.943949 / 1.905602 M | 去掉学习式时相混合 |
| `spp` | 融合支 + `\|P−Q\|`→1×1→`RepSPPBlock`（1/3/5/7 DW 并行，可折叠成单个 7×7 DW + 1×1）→ 相加 → `RepLocalBlock` | 2.118293 / 1.995714 M | 多尺度差分池化 |
| `spp1` | 同上，但 SPP 只有 1×1 窗口 | 2.095509 / 1.983426 M | 分离"多尺度窗口"本身 |
| `cg` | `ChangeGateTARStage`：内容支 `SiLU(1×1[P,Q])`；差分支先把两期各除以局部均值模长（曝光漂移抵消），再 `1×1[d⁺,d⁻]`；门 `g=σ(1×1(pos−neg))` 零初始化，`fused=block(g·n(content)+(1−g)·n(delta))`，两路逐像素跨通道归一 → `RepLocalBlock` | 2.072276 / 2.025539 M | Run10 主线（竞争门） |

\* 指 `--module none` + `--decoder-variant reparam` 时的整网参数。`spp`/`spp1` 额外输出每尺度差分特征 `D_s`，供 `--module difference` 使用；`diff` 在 `return_difference=True` 时也直接给出 `D_s`。

### 上下文：检索与写回（`model/layers/`）

| `--module` | 类 | 几何 | KV 来源 | context 参数 |
|---|---|---|---|---|
| `none` | `AuxiliaryContext` | — | 无检索，只出 guide | 193 |
| `hyperbolic` | `HyperbolicContext` | 曲率 −1 Lorentz 测地平方距离 | `T_s` 的类池化原型 | 11,477 |
| `matched` | `MatchedEuclideanContext` | 同一空间坐标的欧氏平方距离 | 同 H（逐位同构） | 11,477 |
| `difference` | `DifferenceContext` | 同 H | `\|P−Q\|` 的类池化原型（多一层 `LayerNorm(D_s)`） | 11,605 |
| `manifold` | `ManifoldContext` | 同 H，聚合轴可切 `--aggregate {lorentz,euclidean}` | 网格原型，KV 内容跟随 TAR 变体（`cg`/`diff` 时是 `D_s`），另有独立 support 头与 level 通路；`--retrieval-scales` 可只在粗两级检索，`--compact-prototypes` 把原型压到 32，`--support confidence` 换成置信先验，`mean_gate` 零初始化（开局仍去均值） | 见速查表（整网） |
| `qkv` | `EuclideanContext` | 欧氏 | Run5 遗留：压缩方式、读出、门控都不同 | 11,477 |

共同实现（`pyramid_context.py` / `hyperbolic_context.py`）：`X=LN(T_s)`，`p=σ(guide(X))`；按 `p`、`1−p` 分别归一化池化得到 `P⁺/P⁻`；`K=L(W_qk P/√d)`、`V=W_v P`（K、V 共用同一内容统计，`V` 留在欧氏空间），`Q=L(W_qk X/√d)`。打分 `−d²/(2τ²)+log(support)+scale_bias−log(M)`，τ 可学习（初值 0.5）。读出 `U=W_out[gate·(AV−mean_hw(AV))]`，残差 `T'_s=T_s+γ_s·U_s+local_s`（`local` 只在最细两层）。`γ` 零初始化，因此**推理时模块与 CTRL 等价**；训练开始后自行打开。

- `--support {mass,cell,pixel,none}`：类先验 `log(support)` 的来源；`cell` 是 `mass` 的别名。
- `--kv-hierarchy {flat,nested}`：`flat` 为 8/4/4/4（224 KV）；`nested` 为 8/4/2/1 并做父子 2×2 聚合（170 KV）。
- `--freeze-tau`：把 τ 冻结在实测中位数 0.4696。
- `--prototype-pooling {euclidean,lorentz}`（默认 `euclidean`）：`lorentz` 先把每个像素特征 lift 到双曲面再做类条件加权重心（Klein 重心），原型严格落在双曲面上、零参数增量。Run7 单因子实测 `lorentz` 比切空间均值低 **0.387 pp**（SYSU 82.824 vs 83.212），故默认仍是欧氏池化。
- `--aggregate {lorentz,euclidean}`（默认 `lorentz`）：`ManifoldContext` 的加权读出在哪个空间做；`euclidean` 只改聚合，打分仍在流形上。
- `--edge-weight` / `--edge-dilate` / `--edge-detach {none,retrieval}`：GT Sobel 边缘监督的权重、膨胀半径与梯度落点。`retrieval`（**默认**）让检索回写的边缘头输入先 `detach()`，边缘梯度只经 TAR 的 `D_s` 回传；Run6 在 SYSU 上实测 `none` 82.01 → `retrieval` 83.09（+1.09 pp，噪声 0.32），因此默认切断。
- `--context-heads {1,2,4}`：1 头 32 维（默认）；多头时 `head_dim=16`。

### 解码器与预测头（`model/str_dcr.py`、`model/casa_net.py`）

| `--decoder-variant` | 结构 | 部署参数增量 |
|---|---|---|
| `reparam`（默认） | 三级 `RepPairFuse1x1`（concat/sum/diff，可折叠）+ `RepLocalBlock`，再 `refine` | 基线 |
| `plain` | 融合换单 1×1+BN，局部块换普通 DW3+PW1 | −0.037 M |
| `none` | `SingleLevelDecoder`：只在最粗尺度上采样，细尺度只做残差相加 | −0.072 M |

主输出不是单张 64² 图：`1×1(64→1)` 的 logit 先上采样到 128²，与编码器 stem（32 通道）一起进 `FineRefine`（`1×1 → BN → SiLU → DW3×3 → BN → SiLU → 1×1`，末层零初始化 ⇒ 开局恒等），再上采样到 256²；四级深监督权重 `1 / .5 / .25 / .125`。**128² 细节头**（`--detail-head`）：两时相 128² 特征拼接 → 1×1(64) → DW3×3(replicate) → SiLU → 1×1(1，零初始化)，与 `FineRefine` 的输出相加，新增 2,689 参数；`--detail-target {area,nearest-exact}` 决定 GT 下采样方式。guide 头只在训练期提供 0.1 系数（默认）的面积 BCE。

## 参数速查（训练 / 部署，M）

| module | tar | decoder | train / deploy |
|---|---|---|---:|
| none | reparam | reparam | 2.042324 / 1.934019 |
| none | plain | reparam | 1.969356 / 1.934019 |
| none | diff | reparam | 1.939148 / 1.903811 |
| none | spp | reparam | 2.113492 / 1.993923 |
| none | cg | reparam | 2.072276 / 2.025667 |
| hyperbolic / matched | reparam | reparam | 2.053609 / 1.945304 |
| qkv（遗留） | reparam | reparam | 2.059736 / 1.951431 |
| manifold | cg | reparam | 2.090099 / 2.043490 |

部署保留 `FineRefine` 的两个 BN（折它们会把死通道的 1e7 放大带进 FP32，见 `casa_net.FineRefine.switch_to_deploy` 的注释），因此 deploy 比 Run7 同臂多 128 个参数。`decoder=none`（单级解码）与 `manifold`+`diff` 的组合随 Run10 主干变化，本轮未重测，读数以 `train_scripts/CASA-MinorGoal/Run10/` 的 smoke 为准。`--prototype-pooling`、`--retrieval-scales`、`--compact-prototypes`、`--support` 只改缓冲与打分形状，不改参数量。
