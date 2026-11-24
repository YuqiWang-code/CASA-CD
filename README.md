# CASA-CD

普通RGB双时相影像的全监督遥感二值变化检测。**当前主线是 CASA-MinorGoal / Run2**：TAR 加多尺度差分池化（`--tar-variant spp`）、检索的 KV 换成 `|P−Q|` 类条件原型（`--module difference`）。几何与压缩的对照线归档在 **CASA-Pyramid / Run12–Run16**，正向基线 **Run4 完整保留**。

**当前代码口径是 Run7**（`models/` 已回退到 `casa_minor_run7_manifold_pooling_v1`）：Run8 的 CD²-TAR、Run9 的 VMamba/ViT 换骨干、Run10 的层次原型都已移出 `models/`，只留在 git 历史、`train_scripts/` 与下文对应章节里备查，本树下不可直接复跑。

三 seed 配对结果（WHU/SYSU，验证集选模）：差分 KV 相对 CTRL **+0.679 / +0.750 pp**（t-LCB +0.460 / +0.474），是唯一在双库显著的效果；多尺度窗口本身不承载收益（`spp1` 单 1×1 与 `diff` 无 SPP 都在噪声内）。双曲几何无稳定精度收益：Run14 十二对配对均值 −0.167 pp。检索结构始终是完整 Q（5,440 个位置）+ 压缩 KV（224 个类条件原型），[设计与公式](models/HYPERBOLIC_DESIGN.md)。

## 项目定位与固定研究原则

本项目面向211高校硕士的小论文研究，目标是在LEVIR、SYSU、WHU、CDD四库取得有竞争力的精度，同时控制推理参数、计算、延迟与显存。贡献聚焦两个独立版本：双曲压缩QKV、普通变化感知上下文压缩QKV，保持完整Q空间位置，仅压缩KV。

项目F1努力目标为LEVIR≥92%、SYSU≥84%、WHU≥95%、CDD≥97.5%。这些数值是研究目标，不是已核实的统一SOTA界限。先证明模块有效、机制连贯、收益可归因，再争取领先指标。有效性优先于绝对最低参数，允许为有用表征小幅增参；参数目标≤5M，当前部署约2M。较少参数不等于更快推理。

结构须对应可验证的实际问题，保持简洁。避免无关组件、特殊激活函数和额外几何运算堆叠；不将损失、剪枝或量化包装成主要创新。修改前核对日志、运行源码与历史经验，修改后审查公式、卷积/池化、融合/残差、维度与前后梯度，并用真实数据短训检查。

同轮各组使用相同骨干、增强、优化、训练预算及选模规则，专用监督单独记录。选模划分由 `--selection-split` 决定：CASA-Pyramid / Run12–Run13 与 MinorGoal 的解耦臂用 test 选模（结果标注 test 参与选模），Run14–Run16 与 MinorGoal 的 H/E、SPP、DIFF 主臂用 val 选模（`independent_test=true`）。正式复测主阈值固定 0.5，次指标阈值在 val 校准。结构设计与干预使用 train/val，跨协议差异与模块收益分别解释。

文献优先2024–2026年相关高水平工作，基础理论追溯原始论文，核实题名、年份、venue与官方代码。参考材料持续维护于[文献索引](docs/参考文献/文献索引.md)，核心代码与许可证放在`others/`。工作约定见[AGENTS.md](AGENTS.md)，研究对话设置见[ChatGPT_Project_Settings.md](docs/ChatGPT_Project_Settings.md)。

## 项目导航

| 要找的内容 | 位置 |
|---|---|
| 主线模型、训练与测试 | [models/](models/)，模型入口 `model/casa_net.py`，训练入口 `train_CASA.py` |
| 各轮实验矩阵、调度与收集 | [train_scripts/CASA-Pyramid/](train_scripts/CASA-Pyramid/) 与 [train_scripts/CASA-MinorGoal/](train_scripts/CASA-MinorGoal/) 下对应的 `RunN/` |
| 服务器连接、环境和状态 | [.codex/SERVER.md](.codex/SERVER.md)、[.codex/STATE.md](.codex/STATE.md)，实时状态用 `.codex/status.py` |
| 日志、指标、曲线和冻结源码 | 本地 `outputs/<tag>/<Run>/`，已归档材料见 [docs/experiments/](docs/experiments/)、[指标表](docs/experiment_metrics.xlsx) 与 [TXT 快照](docs/temporary/) |
| 下载、汇表与导出工具 | [tools/](tools/) |
| 独立的 MoE 探索线 | [README_MoE.md](README_MoE.md) 与 [models_v1/](models_v1/) |

代码仓库为 [YuqiWang-code/CASA-CD](https://github.com/YuqiWang-code/CASA-CD)，训练服务器为 RSML-3。

## 小目标改进：CASA-MinorGoal / Run1：多尺度差分池化与差分条件检索

前几轮把几何与检索的对照做完后，问题落在两处结构而不是距离函数：**双曲侧的 KV 来自融合特征，回答不了"这里变没变"**，检索在信息不足时退化成去均值恒等；**TAR 只有一个 1×1 的时相混合**，看不到小目标需要的多尺度上下文。Run1 针对这两点各改一处，其余（骨干、四尺度拓扑、DCR、损失、增广、40K 预算、验证集选模）全部不动。

- `--tar-variant spp`：TAR 每尺度在时相融合之外并行一条 `|P−Q|` 支路——1×1 升维后接 `RepSPPBlock`，四支并行深度卷积（1×1 / 3×3 / 5×5 / 7×7，3/5/7 支 BN 从 0 开始），训练期多分支、部署折叠成**单个 7×7 DW + 1×1**，实测 `max_diff 1.19e-7`。
- `--module difference`：新增 `DifferenceContext`，与 `HyperbolicContext` 逐项同构，只多一层 `LayerNorm(D_s)`；Q 仍是 `T_s` 的全位置，KV 换成 `|P−Q|` 的类条件池化原型（224 个），使注意力权重本身编码变化强度。
- 依据的 2024–2026 工作（SET、BoltzFormer、OKNet、Hyperbolic Change Monitoring、CoSA、SCA-Net 等）存于 [参考文献](docs/参考文献/MinorGoal_差分挖掘与多尺度池化/)，核心代码在 `others/`。

| 数据集 | 臂 | 训练/部署参数（M） | Recall | Precision | F1（%） | Δ（pp） |
|---|---|---:|---:|---:|---:|---:|
| WHU | CTRL | 2.047 / 1.936 | 93.15 | 94.10 | 93.6234 | — |
| WHU | `SPP` | 2.118 / 1.996 | 92.89 | 94.51 | 93.6925 | +0.069 |
| WHU | `DIFF` | 2.130 / 2.007 | 93.15 | **95.57** | **94.3468** | **+0.723** |
| SYSU | CTRL | 2.047 / 1.936 | 80.07 | 83.46 | 81.7290 | — |
| SYSU | `SPP` | 2.118 / 1.996 | 80.20 | **85.56** | **82.7915** | **+1.062** |
| SYSU | `DIFF` | 2.130 / 2.007 | 81.08 | 84.30 | 82.6630 | **+0.934** |
| LEVIR | CTRL | 2.047 / 1.936 | 89.76 | 91.96 | 90.8488 | — |
| LEVIR | `SPP` | 2.118 / 1.996 | 90.43 | 91.70 | 91.0612 | +0.212 |
| LEVIR | `DIFF` | 2.130 / 2.007 | 90.12 | 91.59 | 90.8535 | +0.005 |

五条结论：

1. **多尺度空间池化有效，且增益在精度侧。** SYSU **+1.062 pp**（P 83.46 → 85.56，R 几乎不动），是本项目第一次在单库上把控制臂拉开 1 pp 以上；LEVIR +0.212、WHU +0.069 落在噪声内（LEVIR 复现对照差 0.197 pp，WHU 噪声约 0.18 pp，SYSU 约 0.32 pp）。机制上是"少报错"，不是"多检出"。
2. **双曲不缺几何，缺的是 KV 内容。** 换成差分原型后 WHU **+0.723 pp**，P 从 94.10 升到 95.57；SYSU +0.934 pp。同一套检索在 KV 用融合特征时与欧氏双胞胎无差（Run14 十二对均值 −0.167 pp），换上差分内容后才有信号——这解释了此前"双曲无收益"的结论其实测的是"检索内容没有差分"。
3. **两处改动不叠加。** `DIFF` 减 `SPP`：WHU **+0.654**、SYSU **−0.129**、LEVIR **−0.208**。SPP 已经把多尺度差分证据写进主干，再叠一层差分检索只在 WHU 有额外收益，在 SYSU/LEVIR 反而略降。
4. **代价必须一起报。** 部署参数 1.936 → 1.996（SPP，+3.1%）→ 2.007 M（DIFF）；batch1 256² FP32 延迟 5.880 → 5.987 ms（+1.8%）→ 8.883 ms（**+51%**）。按性价比，SPP 单独可用；DIFF 只在 WHU 值这 51%。
5. **可信度边界。** 单 seed；`SYSU +1.06`、`DIFF/WHU +0.72`、`DIFF/SYSU +0.93` 超出噪声，`SPP/WHU +0.07`、`SPP/LEVIR +0.21`、`DIFF/LEVIR +0.005` 不能当信号。另外 SPP/WHU 的 val 峰值 94.709 对应 test 93.693（−1.02 pp），说明该臂在 WHU 上踩着噪声点选模；SYSU 的最优 epoch 是 6–15，前几轮即到顶、后续全是过拟合。

[实验矩阵与结果](train_scripts/CASA-MinorGoal/Run1/README.md) · 逐项结果见 [指标表](docs/experiment_metrics.xlsx) 与 [TXT 快照](docs/temporary/models_and_metrics_CASA-MinorGoal_Run1.txt)。

## 小目标改进：CASA-MinorGoal / Run2：三 seed 配对与解耦臂

Run1 是单 seed 的两条改动，Run2 把它做成可以下结论的形状：三条主臂 `CTRL` / `SPP` / `DIFF` 在 seed 2333/2334/2335 各跑一遍（WHU、SYSU，共 18 项），再加三个只跑 seed 2333 的解耦臂——`SPP1`（SPP 只留 1×1 窗口）、`DIFF_SPP1`（差分检索 + 1×1 池化）、`DIFF_NOSPP`（`--tar-variant diff`，没有 SPP 阶段）。全部验证集选模、阈值 0.5。

| 数据集 | CTRL（三 seed） | SPP（三 seed） | DIFF（三 seed） | DIFF−CTRL |
|---|---|---|---|---:|
| WHU | 93.6234 / 93.6438 / 93.1478（均 93.4717） | 93.6925 / 93.8540 / 93.5517（93.6994） | 94.3468 / 94.4244 / 93.6808（94.1507） | **+0.679**（sd 0.130，LCB +0.460） |
| SYSU | 81.7290 / 82.0917 / 81.8966（81.9058） | 82.7915 / 82.2689 / 82.3727（82.4777） | 82.6630 / 82.7106 / 82.5949（82.6562） | **+0.750**（sd 0.164，LCB +0.474） |

预登记门限 Δ*：WHU 0.36 pp、SYSU 0.47 pp（t(0.95,2)=2.92，按 3 seed 配对差）。四条判读：

1. **差分 KV 是唯一双库显著的改动。** DIFF−CTRL 在 WHU +0.679（LCB +0.460）、SYSU +0.750（LCB +0.474），两个下界都过了门限。
2. **SPP 单独不显著。** SPP−CTRL：WHU +0.228（LCB −0.056，无效应）、SYSU +0.572（sd 0.450，仅方向性）。DIFF−SPP 只有 WHU +0.451（LCB −0.024）算方向性，SYSU +0.178 无效应。
3. **多尺度窗口本身不承载收益。** `SPP1` 换成单 1×1 后 WHU 94.0123（对 SPP +0.32）、SYSU 82.7656（对 SPP −0.03）；`DIFF_NOSPP` 去掉 SPP 后 WHU 94.3919（对 DIFF +0.05）、SYSU **83.4440**（对 DIFF +0.78）。即"1×1/3×3/5×5/7×7 的窗口集"没有可测贡献，起作用的是差分内容本身。
4. **收益仍来自同一套检索机制，而不是几何。** DIFF 与 H 逐项同构（只多一层 `LayerNorm(D_s)`），所以 H/E 配对结论（几何无收益）与差分 KV 的正收益不矛盾：换的是 KV 内容，不是距离函数。

口径提醒：三条主臂共用同一实现、同一骨干、同一预算；`DIFF_NOSPP` / `SPP1` / `DIFF_SPP1` 只有 seed 2333，不能当作三 seed 结论，只用于回答"哪一部分在起作用"。

[实验矩阵与结果](train_scripts/CASA-MinorGoal/Run2/README.md) · 逐项见 [指标表](docs/experiment_metrics.xlsx) 与 [TXT 快照](docs/temporary/models_and_metrics_CASA-MinorGoal_Run2/)。

## 归因实验：CASA-MinorGoal / Run3：模块、先验、边界各值多少

Run2 之后剩下的问题是"收益到底来自模块存在、KV 内容，还是训练期多出来的容量"。Run3 用四条定向臂回答，不再铺笛卡尔积（协议 `casa_minor_run3_attribution_v1`，只用 GPU1 两槽）：

| 臂 | 配置 | 要回答的问题 | 预登记判据 |
|---|---|---|---|
| `ARM_C/WHU`、`ARM_C/SYSU` | `hyperbolic` + `spp`，seed 2334 | 同一模块但 KV 换回 `T_s`：模块存在的效应有多大 | 对同 seed `SPP` 的 Δ ≥ +0.36（WHU）/ +0.47（SYSU）才算模块效应主导；\|Δ\| ≤ 噪声（0.18/0.32）算无模块效应 |
| `SUPP_PIX/SYSU` | 同上 + `--support pixel` | 类先验是不是第二个杠杆 | 对 `ARM_C` Δ ≥ +0.47 才认 |
| `DETAIL/LEVIR` | `difference` + `spp` + `--detail-head`，seed 2333 | 128² 边界分支是否值得 | 总 F1 ≥ 91.2135、0 px 边界 F1 ≥ 62.486、内部 F1 ≥ 96.169、边界占正类 31.41% |

`DETAIL` 同时改了边界分支与一项额外深监督，属已登记的混杂；要单独归因需再跑"只开 `--detail-head` 不加 detail 深监督"的分离臂。每项测试时都落 `per_image_cm.npz`（`index/tp/fp/fn` + list 哈希等采信字段），供图像级 bootstrap；判据与门限写死在 `train_scripts/CASA-MinorGoal/Run3/config.py` 的 `GATES`/`LEVIR_DETAIL_GATES`，`compare.py` 直接出裁决。

结果（2026-11-18 完成，4/4 无失败；val 选模、τ=0.5、single seed）：

| 臂 | 库 | F1（%） | Δ vs 同 seed 基线（pp） | 裁决 |
|---|---|---|---|---|
| `ARM_C` | WHU | 93.5646 | −0.289（vs Run2 SPP 93.8540） | 未达门限，方向为负 |
| `ARM_C` | SYSU | **83.3700** | **+1.101**（vs Run2 SPP 82.2689） | effect confirmed |
| `SUPP_PIX` | SYSU | 82.4219 | −0.948（vs `ARM_C`） | effect reversed |
| `DETAIL` | LEVIR | 91.2611 | +0.408（vs Run1 DIFF 90.8535） | 总门通过，边界门未通过 |

`DETAIL` 四道门：总 F1 91.2611 ≥ 91.2135 通过；内部 F1 96.7172 ≥ 96.1690 通过；**≤0 px 边界 F1 60.6399 < 62.4860 未通过**（边界占正类 31.41%）。

三条判读：

1. context 模块本身只在 SYSU 上有量级：对同 seed SPP +1.10 pp、对同 seed CTRL +1.28 pp；WHU 上比 SPP 低 0.29、比 CTRL 低 0.08——符号随库翻转，与 Run10/Run12/Run14 的模式一致。`ARM_C` 只有 seed 2334，要作结论需补 2333/2335。
2. `--support pixel` 比 mass 差 0.95 pp，且 Recall 79.96 与 Precision 85.04 相对 `ARM_C` 的 81.16 / 85.71 双降——不是阈值偏移，是召回真的少了。support 的第二杠杆不在换度量，而在保留 p 参与的池化质量。
3. 细节头仍然没解决边界：总 F1 与内部区域过门，专门负责的 ≤0 px 边界差 1.85 pp。结合 Run12（边界 −0.46）与 Run1（总 F1 仅 +0.005），可以判定 128² 双时相拼接这条支路在当前实现下对边界无效，不是噪声。

口径与产物：四项全部 val 选模（`independent_test=true`），部署折叠 `maxdiff ≤ 1.9e-6`、二值零差异；GPU1 两槽共 6 h。快照与结构图见 [Run3 快照目录](docs/temporary/models_and_metrics_CASA-MinorGoal_Run3/)，逐项见 [指标表](docs/experiment_metrics.xlsx)。

## 模块消融：CASA-Pyramid / Run16：TAR 与 DCR 在三个历史版本上的去留

Run15 判定了几何，这一轮判定另外两个结构：TAR 的学习式时相混合（concat / P+Q / Q−P 三支可折叠 1×1）与 DCR 的多级 coarse→fine 解码。两条臂分别换成固定 `|P−Q|`+1×1（`TAR_DIFF`）和单层上采样（`DCR_SINGLE`），四尺度拓扑、通道、损失、增广与 40K 预算不变，控制值直接复用 Run15 同版本 `none`（同为验证集选模）。在 `models_Run1/`、`models_Run4/`、`models_Run10/` 三棵历史源码树上各跑 WHU/SYSU，共 12 项，2026-11-15 23:56 完成，无失败。

| 版本 | 臂 | WHU F1（%） | Δ（pp） | SYSU F1（%） | Δ（pp） |
|---|---|---:|---:|---:|---:|
| Run1 | `TAR_DIFF` | 93.7435 | +0.126 | 82.1943 | +0.410 |
| Run1 | `DCR_SINGLE` | 93.2805 | −0.337 | 80.7502 | −1.034 |
| Run4 | `TAR_DIFF` | 94.1642 | +0.807 | 82.6838 | −0.379 |
| Run4 | `DCR_SINGLE` | 93.9531 | +0.596 | 79.9029 | −3.159 |
| Run10 | `TAR_DIFF` | 94.0826 | −0.005 | 82.5632 | +1.468 |
| Run10 | `DCR_SINGLE` | 93.5406 | −0.547 | 81.4424 | +0.347 |

三条结论：

1. **TAR 的学习式时相混合不承载精度。** 六个配对里平均 **+0.405 pp**（+0.126/+0.410、+0.807/−0.379、−0.005/+1.468），没有任何一项掉出复跑噪声（0.18–0.32 pp），叠加 Run14 的 −0.297/+0.181/−0.242，整体仍为正。换成固定绝对差同时把可训练参数从 2.047125 M 降到 1.943949 M、部署参数从 1.935810 M 降到 1.905602 M，延迟不变。这条臂该写成"可折叠等价实现"，而不是"必须保留的表达力"。
2. **DCR 是唯一平均为正贡献的模块，但幅度依赖基线。** 六项平均 **−0.689 pp**，四负两正，最差 `Run4/SYSU` −3.159 pp，最好 `Run4/WHU` +0.596 pp；与 Run14 的 WHU −0.212、SYSU −0.611、LEVIR −0.876 合起来看，多级解码在小目标密集、边界碎的 SYSU 上最不能少。结论应写成"平均有效、逐库逐基线报告"，不能写成"处处有效"。
3. **两个模块的符号都随库翻转，所以必须分库下结论。** TAR 在 WHU 为 +0.126/+0.807/−0.005、在 SYSU 为 +0.410/−0.379/+1.468；DCR 在 WHU −0.337/+0.596/−0.547、在 SYSU −1.034/−3.159/+0.347。三个版本的控制臂本身相差 WHU 0.73 pp、SYSU 1.97 pp，跨版本只能比版本内 Δ。

[实验矩阵与结果](train_scripts/CASA-Pyramid/Run16/README.md) · 逐项结果见 [指标表](docs/experiment_metrics.xlsx) 与 [TXT 快照](docs/temporary/models_and_metrics_CASA-Pyramid_Run16.txt)。

## 换骨干：CASA-MinorGoal / Run9：VMamba-Tiny 与 ViT-B/16 四层

前八轮都在 MobileNetV2 上做模块增减，无法回答"这些模块是不是只在浅层 CNN 特征上成立"。Run9 只换骨干，其它开关不动：**VMamba-Tiny**（`dims=(96,192,384,768)`、`depths=(2,2,5,2)`，29.94 M，ImageNet 预训练）与 **ViT-B/16 只取前 4 块**（`vit_base_patch16_224.augreg_in21k_ft_in1k`，前向 ≈30.9 M，位置编码 197→257 bicubic 重采样）。每个骨干做冻结/解冻 × 四臂（`CTRL` / `TAR_ONLY`=cd2 / `HRA_ONLY`=manifold 检索 / `TAR_HRA`）× SYSU/LEVIR，共 32 项；协议 `casa_minor_run9_backbone_v1`，bs16、20k step、seed 2334、val 选模、阈 0.5，两张 5090 各两槽。冻结臂只训桥、解码器与头（**0.212–0.581 M**），总参数 30.15–30.88 M。

**块内配对 Δ（对同 (backbone, freeze) 的 `CTRL`，门限 0.36/0.36/0.47，噪声 0.32 pp）**

| 块 | 库 | `CTRL` | `TAR_ONLY` | `HRA_ONLY` | `TAR_HRA` |
|---|---|---:|---:|---:|---:|
| VMamba 冻结 | SYSU | 79.03 | 78.65 (−0.38) | **74.78 (−4.24)** | 78.92 (−0.11) |
| VMamba 解冻 | SYSU | 81.00 | 81.21 (+0.21) | **82.31 (+1.31)** | 80.57 (−0.43) |
| ViT-B 冻结 | SYSU | **82.52** | 81.91 (−0.62) | 79.40 (−3.12) | 82.49 (−0.04) |
| ViT-B 解冻 | SYSU | 78.37 | 78.89 (+0.52) | 77.71 (−0.66) | 78.72 (+0.35) |
| VMamba 冻结 | LEVIR | 81.43 | 81.63 (+0.20) | **63.18 (−18.25)** | 81.66 (+0.23) |
| VMamba 解冻 | LEVIR | **90.70** | 89.99 (−0.71) | 90.43 (−0.27) | 90.67 (−0.03) |
| ViT-B 冻结 | LEVIR | 85.44 | 85.97 (+0.53) | **75.06 (−10.38)** | 86.21 (+0.77) |
| ViT-B 解冻 | LEVIR | 86.04 | 86.64 (+0.60) | 86.21 (+0.17) | **86.83 (+0.79)** |

三条结论：

1. **检索模块依赖骨干更新，不能冻结着用。** `HRA_ONLY` 在四个冻结块上全部塌方：SYSU −3.12/−4.24、LEVIR −10.38/−18.25 pp。原因是它的 KV 是"变化证据原型"：骨干不随任务更新时，原型与查询来自同一套未适配特征，写回等于往主干加噪声；同一模块在 VMamba 解冻块是 SYSU **+1.31**（唯一过门），LEVIR 也只 −0.27。Run3 的 `ARM_C` 在 MobileNet 上 +1.10，前提同样是骨干在训。
2. **冻结本身的代价比模块之间的差大一个量级。** 同为 `CTRL`，VMamba 冻结→解冻在 SYSU 是 79.03→81.00（+1.97）、在 LEVIR 是 81.43→**90.70**（+9.27）。所以"冻结骨干省算力"在这套四尺度检索结构上不成立：冻结臂的绝对精度离可用线太远，横向比较只能在块内做。
3. **ViT-B 全参微调在 SYSU 上掉 4.15 pp**（82.52→78.37）：SYSU-CD 训练集只有 8 对，5e-4 全参更新把 ImageNet 特征带跑偏；LEVIR 上反而 +0.60（85.44→86.04）。于是两个库的最优骨干不同——SYSU 是 ViT-B 冻结（82.52，全矩阵最高），LEVIR 是 VMamba 解冻（90.70）。cd2 桥（`TAR_ONLY`）在 16 个块×库单元里最大只到 +0.60 pp（ViT-B 解冻/LEVIR），叠加臂最大 +0.79，与 MobileNet 上的 +0.6 同量级，**没有**因为换骨干而变成主要因素；`TAR_HRA` 在 VMamba 上还比 `HRA_ONLY` 低 1.74（SYSU），两模块互相拆台。

**工程口径**：折叠容差按骨干分档（MobileNet 2e-4 / ViT 2e-3 / VMamba 2e-2，预训练 logits 5–10 会把 FP32 折 BN 的舍入放大到 1e-2 量级），16 臂实测 maxdiff 最大 1.094e-02、二值差 ≤58 px（未训练头在 0.5 附近抖动）。收尾阶段修掉一个真 bug：`evaluate_auxiliary` 里 `encoder(...)[1:]` 把 `(stem, features)` 切成 list，导致 16 臂跑完 20k step 后全部崩在收尾；并口实测峰值 cd2+manifold ≈15 GiB、cd2 ≈12 GiB、轻臂 ≈6 GiB，一卡只能两条。

产物：`outputs/CASA-MinorGoal/Run9/{comparison.md, comparison.json, arm_params.txt}`、`docs/temporary/models_and_metrics_CASA-MinorGoal_Run9.txt`（46 个代码文件 + 428 行指标）、`docs/experiment_metrics.xlsx`（428 行）。

## 大小目标解耦：CASA-MinorGoal / Run8：CD²-TAR

Run7 之后的两处判断：桥的增益来自"判别边界"而非"小目标证据"，检索的增益只在召回。Run8 把 TAR 改成两步解耦的桥（CD²-TAR，Change-Decouple → Dimension-Decouple），检索侧用流形上压缩原型（Run7 已引入）。四臂同 seed、同调度、val 选模、阈定 0.5：

| 臂 | TAR 桥 | 流形检索 | SYSU F1 | Δ vs CTRL | LEVIR F1 | Δ vs CTRL |
|---|---|---|---:|---:|---:|---:|
| `CTRL` | reparam | — | 81.41 | — | 90.84 | — |
| `TAR_ONLY` | **CD²（cd2）** | — | **82.06** | **+0.64** | **91.02** | +0.19 |
| `HRA_ONLY` | reparam | manifold | 81.85 | +0.44 | 90.73 | −0.10 |
| `TAR_HRA` | **CD²** | manifold | **82.15** | **+0.74** | **91.04** | +0.20 |

判定（噪声 0.32 pp）：SYSU 上 CD² 桥 +0.64、检索 +0.44、叠加 +0.74，三条全过门；LEVIR 上三项都在噪声内。叠加并不等于两项相加（0.64+0.44=1.08 > 0.74），说明两个模块在 SYSU 上争用同一批边界像素。

**按 GT 变化面积分层（test，Run8，`<1k px` = 小目标尺度）**

| 库 | 臂 | <1k px 的 P / R / F1 | 1k–9k px F1 | >9k px F1 |
|---|---|---|---:|---:|
| SYSU | `CTRL` | 0.29 / 0.388 / 0.332 | 0.638 | 0.843 |
| SYSU | `TAR_HRA` | 0.39 / 0.387 / **0.389** | 0.645 | 0.849 |
| LEVIR | `CTRL` | — / 0.574 / 0.597 | 0.889 | 0.921 |
| LEVIR | `TAR_HRA` | — / **0.610** / **0.641** | 0.889 | 0.923 |

小目标那一档的召回基本不动（SYSU 0.388→0.387），但 F1 从 0.332 抬到 0.389，全部来自精度（0.29→0.39）；LEVIR 小目标档是召回抬（0.574→0.610）。所以 CD² 在 SYSU 上把"小变化误报"压下去，在 LEVIR 上把"小变化漏检"补回来——两库机制不同，不是同一个旋钮。

产物：`outputs/CASA-MinorGoal/Run8/{comparison.md, area_strata.json}`、`docs/temporary/models_and_metrics_CASA-MinorGoal_Run8/`（代码 + 指标快照）、`docs/experiment_metrics.xlsx`（396 行）。

## 双曲压缩与 2×2 模块消融（含池化轴补跑）：CASA-MinorGoal / Run7

Run6 之后两个模块都还在，但原型压缩仍是在切空间做加权平均、再映射回双曲面。Run7 把这一步搬到流形上（`--prototype-pooling lorentz`：`K = L(Σw·x)` 的加权 Klein 重心，严格落在双曲面上），并删掉 TAR 里"差异特征 → 共有特征"之间的可学习标量 `β`（差分以单位权重进入融合）。两处都是零参数增量：流形池化是纯几何操作，删 `β` 让每个 `cg` 臂各少 4 个参数。

四臂 2×2（`CTRL` / `TAR_ONLY` / `HRA_ONLY` / `TAR_HRA`），seed 2334、val 选模、test τ=0.5、两库各四条：

| arm | 模块 | TAR | 池化 | SYSU F1 | Δ | LEVIR F1 | Δ | train / deploy (M) |
|---|---|---|---|---:|---:|---:|---:|---|
| `CTRL` | none | reparam | euclid | 81.392 | — | 90.885 | — | 2.0471 / 1.9358 |
| `TAR_ONLY` | none | cg | euclid | 81.995 | +0.603 | 91.095 | +0.209 | 2.0935 / 2.0438 |
| `HRA_ONLY` | manifold | diff | lorentz | 82.533 | +1.141 | 90.851 | −0.035 | 1.9618 / 1.9234 |
| `TAR_HRA` | manifold | cg | lorentz | **83.016** | **+1.625** | 90.713 | −0.172 | 2.1113 / 2.0617 |

`compare.py` 的配对裁决（门限 TAR 0.36 / HRA 0.36 / 联合 0.47，噪声 0.32 pp）：

- **SYSU 三条全过门**：TAR +0.603、HRA +1.141、联合 +1.625；HRA 是三者里最干净的（3.6 倍噪声），TAR 只比噪声高 0.28 pp。
- **两个模块不可加**：0.603 + 1.141 = 1.744，实测 1.625，少 0.12 pp。机制上也能看出来——`TAR_ONLY` 是纯精度（P 85.24→87.05，R −0.38），`HRA_ONLY` 是纯召回（R +1.30），两者都在修同一段边界/召回，叠加只多 0.12。
- **LEVIR 三条全在噪声内**（+0.209 / −0.035 / −0.172），与 Run6 的 LEVIR 结论一致：这套改动的有效域只有 SYSU。
- **口径提醒**：这一轮 `CTRL` 是 81.392，比 Run2 同 seed 的 `CTRL_S2334`（82.092）低 0.70 pp、比 Run1 的 `CTRL`（81.729）低 0.34 pp。四个 Δ 是同 run 配对差，可信；`83.016` 这个绝对水平不能和历史上 83.37（Run3 `ARM_C`）直接比。

补跑（同属 Run7）：2×2 里两条带检索的臂同时带着流形池化，回答不了"重心本身值不值"。固定 `manifold + cg + lorentz`、只切 `--prototype-pooling`，SYSU 单库、seed 2334、2.77 h：

| arm | 池化 | F1 | Δ | R / P | 训练/部署 (M) |
|---|---|---:|---:|---|---|
| `POOL_EUCLID` | 切空间均值 → lift | **83.212** | — | 80.50 / 86.11 | 2.1113 / 2.0617 |
| `POOL_LORENTZ` | 先 lift → Klein 重心 | 82.824 | **−0.387** | 79.46 / 86.48 | 2.1113 / 2.0617 |

两臂参数逐位相同，裁决 **worse（≤ −0.36，噪声 0.32）**：压缩位置不是杠杆。与 `EUCLID_ALL` 82.504 vs `EDGE_ALL` 82.258（聚合换欧氏 +0.246）方向一致；`POOL_LORENTZ` 与 `TAR_HRA` 等价配置差 0.19 pp，故严谨写法是"无正效应、可能略负"。机制同向：重心偏保守（P +0.37、R −1.04）。

于是这一轮可以写成三句：**模块有用（TAR +0.60 / HRA +1.14 / 联合 +1.63，均过门，但不可加），把压缩搬进双曲面没用（−0.39），双曲几何的作用仍在打分、不在内容压缩。**

产物：`train_scripts/CASA-MinorGoal/Run7/`、`docs/temporary/models_and_metrics_CASA-MinorGoal_Run7/`（代码+指标快照、结构图）、`docs/experiment_metrics.xlsx`（388 行，`CASA-MinorGoal` 51 行，其中 Run7 10 行）。

## 梯度落点与杠杆排查：CASA-MinorGoal / Run6

Run5 的 +0.47 pp 过门，但另一会话的干预显示 SYSU 的 EDGE 权重高度依赖粗尺度回写，且 87.3% 的净新增正确检出在边界带以外。Run6 先只改一件事：保留两个边缘头与全部损失权重，把检索回写 `U_s` 在喂给边缘头之前 `detach()`，边缘梯度只能经由 TAR 的 `D_s` 回传。确认方向后再加两个开关做杠杆排查——`--edge-scales {all,fine}`（只监督 `D₁,D₂`）与 `--aggregate {lorentz,euclidean}`（把 Lorentz 重心换成普通加权均值）。

| arm | 设定 | SYSU F1 | R / P | best ep | Δ vs `EDGE_ALL` |
|---|---|---:|---|---:|---:|
| `EDGE_ALL` | 四级监督 + Lorentz 重心（参照） | 82.258 | 77.99 / 87.02 | 33 | — |
| `EDGE_FINE` | 只监督 `D₁,D₂` + Lorentz | 82.351 | 78.48 / 86.62 | 25 | +0.093 |
| `EUCLID_ALL` | 四级 + 普通加权均值 | 82.504 | 80.55 / 84.56 | 7 | +0.246 |
| `EDGE_TAP_CUT` | 与 `EDGE_ALL` 逐位同配置（第一批） | **83.090** | 79.66 / 86.83 | 33 | +0.832 |
| `EDGE_FULL` | 边缘梯度回写 `U_s` | 82.006 | 78.97 / 85.28 | 19 | −0.252 |

val 选模、test τ=0.5、seed 2334，两臂参数逐位相同（2,111,417 / 2,061,668），折叠 `5.96e-8`、二值差 0。三条结论：

1. **方向可信、幅度不可信**。`EDGE_ALL` 与 `EDGE_TAP_CUT` 是逐位相同的配置，复跑差 **0.832 pp**，比"切断回写"的效应（+1.085）只小 0.25 pp。能站住的是 `EDGE_FULL` 82.006 为五条最低，即"别让边缘梯度改回写"；站不住的是"+1.1 pp"。`models/` 的默认值仍改为 `edge_detach="retrieval"`（`HyperbolicContext`、`ManifoldContext`、`DifferenceContext`、`CASANet`、`--edge-detach`）。
2. **尺度不是杠杆**：只监督细两级 +0.093 pp，val 上高 0.64 而 test 上消失，只把选模落点从 ep33 推到 ep25。
3. **几何不是杠杆**：欧氏均值 +0.246 pp（噪声内），唯一稳定痕迹是操作点换位——重心 P 87.02 / R 77.99，均值 P 84.56 / R 80.55，即重心把判定推向更保守，而不是提高可分性。

LEVIR 侧补一条同配置（四级监督 + 切断回写 + Lorentz 重心）：**91.009**（R 89.83 / P 92.23 / IoU 83.63，best ep 162，val 91.59），与 Run5 `CG_HRA` 90.990、`CG_HRA_EDGE` 91.008 三条同 seed 相差 0.02 pp。**这套改动的作用域就是 SYSU 的边界带，LEVIR 上归零。**

[实验矩阵](train_scripts/CASA-MinorGoal/Run6/) · [快照与结果表](docs/temporary/models_and_metrics_CASA-MinorGoal_Run6/) · 逐项结果见 [指标表](docs/experiment_metrics.xlsx)。

## 版本消融：CASA-Pyramid / Run15：三个历史版本的同口径复现

把 Run1（Poincaré 球 + `qkv` 臂）、Run4（直接 Lorentz + `qkv` 臂）、Run10（后验压缩 + `matched` 欧氏双胞胎）三个版本的源码落到 `models_Run1/`、`models_Run4/`、`models_Run10/`（`SOURCE.md` 记来源与 sha256），统一在 WHU/SYSU 上按**验证集选模**各跑一遍控制臂与几何臂，共 20 项，2026-11-15 13:59 完成，无失败（中途一次显存 OOM 已提预留后重跑）。

| 版本 | 臂 | WHU F1（%） | Δ（pp） | SYSU F1（%） | Δ（pp） |
|---|---|---:|---:|---:|---:|
| Run1 | `none`（控制） | 93.6175 | — | 81.7839 | — |
| Run1 | `hyperbolic`（Poincaré） | 94.1792 | +0.562 | 83.1153 | +1.331 |
| Run1 | `qkv` | 94.4367 | +0.819 | 81.6452 | −0.139 |
| Run4 | `none`（控制） | 93.3574 | — | 83.0623 | — |
| Run4 | `hyperbolic`（Lorentz） | 93.9368 | +0.579 | 83.1173 | +0.055 |
| Run4 | `qkv` | 94.1704 | +0.813 | 81.9697 | −1.093 |
| Run10 | `none`（控制） | 94.0877 | — | 81.0951 | — |
| Run10 | `hyperbolic`（后验压缩） | 93.9819 | −0.106 | 82.6912 | +1.596 |
| Run10 | `matched`（同构欧氏） | 93.6482 | −0.440 | 82.7001 | +1.605 |
| Run10 | `qkv` | 94.0871 | −0.001 | 82.6776 | +1.583 |

三条结论：

1. **几何本身没有精度收益。** Run10 的 `matched` 与 `hyperbolic` 参数形状、初始化、读出、门控逐位一致，只把测地距离换成欧氏距离：WHU 差 **+0.334 pp**、SYSU 差 **−0.009 pp**。把三个独立实现放在一起看，同构配对差为 Run10 +0.334/−0.009、Run14 12 对均值 −0.167（sd 0.64）、Run12 WHU +0.118/+0.330，全部跨零且落在 0.18–0.32 pp 的复跑噪声内。
2. **SYSU 上的正增益来自"加了一个 context 模块"，不是双曲。** Run10 三条臂（`hyperbolic`/`matched`/`qkv`）在 SYSU 分别 +1.596/+1.605/+1.583，彼此相差 ≤0.03 pp；而它的控制臂 81.0951 是三个版本里最弱的。基线越弱，模块带来的提升越大，这与"几何更优"无关。
3. **WHU 与 SYSU 的排序在两库之间不稳定。** WHU 上 `qkv` 臂在三个版本都高于控制（+0.819/+0.813/−0.001），`hyperbolic` 在 Run1/Run4 为 +0.562/+0.579、在 Run10 为 −0.106；SYSU 上 `qkv` 在 Run1/Run4 为负。跨版本比较必须用版本内 Δ，不能直接比绝对 F1——三个版本的控制臂本身就相差 0.73 pp（WHU 93.36–94.09）和 1.97 pp（SYSU 81.10–83.06）。

这些副本只加了 `--selection-split` 补丁（默认 `test` 保持原行为），模型与损失未改；Run1 原生的 BGR 通道顺序与 Poincaré 实现原样保留。

[实验矩阵与结果](train_scripts/CASA-Pyramid/Run15/README.md) · 逐项结果见 [指标表](docs/experiment_metrics.xlsx) 与 [TXT 快照](docs/temporary/models_and_metrics_CASA-Pyramid_Run15.txt)。

## 实验：CASA-Pyramid / Run14：配对 seed 与真·去模块臂

Run14 在四库上用同 seed 初始化跑 H/E 配对（3 seed）与两条"真去掉模块"的臂，2026-11-14 14:32 完成 30 项。H/E 用验证集选模，得无偏 test 分数；去模块臂仍用 test 选模，直接与 Run13 阶梯衔接。

| 库 | H 三 seed 均值 | E 三 seed 均值 | H − E（pp） |
|---|---:|---:|---:|
| WHU | 93.1903 | 93.8825 | −0.692 |
| SYSU | 81.8998 | 81.8426 | +0.057 |
| LEVIR | 90.9503 | 90.9729 | −0.023 |
| CDD | 96.9370 | 96.9457 | −0.009 |

12 组配对差均值 **−0.167 pp**（sd 0.64，标准误 0.19，t=−0.90）：双曲几何在等价参数、等价延迟下没有精度收益。去模块臂相对 Run13 控制臂：把学习式时相混合换成固定 `|P−Q|` 后 WHU −0.297、SYSU +0.181、LEVIR −0.242 pp；把多级解码换成单层上采样后 WHU −0.212、SYSU −0.611、LEVIR −0.876 pp——粗解码对 LEVIR 的损失最大，是唯一在两轮里都稳定为负的移除。

[实验矩阵与结果](train_scripts/CASA-Pyramid/Run14/README.md)。

## 消融实验：CASA-Pyramid / Run13：TAR 与 DCR 的可折叠多分支

两条单因子臂，控制支路为 `module=none`、1 头、无细节头（Run12 的 `CTRL_ANCHOR`/`CTRL_REPEAT`/`DETAIL_OFF` 配置）。`TAR_PLAIN` 把 `TemporalRep1x1` 的 concat/P+Q/Q−P 三支换成单个 1×1+BN、`RepLocalBlock` 换成普通 DW3×3+PW1×1+BN；`DCR_PLAIN` 把 `RepPairFuse1x1` 的 concat/sum/diff 换成单个 1×1+BN，局部块同样退化。四尺度拓扑、通道、解码顺序、损失、增广与 40K 预算均不变。

北京时间 **2026-11-12 22:00** 完成 6 项（两臂 × WHU/SYSU/LEVIR），无失败：

| 数据集 | 臂 | F1（%） | 控制 F1（%） | Δ（pp） | 同配置复跑跨度（pp） |
|---|---|---:|---:|---:|---:|
| WHU | `TAR_PLAIN` | 94.7843 | 94.6403 | +0.144 | 0.177 |
| WHU | `DCR_PLAIN` | 94.7339 | 94.6403 | +0.094 | 0.177 |
| SYSU | `TAR_PLAIN` | 82.3698 | 82.5512 | −0.181 | 0.319 |
| SYSU | `DCR_PLAIN` | 82.3239 | 82.5512 | −0.227 | 0.319 |
| LEVIR | `TAR_PLAIN` | 91.0016 | 91.0455 | −0.044 | 单次控制 |
| LEVIR | `DCR_PLAIN` | 91.1427 | 91.0455 | +0.097 | 单次控制 |

全部差值都在同配置复跑跨度以内。这些分支只在训练期存在，折叠后三臂部署参数同为 1,935,810，RTX 5090 单张 256²、FP32、batch 1 中位延迟 3.678 / 3.680 / 3.672 ms，训练参数分别为 2.0471M / 1.9742M / 2.0103M。因此 TAR 与 DCR 的多分支可作为"可折叠的实现选择"表述，不能作为精度来源；H/E 几何与细节头的对照仍取 Run12。

[实验矩阵与结果](train_scripts/CASA-Pyramid/Run13/README.md) · 逐项结果见 [指标表](docs/experiment_metrics.xlsx) 与 [TXT 快照](docs/temporary/models_and_metrics_CASA-Pyramid_Run13.txt)。

## 主线实验：CASA-Pyramid / Run12

GPU 1 运行 26 项。WHU/SYSU 比较四头双曲、同构欧式、CTRL 和一头复跑。WHU 增加等宽 2×16 对照、seed 2334、二值细节监督及上下文交互；WHU/LEVIR 分别比较 128² 细节头有无，部署增加 2,689 参数。保留完整 40K 预算，每轮以 test F1 选最优权重。val 校准范围为 0.10–0.70。

逐轮 test 选模分数、最终 test 复测与独立 val 诊断分栏记录。主分析使用固定窗口与逐轮配对差，末 10 轮和共享的探索窗口作为补充。

[实验矩阵](train_scripts/CASA-Pyramid/Run12/README.md) · [冻结验证与阈值分析](docs/experiments/CASA-Pyramid/Run12/README.md) · [Run11 归因方案](docs/temporary/CASA-Pyramid_Run11_结果归因与Run12方案.md)。

### 阶段结果

北京时间 **2026-11-11 12:29**，正式完成 **18/26** 项，WHU 全部完成。GPU 1 同时运行 SYSU 四头双曲和 CTRL 锚点，分别完成 81/107 和 42/107 轮，另有 6 项排队。队列无失败，GPU 1 为 66°C，显存占用 27,683/32,607 MiB。18 项均有完整 `TEST RESULTS`，部署折叠后首级二值预测差异为 0。结果暂存于服务器，待全轮收齐后归档。

| WHU 配置 | seed 2333 F1（%） | seed 2334 F1（%） |
|---|---:|---:|
| CTRL 锚点 / 同 seed 复跑 | 94.6403 / 94.8173 | 94.5184 |
| 单头双曲 / 欧式 | 94.7712 / 94.6534 | 94.8220 / 94.4923 |
| 双头双曲 / 欧式 | 94.3130 / 94.0082 | — |
| 四头双曲 / 欧式 | 94.3013 / 94.5422 | 94.5969 / 94.2547 |
| 无细节头 / 软标签 / 硬标签 | 94.7832 / 94.2791 / 94.6561 | — |
| 单头双曲＋细节头 / 单头欧式＋细节头 | 94.2854 / 94.2914 | — |

单头双曲在两个 seed 上均高于匹配欧式，差值为 +0.118 和 +0.330 个百分点。同 seed 的 CTRL 锚点与复跑相差 0.177 个百分点。四头双曲在两个 seed 上均低于同 seed 单头双曲。相对无细节头，软标签细节头的总体 F1 和 1px 边界 F1 分别下降 0.504 和 0.460 个百分点。硬标签细节头的 1px 边界 F1 提高 0.247 个百分点，总体 F1 下降 0.127 个百分点。以上为 test 参与选模的同集复测结果，完整比较使用预设固定窗口和同 seed 对照。

### 参数与部署计算量

| 配置 | 训练参数（M） | 部署参数（M） | 部署 GFLOPs（双曲 / 欧式） |
|---|---:|---:|---:|
| CTRL | 2.0471 | 1.9358 | 1.7341 |
| CTRL＋细节头 | 2.0498 | 1.9385 | 1.8223 |
| 单头 QKV（1×32） | 2.0584 | 1.9471 | 2.0243 / 2.0158 |
| 双头 QKV（2×16） | 2.0585 | 1.9472 | 2.0396 / 2.0226 |
| 四头 QKV（4×16） | 2.0651 | 1.9537 | 2.2980 / 2.2638 |
| 单头 QKV＋细节头 | 2.0611 | 1.9498 | 2.1124 / 2.1039 |

FLOPs 以本轮冻结源码的折叠部署图统计，输入为一对 256×256 RGB 图像，一次乘加按 2 FLOPs。PyTorch profiler 计入卷积、矩阵乘和部分逐元素运算，双曲距离中的 `sqrt`、`log1p` 等算子未计入。单头 QKV 相对 CTRL 净增 11,284 个参数。RTX 5090 上 CTRL、单头双曲和单头欧式的 FP32、batch 1 实测中位延迟分别为 3.72、6.56 和 6.13 ms。[测量设置与各配置延迟](docs/experiments/CASA-Pyramid/Run12/README.md#修订版部署实测)。

从仓库根目录用 `python .codex/status.py --tag CASA-Pyramid --run Run12` 查实时状态。收齐后用 `python train_scripts/CASA-Pyramid/Run12/collect_results.py` 下载完整正式结果、更新指标表和 TXT，并导出固定窗口报告及逐轮记录。

## 归档实验：CASA-Pyramid / Run11

一阶guide内容池化，保留可变半径、完整Q、224KV、跨尺度门控和细节分支。H/E/CTRL分别对照首级内部正样本BCE权重1/1.25，共24项，仅GPU1。结构比较与训练损失比较分开解释，辅助统一0.1 raw-guide BCE。实现与检查见[Run11脚本](train_scripts/CASA-Pyramid/Run11/README.md)。

<!-- CASA_PYRAMID_RUN11_RESULTS_START -->

2026-11-10 10:36:14 CST 完成全部 **24/24** 项正式测试。下表为 test 选模后的复测 F1（%）。

| 数据集 | H_SIMPLE | H_INTERIOR | E_SIMPLE | E_INTERIOR | M0_CTRL | CTRL_INTERIOR |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR | 91.04 | 91.04 | 91.14 | 91.04 | 91.05 | 91.04 |
| SYSU | 83.52 | 83.09 | 83.03 | 82.77 | 83.02 | 82.97 |
| WHU | 94.59 | 94.33 | 94.64 | 94.37 | 94.62 | 94.44 |
| CDD | 96.93 | 96.94 | 96.93 | 96.91 | 96.94 | 96.93 |

<!-- CASA_PYRAMID_RUN11_RESULTS_END -->

SYSU 双曲简化版相对同轮 CTRL 的 F1 提高 **0.50 个百分点**，主要来自误报减少。SYSU 后期评估回落仍在，内部正样本加权使三种结构的内部召回均下降。WHU 的欧式简化版提高 1 像素边界 F1 **0.62 个百分点**。24 项部署折叠检查通过，首级二值预测差异为 0。

[完整日志、逐轮指标与曲线](docs/experiments/CASA-Pyramid/Run11/README.md) · [结果分析](docs/experiments/CASA-Pyramid/Run11/RESULT_ANALYSIS.md) · [冻结源码与指标](docs/temporary/models_and_metrics_CASA-Pyramid_Run11.txt) · [指标表](docs/experiment_metrics.xlsx)。

## 归档实验：CASA-Pyramid / Run10：几何归属与空间检索

每区域以平方guide权重构造变化/背景P0，用距离差`δ=(d_bg²−d_change²)/(2τ²)`修正归属`r=sigmoid(g+δ)`，再按r与1−r一阶池化P1，同时生成K与欧式V。完整Q、网格(8,4,4,4)、224KV、1头32维，压缩温度初始0.5且可学习。H/E训练2.058410M、部署1.947095M，context 11,478参数。

写回为`W_out[gate·(AV−mean_hw(AV))]`，保留TAR和细层局部残差。不同query检索相同时，全局更新为0，使该分支依赖空间相关的检索内容。H/E在归属修正和最终检索两处分别使用双曲/匹配欧式距离，其余结构一致。

```text
四尺度TAR → LN → 完整Q / 平方guide加权的两类初始P0
  → 本区域变化/背景距离差 → 修正软归属r
  → r与1−r一阶压缩P1 → 同源K/V，共224个
  → 全局检索AV → 减去同图同尺度空间均值 → 内容门控/输出投影
  → TAR残差 + 局部高通残差 → DCR
```

| 范围 | 训练组 | 比较目的 |
|---|---|---|
| LEVIR、SYSU、WHU、CDD | H_POSTERIOR、E_POSTERIOR、M0_CTRL，各3项 | 相同压缩结构下的几何贡献与模块净收益 |

正式计划共12项，仅GPU1，优先LEVIR→SYSU→WHU→CDD，按实测显存补位。协议`casa_pyramid_40k_supervisedposterior_v10`：H/E为main＋0.05 raw guide BCE＋0.05区域后验BCE，CTRL为main＋0.1 raw guide BCE；检索BCE为0，辅助目标均使用四尺度面积标签。共同batch32、40K、seed/context seed2333；正式从MobileNetV2预训练全新初始化，逐epoch test选模。辅助系数总预算相同，监督目标及梯度路径不同。

<!-- CASA_PYRAMID_RUN10_RESULTS_START -->

状态更新：2026-11-02 15:45:32 CST；队列 `complete`；正式结果 **12/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `H_POSTERIOR` | 完成 | 91.01 | 83.50 |
| LEVIR | `E_POSTERIOR` | 完成 | 91.12 | 83.68 |
| LEVIR | `M0_CTRL` | 完成 | 91.01 | 83.50 |
| SYSU | `H_POSTERIOR` | 完成 | 82.67 | 70.47 |
| SYSU | `E_POSTERIOR` | 完成 | 82.86 | 70.73 |
| SYSU | `M0_CTRL` | 完成 | 82.93 | 70.84 |
| WHU | `H_POSTERIOR` | 完成 | 94.18 | 89.01 |
| WHU | `E_POSTERIOR` | 完成 | 94.65 | 89.84 |
| WHU | `M0_CTRL` | 完成 | 94.80 | 90.12 |
| CDD | `H_POSTERIOR` | 完成 | 96.93 | 94.05 |
| CDD | `E_POSTERIOR` | 完成 | 96.95 | 94.08 |
| CDD | `M0_CTRL` | 完成 | 96.95 | 94.07 |

<!-- CASA_PYRAMID_RUN10_RESULTS_END -->

Run10已于2026-11-01 03:44 CST完成12/12项。结果已下载、汇表并冻结，指标表275条，原263条逐单元格保留。双曲/匹配欧式相对CTRL四库平均F1分别−0.2199/−0.0291pp；WHU双曲下降0.6195pp，SYSU后期泛化和漏检问题仍在，未形成整体改善。Run4继续作为正向基线。[完整训练记录](docs/experiments/CASA-Pyramid/Run10/README.md) · [正式分析](docs/experiments/CASA-Pyramid/Run10/RESULT_ANALYSIS.md) · [冻结TXT](docs/temporary/models_and_metrics_CASA-Pyramid_Run10.txt)。

[Run10脚本](train_scripts/CASA-Pyramid/Run10/README.md) · [机制、证据与实验安排](docs/temporary/CASA_Pyramid_Run10_区域几何归属方案.md) · [配对短训记录](docs/experiments/CASA-Pyramid/Run10/selection/centered/probe_comparison.md) · [参考文献](docs/参考文献/双曲类别归属/README.md)。后验监督是压缩归属的任务约束，半径不预设语义层级；双曲度量的额外收益由配对结果判断。

## 归档实验：CASA-Pyramid / Run9

2026-10-31 13:22 CST停止，0项完整TEST，源码基线`3a04aca`及日志、权重保留。固定256张LEVIR val的冻结权重干预中，完整模型F1为91.0789%，关闭全局上下文为91.1318%，关闭局部分支为90.8255%；全局分支尚无净纠错收益。同区域β1精炼强化接近同类均值的内容，缺少变化与背景的相对比较。

[冻结诊断](docs/experiments/CASA-Pyramid/Run10/diagnosis/README.md) · [Run9记录](docs/experiments/CASA-Pyramid/Run9/README.md) · [实际源码快照](docs/temporary/models_and_metrics_CASA-Pyramid_Run9.txt)。停止项及验证干预不计正式指标。

## 归档实验：CASA-Pyramid / Run8

按[Run7深度分析方案](docs/temporary/CASA_Pyramid_Run7_深度分析_调研与改进方案.md)分阶段开展。首阶段仅SYSU/WHU的H_CLASS与H_NONE，共4项，物理GPU1；完整224KV、共享K/V、独立内容门控与局部分支不变，唯一目标差异为retrieval BCE系数0.1→0，guide固定0.1。run/context seed均为2333，loader使用独立配对随机流；两臂均从预训练骨干全新初始化。

匹配欧式距离、β=0/1区域内一次精炼及显式seed/辅助权重接口已实现。首阶段β保持0；其后共同目标、几何与精炼对照见Run9。现有test逐epoch选模口径继续保留，完整val用于诊断，不称独立盲测。

<!-- CASA_PYRAMID_RUN8_RESULTS_START -->

状态更新：2026-10-31 12:09:18 CST；队列 `complete`；正式结果 **4/4**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| SYSU | `H_CLASS` | 完成 | 82.77 | 70.61 |
| SYSU | `H_NONE` | 完成 | 82.61 | 70.38 |
| WHU | `H_CLASS` | 完成 | 94.36 | 89.33 |
| WHU | `H_NONE` | 完成 | 94.57 | 89.70 |

<!-- CASA_PYRAMID_RUN8_RESULTS_END -->

首阶段于2026-10-31 01:03:56 CST完成，4/4项无失败。检索监督使SYSU F1提高0.1589pp、WHU降低0.2072pp，WHU的1px边界F1降低1.0802pp；尚无跨库一致收益。[归因分析](docs/experiments/CASA-Pyramid/Run8/analysis.md)。全部部署折叠通过，最大误差1.3709e−6、二值差0。指标表共**263条**：253条历史＋Run8四项＋独立MoE Run2六项，历史单元格全部保留。

[Run8脚本与阶段](train_scripts/CASA-Pyramid/Run8/README.md) · [冻结诊断与启动记录](docs/experiments/CASA-Pyramid/Run8/README.md) · [冻结源码快照](docs/temporary/models_and_metrics_CASA-Pyramid_Run8.txt)。

## 归档实验：CASA-Pyramid / Run7

协议`casa_pyramid_40k_retrieval_v7`。四库各双曲/CTRL共8项，物理GPU0；LEVIR→SYSU→WHU→CDD优先，按实测显存与1GiB留量补位。正式训练从MobileNetV2预训练全新初始化。

```text
A/B → 共享MobileNetV2 → 四级TAR → LN
  ├─ 完整Q：共享32维投影 → Lorentz提升
  └─ 变化/背景区域压缩 → 同一内容P → K与欧式V
       四尺度128/32/32/32，共224KV
  → 双曲条件检索AV → 内容门控g → W_out[g(AV−V_query)]
  + 原TAR残差 + 细两层局部高通 → DCR → 共享预测头
  训练辅助：变化/背景注意力总权重的log-odds → 面积标签BCE
```

Q/K均由`W_qkY/√32`直接提升，半径由投影范数学习；不预设置信度或语义层级。K/V使用同一P，取消密集点质心。类别检索辅助监督复用注意力logits；采用量由内容门控学习。双曲为主损失＋0.1 guide＋0.1检索BCE，CTRL为主损失＋0.2 guide。骨干、增强、优化器与40K预算相同；同轮收益包含检索结构和辅助监督。

| 变体 | 训练参数M | 部署参数M | context参数 |
|---|---:|---:|---:|
| 双曲 | 2.058409 | 1.947094 | 11,477 |
| CTRL | 2.047125 | 1.935810 | 193 |

16项LEVIR/SYSU完整train/val短训均完成12epoch。选定双曲的最佳val F1为89.4910%/81.2207%，CTRL为89.3840%/80.5914%；相对差值+0.107/+0.629pp。同结构欧式度量对照为89.5358%/80.4533%。旧Run6为89.4639%/81.4482%，新候选减少一半KV，SYSU最佳值略低。固定LEVIR子集在此阶段未观察到全局分支收益；正式训练中SYSU后期泛化不足仍存在。短训用于选型，正式结论见本轮完整结果。

<!-- CASA_PYRAMID_RUN7_RESULTS_START -->

Run7于2026-10-28 16:00:08 CST启动，2026-10-29 00:38:32 CST完成，历时8小时38分24秒。8/8项正式训练与测试完成，无失败，GPU0队列已退出。全部部署折叠检查通过，最大绝对差3.397e-06，二值差均为0。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 双曲相对CTRL (pp) |
|---|---:|---:|---:|
| LEVIR | 91.1058 | 91.1316 | +0.026 |
| SYSU | 82.7919 | 82.8851 | +0.093 |
| WHU | 94.7471 | 94.5274 | -0.220 |
| CDD | 96.9528 | 96.9054 | -0.047 |

双曲相对同轮CTRL在LEVIR、SYSU提高0.026、0.093个百分点，WHU、CDD降低0.220、0.047个百分点；四库等权平均差值为-0.037个百分点。SYSU精确率提高、召回降低，WHU的1px边界F1降低约0.370个百分点。当前尚未形成四库一致的净收益。

完整日志、逐轮CSV、曲线、最佳epoch和模块诊断已发布。本轮收集时指标表229条，原221条历史记录逐单元格保留；Run7冻结TXT来自实际source_code.tar.gz，current TXT来自工作源码。

[完整训练记录](docs/experiments/CASA-Pyramid/Run7/README.md) · [结果分析](docs/experiments/CASA-Pyramid/Run7/analysis.md) · [冻结快照](docs/temporary/models_and_metrics_CASA-Pyramid_Run7.txt)。

<!-- CASA_PYRAMID_RUN7_RESULTS_END -->

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run7
python train_scripts/CASA-Pyramid/Run7/collect_results.py
```

[Run7脚本](train_scripts/CASA-Pyramid/Run7/README.md) · [筛选证据](docs/experiments/CASA-Pyramid/Run7/selection/selection_summary.md)。日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run7`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run7`。

### Run6停止记录

<!-- CASA_PYRAMID_RUN6_RESULTS_START -->

Run6于2026-10-28 12:46 CST停止，正式结果0/8；LEVIR双曲/CTRL完成138/162个epoch、SYSU CTRL完成95个epoch，源码、日志和checkpoint保留。停止项不计入指标表。[运行脚本](train_scripts/CASA-Pyramid/Run6/README.md)。

<!-- CASA_PYRAMID_RUN6_RESULTS_END -->

## Run5归档训练模型

```text
双时相RGB A/B → 共享MobileNetV2原生四尺度 → TAR至64通道
  ├─ 原TAR残差
  └─ 共享LN
       ├─ 完整Q：5440个位置，4头×16维
       ├─ 变化/背景概率平方选择 → 128/32/32/32个KV
       │    普通：池化后投影K/V，欧氏距离
       │    双曲：直接Lorentz映射与质心K，欧式V，测地距离
       └─ 每个Q的自身候选 + 224个原型 → 联合softmax
            → Σ A_ij(V_j−V_i) → 输出投影与逐通道残差
  + 细两层高通DW3×3/PW1×1局部残差
  → DCR逐级解码 → 共享预测头，训练四级/推理一级
```

Run5针对Run4实测的稀疏变化混合：区域权重由`m/Σm`改为`(m+eps)²/Σ(m+eps)²`，m分别为p与1−p。类别支持度为`Σw·m²`；均匀模糊区域的两原型总支持低于明确区域。Q自身以距离0、支持度1加入softmax，压缩内容的总权重可小于1；检索不匹配时减少写回，原TAR与局部细节继续保留。新增机制不增加可学习参数。

双曲曲率固定−1，直接坐标`(sqrt(1+||z||²),z)`无硬半径裁剪，K使用Lorentz归一化质心，V保持欧式。两版共享压缩、支持度和自参照规则；几何作用通过同轮比较验证。没有把四尺度直接解释为语义层次。

| 变体 | 训练参数 | 折叠部署参数 | context参数 |
|---|---:|---:|---:|
| M0_CTRL | 2,047,125 | 1,935,810 | 193 |
| M1_HYPER_QKV | 2,064,537 | 1,953,222 | 17,605 |
| M2_CA_QKV | 2,064,537 | 1,953,222 | 17,605 |

完整公式、逐条问题核实及四库验证干预见[Run5设计](docs/temporary/CASA-Pyramid_Run5_design.md)。实现与来源见[models](models/README.md)、[SOURCES](models/SOURCES.md)，论文见[Run5调研](docs/参考文献/Run5_原型选择与自参照检索/README.md)。

## 固定训练与评价设置


| 设置 | 值 |
|---|---|
| 输入 / batch / seed | RGB 256×256 / 32 / 2333 |
| 初始化 | MobileNetV2预训练；每项全新初始化；各组共同路径初值一致 |
| 读取与归一化 | A/B各自BGR→RGB，保持时相顺序；除255；ImageNet均值(.485,.456,.406)、标准差(.229,.224,.225) |
| resize / mask | 图像双线性，标签最近邻；标签gray≥128 |
| 裁剪 | 训练以0.5概率四边各裁0～8像素，再恢复256；A/B/标签同步 |
| 翻转 / 旋转 | 水平、垂直各0.5；均匀选择0/90/180/270度，A/B/标签同步 |
| 时相交换 | 训练以0.5概率交换A/B；标签保持对应空间位置 |
| 光度增强 / 评估增强 | 无独立时相颜色扰动；评估无随机增强、无TTA |
| 优化器 | Adam，全网lr5e-4，betas(.9,.99)，eps1e-8 |
| Weight decay | context卷积/线性weight为解耦衰减1e-4，其他context为0；网络其余保持Adam耦合L2衰减1e-4 |
| 学习率 / 预算 | 前200步warmup，之后poly^0.9，骨干倍率1.0；40K按完整epoch上取整 |
| 主损失 | 四级 BCEWithLogits + sigmoid Dice，等权求和；`--detail-head` 时额外监督 128² 合成 logit（area 或 nearest-exact） |
| 辅助损失 | 0.1 raw guide BCE，使用四尺度面积标签；检索 BCE 系数默认 0（`--retrieval-weight`） |
| 精度 | FP32，AMP关闭；cuDNN TF32开启，matmul TF32关闭，benchmark开启 |
| 评估与选模 | 每 epoch 在 `--selection-split` 指定的划分上评估（test 或 val），第一级 F1 最高选 best，相同值更新 |
| 最终结果 | 折叠 best 后在 test 复测，主指标阈值 0.5，次指标使用 val 校准阈值；val 选模的项记 `independent_test=true`，test 选模的项不得称独立盲测 |
| 部署检查 | strict_fp32_fold_v1；关闭TF32检查，首级最大差阈值2e-4并记录二值差 |

参考[ChangeMamba的2024官方训练脚本](https://github.com/ChenHongruixuan/ChangeMamba/blob/72597fa4a1fde0bffa94c94302a6dc848898be13/changedetection/script/train_MambaBCD.py)使用test参与选模；其原实现每500步评估并按Kappa选模，本项目采用每epoch与F1。该设置不代表所有RSCD工作的统一规范。

| 数据集 | Train / Val / Test | epoch / 实际步数 |
|---|---:|---:|
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | 181 / 40,182 |
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | 107 / 40,125 |
| WHU-CD-256 | 5,947 / 743 / 744 | 217 / 40,145 |
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | 129 / 40,248 |

## 归档结果：CASA-Pyramid / Run5

四库各CTRL、双曲、普通，共12项；LEVIR→SYSU→WHU→CDD优先，两张GPU按实测显存和1GiB留量并行补位。正式权重从MobileNetV2预训练全新初始化，不继承Run4或探针权重。共同优化、增强与损失保持Run4设置。

<!-- CASA_PYRAMID_RUN5_RESULTS_START -->

Run5于2026-10-27 12:54:33 CST启动，2026-10-27 22:10:22 CST完成，历时9小时15分49秒。12/12项正式训练及测试完成，无失败；调度器已退出，两张GPU空闲。全部部署折叠检查通过，最大绝对差3.457e-06，二值差均为0。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) | 双曲/普通相对CTRL (pp) |
|---|---:|---:|---:|---:|
| LEVIR | 91.0562 | 91.0931 | 91.2324 | +0.037 / +0.176 |
| SYSU | 82.6402 | 82.3319 | 82.9401 | -0.308 / +0.300 |
| WHU | 94.3515 | 94.4315 | 94.4922 | +0.080 / +0.141 |
| CDD | 96.9524 | 96.9051 | 96.8727 | -0.047 / -0.080 |

普通版在LEVIR、SYSU、WHU相对同轮CTRL提高0.176、0.300、0.141个百分点，CDD降低0.080个百分点；双曲版在LEVIR、WHU提高0.037、0.080个百分点，在SYSU、CDD降低0.308、0.047个百分点。普通版四库均高于双曲版。相对Run4，普通版LEVIR/SYSU绝对F1略升、WHU/CDD下降；双曲版四库均降低，同轮净收益也减弱。Run5数值稳定，但尚未证明平方选择与自身竞争优于Run4的组合。Run4继续保留为正向基线。

[完整日志与曲线](docs/experiments/CASA-Pyramid/Run5/README.md) · [结果分析](docs/experiments/CASA-Pyramid/Run5/analysis.md) · [冻结源码TXT](docs/temporary/models_and_metrics_CASA-Pyramid_Run5.txt)

<!-- CASA_PYRAMID_RUN5_RESULTS_END -->

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run5
python train_scripts/CASA-Pyramid/Run5/collect_results.py
python tools/models_to_txt.py --tag CASA-Pyramid --run current
```

[Run5脚本与状态](train_scripts/CASA-Pyramid/Run5/README.md)。本地输出`outputs/CASA-Pyramid/Run5`；服务器日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run5`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run5`。

## 保留基线：CASA-Pyramid / Run4

2026-10-26完成12/12，历时9小时00分28秒，源码基线`eb3babe`。训练数值稳定，部署折叠全部通过、二值差为0。两版在LEVIR、SYSU、WHU均比同轮CTRL提高，CDD基本持平。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) | 双曲/普通相对CTRL (pp) |
|---|---:|---:|---:|---:|
| LEVIR | 90.9482 | 91.1166 | 91.1429 | +0.168 / +0.195 |
| SYSU | 82.3165 | 82.7753 | 82.8930 | +0.459 / +0.576 |
| WHU | 94.1891 | 94.6085 | 94.7247 | +0.419 / +0.536 |
| CDD | 96.9150 | 96.9384 | 96.8998 | +0.023 / −0.015 |

Run4是正向结果，普通版在前三库略高于双曲版。SYSU后期训练损失下降而选模损失上升仍存在，普通版最佳epoch为3。较大残差和跨尺度交互未自动带来更多收益；固定val干预确认建筑库检索与局部支路均有作用，原型内容纯度仍有改善空间。

RTX5090、256×256、FP32 batch1，30预热/100次中位数的Run4 CTRL/H/E推理延迟为3.83/6.79/5.78ms。相同测量设置下，Run5 CTRL/H/E为3.75/6.83/5.84ms，参数量不变。

[Run4完整日志、曲线与诊断](docs/experiments/CASA-Pyramid/Run4/README.md) · [结果分析](docs/experiments/CASA-Pyramid/Run4/analysis.md) · [冻结源码TXT](docs/temporary/models_and_metrics_CASA-Pyramid_Run4.txt) · [配置说明](train_scripts/CASA-Pyramid/Run4/README.md)

## 历史版本

| 轮次 | 保留结论 | 完整材料 |
|---|---|---|
| CASA-Pyramid Run3 | 配对双库、双曲差异检索；SYSU/CDD小幅收益，建筑库效果不足，共同LR/loss同时变化 | [记录与分析](docs/experiments/CASA-Pyramid/Run3/README.md) |
| CASA-Pyramid Run2 | 224KV、共享Q/K与欧式V；LEVIR/WHU受益，SYSU下降 | [记录与分析](docs/experiments/CASA-Pyramid/Run2/README.md) |
| CASA-Pyramid Run1 | TAR后128KV跨尺度检索；初步验证结构，四库收益不一致 | [完整记录](docs/experiments/CASA-Pyramid/Run1/README.md) |
| 早期CASA与CASA-WDMF | 模块、优化与几何经验保留在各轮冻结TXT | [历史快照](docs/temporary/) / [指标表](docs/experiment_metrics.xlsx) |

指标表收录 299 条正式记录，其中 Run11 新增 24 条。正式指标只取完整 TEST RESULTS。历史冻结 TXT 使用对应运行源码，current TXT 使用工作源码。

## 已知限制

Run11 为单 seed，使用 test 选模后同集复测。每区域每类一个原型可能合并不同外观，空间中心化可能削弱大面积同类变化需要的共享信息。模块收益按同轮配对结果评价，边界、内部召回和总体 F1 分开报告。[Run11 结果分析](docs/experiments/CASA-Pyramid/Run11/RESULT_ANALYSIS.md)。
