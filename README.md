# CASA-CD

普通RGB双时相影像的全监督遥感二值变化检测。当前主线 **CASA-Pyramid / Run12（四头检索与高分辨率细节对照）**。**Run4作为正向基线完整保留**。

双曲版采用完整 Q、224 个跨尺度 KV 和直接 Lorentz 投影，K/V 共用一阶 guide 池化内容，空间中心化检索后门控写回。Run12 对照 1×32、2×16 和 4×16，H/E 各配置同构，部署约 1.95M 参数。[设计与公式](models/HYPERBOLIC_DESIGN.md)。

## 项目定位与固定研究原则

本项目面向211高校硕士的小论文研究，目标是在LEVIR、SYSU、WHU、CDD四库取得有竞争力的精度，同时控制推理参数、计算、延迟与显存。贡献聚焦两个独立版本：双曲压缩QKV、普通变化感知上下文压缩QKV，保持完整Q空间位置，仅压缩KV。

项目F1努力目标为LEVIR≥92%、SYSU≥84%、WHU≥95%、CDD≥97.5%。这些数值是研究目标，不是已核实的统一SOTA界限。先证明模块有效、机制连贯、收益可归因，再争取领先指标。有效性优先于绝对最低参数，允许为有用表征小幅增参；参数目标≤5M，当前部署约2M。较少参数不等于更快推理。

结构须对应可验证的实际问题，保持简洁。避免无关组件、特殊激活函数和额外几何运算堆叠；不将损失、剪枝或量化包装成主要创新。修改前核对日志、运行源码与历史经验，修改后审查公式、卷积/池化、融合/残差、维度与前后梯度，并用真实数据短训检查。

同轮各组使用相同骨干、增强、优化、训练预算及选模规则，专用监督单独记录。Run12 使用每轮 test F1 选权重，正式复测主阈值固定 0.5，次指标阈值在 val 校准。复测沿用 test 选模口径。结构设计与干预使用 train/val，跨协议差异与模块收益分别解释。

文献优先2024–2026年相关高水平工作，基础理论追溯原始论文，核实题名、年份、venue与官方代码。参考材料持续维护于[文献索引](docs/参考文献/文献索引.md)，核心代码与许可证放在`others/`。工作约定见[AGENTS.md](AGENTS.md)，研究对话设置见[ChatGPT_Project_Settings.md](docs/ChatGPT_Project_Settings.md)。

## 项目导航

| 要找的内容 | 位置 |
|---|---|
| 主线模型、训练与测试 | [models/](models/)，模型入口 `model/casa_net.py`，训练入口 `train_CASA.py` |
| 各轮实验矩阵、调度与收集 | [train_scripts/CASA-Pyramid/](train_scripts/CASA-Pyramid/) 下对应的 `RunN/` |
| 服务器连接、环境和状态 | [.codex/SERVER.md](.codex/SERVER.md)、[.codex/STATE.md](.codex/STATE.md)，实时状态用 `.codex/status.py` |
| 日志、指标、曲线和冻结源码 | 本地 `outputs/<tag>/<Run>/`，已归档材料见 [docs/experiments/](docs/experiments/)、[指标表](docs/experiment_metrics.xlsx) 与 [TXT 快照](docs/temporary/) |
| 下载、汇表与导出工具 | [tools/](tools/) |
| 独立的 MoE 探索线 | [README_MoE.md](README_MoE.md) 与 [models_v1/](models_v1/) |

代码仓库为 [YuqiWang-code/CASA-CD](https://github.com/YuqiWang-code/CASA-CD)，训练服务器为 RSML-3。

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
| 主损失 | 四级 BCEWithLogits + sigmoid Dice，等权求和。Run12 DETAIL_ON 额外监督 128² 合成 logit |
| 辅助损失 | Run12 各组统一 0.1 raw guide BCE，使用四尺度面积标签，检索 BCE 系数为 0 |
| 精度 | FP32，AMP关闭；cuDNN TF32开启，matmul TF32关闭，benchmark开启 |
| 评估与选模 | Run12 每 epoch 在 test 评估，第一级 F1 最高选 best，相同值更新 |
| 最终结果 | 折叠 best 后在同一 test 复测，主指标阈值 0.5，次指标使用 val 校准阈值 |
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
