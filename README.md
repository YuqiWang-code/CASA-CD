# CASA-CD

普通RGB双时相影像的全监督遥感二值变化检测。当前开发与实验轮次 **CASA-Pyramid / Run5**，协议`casa_pyramid_40k_selective_selfref_v5`。**Run4作为正向基线完整保留**。

## 项目定位与固定研究原则

本项目面向211高校硕士的小论文研究，目标是在LEVIR、SYSU、WHU、CDD四库取得有竞争力的精度，同时控制推理参数、计算、延迟与显存。贡献聚焦两个独立版本：双曲压缩QKV、普通变化感知上下文压缩QKV，保持完整Q空间位置，仅压缩KV。

项目F1努力目标为LEVIR≥92%、SYSU≥84%、WHU≥95%、CDD≥97.5%。这些数值是研究目标，不是已核实的统一SOTA界限。先证明模块有效、机制连贯、收益可归因，再争取领先指标。有效性优先于绝对最低参数，允许为有用表征小幅增参；参数目标≤5M，当前部署约2M。较少参数不等于更快推理。

结构须对应可验证的实际问题，保持简洁。避免无关组件、特殊激活函数和额外几何运算堆叠；不将损失、剪枝或量化包装成主要创新。修改前核对日志、运行源码与历史经验，修改后审查公式、卷积/池化、融合/残差、维度与前后梯度，并用真实数据短训检查。

同轮三组使用相同骨干、增强、损失、训练预算和选模规则。当前按每epoch的test F1选模，最终复测属于**test参与选模的结果**，不能称独立盲测。结构设计与干预使用train/val；阈值固定0.5，不按测试成绩选seed。单seed小差异不作为稳定领先证据，跨协议涨幅与模块收益分别解释。

文献优先2024–2026年相关高水平工作，基础理论追溯原始论文，核实题名、年份、venue与官方代码。参考材料持续维护于[文献索引](docs/参考文献/文献索引.md)，核心代码与许可证放在`others/`。工作约定见[AGENTS.md](AGENTS.md)，研究对话设置见[ChatGPT_Project_Settings.md](docs/ChatGPT_Project_Settings.md)。

## Run5模型

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
| 初始化 | MobileNetV2预训练；每项全新初始化；三组共同路径初值一致 |
| 读取与归一化 | A/B各自BGR→RGB，保持时相顺序；除255；ImageNet均值(.485,.456,.406)、标准差(.229,.224,.225) |
| resize / mask | 图像双线性，标签最近邻；标签gray≥128 |
| 裁剪 | 训练以0.5概率四边各裁0～8像素，再恢复256；A/B/标签同步 |
| 翻转 / 旋转 | 水平、垂直各0.5；均匀选择0/90/180/270度，A/B/标签同步 |
| 时相交换 | 训练以0.5概率交换A/B；标签保持对应空间位置 |
| 光度增强 / 评估增强 | 无独立时相颜色扰动；评估无随机增强、无TTA |
| 优化器 | Adam，全网lr5e-4，betas(.9,.99)，eps1e-8 |
| Weight decay | context卷积/线性weight为解耦衰减1e-4，其他context为0；网络其余保持Adam耦合L2衰减1e-4 |
| 学习率 / 预算 | 前200步warmup，之后poly^0.9，骨干倍率1.0；40K按完整epoch上取整 |
| 主损失 | 四级BCEWithLogits + sigmoid Dice，等权求和 |
| 评分损失 | 四尺度面积平均标签的BCEWithLogits求和×0.2，三组一致 |
| 精度 | FP32，AMP关闭；cuDNN TF32开启，matmul TF32关闭，benchmark开启 |
| 评估与选模 | 每epoch（包括0）在test评估，第一级F1最高选best，相同值更新 |
| 最终结果 | 折叠best后在同一test复测；明确标记test参与选模、非独立测试 |
| 部署检查 | strict_fp32_fold_v1；关闭TF32检查，首级最大差阈值2e-4并记录二值差 |

参考[ChangeMamba的2024官方训练脚本](https://github.com/ChenHongruixuan/ChangeMamba/blob/72597fa4a1fde0bffa94c94302a6dc848898be13/changedetection/script/train_MambaBCD.py)使用test参与选模；其原实现每500步评估并按Kappa选模，本项目采用每epoch与F1。该设置不代表所有RSCD工作的统一规范。

| 数据集 | Train / Val / Test | epoch / 实际步数 |
|---|---:|---:|
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | 181 / 40,182 |
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | 107 / 40,125 |
| WHU-CD-256 | 5,947 / 743 / 744 | 217 / 40,145 |
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | 129 / 40,248 |

## 当前实验：CASA-Pyramid / Run5

四库各CTRL、双曲、普通，共12项；LEVIR→SYSU→WHU→CDD优先，两张GPU按实测显存和1GiB留量并行补位。正式权重从MobileNetV2预训练全新初始化，不继承Run4或探针权重。共同优化、增强与损失保持Run4设置。

<!-- CASA_PYRAMID_RUN5_RESULTS_START -->

正式队列于2026-10-27 12:54:33 CST启动，源码基线`b2ebe9c`，supervisor PID 1912750。两卡首批5项已完成至少3个训练及评价epoch，主损失下降，未见NaN、OOM或非有限梯度。正式结果0/12，历史指标209条保持不变。

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

已收集正式指标209条，Run4原始源码与全部历史记录保留。Run5新增12条收齐后应为221条；如有其他新增记录全部保留。正式指标只取完整TEST RESULTS，探针与验证干预不入表。历史冻结TXT用对应运行源码，current TXT用工作源码。

## 已知限制

当前单seed且test参与选模；小幅差异需要进一步复核。Run5平方分配仍可能放大guide高分误报，自身竞争也不是独立可靠性估计；同类单原型不能表示任意多种外观。SYSU长期泛化与双曲几何额外收益仍需正式结果判断。详细证据与方案边界集中在Run5设计文档中。
