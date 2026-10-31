# CASA-Pyramid 工作源码

CASA-Pyramid/Run9使用224个跨尺度原型、32维关系投影及一次区域内内容精炼，协议`casa_pyramid_40k_refinement_v9`。四库比较精炼双曲、严格匹配欧式与CTRL；SYSU/WHU补充未精炼欧式，对照已有Run8 H_NONE。普通QKV的Run5实现独立保留。

共享MobileNetV2输出24/32/96/320通道原生特征，经四级TAR统一为64通道；上下文模块位于TAR之后、DCR之前。双曲和普通模块保留全部5440个空间查询，推理使用第一级预测。

## 模块

| 变体 | 压缩与关系 | 内容写回 |
|---|---|---|
| 精炼双曲QKV | 网格8/4/4/4，每区域变化/背景各一个原型，共128/32/32/32＝224KV；类别平方分配后做一次内容精炼；Q/K共享32维投影，经`/sqrt(32)`直接Lorentz提升；K/V来自同一内容P | 仅在原型间softmax；独立66→1内容门控；`W_out[g(AV−V_query)]` |
| 匹配欧式 | 与双曲共享投影、原型、支持度、温度、门控和局部分支；只将平方测地距离改为空间坐标的欧氏平方距离 | 与双曲相同 |
| 普通QKV（Run5） | 224KV、四头×16维、平方类别选择、欧氏Q/K、欧式V | self零logit联合softmax、二阶支持度；`W_out[AV−mass·V_query]` |
| CTRL | 同一LN/guide训练监督 | TAR直接进入DCR |

双曲K由P投影后提升，V由P线性投影，不使用密集流形点质心。内在半径由投影范数决定，不预设为置信度或语义层级。门控根据本地内容、检索差值及本地/检索占比预测学习采用量。两种模块均保留原TAR残差，以及S1/S2的`PW1(DW3(X−Avg3(X)))`局部分支；逐通道gamma零初始化，投影正常初始化。

| 变体 | 训练参数M | 折叠部署参数M | context参数 |
|---|---:|---:|---:|
| 双曲 / 匹配欧式（β=0或1） | 2.058409 | 1.947094 | 11,477 |
| 普通QKV | 2.064537 | 1.953222 | 17,605 |
| CTRL | 2.047125 | 1.935810 | 193 |

## 训练与接口

四级主损失为BCEWithLogits＋Dice。Run9所有组显式使用`--guide-weight 0.1 --retrieval-weight 0`，统一为`L_seg+0.1 L_guide`。guide监督标签是对应分辨率的变化面积均值。类别检索辅助输出保留用于诊断，当前不进入优化目标；`predict()`不计算该输出。Run8的H_NONE采用相同目标，其SYSU/WHU结果作为β=0双曲参照。

RGB256、batch32、seed2333、40K完整epoch；Adam lr5e−4、betas(.9,.99)、eps1e−8，200步warmup和poly0.9。context卷积/线性权重解耦WD1e−4，其余context参数WD0，主体耦合L2 WD1e−4。训练使用同步裁剪、水平/垂直翻转、90°旋转、时相交换及ImageNet归一化。

`forward`返回四级概率；`return_aux=True`返回四级logits和辅助输出；`predict`返回一级概率。每epoch在test上按F1选模，最终测试标注test参与选模。部署折叠采用`strict_fp32_fold_v1`，最大误差阈值2e−4并记录二值差。

## 文件组织

| 文件 | 用途 |
|---|---|
| `model/casa_net.py` | 骨干、TAR、context、DCR及预测 |
| `model/layers/hyperbolic_context.py` | 单次内容精炼、同内容K/V、匹配H/E检索、内容门控与辅助输出 |
| `model/layers/lorentz_ops.py` | 曲率−1 Lorentz提升与稳定平方测地距离 |
| `model/layers/pyramid_context.py` | 普通QKV压缩、读出及共享算子 |
| `train_CASA.py` | 损失、优化、选模、最终测试及训练记录 |
| `profile_CASA.py` | 实际训练显存与步耗时 |
| `test_refinement_health.py` | β=1两种度量的主任务梯度、时相输入与部署折叠 |

## 已知限制

RTX5090、双时相256输入、batch1、FP32，cuDNN TF32开启、matmul TF32关闭，30次预热/100次中位数：CTRL 3.863ms、54.76MiB峰值allocated；双曲6.918ms、73.90MiB。测量使用随机权重结构，完整记录见[Run7效率](../docs/experiments/CASA-Pyramid/Run7/inference_benchmark.json)。

原型的变化/背景类别名来自guide分配，不保证P纯净；单次精炼也不能保证保留稀疏目标或完整表达多模态。半径变化、非零梯度和注意力差异不能单独证明双曲关系有效。Run9相对CTRL衡量完整模块收益，匹配H/E衡量距离的作用，SYSU/WHU的β对照衡量精炼作用。短训检查训练健康，精度与效率以正式测量为准。TAR包含有序concat及带符号差分，时相交换由数据增强覆盖，不是结构上的严格对称。

[设计与公式](HYPERBOLIC_DESIGN.md) · [Run9](../train_scripts/CASA-Pyramid/Run9/README.md) · [来源](SOURCES.md)

## 精炼与匹配度量接口

`module="matched"`为同结构欧式对照，与Run5普通QKV分别实现。`refinement_beta=0`精确返回原压缩权重；β=1在区域内按`exp(cosine(X,P0))`做一次重加权，K/V和support同步更新，参数不增加。LN在压缩前，cosine的eps为1e−6，初始原型和分配权重均保留梯度。Run9精炼两臂固定β=1，E_BASE固定β=0。

训练入口显式记录run/context seed、配对loader随机流和精炼配置。每项完整训练结束后，在val分别测量best/last的guide与retrieval BCE，保存在auxiliary_validation.json。诊断中的`compression_weight_tv`比较精炼前后区域权重，`compression_effective_pixels`为`1/Σw²`的均值，`prototype_refinement_to_feature_rms`为原型改变量相对LN特征RMS；它们衡量压缩的实际改动，不代表精度增益。`gate_mean`表示门控，`prototype_attention_mass_mean`表示原型注意力总量，`value_difference_rms`表示未归一化内容差值。

[Run8阶段与配置](../train_scripts/CASA-Pyramid/Run8/README.md) · [Run7正式锚点](../docs/experiments/CASA-Pyramid/Run7/README.md)
