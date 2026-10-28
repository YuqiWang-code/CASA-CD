# CASA-Pyramid 工作源码

双曲版采用224个跨尺度原型、一个32维Lorentz空间及类别检索监督，用于CASA-Pyramid/Run7，协议`casa_pyramid_40k_retrieval_v7`。普通QKV保持Run5结构。

共享MobileNetV2输出24/32/96/320通道原生特征，经四级TAR统一为64通道；上下文模块位于TAR之后、DCR之前。双曲和普通模块保留全部5440个空间查询，推理使用第一级预测。

## 模块

| 变体 | 压缩与关系 | 内容写回 |
|---|---|---|
| 双曲QKV | 网格8/4/4/4，每区域变化/背景各一个原型，共128/32/32/32＝224KV；Q/K共享32维投影，经`/sqrt(32)`直接Lorentz提升；K/V来自同一内容P | 仅在原型间softmax；独立66→1内容门控；`W_out[g(AV−V_query)]` |
| 普通QKV（Run5） | 224KV、四头×16维、平方类别选择、欧氏Q/K、欧式V | self零logit联合softmax、二阶支持度；`W_out[AV−mass·V_query]` |
| CTRL | 同一LN/guide训练监督 | TAR直接进入DCR |

双曲K由P投影后提升，V由P线性投影，不使用密集流形点质心。内在半径由投影范数决定，不预设为置信度或语义层级。门控根据本地内容、检索差值及本地/检索占比预测学习采用量。两种模块均保留原TAR残差，以及S1/S2的`PW1(DW3(X−Avg3(X)))`局部分支；逐通道gamma零初始化，投影正常初始化。

| 变体 | 训练参数M | 折叠部署参数M | context参数 |
|---|---:|---:|---:|
| 双曲QKV | 2.058409 | 1.947094 | 11,477 |
| 普通QKV | 2.064537 | 1.953222 | 17,605 |
| CTRL | 2.047125 | 1.935810 | 193 |

## 训练与接口

四级主损失为BCEWithLogits＋Dice。双曲辅助项为`0.1 × guide BCE + 0.1 × 类别检索BCE`；CTRL和普通QKV为`0.2 × guide BCE`。类别检索logit由变化/背景原型的注意力logits分别做logsumexp后相减，监督标签是对应分辨率的变化面积均值。辅助项不增加可学习预测头，`predict()`不计算类别检索辅助输出。

RGB256、batch32、seed2333、40K完整epoch；Adam lr5e−4、betas(.9,.99)、eps1e−8，200步warmup和poly0.9。context卷积/线性权重解耦WD1e−4，其余context参数WD0，主体耦合L2 WD1e−4。训练使用同步裁剪、水平/垂直翻转、90°旋转、时相交换及ImageNet归一化。

`forward`返回四级概率；`return_aux=True`返回四级logits和辅助输出；`predict`返回一级概率。每epoch在test上按F1选模，最终测试标注test参与选模。部署折叠采用`strict_fp32_fold_v1`，最大误差阈值2e−4并记录二值差。

## 文件组织

| 文件 | 用途 |
|---|---|
| `model/casa_net.py` | 骨干、TAR、context、DCR及预测 |
| `model/layers/hyperbolic_context.py` | 同内容K/V、双曲检索、内容门控与类别检索辅助输出 |
| `model/layers/lorentz_ops.py` | 曲率−1 Lorentz提升与稳定平方测地距离 |
| `model/layers/pyramid_context.py` | 普通QKV压缩、读出及共享算子 |
| `train_CASA.py` | 损失、优化、选模、最终测试及训练记录 |
| `profile_CASA.py` | 实际训练显存与步耗时 |

## 已知限制

原型的变化/背景类别名来自guide分配，不保证P纯净。半径变化、非零梯度和注意力差异不能单独证明双曲关系有效。相对CTRL的收益同时包含结构与辅助监督的影响，几何贡献需要同结构欧式度量对照；短训用于选择候选，完整精度及效率以正式测量为准。

[设计与公式](HYPERBOLIC_DESIGN.md) · [Run7](../train_scripts/CASA-Pyramid/Run7/README.md) · [来源](SOURCES.md)
