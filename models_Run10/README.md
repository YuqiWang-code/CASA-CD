# CASA-Pyramid 工作源码

CASA-Pyramid/Run10使用224个跨尺度原型、32维关系投影及一次区域内类别归属修正，协议`casa_pyramid_40k_supervisedposterior_v10`。四库比较双曲、严格匹配欧式与CTRL。普通QKV的Run5实现独立保留。

共享MobileNetV2输出24/32/96/320通道原生特征，经四级TAR统一为64通道；context位于TAR之后、DCR之前。完整Q保留5440个位置，推理使用第一级预测。

## 模块

| 变体 | 压缩与关系 | 内容写回 |
|---|---|---|
| 双曲QKV | 网格8/4/4/4，每区域变化/背景两个原型，共224KV；平方guide权重形成初始类中心，双曲距离修正像素类别归属，修正概率一阶汇聚P；Q/K共享32维投影，直接Lorentz提升；K/V同源 | 独立66→1内容门控；`W_out[g(AV−mean_spatial(AV))]` |
| 匹配欧式 | 与双曲相同的投影、原型、两个温度、支持度、门控和局部分支；压缩及全局检索均使用空间坐标欧氏平方距离 | 与双曲相同 |
| 普通QKV（Run5） | 224KV、四头×16维、平方类别选择、欧氏Q/K、欧式V | self零logit联合softmax；`W_out[AV−mass·V_query]` |
| CTRL | 同一LN/guide结构，原guide监督 | TAR直接进入DCR |

令`p=sigmoid(guide(X))`，`X=LN(T)`。每区域按`(p+1e−6)²`及`(1−p+1e−6)²`构造初始类中心P0。密集Q分别计算到本区域两类K0的距离，得到`delta=(d_background²−d_change²)/(2 tau_comp²)`；修正归属`r=sigmoid(guide_logits+delta)`。最终P1按r、1−r的一阶权重汇聚，支持度按相同权重汇聚原p、1−p；K/V均从P1生成。

压缩温度是独立可学习标量，构造参数`compression_temperature`和CLI `--compression-temperature`指定初始值，默认0.5。全局检索温度单独学习。完整更新可微，新增参数仍仅1个；不增加空间邻域或token。

检索内容C在每个样本、每个尺度内减去空间均值，均值参与反传；注意力与query位置无关时，全局更新为零。门控随位置变化，门控后更新的空间均值可以不为零。门控依据32维本地V、32维中心化检索内容及两个变化占比预测采用量。保留TAR残差，以及S1/S2的`PW1(DW3(X−Avg3(X)))`局部分支；逐通道gamma零初始化，投影正常初始化。

| 变体 | 训练参数M | 折叠部署参数M | context参数 |
|---|---:|---:|---:|
| 双曲 / 匹配欧式 | 2.058410 | 1.947095 | 11,478 |
| 普通QKV | 2.064537 | 1.953222 | 17,605 |
| CTRL | 2.047125 | 1.935810 | 193 |

## 训练与接口

四级主损失为BCEWithLogits＋Dice。训练入口默认`--guide-weight 0.1 --retrieval-weight 0`，Run10脚本同时显式记录该设置，guide标签为对应位置的变化面积均值：

| 变体 | 总目标 |
|---|---|
| 双曲 / 匹配欧式 | `L_seg + 0.05 BCE(raw_logits, y_area) + 0.05 BCE(posterior_logits, y_area)` |
| CTRL | `L_seg + 0.1 BCE(raw_logits, y_area)` |

H/E监督目标同构；CTRL的总辅助预算相同，目标为原guide。`posterior_logits=guide_logits+delta`，按区域顺序还原空间位置后直接计算BCEWithLogits，不经`logit(sigmoid)`。标签只进入损失。类别检索输出用于诊断，`predict()`不计算辅助输出。

RGB256、batch32、seed2333、40K完整epoch；Adam lr5e−4、betas(.9,.99)、eps1e−8，200步warmup和poly0.9。context卷积/线性权重解耦WD1e−4，其余context参数WD0，主体耦合L2 WD1e−4。同步裁剪、水平/垂直翻转、90°旋转、时相交换及ImageNet归一化保持固定。

`forward`返回四级概率；`return_aux=True`返回四级logits和辅助输出；`predict`返回一级概率。每epoch在test上按F1选模，最终结果标注test参与选模。部署折叠采用`strict_fp32_fold_v1`，阈值2e−4并记录二值差。manifest/profile包含`GUIDE_OBJECTIVE`对应的`guide_objective`、`guide_effective_coefficients`、`compression_temperature_init`、初始平方/最终一阶分配及随机流。训练日志`guide`表示实际guide目标；`guide_raw`和`guide_posterior`分别记录两项BCE，CTRL不输出`guide_posterior`。

## 文件组织

| 文件 | 用途 |
|---|---|
| `model/casa_net.py` | 骨干、TAR、context、DCR及预测 |
| `model/layers/posterior_compression.py` | 初始置信锚点、几何类分数、归属修正与共享内容汇聚 |
| `model/layers/hyperbolic_context.py` | 匹配H/E压缩及检索、内容门控与辅助输出 |
| `model/layers/lorentz_ops.py` | 曲率−1 Lorentz提升与稳定平方测地距离 |
| `model/layers/pyramid_context.py` | 普通QKV及共享算子 |
| `train_CASA.py` / `profile_CASA.py` | 训练测试、记录及实际显存/步耗时 |
| `test_classposterior_compression.py` | 常量场、类交换、修正方向、同权内容与梯度 |
| `test_classposterior_health.py` | 主任务梯度、残差启动、时相输入与部署折叠 |
| `test_centered_readout.py` / `test_posterior_supervision.py` | query无关检索的零更新、后验监督梯度与面积标签对齐 |

## 诊断与限制

诊断记录delta均值/绝对值/离散程度、r−p的方向和幅度、posterior熵、低于0.01及高于0.99的饱和比例，以及两类有效像素数、空原型比例、P1改变量、门控和残差强度。GT分组的纠错与误改在前向完成后的分析中测量。

归属修正依赖当前特征与guide形成的类中心，不是独立校准的贝叶斯后验；错误guide也可能产生错误锚点。一个区域只有两个原型，修正不保证纯净内容或完整多模态表示。半径不预设为置信度或层级；非零梯度、分配变化不代表精度提升。匹配H/E衡量度量在压缩与检索中的联合作用；CTRL比较完整模块。TAR有序concat及带符号差分，时相交换由增强覆盖，不是严格结构对称。效率以当前实测为准。

[设计与公式](HYPERBOLIC_DESIGN.md) · [Run10](../train_scripts/CASA-Pyramid/Run10/README.md) · [来源](SOURCES.md)
