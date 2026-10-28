# CASA-Pyramid 工作源码

双曲版用于CASA-Pyramid/Run6，协议`casa_pyramid_40k_taskgeometry_v6`；普通QKV与CTRL沿用Run5结构。Run6只训练双曲和CTRL共8项，限定GPU0。已完成Run5的结果与冻结TXT对应原归档。

主体：共享MobileNetV2原生24/32/96/320通道→四级TAR至64通道→融合后上下文→DCR→共享预测头。所有版本保留完整5440个Q，推理使用第一级预测。

## 模块

| 变体 | 内容与几何 | 路由与采用 |
|---|---|---|
| 双曲候选 | 每区变化/背景各两个外观子原型，共448KV；K/V来自同一压缩内容；一个32维Lorentz空间，半径`asinh(1+p)`关联guide预测变化占比 | 双曲条件softmax路由；66→1内容门控独立学习采用量；`W_out[g(AV−V_query)]` |
| 普通QKV（Run5） | 平方类别选择，224KV，四头×16维，欧氏Q/K和欧式V | self零logit联合softmax，二阶支持度，`W_out[AV−mass·V_query]` |
| CTRL | 同一LN/guide辅助监督 | TAR原样交给DCR |

双曲版的p来自既有面积平均标签BCE监督，表示预测变化占比；p=.5不等同低置信。方向使用归一化Q/K投影，空间范数为1+p；零方向取原点。K由内容P投影后提升，不使用密集点质心，因此不把区域散布混入半径。V保持欧式。门控读取本地内容、检索差值和本地/上下文占比，通过主分割任务学习；绝对距离不作为采用量。

两版都保留原TAR残差及S1/S2的`PW1(DW3(X−Avg3(X)))`局部分支。gamma零初始化，输出投影正常初始化；常量场全局差值为0。几何、半径与门控均不能替代被压缩掉的内容，双子原型也可能趋同；完整设计与逐项处理见[HYPERBOLIC_DESIGN](HYPERBOLIC_DESIGN.md)。

| 变体 | 训练参数M | 折叠部署参数M | context参数 |
|---|---:|---:|---:|
| 双曲候选 | 2.058474 | 1.947159 | 11,542 |
| 普通QKV | 2.064537 | 1.953222 | 17,605 |
| CTRL | 2.047125 | 1.935810 | 193 |

## 文件组织

| 文件 | 用途 |
|---|---|
| `model/casa_net.py` | 骨干、TAR、context、DCR和预测 |
| `model/layers/hyperbolic_context.py` | 占比半径、双子原型、内容一致的K/V和独立门控 |
| `model/layers/lorentz_ops.py` | 曲率−1 Lorentz点与稳定平方测地距离 |
| `model/layers/pyramid_context.py` | 普通QKV压缩、自参照读出及共享局部算子 |
| `model/layers/euclidean_context.py` | 普通欧式距离检索 |
| `train_CASA.py` | 共同训练、guide监督、评价与协议记录 |
| `profile_CASA.py` | 同配置训练显存与步耗时 |
| `analyze_hyperbolic.py` | train/val固定样本的径向、角向、度量与门控干预及错误分组 |
| `test_hyperbolic_context.py` | 双曲候选的公式、内容一致性、采用规则与梯度检查 |

## 训练与检查

共同设置仍为RGB256、batch32、seed2333、Adam全网lr5e−4、betas(.9,.99)、eps1e−8，200warmup/poly^.9、40K整epoch。context Conv/Linear权重解耦WD1e−4，其他context WD0；网络其余保持Adam耦合L2 WD1e−4。四级BCEWithLogits+Dice＋.2倍guide面积标签BCE；未增加门控、曲率或层次损失。

RGB ImageNet归一化；训练同步裁剪、水平/垂直翻转、90度旋转及时相交换。当前共同训练入口每epoch在test选F1最高权重，结果须标为test参与选模；分析工具仅使用train/val。

模型、梯度、常量/零场、稀疏类别、共同logit平移不变、稀有匹配采用与整网折叠检查通过。LEVIR/SYSU各12个真实batch检查中，Q/K、guide、外观分配和门控的主任务梯度非零且有限。该检查不作为正式训练或精度证据。Run6在完整LEVIR 6epoch与SYSU 4epoch健康探针通过后启动；探针与正式训练均限定GPU0。

部署检查使用strict_fp32_fold_v1，首级误差阈值2e−4并记录二值差。接口保持`forward`四级概率、`return_aux=True`四级logits和guide、`predict`一级概率。

## 效率与已知限制

RTX5090、RGB256、FP32、batch1，30预热/100次中位数，随机权重结构测量：双曲候选6.91ms、峰值allocated 102.11MiB。Run5双曲为6.83ms、158.86MiB；新结构减少推理显存，当前未取得延迟下降。FLOPs只计支持的卷积/矩阵乘，不能表示全部几何算术。

半径含义、K/V对应和采用规则已明确；精度、泛化和双曲独立贡献仍需正式训练及相同协议对照。新双曲权重与Run5不兼容；历史权重继续使用归档源码。

[逐项设计与诊断](HYPERBOLIC_DESIGN.md) · [来源](SOURCES.md) · [历史Run5](../train_scripts/CASA-Pyramid/Run5/README.md)
