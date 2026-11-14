# CASA-Pyramid

共享MobileNetV2原生四尺度（24/32/96/320通道）→ TAR时相融合至64通道 → 跨尺度上下文修正 → DCR → 共享预测头。context接收四层TAR变化特征；骨干结束于320通道，不使用HFEA。训练四级深监督，`predict()`输出最终一级变化概率。

当前工作源码对应Run4，协议`casa_pyramid_40k_relative_adamwctx_v4`，脚本位于`train_scripts/CASA-Pyramid/Run4/`。设计见[Run4设计文档](../docs/temporary/CASA-Pyramid_Run4_design.md)，参考材料见[文献索引](../docs/参考文献/文献索引.md)。历史冻结快照保持对应轮次实现。

## 文件组织

| 文件 | 职责 |
|---|---|
| `model/casa_net.py` | 骨干、TAR、context、DCR与预测头的数据流 |
| `model/layers/pyramid_context.py` | 变化/背景压缩、可靠性与尺度先验、欧式V、相对回写和细层局部分支 |
| `model/layers/euclidean_context.py` | 共享Q/K投影、欧氏距离检索 |
| `model/layers/hyperbolic_context.py` | 直接Lorentz映射、流形K池化、双曲距离检索 |
| `model/layers/lorentz_ops.py` | 曲率−1的映射、归一化质心与平方测地距离 |
| `train_CASA.py` | 训练协议、选模、诊断及部署检查 |
| `profile_CASA.py` | 实际配置的训练显存及步耗时测量 |

## 共同输入与压缩

四层TAR共用通道LayerNorm，`X_s=LN(T_s)`。Q/K使用同一个无bias的64→64投影，四尺度共享，拆为4头×16维并除以√16。保留全部5440个query位置，只有K/V压缩。

| 尺度 | 分辨率 | 区域网格 | 每区域像素数 | 变化/背景token |
|---|---|---|---:|---:|
| S1 | 64×64 | 8×8 | 64 | 64＋64=128 |
| S2 | 32×32 | 4×4 | 64 | 16＋16=32 |
| S3 | 16×16 | 4×4 | 16 | 16＋16=32 |
| S4 | 8×8 | 4×4 | 4 | 16＋16=32 |

共享guide从X预测变化概率p。每区域对`m=p`和`m=1−p`分别构造一个原型：

```text
w_i = (m_i + 1e−6) / Σ_j(m_j + 1e−6)
P = Σ_i w_i X_i
r = Σ_i m_i² / max(Σ_i m_i, 1e−6)
```

r是按类别概率加权的置信度，不是面积占比；少量高置信变化不因面积小而直接降低prior。分配权重和prior保留梯度。224个token联合检索：

```text
logit(q,k) = −d(q,k)² / (2 tau_head²)
             + log(max(r_k,1e−6))
             + scale_bias[head,query_scale,source_scale]
             − log(source_token_count)
A = softmax(logit)
```

温度为每头一个softplus正参数，初值0.5。数量先验修正128/32/32/32布局的token数偏差，内容与可学习尺度偏置仍决定实际偏好。

## 内容回写与局部细节

V为欧式64维，先池化X，再做无bias线性投影。完整query使用同一个V投影，回写检索内容相对当前内容的差值：

```text
U_s = W_out(reshape(A_s @ V) − W_V X_s)
T'_s = T_s + gamma_context[s] ⊙ U_s
```

常量内容的修正为零；一般输入下的实际幅度仍由学习决定。S1/S2增加局部高频分支，平均滤波和深度卷积均使用replicate padding：

```text
H_s = X_s − AvgPool3×3(X_s)
L_s = PW1×1(DW3×3(H_s))
T'_s = T'_s + gamma_detail[s] ⊙ L_s
```

该高通来自TAR变化特征，不是RGB梯度或额外A/B绝对差分。没有独立边界预测头和边界辅助损失。`gamma_context`为4×64、`gamma_detail`为2×64，逐通道零初始化；支路卷积和输出投影正常初始化，初始输出等于TAR。

## 普通QKV

`euclidean_context.py`采用共享Q/K坐标的平方欧氏距离。完整Q由X密集投影；K先池化X再投影。LN在池化前，同一组权重下池化与无bias线性投影交换等价。

变化与背景原型进入同一个224-token softmax。没有独立对比差分库，也不通过两个独立softmax分别广播均值与对比内容。

## 双曲QKV

`hyperbolic_context.py`把共享投影输出z作为Lorentz空间坐标：

```text
x = (sqrt(1 + ||z||²), z)
<x,x>_L = −1
radius(x) = asinh(||z||)
```

曲率固定−1；不逐token单位化投影输出，不做径向硬裁剪。每个K使用与V一致的变化/背景权重，将密集Lorentz点加权求和后归一化。该操作是Lorentz归一化质心，不称任意多点的精确Fréchet均值。

检索仅使用平方测地距离`acosh(−<q,k>_L)²`。实现从空间弦差与时间差计算`cosh(d)−1`，在同点附近使用解析级数，使平方距离零点导数有限。V保持欧式内容，不做输出Log映射。实际几何范围通过半径、流形误差及任务梯度诊断记录。

## 训练与数据

- `forward(a,b)`返回四级概率；`return_aux=True`返回四级分割logits与四层`guide_logits`。`predict()`只计算最终一级预测；TAR/DCR在eval下折叠。
- 四级主损失均为BCEWithLogits＋sigmoid Dice，求和；guide为面积平均标签BCE，权重0.2。CTRL保留同样的LN与guide监督，TAR直接进入DCR。
- 256×256、batch32、seed2333、40K目标按整epoch上取整。Adam基础LR5e−4，预训练骨干倍率1.0，betas(0.9,0.99)、eps1e−8；前200步warmup、poly指数0.9。三组共同设置一致。
- context内全部Conv2d/Linear权重使用独立参数组：`decoupled_weight_decay=True`、WD1e−4，即衰减按`weight *= 1−lr·wd`作用于参数，不进入Adam梯度矩估计。LN、bias、温度、gamma及尺度偏置WD0；encoder与body保留原Adam耦合L2衰减WD1e−4。实现使用服务器PyTorch2.14的[Adam参数组接口](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)。FP32训练，AMP关闭。
- RGB双线性缩放、标签最近邻；标签灰度≥128。像素除255，再用ImageNet均值`[0.485,0.456,0.406]`、标准差`[0.229,0.224,0.225]`归一化。
- 训练：0.5概率裁剪四边各0～8像素后恢复；水平/垂直各0.5概率翻转；等概率0/90/180/270°旋转；0.5概率时相交换。空间变换同步，不做独立光度扰动。评价仅缩放和归一化；训练shuffle、drop_last，评价保留全部样本。
- 每epoch在test选最高F1权重，相同更新，结束复测best；结果属于test参与选模，非独立盲测。训练/验证材料优先用于机制诊断。

## 诊断与效率

正式训练前，LEVIR的双曲/普通模块各短训6个完整epoch，SYSU各4个完整epoch，在val评价并检查损失、任务梯度和分支状态；短训不计正式指标，正式训练从预训练权重全新初始化。

两版记录context/local残差RMS及相对TAR幅度、guide概率、原型可靠性、两类分配差异、Q/K与V权重RMS、温度、注意力熵、来源尺度质量及query差异。双曲另记录query/key半径、半径CV和流形误差。每epoch模块诊断取评价首个batch，边界及内部指标按全split累计；单个熵、gamma或幅度不能独立证明模块失活或贡献为正。

参数、显存和延迟以本轮测量为准，不沿用历史结构数值。目标是低开销下的有效、可归因提升；允许为有用表征增加少量参数，不以极端压参替代精度验证。
