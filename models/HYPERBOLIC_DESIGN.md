# 双曲上下文：区域内容精炼与匹配几何检索

当前协议`casa_pyramid_40k_refinement_v9`。模块处理TAR融合后的变化特征，保持完整Q、压缩KV，以主分割损失和guide监督学习跨尺度上下文。Run9对区域类别原型做一次内容关联精炼，并以同结构欧式距离对照检验几何作用。曲率固定为−1，内容V保持欧式。

```text
A/B → 共享MobileNetV2 → 四尺度TAR → T_s
  ├─ 原TAR残差
  └─ X_s = LN(T_s)
       ├─ guide p → 区域变化/背景平方分配 → P0 → 单次cosine精炼 → 224个共同内容P
       │                                      ├─ K：共享投影 → Lorentz提升
       │                                      └─ V：欧式线性投影
       └─ 完整Q：共享投影 → Lorentz提升
            → 平方双曲距离＋类别支持度＋尺度先验 → A
            → R=AV → 内容门控g → W_out[g(R−V_query)] → gamma
  + S1/S2局部高通DW3/PW1残差
  → DCR → 四级训练 / 一级推理
```

## 压缩与几何

四尺度网格为8/4/4/4，每区域生成变化和背景两个原型，数量128/32/32/32，总数224。对类别隶属度`m_i=p_i`或`1−p_i`：

```text
w0_i = (m_i+eps)^2 / Σ_l(m_l+eps)^2
P0 = Σ_i w0_i X_i
w_i = w0_i exp(β cosine(X_i,P0)) / Σ_l[w0_l exp(β cosine(X_l,P0))]
P = Σ_i w_i X_i,  β=1（精炼）或0（原压缩）
r = max(Σ_i w_i m_i, eps),  eps=1e−6
```

LN位于池化前，分配权重保留梯度。r是类别支持度；面积标签下p=.5可以表示半面积变化，不直接解释为低置信。

完整Q与压缩K共用一个32维投影：

```text
z(Y) = W_qk Y / sqrt(32)
H(Y) = (sqrt(1+||z(Y)||²), z(Y))
Q_i = H(X_i), K_j = H(P_j)
V_j = W_v P_j, V_i = W_v X_i
rho(Y) = asinh(||z(Y)||)
```

K/V共享同一P，K不经过密集双曲点质心聚合。改变区域散布但保持P不变时，K和V保持不变。半径连续可学习，不做单位方向归一化或径向硬裁剪；它表示投影幅度形成的几何坐标，不预设为置信度、抽象层级或目标大小。

## 路由、内容采用与局部细节

```text
l_ij = −d_H(Q_i,K_j)^2/(2 tau²)
       + log(r_j) + b[query_scale,source_scale] − log(M_source)
A_i = softmax_j(l_ij)
R_i = Σ_j A_ij V_j
p_ctx_i = Σ_j A_ij sigmoid(W_guide P_j+b_guide)
g_i = sigmoid(G([V_i, R_i−V_i, p_i, p_ctx_i]))
U_i = W_out[g_i(R_i−V_i)]
T'_s = T_s + gamma_context_s ⊙ U_s + local_s
```

tau采用softplus参数化，初值0.5。数量校正抵消不同源尺度原型数量造成的先验偏好，尺度偏置仍可学习。注意力只在224个压缩原型间归一化；门控G为66→1卷积，独立学习内容采用量，不读取距离的总匹配量。

S1/S2保留`local=gamma_detail⊙PW1(DW3(X−Avg3(X)))`。原TAR与局部支路提供细节路径；常量内容下全局差值为0。gamma零初始化、输出投影正常初始化，内部支路在gamma开始学习后获得主任务梯度。

## 共同训练目标与检索诊断

Run9的CTRL、双曲和匹配欧式均使用相同训练目标：

```text
L = L_seg + 0.1 L_guide
```

`L_seg`为四级BCEWithLogits＋Dice；guide BCE使用对应分辨率的GT面积均值，并在四尺度求和。保留`e_i=logsumexp_change(l_ij)−logsumexp_background(l_ij)`作为检索类别诊断；其BCE权重为0。e复用检索logits，不增加可学习预测头，推理时不计算该辅助输出。

Run8中检索监督对SYSU与WHU的影响方向不同。Run9固定同一个简洁目标，从而将度量和精炼作为独立比较因素。SYSU/WHU的β=0双曲参照来自Run8 H_NONE；数据、初始化、优化和完整训练预算保持一致。

## 验证与限制

检查内容包括流形恒等式、同点平方距离及梯度、压缩与投影一致性、类别原型索引、log-odds、全部支路任务梯度、常量输入和部署折叠。train/val短训比较主损失、F1、召回、边界、半径、注意力和最终残差；正式实验从预训练骨干全新初始化。

原型类别名来自guide，不保证内容纯净；一类一个区域原型仍会混合多种外观。共享内容保证K/V统计对应，但不能保证被检索的V一定有利于纠错。门控不是最终残差幅度上界，半径和路由变化也不是精度收益证据。完整双曲路径的有效性与双曲度量自身的贡献分别报告。

## 一次区域内关联精炼

由原权重w0得到P0；对同一区域内每个类别，w1∝w0·exp(β cosine(X,P0))，沿区域像素重新归一化。K和V均从P1=Σw1X生成，support同步为Σw1m。cosine使用eps=1e−6，不新增投影或token。β=0精确返回原权重，β=1是预先固定的单次更新候选；常量场和零场保留恒等读出，分配及内容梯度均可传播。

四库比较β=1双曲、β=1匹配欧式与CTRL；SYSU/WHU额外训练β=0匹配欧式，结合Run8 H_NONE组成两种度量×两种压缩的对照。匹配欧式仅将`d_H²`替换为Lorentz空间坐标的欧氏平方距离，温度、初始化、K/V、支持度、门控、局部分支与预算均相同。精炼的作用与度量的作用分别报告。
