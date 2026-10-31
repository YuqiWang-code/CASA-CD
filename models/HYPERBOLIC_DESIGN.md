# 双曲上下文：区域类别归属修正与跨尺度检索

协议`casa_pyramid_40k_supervisedposterior_v10`。模块处理TAR融合后的变化特征，完整Q、224个KV、一头32维；双曲几何参与区域内变化/背景归属和跨尺度检索。

```text
四尺度TAR → X=LN(T) → guide logits → p
    │            │                 │
    │            │           平方置信权重
    │            │                 ↓
    │            ├→ 完整Q        初始P0 → K0(两类)
    │            │                 │
    │            └──── 同区域两类距离差
    │                              ↓
    │        guide logits + (d_background²−d_change²)/(2 tau_comp²)
    │                              ↓ sigmoid
    │                        修正归属 r / 1−r
    │                              ↓ 一阶内容汇聚
    │                         P1、support
    │                          ┌───┴───┐
    │                          K     欧式V
    │            完整Q → 224KV跨尺度检索
    │                          ↓
    │                  内容门控 × (AV−mean_spatial(AV))
    │                          ↓ W_out、gamma
    └───────── TAR残差 + 全局修正 + 浅层局部修正
                               ↓
                         DCR → 变化预测
```

## 置信锚点与归属修正

`X=LN(T)`，guide输出`b`，`p=sigmoid(b)`。每个网格区域分别使用变化/背景的平方置信权重：

```text
m_i+ = p_i; m_i− = 1−p_i
w0_ic = (m_ic+1e−6)² / Σ(k∈region(i))(m_kc+1e−6)²
P0_rc = Σ(i∈r) w0_ic X_i
z(X) = W_qk X / √32
L(z) = (√(1+||z||²), z)
Q_i = L(z(X_i)); K0_rc = L(z(P0_rc))
delta_i = [d(Q_i,K0_r−)² − d(Q_i,K0_r+)²] / (2 tau_comp²)
r_i = sigmoid(b_i + delta_i)
```

比较对象只有本区域的两个类中心。接近变化中心时delta为正，接近背景中心时为负；两类中心重合时delta为零，r=p。初始锚点保留高置信位置的较高占比，最终一阶分配保留较软的混合内容。

```text
u_i+ = r_i; u_i− = 1−r_i
w1_ic = u_ic / Σ(k∈region(i)) u_kc
P1_rc = Σ(i∈r) w1_ic X_i
support_rc = Σ(i∈r) w1_ic m_ic
K_rc = L(W_qk P1_rc / √32)
V_rc = W_v P1_rc
```

若数值饱和导致某类别总质量为零，其汇聚内容为零；记录空原型比例，支持度使用1e−6下限。正常化、P0、几何分数及P1全程保留梯度。直接使用guide logits，避免sigmoid再logit的端点数值误差。

`tau_comp=softplus(raw_compression_temperature)`，默认初始化0.5；初始化值可由训练参数指定并写入manifest。它与全局检索温度独立。网格8/4/4/4对应128/32/32/32个KV。每像素的压缩距离仅两个候选，无邻域展开或scatter。

## 度量与内容写回

双曲空间曲率−1，使用直接Lorentz提升及稳定平方测地距离。内在半径为`asinh(||W_qk X/√32||)`。K和V来自同一P1，不使用密集流形点质心。匹配欧式在压缩与检索两处都改用空间坐标平方欧氏距离，其余结构和训练保持一致。

```text
logits_ij = −d(Q_i,K_j)²/(2 tau_retrieval²)
            + scale_bias[source(j)] − log(M_source(j)) + log(support_j)
A_i = softmax_j(logits_ij)
C_i = Σ_j A_ij V_j; V_query_i = W_v X_i
Δ_i = C_i − mean_spatial(C)
p_retrieved_i = Σ_j A_ij sigmoid(guide(P1_j))
g_i = sigmoid(W_gate[V_query_i, Δ_i, p_i, p_retrieved_i])
T'_i = T_i + γ_i W_out(g_i Δ_i) + local_i
```

检索内容在每个样本、每个尺度内减去空间均值，均值参与反传；相同注意力在所有位置产生的全局修正为零。距离控制相对检索；独立门控学习采用量。门控随空间位置变化，因此最终更新不要求空间均值为零。S1/S2保留`PW1(DW3(X−Avg3(X)))`；gamma零初始化，投影正常初始化。

## 训练、诊断与解释

双曲/匹配欧式使用`L_seg+0.05BCE(raw_logits,y_area)+0.05BCE(posterior_logits,y_area)`。CTRL使用`L_seg+0.1BCE(raw_logits,y_area)`；总辅助预算相同，H/E目标同构，CTRL保留原guide目标。后验logits直接使用`guide_logits+delta`，按实际区域顺序还原B1HW；标签只进入损失。四级主损失为BCEWithLogits＋Dice。每epoch在test选模，val用于分析。

记录delta、r−p、熵、两端饱和率、每类有效像素数、空原型、原型改变量以及最终残差。实际GT占比与纠错统计使用最终w1，在图像前向完成后计算。模型诊断不读取标签。

该式是guide logit与几何类分数的可学习组合；不将r解释为已经校准的真实后验概率。P0错误或最终饱和可能导致误改，须按GT变化、背景和混合位置检查。半径及分配差异不能单独证明任务收益。H/E比较两处度量的联合贡献，CTRL比较完整模块。几何映射的冻结干预会同时改变压缩与检索，分析需说明作用范围。
