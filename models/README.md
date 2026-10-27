# CASA-Pyramid

当前工作源码对应Run5，协议`casa_pyramid_40k_selective_selfref_v5`。共享MobileNetV2原生24/32/96/320通道→TAR至64通道→选择性跨尺度上下文→DCR→共享预测头。保持完整Q与压缩KV；Run4实际源码保存在其归档与冻结TXT中。

## 文件组织

| 文件 | 职责 |
|---|---|
| `model/casa_net.py` | 共享骨干、TAR、context、DCR与预测头 |
| `model/layers/pyramid_context.py` | 平方概率压缩、支持度、自参照注意力、欧式内容与细层局部残差 |
| `model/layers/euclidean_context.py` | 完整欧氏Q、池化后K投影、平方欧氏距离 |
| `model/layers/hyperbolic_context.py` | 直接Lorentz映射、质心K、平方测地距离 |
| `model/layers/lorentz_ops.py` | 曲率−1几何与同点平方距离解析极限 |
| `train_CASA.py` | 固定训练、test选模、损失、诊断与部署检查 |
| `profile_CASA.py` | 同配置batch32训练显存与步耗时 |

## 内容压缩

四层TAR共用通道LayerNorm：`X=LN(T)`；guide预测p。四尺度分辨率64²/32²/16²/8²，网格8/4/4/4，每区域变化和背景各一个原型，共128/32/32/32=224KV。完整Q为5440个位置、4头×16维；共享Q/K无bias投影，除以√16。

```text
m = p 或 1−p
w_i = (m_i+eps)² / Σ_l(m_l+eps)²
P = Σ_i w_i X_i
r = max(Σ_i w_i m_i², eps), eps=1e−6
V = W_v P
```

固定概率平方降低区域内大量低分背景的累计混入，不增加特征激活函数。r是二阶类别支持度；若p整区相同，两原型总支持为p²+(1−p)²，模糊p=.5时低于明确类别。分配和支持度保留梯度；LN在池化前，无bias线性投影可与同权池化交换。支持度仍依赖guide，不能当作独立正确性证据。

## 普通与双曲检索

普通：`Q=W_qk X/√16`，`K=W_qk P/√16`，距离为平方欧氏距离。双曲：先将密集投影z提升为`L(z)=(sqrt(1+||z||²),z)`，再按w构造Lorentz归一化质心K；半径`asinh(||z||)`可变且无硬裁剪。检索使用平方测地距离，近同点用同一公式的解析级数保留有限导数。V保持欧式。Lorentz归一化质心不称一般精确Fréchet均值。

```text
l_ij = −d(Q_i,K_j)²/(2 tau_head²) + log(r_j)
       + scale_bias[head,query_scale,source_scale] − log(M_source)
A_i = softmax([l_i1,...,l_i224,0])
g_i = Σ_j=1..224 A_ij
U_i = W_out(Σ_j A_ij V_j − g_i W_v X_i)
```

末项为query自身：K=Q、V=W_vX，距离0、支持1，因此logit固定0。自身和压缩内容竞争，g_i由图像、位置与head共同决定。回写等价于联合读出减自身V；减项须乘g_i。所有内容相同则修正为零，候选远或支持低时可减少改写。不是另一个共享全局KV，跨尺度原型仍为224个。

## 残差与细节

```text
T'_s = T_s + gamma_context[s] ⊙ U_s
S1/S2另加 gamma_detail[s] ⊙ PW1(DW3(X_s−Avg3(X_s)))
```

平均滤波及DW3使用replicate padding。gamma_context为4×64，gamma_detail为2×64，均零初始化；投影与卷积正常初始化。CTRL保留相同LN/guide监督，TAR直接进入DCR。TAR与DCR保持Run4结构重参数化及SiLU；三组共同路径初始化一致。

两模块训练2.064537M、部署1.953222M、context17,605参数；CTRL为2.047125M、1.935810M、193参数。Run5机制不增加可学习参数；运行效率单独实测。

## 训练、诊断与检查

Run4共同协议保持：RGB256、batch32、seed2333、Adam全网lr5e−4，betas(.9,.99)、eps1e−8，200warmup后poly^.9，40K整epoch取整。context Conv/Linear权重解耦WD1e−4，其余context WD0；encoder/body保留耦合L2 WD1e−4。四级BCEWithLogits+Dice加.2倍四级guide面积标签BCE，无边界辅助损失。

RGB ImageNet归一化，标签gray≥128；同步裁剪、水平/垂直翻转、90°旋转与时相交换，评估无随机增强。每epoch test/F1选模并复测，test参与选模，非独立盲测。完整设置见[根README](../README.md)。

`forward`返回四级概率，`return_aux=True`返回四级logits及guide_logits；`predict`输出一级概率。部署在eval折叠TAR/DCR，strict_fp32_fold_v1检查首级差≤2e−4并记录全部四级和二值差。

日志保留残差、权重、梯度与几何诊断，并新增上下文采用量g、self质量、检索V和自身V的幅度/余弦、读出空间差异。entropy、scale_mass与query_attention_tv均基于**给定采用上下文后的条件分布**；实际采用比例另看context_mass。首batch诊断不能代替全split任务指标。

测试覆盖稀疏/模糊区域、压缩投影交换、零初始化、常量保持、自参照拒绝、跨尺度与全部参数梯度、Lorentz数值及部署折叠。LEVIR每模块6epoch、SYSU每模块4epoch真实train/val短训用于检查健康，正式从预训练骨干全新初始化。

[Run5逐项核实与设计](../docs/temporary/CASA-Pyramid_Run5_design.md) · [来源](SOURCES.md) · [脚本](../train_scripts/CASA-Pyramid/Run5/README.md)
