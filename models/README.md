# CASA-Pyramid

共享MobileNetV2原生四尺度（24/32/96/320通道）→ TAR时相融合至64通道 → 配对跨尺度上下文 → DCR → 共享预测头。context同时接收四层A/B编码特征和TAR输出；骨干结束于320通道，不使用HFEA。训练四级深监督，`predict()`输出最终一级变化概率。

当前工作源码对应Run3，协议`casa_pyramid_40k_pairbanks_contrastwd0_v3`，脚本位于`train_scripts/CASA-Pyramid/Run3/`。设计依据见[Run3设计文档](../docs/temporary/CASA-Pyramid_Run3_design.md)。各历史冻结快照保留对应轮次的实现。

## 共同输入与压缩规模

同层A/B共用输入通道LayerNorm和`C_i→32`投影。两版保留全部5440个query位置，采用四个attention头及224个跨尺度上下文token。guide以预测变化概率构造原型，推理不需要标签。

| 尺度 | 分辨率 | 区域网格 | 区域内位置 | 普通版：均值/对比 | 双曲版：整体/变化 |
|---|---|---|---:|---:|---:|
| S1 | 64×64 | 8×8 | 64 | 64＋64 | 64＋64 |
| S2 | 32×32 | 4×4 | 64 | 16＋16 | 16＋16 |
| S3 | 16×16 | 4×4 | 16 | 16＋16 | 16＋16 |
| S4 | 8×8 | 4×4 | 4 | 16＋16 | 16＋16 |

各来源尺度加可学习偏置及`-log(该尺度token数)`，保持相同内容时的尺度先验一致。上下文系数`gamma_context`四个、细节系数`gamma_detail`两个均零初始化；输出投影正常初始化。

## 普通模块：区域均值与变化对比双库

`model/layers/paired_euclidean_context.py`实现独立模块。A/B投影后绝对差分为`D`，融合表示为`X=LN(T+W_D D)`；`W_D:32→64`跨尺度共享。guide输出变化概率p。

每个区域构造均值`P0=mean(X)`，以及带一个位置伪计数的类别均值：

```text
P+ = (ΣpX + P0) / (Σp + 1)
P− = (Σ(1−p)X + P0) / (Σ(1−p) + 1)
PΔ = P+ − P−
```

常量guide产生零对比原型；缺少某类支持时，其均值向区域均值收缩。固定伪计数1属于压缩设计，稀疏变化仍可能被平滑，因此细层另保留未压缩D。

完整Q为4×16维；均值库与对比库分别具有112个K/V，各自采用独立K投影和一次scaled-dot-product softmax。每库V为4×8维，两个32通道读出拼接后投影至64通道。Q/K/V均无bias，K/V先池化再投影；对比V不添加带偏置的归一化。两库分别归一化，没有类别支持度`log(r)`先验。

S1/S2计算固定3×3二项式滤波的高通`H=D−blur(D)`，reflect padding。共享128→1边界头输入`[X,D,H]`，得到b；回写：

```text
S1/S2: T' = T + gamma_context·O + gamma_detail·W_D(D + bH)
S3/S4: T' = T + gamma_context·O
```

边界门只调节高通部分，完整D和区域路径始终保留。

## 双曲模块：成对中点与差异联合检索

`model/layers/paired_hyperbolic_context.py`实现独立模块。A/B投影各拆为4×8维、除以√8，经曲率−1、切空间范数上限2.3的Exp映射并转换为Lorentz坐标。`lorentz_pair_ops.py`计算：

```text
m = 两时相的精确等权测地中点
v = PT(m→原点, Log_m(B)) 的8维空间坐标
u = |v|
X = LN(T + W_D u)
```

中点与有向半位移由闭式公式获得，不逐像素迭代。对固定TAR输入交换A/B，m不变、v反号、u不变；整网TAR包含有序融合，不能据此声称全网严格交换不变。

每区域使用整体均匀权重与`(p+1e-6)/Σ(p+1e-6)`变化权重，分别压缩成两个token。每个token的中点K、幅度K和欧式V使用相同权重。区域中点K使用归一化Lorentz质心；它与精确两点Fréchet中点是不同操作，不称任意多点Fréchet均值。恒定p时两类token重合，变化分支不加入普通版的单位伪计数。

每个完整query联合检索中点与幅度：

```text
logit = −d_H(m, K_mid)/(2·tau_mid) − ||u−K_diff||/tau_diff
        + scale_bias − log(source_token_count)
```

两个每头正温度初值均为0.2，使用一个224-token softmax。V为4×16维欧式内容，直接`A @ V`聚合。该评分组合双曲中点关系和共同切空间幅度关系。

S1/S2采用`H=|v−blur(v)|`，边界头输入`[X,u,H]`，细节残差为`gamma_detail·W_D(u+bH)`；TAR与上下文残差仍保留。u及高通绝对值不保留完整方向信息。

## 训练与推理

- `forward(a,b)`返回四级概率；`return_aux=True`返回四级分割logits和`{'guide_logits':四层, 'boundary_logits':细两层}`。`predict(a,b)`只产生最终一级全分辨率变化概率；TAR/DCR在eval下折叠。
- 主损失为四级BCEWithLogits＋Dice求和；guide为面积平均标签BCE，权重0.2；边界为BCE＋Dice，权重0.05。边界标签为原始mask的3×3膨胀减腐蚀，外侧replicate，再以max pooling得到细两层边界占据标签。
- CTRL保留共享LayerNorm、guide及65参数边界头，TAR直接进入DCR。三组使用同样的辅助目标与损失系数；CTRL边界输入为64通道TAR归一化特征，模块版为128通道配对表示。
- 256×256、batch32、seed2333、40K取整到完整epoch。Adam betas(0.9,0.99)、eps1e-8；新层基础LR5e-4，预训练骨干LR5e-5。全局前200步warmup、poly指数0.9；每步保留骨干`lr_scale=0.1`。
- 普通模块的`context.k_contrast.weight`与`context.v_contrast.weight`采用WD0；context内其余Conv2d/Linear权重WD1e-4，LayerNorm、bias、温度、gamma与尺度偏置WD0；网络其余参数WD1e-4。FP32训练，AMP关闭。
- 每epoch用`test.txt`的最高F1选模（相同更新），结束后复测best；结果明确为test参与选模，非独立盲测。方案诊断优先使用train/val材料。

## 数据预处理与增强

RGB图像双线性缩放至256×256，标签最近邻、灰度≥128二值化。像素除255，ImageNet RGB均值`[0.485,0.456,0.406]`、标准差`[0.229,0.224,0.225]`归一化。

训练开启：0.5概率裁剪四边各0～8像素后恢复尺寸；水平/垂直各0.5概率翻转；等概率0/90/180/270°旋转；0.5概率交换A/B。空间变换同步，不添加独立光度扰动。评价仅缩放与归一化；训练shuffle并drop_last，评价保留全部样本。

## 参数与诊断

| 模式 | 训练参数 | 部署参数 | context参数 |
|---|---:|---:|---:|
| CTRL | 2,047,190 | 1,935,875 | 258 |
| 双曲中点—差异QKV | 2,073,620 | 1,962,305 | 26,688 |
| 普通均值—对比QKV | 2,085,964 | 1,974,649 | 39,032 |

参数按当前结构统计；训练峰值显存及步耗时由`profile_CASA.py`测量，折叠单输出延迟由`tools/benchmark_casa.py`测量，Run2效率数值不能代替Run3实测。

诊断分别记录context/local残差RMS、guide与边界输出、均值/对比库注意力、来源尺度质量和query差异。普通模块还记录两库读出内容RMS，以及`k_mean/k_contrast/v_mean/v_contrast`权重RMS，观察对比检索与内容是否保持有效幅度。双曲另记录中点/差异logit贡献、半径CV、裁剪比例、压缩前后差异幅度。每轮诊断取评价首个batch，结合任务梯度和全split边界/内部指标判断状态。
