# CASA-Pyramid Run10：几何归属与空间检索

日期：2026-10-31。工作协议：`casa_pyramid_40k_supervisedposterior_v10`。当前为六项配对短训审阅，正式结果0/12；指标表保留263条正式记录。

## 设计目标与结构

几何承担两个具体任务：在固定区域内修正变化/背景内容的压缩归属，以及为不同空间位置检索不同的跨尺度内容。区域后验接受面积标签监督；检索内容按空间中心化后写回，保留TAR和细层局部残差。

完整Q、网格(8,4,4,4)、各尺度128/32/32/32个KV，共224个；H/E均为1头32维，压缩温度初始0.5且可学习。H/E训练参数2.058410M、部署1.947095M，context为11,478参数。普通QKV历史版保持Run5，不参与本轮H/E几何配对。

```text
四尺度TAR T_s → LN → X_s ────────────────────→ 完整Q
                      │                         │
                      ├→ guide logit g → p      │
                      │                  │      │
                      └→ 平方guide池化 → 两类P0 → 两类K0
                                               │
                                  变化/背景距离差δ
                                               │
                                  r = sigmoid(g+δ)
                                               │
                       r与1−r一阶池化X → 两类P1
                                               │
                                      同源K / 欧式V
                                               │
                                完整Q检索224KV → AV
                                               │
                           C = AV − mean_hw(AV)
                                               │
                             内容门控 → 输出投影
                                               │
                         TAR残差 + 局部残差 → DCR

训练任务约束：面积标签监督原g与修正后的g+δ
```

## 公式与数据流

### 高置信初始类别锚点

令 \(X=\operatorname{LN}(T)\)、\(g=\operatorname{guide}(X)\)、\(p=\sigma(g)\)，\(m_{i,+}=p_i\)、\(m_{i,-}=1-p_i\)。每个固定区域R内：

\[
w^0_{i,c}=\frac{(m_{i,c}+\varepsilon)^2}{\sum_{j\in R}(m_{j,c}+\varepsilon)^2},\qquad
P^0_c=\sum_{i\in R}w^0_{i,c}X_i,\qquad \varepsilon=10^{-6}.
\]

P0以较高guide分数的内容作为初始参考。构造例中，真实变化占1%、其p=0.5、背景p=0.02时，正类一阶池化的真变化权重约20.16%，平方池化约86.33%。这说明平方权重能够减轻该稀疏场景的初始背景混入，实际锚点质量仍取决于guide。

Q/K共享64→32投影，直接提升到曲率−1的Lorentz空间：

\[
L(z)=(\sqrt{1+\|z\|^2},z),\qquad
Q_i=L(W_{qk}X_i/\sqrt{32}),\qquad K^0_c=L(W_{qk}P^0_c/\sqrt{32}).
\]

### 区域几何归属

\[
\delta_i=\frac{d(Q_i,K^0_-)^2-d(Q_i,K^0_+)^2}{2\tau_c^2},\qquad
r_i=\sigma(g_i+\delta_i).
\]

距离更接近变化原型时，δ为正，提高变化归属；接近背景时相反。两类中心相同或两类距离相等时恢复r=p。实现直接保留g+δ的原始logit，r与1−r互补，每个像素分配给两个类别。

H使用平方测地距离；E使用同一投影的空间坐标平方欧氏距离。几何原型与guide来自同一X，r是经过任务监督的压缩归属，不作为独立证据融合概率或最终分割输出。

### 同源K/V与支持度

令 \(u_{i,+}=r_i\)、\(u_{i,-}=1-r_i\)：

\[
w^1_{i,c}=\frac{u_{i,c}}{\sum_{j\in R}u_{j,c}},\qquad
P^1_c=\sum_{i\in R}w^1_{i,c}X_i,
\]

\[
s_c=\sum_iw^1_{i,c}m_{i,c},\qquad
K_c=L(W_{qk}P^1_c/\sqrt{32}),\qquad V_c=W_vP^1_c.
\]

P1使用一阶软归属，K和V共享同一内容；support以相同w1聚合原guide，保持内容与支持来源一致。全局注意力保留负距离平方、log支持度、可学习尺度偏置和源尺度token数量校正。H/E在归属和检索两处切换距离，其余结构一致。

### 空间检索写回

对每个样本、每个query尺度分别计算：

\[
R_i=(AV)_i,\qquad \bar R=\frac1N\sum_iR_i,\qquad C_i=R_i-\bar R=(A_i-\bar A)V,
\]

\[
a_i=\sigma\!\left(W_g[V_i,C_i,p_i,\hat p_i]+b_g\right),\qquad
T'_i=T_i+\gamma_s\odot W_o(a_iC_i)+\text{local}_i.
\]

\(\hat p_i\)为同一注意力检索得到的原型guide值；Vquery保留在门控输入中。Wout无bias，γ逐通道零初始化，投影正常初始化。均值参与反传，细两层局部高通保持。

若所有query采用相同注意力，C为0，全局写回也为0。空间中心化不等同于局部高通；门控随位置变化，最终残差的空间均值不一定为0。该结构要求写回依赖空间相关的检索内容，但是否有益仍由最终预测判断。

## 监督、训练与部署

\[
\mathcal L_{H/E}=\mathcal L_{seg}+0.05\mathcal L_{raw}+0.05\mathcal L_{posterior},\qquad
\mathcal L_{CTRL}=\mathcal L_{seg}+0.1\mathcal L_{raw}.
\]

主损失为四级BCEWithLogits与sigmoid Dice之和。raw和posterior分别以g、g+δ计算四尺度面积标签BCE；检索BCE系数为0。后验logit按区域逆排列恢复空间位置后监督，GT只进入loss和前向后的分析。

后验监督是归属的任务约束，直接向QK及压缩温度传递梯度。H/E与CTRL的辅助系数总预算相同，监督目标和梯度路径不同；不将辅助损失本身作为主要创新。δ变大或归属变尖锐不代表监督目标更好。

共同配置：RGB256×256、batch32、seed/context seed2333、MobileNetV2预训练全新初始化、Adam全网lr5e-4、40K整epoch取整，增强与优化器设置见[根README](../../README.md)。正式计划LEVIR→SYSU→WHU→CDD，每库H/E/CTRL共12项，仅物理GPU1，按实测显存并行。正式每epoch在test选模，结果标注test参与选模；设计、短训和干预使用train/val，探针权重不进入正式训练。

## 六项配对短训

LEVIR、SYSU各运行三项，每项4个完整epoch；压缩温度初始均为0.5。

| 组别 | 读出 | 辅助监督 | 对应比较 |
|---|---|---|---|
| H_CENTER_RAW | 空间中心化 | 0.1 raw BCE | 与H05参照比较读出作用 |
| H_CENTER_SUP | 空间中心化 | 0.05 raw＋0.05 posterior BCE | 与H_CENTER_RAW比较监督作用 |
| E_CENTER_SUP | 空间中心化 | 0.05 raw＋0.05 posterior BCE | 与H_CENTER_SUP比较完整几何路径 |

配对参照为同seed、同预算的H05/E05与CTRL。原始记录及独立raw/posterior BCE见[短训比较](../experiments/CASA-Pyramid/Run10/selection/centered/probe_comparison.md)。短训不入正式指标表，正式启动待六项完整结果与冻结内容审阅。

### 已有LEVIR证据

数值为百分比，末轮为第4轮，末三轮均值为epoch2–4的完整val F1。

| 组别 | 末轮F1 | 末轮1px边界F1 | 末三轮平均F1 |
|---|---:|---:|---:|
| H05参照 | 86.6635 | 69.9260 | 84.8236 |
| H_CENTER_RAW | 85.6358 | 68.6807 | 85.3221 |
| H_CENTER_SUP | 86.5958 | 70.4766 | 85.2056 |

SUP相对RAW改善末轮F1与边界F1，末三轮平均F1略低；相对H05，末轮F1基本接近、边界较高、末三轮均值较高。当前证据支持后验监督改善该候选的部分表现，尚不能据此判断长期或跨库优势。

冻结H_CENTER_SUP权重在固定256张LEVIR val上的F1为：完整87.82481%、关闭全局87.7904%、打乱query与注意力行对应87.72356%。它与上表的完整val范围不同。全局分支的FP/FN存在取舍，F1略升不能等同于总像素错误数下降。

### 有效性判断

- 同时观察原p与修正r的软标签误差、纠正与新增误归属；不能仅用r变尖或半径变化判定有效。
- 比较P0/P1的变化内容纯度与变化质量保留；压缩质量和最终采用效果分别分析。
- 比较同预算完整val的F1、Recall、边界指标和损失；冻结干预检查最终纠错及其错误类型。
- 中心化后mean_attention与global_off相同是结构性质，不作为新增几何证据。H/E训练对照以及保留注意力分布的query对应打乱用于进一步辨别。

## 文献依据

- [ProtoNet，NIPS2017](https://papers.nips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)：负原型距离的类别softmax形式；两类分数差可写成sigmoid。
- [Hyperbolic Image Embeddings，CVPR2020](https://arxiv.org/abs/1904.02239v2)：双曲原型用于视觉类别关系评分的先例。
- [ProtoSeg，CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Rethinking_Semantic_Segmentation_A_Prototype_View_CVPR_2022_paper.html)：像素与类别原型关系用于分割，并讨论同类多代表。

本地全文与官方实现见[文献说明](../参考文献/双曲类别归属/README.md)和[文献索引](../参考文献/文献索引.md)。这些文献提供机制依据，不证明当前模块具备新颖性、双曲优越性或必然精度收益。历史候选源码、短训及干预分别保留在`selection/neighborhood`、`selection/classposterior`与`selection/centered`。

## 局限

- guide与初始原型同源，错误可能相互强化；后验监督不能保证泛化正确。
- 原p很低的真变化即使得到较大r，其support仍可能偏低；后验改善未必转化为最终检索收益。
- 每区域每类仍只有一个代表，不能完整保留任意多种外观。半径不预设层级、置信或变化强度含义。
- 空间中心化可能削弱大面积同类变化的共享上下文。γ零初始化时需确认实际任务梯度与早期写回，不能只看辅助分支梯度。
- raw与posterior拟合同一软标签，δ趋近0也可能降低损失；这不证明几何关系得到更充分利用。
- 当前四轮、单seed和部分冻结子集证据不支持稳定普适提升。参数轻量不等于推理更快，效率以本实现的实测记录为准。
