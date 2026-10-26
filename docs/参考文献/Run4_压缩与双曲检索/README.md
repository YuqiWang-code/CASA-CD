# CASA-Pyramid Run4 定向调研

检索日期：2026-10-26。围绕Run3的建筑边界收益不足、强上下文回写、双曲输入裁剪及同尺度检索集中，核验五篇官方公开全文。保留完整Q、压缩KV；参考文献支持机制选择，精度收益由CASA同轮对照决定。

## 已归档文献

| 本地PDF | 正式信息与原始来源 | 使用范围 |
|---|---|---|
| [Hypformer](Hypformer(KDD2024,arXiv版).pdf) | Menglin Yang等，*Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space*，KDD 2024，[DOI](https://doi.org/10.1145/3637528.3672039)、[作者全文](https://arxiv.org/abs/2407.01290)、[官方代码](https://github.com/Graph-and-Geometric-Learning/hyperbolic-transformer) | Eq.9/Eq.13直接构造空间与时间坐标，减少Exp/Log往返；PDF为2025-08修订版 |
| [Robust Hyperbolic Learning](Robust-Hyperbolic-Learning(NeurIPS2025).pdf) | Ahmad Bdeir等，*Robust Hyperbolic Learning with Curvature-Aware Optimization*，NeurIPS 2025主会，[官方页](https://papers.nips.cc/paper_files/paper/2025/hash/75817c38dd91ccef8add921b2eb4284a-Abstract-Conference.html) | 约束、曲率与优化必须一致；裁剪和归一化影响可表达范围。其Riemannian AdamW针对直接流形参数，不可直接解释CASA欧氏投影参数的WD |
| [FreqFusion](FreqFusion(TPAMI2024,arXiv版).pdf) | Linwei Chen等，*Frequency-Aware Feature Fusion for Dense Image Prediction*，IEEE TPAMI 46(12):10763–10780，2024，[DOI](https://doi.org/10.1109/TPAMI.2024.3449959)、[作者全文](https://arxiv.org/abs/2408.12879)、[官方代码](https://github.com/Linwei-Chen/FreqFusion) | 区分内部特征不一致与边界高频缺失；细节恢复必须同时考虑区域一致性 |
| [Change3D](Change3D(CVPR2025).pdf) | Duowang Zhu等，*Change3D: Revisiting Change Detection and Captioning from A Video Modeling Perspective*，CVPR 2025:24011–24022，[CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Change3D_Revisiting_Change_Detection_and_Captioning_from_A_Video_Modeling_CVPR_2025_paper.html)、[项目页](https://zhuduowang.github.io/Change3D/) | 变化检测需任务相关的交互表征；参数轻量与变化表达不必对立。其视频编码器不是CASA本轮的替换目标 |
| [Numerical Stability](Numerical-Stability-Hyperbolic(ICML2023).pdf) | Gal Mishne等，*The Numerical Stability of Hyperbolic Representation Learning*，ICML 2023，PMLR202:24925–24949，[PMLR](https://proceedings.mlr.press/v202/mishne23a.html) | 基础理论：Poincaré与Lorentz均受浮点精度约束，Lorentz优化性质较好不等于无限可表达范围 |

既有[SAT](../baseline/SAT(CVPR2026).pdf)继续提供完整Q、压缩KV的任务结构依据；其状态为CVPR2026 Findings。既有[Deep Change Monitoring](../流形空间spd黎曼几何/Deep%20Change%20Monitoring（CVPR2025）.pdf)提供细粒度树木变化的双曲表示证据，不能据此外推所有建筑或地表变化都会获得相同收益。

## 对本轮设计的结论

**共同坐标再检索。** Run3原始A/B各尺度独立投影后直接跨尺度匹配，统计和语义差异可能放大本尺度偏好。统一TAR通道后共享Q/K坐标、输入归一化及尺度数量先验，是减少这类自由度的最小改动。它不保证四尺度都会被使用；应测量注意力尺度分布与相对CTRL的实际收益，不强迫均分注意力。

**普通模块侧重压缩与回写。** 变化/背景两原型保留少数变化与区域上下文；池化先于线性K/V投影，在相同分配权重下具有线性等价性。将检索内容写成相对当前query内容的修正，可抵消常量广播；细尺度另保留局部差异卷积，避免压缩上下文承担全部边界恢复。前者是结构假设，后者由FreqFusion关于区域一致性与边界细节的区分提供依据；不是对FreqFusion完整算子的复现。

**双曲模块采用直接Lorentz坐标与单个几何距离。** 设学习到的空间坐标为`z`，映射为`x=(sqrt(1+||z||²),z)`，曲率−1时有`<x,x>_L=−1`、`r=asinh(||z||)`。它保留可变半径，取消Exp前的固定阈值裁剪，同时减少中点、Log和并行移动的计算。Q/K在共同坐标中由双曲距离检索，V保持欧氏内容。投影输出不逐token单位化，以免半径再次固定。K使用流形质心时，应明确是Lorentz归一化质心而非精确多点Fréchet均值。

**共同训练设置单独归因。** Run3的CTRL也在三库下降，不能把全部退步归因于两个模块。恢复历史较好共同协议并保持本轮三组一致，可检验模块净收益；跨轮改变协议后的绝对变化仍不等于模块增益。

## 需要实际验证的不足

- 共享坐标是否改善跨尺度信息利用，须结合验证特征和检索差异；注意力更分散本身不代表更好。
- query相对回写可能产生平滑，必须同时看建筑边界、内部完整性及残差幅度。
- 直接Lorentz映射减少硬裁剪，不消除全部数值风险；半径、距离、同点梯度和任务梯度需要实测。
- 两模块的贡献须在相同训练设置下与CTRL比较；有限参数增量可以接受，但延迟、显存与F1应同时报告。

官方核心代码归档于[others/Hypformer](../../../others/Hypformer/README.md)，保留MIT许可证、版本及原始路径。Run3证据见[结果分析](../../experiments/CASA-Pyramid/Run3/analysis.md)。
