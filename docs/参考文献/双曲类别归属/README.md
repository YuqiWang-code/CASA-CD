# 双曲类别归属：原型距离评分依据

核对日期：2026-10-31。对应 [Run10 区域几何归属方案](../../temporary/CASA_Pyramid_Run10_区域几何归属方案.md)。

| 本地全文 | 正式出处 | 官方来源 | 本轮用途 |
|---|---|---|---|
| [Prototypical Networks for Few-shot Learning](Prototypical-Networks(NIPS2017).pdf) | Jake Snell、Kevin Swersky、Richard Zemel；NIPS 2017 | [会议全文](https://papers.nips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf) / [作者实现](https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py) | 用到类别原型的负距离形成 softmax 类分数；两类情形可写为距离差的 sigmoid。 |
| [Hyperbolic Image Embeddings](Hyperbolic-Image-Embeddings(CVPR2020,arXiv版).pdf) | Valentin Khrulkov、Leyla Mirvakhabova、Evgeniya Ustinova、Ivan Oseledets、Victor Lempitsky；CVPR 2020，6418–6428；本地为作者 v2 | [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Khrulkov_Hyperbolic_Image_Embeddings_CVPR_2020_paper.html) / [作者全文](https://arxiv.org/abs/1904.02239v2) / [官方实现](https://github.com/leymir/hyperbolic-image-embeddings) | 双曲原型分类的视觉先例；用于核对几何距离作为类别关系分数的合法性。 |
| [Rethinking Semantic Segmentation: A Prototype View，复用已有全文](../Run7与MoE_Run1_机制诊断/ProtoSeg(CVPR2022,arXiv版).pdf) | Tianfei Zhou、Wenguan Wang、Ender Konukoglu、Luc Van Gool；CVPR 2022，2582–2593 | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Rethinking_Semantic_Segmentation_A_Prototype_View_CVPR_2022_paper.html) / [官方实现](https://github.com/tfzhou/ProtoSeg) | 像素与类原型关系用于分割；同类多个代表与单均值表达能力的区别。 |

本轮采用距离类分数的基础形式，结合本项目变化 guide 修正压缩归属；代码按本项目公式实现，不引入上述训练框架。P0 由平方 guide 权重构造，P1 使用修正后的一阶软归属，K/V 从同一 P1 构造。

## 适用边界

ProtoNet 的原型来自有标注支持集，CASA 的局部原型来自本图预测 guide；Hyperbolic Image Embeddings 使用自身的双曲平均方式，CASA 使用共同内容池化后直接 Lorentz 提升；ProtoSeg 的类别原型机制与本图局部二类压缩不同。三者支持距离类别评分的先例，不直接证明 CASA 的归属修正有效。

guide 与距离来自同一特征和同一图像，分数相加不构成独立证据的 Bayes 校准。双曲半径不自动代表可靠性，分类关系的解释及精度收益需要本项目的配对对照与冻结干预。
