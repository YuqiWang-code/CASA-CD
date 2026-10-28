# 双曲半径、代表点与监督区域内容

2026-10-28核对的基础原始工作。用于解释公式和监督压缩，不作为候选精度已提高的证据。

| 本地文件 | 正式信息 | 官方来源与范围 |
|---|---|---|
| [Lorentzian Distance Learning](Lorentzian-Distance-Learning(ICML2019).pdf) | Marc Law、Renjie Liao、Jake Snell、Richard Zemel；ICML2019，PMLR97:3672–3681 | [PMLR](https://proceedings.mlr.press/v97/law19a.html)。闭式质心对应平方Lorentzian距离，不等同一般平方测地距离Fréchet中心。 |
| [Object-Contextual Representations](Object-Contextual-Representations(ECCV2020).pdf) | Yuhui Yuan、Xilin Chen、Jingdong Wang；ECCV2020 | [ECVA原文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510171.pdf)。受分割监督的软区域表示和像素-区域内容聚合。 |

本轮未移植额外训练框架；沿用已有Lorentz数值函数。占比半径、同内容K/V及独立纠错门控的实现见[设计说明](../../../models/HYPERBOLIC_DESIGN.md)。
