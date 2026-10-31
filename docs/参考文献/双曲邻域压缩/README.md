# 双曲邻域压缩：基础依据

核对日期：2026-10-31。对应 [CASA-Pyramid Run10 九邻域候选](../../temporary/过去的想法/CASA_Pyramid_Run10_邻域几何压缩方案.md)，记录其八项短训的基础依据。区域内类别归属的文献另见[双曲类别归属](../双曲类别归属/README.md)。

| 本地全文 | 正式出处 | 官方来源 | 本轮采用的依据 |
|---|---|---|---|
| [Superpixel Sampling Networks](Superpixel-Sampling-Networks(ECCV2018).pdf) | Varun Jampani、Deqing Sun、Ming-Yu Liu、Ming-Hsuan Yang、Jan Kautz；ECCV 2018，352–368 | [作者项目页](https://varunjampani.github.io/ssn/) / [作者 PDF](https://varunjampani.github.io/papers/jampani18_SSN.pdf) / [官方实现](https://github.com/NVlabs/ssn_superpixels) | 局部邻域软关联与归一化内容聚合能够参与端到端任务学习；将每像素候选限制在邻近九个代表，控制关联开销。 |
| [Vision Transformer with Super Token Sampling](../Run5_原型选择与自参照检索/Super-Token-Sampling(CVPR2023,arXiv版).pdf) | Huaibo Huang、Xiaoqiang Zhou、Jie Cao、Ran He、Tieniu Tan；CVPR 2023；本地为作者开放版本 | [作者论文](https://arxiv.org/abs/2211.11167) / [官方核心文件](https://github.com/hhb072/STViT/blob/main/models/stvit.py) | 视觉 token 在九邻域代表之间进行关联，再按关联聚合内容；复用已有论文和 [官方核心代码](https://github.com/hhb072/STViT/blob/main/models/stvit.py)。 |

Run10 在变化/背景类别内独立进行九邻域竞争，保持像素类别质量守恒；双曲距离同时参与内容压缩与上下文检索，K/V 共享压缩内容。该组合采用本项目独立实现。SSN 的 Caffe 训练框架不引入项目；STViT 现有代码仅用于研究阅读，许可状态见目录说明，CASA 训练路径不导入。

## 适用边界

两篇基础论文支撑稀疏邻域关联与内容聚合，不构成双曲 RSCD 的精度或 SOTA 保证。Run10 保留完整 Q 直接检索压缩 KV，与 STViT 的超 token 自注意力和回映射路径不同。压缩分配保持可微，任务梯度同时传入几何关系与被聚合内容。
