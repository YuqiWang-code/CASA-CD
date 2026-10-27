# Run5：原型内容与检索采用

检索日期2026-10-27。优先检索2024–2026年官方论文、作者开放全文和代码；2023年的STViT作为压缩机制基础材料。研究问题是压缩内容是否保留稀疏变化，以及query能否避免强制采用不匹配的原型。

| 本地PDF | 正式出处与官方来源 | 与本项目相关的结论 | 本轮采用范围 |
|---|---|---|---|
| [TransNeXt](TransNeXt(CVPR2024,arXiv版).pdf) | Dai Shi，CVPR 2024，17773–17783；[CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_TransNeXt_Robust_Foveal_Visual_Perception_for_Vision_Transformers_CVPR_2024_paper.html)、[作者全文v3](https://arxiv.org/html/2311.17132v3)、[代码](https://github.com/DaiShiResearch/TransNeXt) | §3.1.1把未压缩局部候选和池化候选放入同一softmax，让细粒度与粗粒度内容竞争 | 采用query自身与压缩KV联合归一化；不增加滑窗KV、位置MLP、GLU或learnable tokens |
| [Visual Hierarchies](Visual-Hierarchies(ICCV2025,arXiv版).pdf) | Ziwei Wang等，Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval，ICCV 2025；[CVF正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Learning_Visual_Hierarchies_in_Hyperbolic_Space_for_Image_Retrieval_ICCV_2025_paper.pdf)、[作者全文v2](https://arxiv.org/html/2411.17490v2) | 通过明确的场景/对象/部件包含关系和非对称目标建立视觉层次；双曲空间本身不自动产生任务层次 | 用于界定解释边界。CASA不把四尺度等同语义层级，不强制增加没有标注依据的包含损失 |
| [Super Token Sampling](Super-Token-Sampling(CVPR2023,arXiv版).pdf) | Huaibo Huang等，Vision Transformer with Super Token Sampling，CVPR 2023；[作者全文v2](https://arxiv.org/html/2211.11167v2)、[官方代码](https://github.com/hhb072/STViT) | §3.2.1与算法1通过局部关联更新压缩代表，说明分配内容是token压缩的重要组成部分 | 用于分析固定区域软均值的局限；不移植迭代superpixel、额外token自注意力或其骨干 |

PDF为作者arXiv开放版本，正式引用按会议年份，v2/v3版本日期不替代发表年份。TransNeXt核心与许可证保存在[others/TransNeXt](../../../others/TransNeXt/README.md)。Visual Hierarchies未将未核实的第三方实现作为官方代码；本轮不需要其训练框架。

已复核的历史材料包括[Hypformer、数值稳定与FreqFusion](../Run4_压缩与双曲检索/README.md)、[Deep Change Monitoring](https://arxiv.org/abs/2503.00643)及[SAT官方来源](https://github.com/PhuTran1005/SAT)。它们分别支撑Lorentz参数化、浮点几何边界、局部细节与完整Q/压缩KV的背景，不能替代CASA同轮消融。

Run5的概率平方分配、二阶类别支持度与自参照相对回写是针对Run4实测混合问题的项目设计。文献只提供关联压缩和局部/池化竞争的先例，不被解释为已经证明本设计在四库提高F1。
