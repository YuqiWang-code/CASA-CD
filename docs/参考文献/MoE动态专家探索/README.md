# 动态异构专家与频率处理

核实日期2026-10-28。用途：CASA的`models_v1`探索线，四条候选路径为双曲QKV、普通QKV、原TAR、低高频处理。模型与训练状态以[探索说明](../../../models_v1/README.md)为准。

## 主要文献

| 文献 / 本地全文 | 出处与官方来源 | 本项目采用的依据与边界 |
|---|---|---|
| [From Sparse to Soft Mixtures of Experts](Soft-MoE(ICLR2024).pdf) | ICLR2024；[会议全文](https://proceedings.iclr.cc/paper_files/paper/2024/file/79fea214543ba263952ac3f4e5452b14-Paper-Conference.pdf) | Soft MoE通过dispatch/combine槽位混合工作。单纯四路全图softmax加权属于稠密软门控，不等于该论文算法。 |
| [Routers in Vision Mixture of Experts: An Empirical Study](Vision-MoE-Routers(TMLR2024,arXiv版).pdf) | TMLR2024；[正式OpenReview](https://openreview.net/forum?id=aHk3vctnf1)、[作者开放版](https://arxiv.org/abs/2401.15969) | 路由策略影响视觉MoE精度与效率；分类结果不直接证明异构RSCD专家互补。 |
| [Multi-Task Dense Prediction via Mixture of Low-Rank Experts](MLoRE(CVPR2024).pdf) | CVPR2024:27927–27937；[CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Multi-Task_Dense_Prediction_via_Mixture_of_Low-Rank_Experts_CVPR_2024_paper.html)、[官方代码](https://github.com/YuqiYang213/MLoRE) | 共享路径和低秩专家控制开销。其线性专家融合条件不适用于双曲距离、softmax及内容相关频率门控。 |
| [Wavelet Convolutions for Large Receptive Fields](WTConv(ECCV2024).pdf) | ECCV2024；[ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07137.pdf)、[官方代码](https://github.com/BGU-CS-VIL/WTConv) | 固定小波分解、子带卷积及重建的可实现依据；不预设高频只含真实变化。 |
| [Feature Spectrum Learning for Remote Sensing Change Detection](FeaSpect(CVPR2025).pdf) | CVPR2025:12647–12657；[CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zang_Feature_Spectrum_Learning_for_Remote_Sensing_Change_Detection_CVPR_2025_paper.html) | 在特征频谱中处理双时相风格差异与伪变化；频率专家须同时分析边界保留和干扰抑制。 |
| [SARE-CD: Structure–Appearance Routed Experts for Building Change Detection in Remote Sensing Imagery](SARE-CD(Remote-Sensing2026).pdf) | Remote Sensing18(20):3613，2026-10-20；[出版社](https://www.mdpi.com/2072-4292/18/20/3613)、[DOI](https://doi.org/10.3390/rs18203613) | 已有结构/外观/自适应异构专家与像素软路由；它全部执行三路。不能将“异构专家用于变化检测”作为本项目独创点，也不将LEVIR-CD+成绩直接比较于LEVIR。 |
| SCD-MoE: Adaptive Expert Routing for Efficient Remote Sensing Semantic Change Detection | TGRS64，2026-09-29；[DOI](https://doi.org/10.1109/TGRS.2026.3738239)、[官方代码](https://github.com/HuQ1an/SCD-MoE) | 已核实摘要与交互路由代码，未取得全文。借鉴差分/语义保持专家的任务划分及真实Top-k调用；该工作包含语义多任务目标。 |
| [FreqFusion](../Run4_压缩与双曲检索/FreqFusion(TPAMI2024,arXiv版).pdf) | IEEE TPAMI46(12)，2024；[DOI](https://doi.org/10.1109/TPAMI.2024.3449959) | 已有本地全文。区域一致性与边界细节应分别处理；本项目不复制完整融合框架。 |

核心参考代码为[WTConv](../../../others/WTConv/README.md)与[SCD-MoE](../../../others/SCD-MoE/README.md)，仅保留与探索有关的部分及原始许可证。完整方法取舍与实验安排见[MOE_RESEARCH](../../../models_v1/research/MOE_RESEARCH.md)。

## 本地材料口径

本目录6份PDF。Routers为作者arXiv开放版本，正式出处已由TMLR/OpenReview核实；其余5份来自会议或出版社公开全文。SCD-MoE仅记录摘要和官方代码，不将其当作已阅读全文的文献。
