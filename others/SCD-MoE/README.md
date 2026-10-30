# SCD-MoE 交互路由核心参考

论文：*SCD-MoE: Adaptive Expert Routing for Efficient Remote Sensing Semantic Change Detection*，IEEE TGRS64，2026，DOI[10.1109/TGRS.2026.3738239](https://doi.org/10.1109/TGRS.2026.3738239)。[官方仓库](https://github.com/HuQ1an/SCD-MoE)。

版本`30e3cf23294a1d3122f20db5bf6e59dde90043ce`，2026-10-28读取，Apache-2.0许可证见LICENSE。

`interaction_adaptive_moe.py`摘录官方`changedetection/models/moes/dense_MoE.py`中的`ConvExpert`与`InteractionAdaptiveMoELayer`，保留其实现。仅保留二值任务研究相关的交互核心；训练、数据、多任务语义头、Mamba骨干和CUDA kernel不在本目录。

核心以双时相差与和构造专家输入，按每张图的池化特征选择Top-k，并只调用选中专家。它是图像级路由，不能直接解决同图小目标与大背景需要不同专家的问题。官方论文任务为语义变化检测，指标与CASA二值任务分别解释。

参考调用中的`training`形参独立于`nn.Module.training`；直接使用时应显式传入`training=layer.training`，否则单独调用`eval()`仍可能保留门控噪声。Python逐样本调度也不能直接代表batch1或区域路由的最佳延迟。

本次核实范围为出版社摘要与官方核心代码；论文全文未归档。CASA-MoE方案见[探索调研](../../models_v1/research/MOE_RESEARCH.md)。
