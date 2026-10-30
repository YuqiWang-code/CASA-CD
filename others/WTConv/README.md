# WTConv 核心参考

论文：*Wavelet Convolutions for Large Receptive Fields*，ECCV2024。[官方论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07137.pdf) · [官方仓库](https://github.com/BGU-CS-VIL/WTConv)。

版本`4a84da2218924b3973de2235d641f2ecf969c346`，2026-10-28读取，MIT许可证见LICENSE。

仅保留原始2D核心：`wtconv/wtconv2d.py`、`wtconv/util/wavelet.py`的2D滤波/变换/逆变换，以及导入入口；未保留分类网络、训练脚本、数据及优化后端。参考依赖PyTorch、PyWavelets；CASA当前训练不依赖此目录。

WTConv在低频递归分解中处理四个子带，再逆变换回空间域，并保留基础深度卷积路径。用于核对频率专家的子带组织与重建，不能直接作为“高频增强必然改善变化检测”的证据。

探索线拟采用单级Haar、低频和三方向高频分别处理；该方案还包含变化任务路由，与原WTConv不是同一模块。固定滤波器在探索实现中宜作为buffer登记，参数报告区分可学习参数与固定系数。
