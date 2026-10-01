# ViT-CoMer

- **论文**: *ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions*, CVPR 2024.
- **官方 GitHub**: https://github.com/Traffic-X/ViT-CoMer （下载自 main 分支 zip）
- **保留文件**（取自仓库 segmentation/ 部分的核心 backbone，其余训练/检测/数据/配置代码已删除）:
  - `segmentation/mmseg_custom/models/backbones/comer_modules.py` — 核心机制：CNN 分支生成多尺度卷积特征、CTIBlock 特征交互、deformable attention 输入构造；
  - `segmentation/mmseg_custom/models/backbones/vit_comer.py` — 注入 CoMer 多尺度 CNN 特征的 ViT backbone；
  - `segmentation/mmseg_custom/models/backbones/vit_baseline.py` — 基线 ViT backbone（对比/依赖）；
  - `segmentation/mmseg_custom/models/backbones/base/vit.py` — TIMM VisionTransformer 基类（依赖）。
- **与 CASA-CD 相关**: 遥感变化检测常用 ViT 提取多尺度特征，ViT-CoMer 的"ViT + 卷积多尺度交互"正是变化检测编码器的直接参考。
- **依赖说明**: 依赖 mmcv/mmseg、timm、以及仓库内的 deformable attention CUDA 算子（`ops/`，未复制）与 `mmcv_custom` 包（未复制）。
- **上游许可证**: LICENSE，Apache License 2.0。
- **下载日期**: 2026-10-01。
