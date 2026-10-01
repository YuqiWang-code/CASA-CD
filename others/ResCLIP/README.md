# ResCLIP

- **论文**: *ResCLIP: Residual Attention for Training-free Dense Vision-language Inference*, CVPR 2025.
- **官方 GitHub**: https://github.com/yvhangyang/ResCLIP （下载自 main 分支 zip）
- **保留文件**（评估/数据集/配置/prompts 已删除）:
  - `clip/model.py` — 核心机制之一：改造版 CLIP ViT（`ResidualAttentionBlock` + `attn_strategy='resclip'` 的 residual attention，用高斯加权的跨层残差注意力校正注意力图）；
  - `resclip_segmentor.py` — 核心机制之二：ResCLIP 训练无关密集推断 segmentor（SFR/RCS 特征精修、滑窗密集预测，依赖 mmseg/mmengine）；
  - `pamr.py` — PAMR 像素自适应精修后处理（密集预测常用后处理，上游 visinf/1-stage-wseg）。
- **与 CASA-CD 相关**: 变化检测常引入 CLIP/文本先验做训练无关语义引导，ResCLIP 的 residual attention 是提升冻结 CLIP 密集特征质量的关键技巧。
- **上游许可证**: LICENSE，MIT License（pamr.py 来自 visinf/1-stage-wseg，Apache 2.0）。
- **下载日期**: 2026-10-01。
