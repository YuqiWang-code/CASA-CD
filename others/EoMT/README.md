# EoMT

- **论文**: *EoMT: Encoder-only Mask Transformer* (CVPR 2025 Highlight).
- **官方 GitHub**: https://github.com/tue-mps/EoMT （下载自 master 分支 zip；main 不存在）
- **保留文件**（训练/数据集/损失代码已删除）:
  - `models/eomt.py` — 核心机制：Encoder-only Mask Transformer（可学习 query + 渐进式尺度块堆叠，把 mask transformer 解码器折叠进编码器）；
  - `models/scale_block.py` — 核心模块：ScaleBlock（逐块 2x 上采样，逐步恢复高分辨率 mask）；
  - `models/vit.py` — 冻结 ViT 编码器封装（DINOv2 等，依赖 timm/transformers）。
- **与 CASA-CD 相关**: 变化检测本质是密集 mask 预测；EoMT 提供了不引入单独解码器、直接在编码器内产出高分辨率 mask 的高效结构。
- **上游许可证**: LICENSE，MIT License。
- **下载日期**: 2026-10-01。
