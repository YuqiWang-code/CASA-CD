# Token Cropr

- **论文**: *Token Cropr: Faster ViTs for Quite a Few Tasks*, CVPR 2025.
- **官方 GitHub**: https://github.com/benbergner/cropr （下载自 main 分支 zip）
- **保留文件**（训练引擎/数据/配置代码已删除）:
  - `cls/cropr.py` — 核心机制：Token Cropr 剪枝头（CrossAttention + 可学习 query + token 重要性排序剪枝）；
  - `segm/cropr.py` — 分割任务版 Cropr 头（与 cls 版略有差异）；
  - `cls/vision_transformer.py` — 改造版 timm ViT：展示 Cropr 头逐块插入、剪枝 token 如何在 ViT 前向中被丢弃的集成方式。
- **与 CASA-CD 相关**: 遥感影像 token 数巨大，Token Cropr 的轻量剪枝头可作为变化检测 ViT 编码器的加速手段。
- **上游许可证**: LICENSE，MIT License。
- **下载日期**: 2026-10-01。
