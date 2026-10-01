# LiFT

- **论文**: *LiFT: A Surprisingly Simple Lightweight Feature Transform for Dense ViT Descriptors*, ECCV 2024.
- **官方 GitHub**: https://github.com/saksham-s/lift （项目主页 http://www.cs.umd.edu/~sakshams/LiFT/ 链接的官方仓库，下载自 main 分支 zip）
- **保留文件**（训练/各评测脚本已删除）:
  - `lift.py` — 核心机制：LiFT 模型定义（小 CNN 编码细粒度图像特征 + 上采样 CNN 与 ViT 特征多阶段融合，把稀疏 ViT 特征变换为 dense 特征）；
  - `lift_utils.py` — LiFT 的模型配置/推理设置工具；
  - `extractor.py` — 冻结 ViT dense 特征提取器（LiFT 的输入来源，依赖 timm/torchvision）。
- **与 CASA-CD 相关**: 变化检测需要高质量 dense 特征；LiFT 提供了用极低成本把预训练 ViT 特征变 dense 的轻量特征变换模块。
- **上游许可证**: LICENSE，MIT License。
- **下载日期**: 2026-10-01。
