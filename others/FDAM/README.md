# FDAM

- **论文**: *FDAM: Frequency-Dynamic Attention Modulation for Dense Prediction*, ICCV 2025.
- **官方 GitHub**: https://github.com/Linwei-Chen/FDAM （下载自 main 分支 zip）
- **保留文件**（configs/tools/数据集等全部删除，仅保留核心 backbone 1 个文件）:
  - `deit_fdam.py` — 核心机制全部在此文件：`GroupDynamicScale`（逐组动态缩放）与 `AttentionwithAttInv`（注意力逆变换）：对低频/高频分量做频率动态调制（`dy_freq` 调制）后经注意力逆变换恢复密集空间分辨率，即 FDAM 的频率-动态注意力调制。
- **与 CASA-CD 相关**: 变化检测对边界/高频细节敏感，FDAM 的频率域注意力调制可作为提升 ViT 密集预测细节恢复的参考模块。
- **依赖说明**: 依赖 mmseg/mmcv、timm、einops（文件中为独立 ViT backbone，含全部 Block/Attention 定义，但注册依赖 mmseg BACKBONES）。
- **上游许可证**: 未标注（仓库无 LICENSE 文件）。
- **下载日期**: 2026-10-01。
