# TransNeXt 聚合注意力参考

- 论文：Dai Shi, **TransNeXt: Robust Foveal Visual Perception for Vision Transformers**, CVPR 2024, 17773–17783。
- [官方仓库](https://github.com/DaiShiResearch/TransNeXt)，版本 `c8a99743b60ac94ac8d2bf66ffe164a440dcfe21`，下载于2026-10-27。
- `attention_native.py`保留上游`classification/attention_native.py`：PyTorch实现的局部/池化联合softmax、相对位置偏置和内容聚合，文件内容保持上游版本；依赖PyTorch与NumPy。
- `LICENSE`为上游Apache-2.0许可证。训练、数据、分类骨干、CUDA扩展与预训练权重不在本参考目录内。

CASA Run5参考其局部与压缩内容在同一softmax内竞争的原则。CASA只加入query自身这个未压缩候选，并使用变化引导原型、欧氏/双曲距离及相对写回；不是TransNeXt的完整聚合注意力。训练实现不导入本目录。
