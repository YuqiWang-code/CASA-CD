# TinyViM（参考实现，简化提取版）

- 论文：**TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba**（Xiaowen Ma, Zhenliang Ni, Xinghao Chen, Huawei Noah's Ark Lab），ICCV 2025，CCF-A。
- 官方仓库：https://github.com/xwmaxwma/TinyViM （arXiv:2411.17473）
- 官方 checkpoint：TinyViM-S 1000e（ImageNet-1K Top-1 80.3%）——本课题使用文件
  `pretrained_weight/tinyvim_s_1000e.pth`（sha256 首16 `f63e58da8109dbb1`，取 `model_ema` 权重）。
- 许可证：官方仓库未提供 LICENSE 文件（论文/代码仅作研究参考）。
- 下载/简化日期：2026-10-03。

## 保留内容

| 文件 | 说明 |
|---|---|
| `model/tinyvim.py` | TinyViM-S/B/L 主干定义（stem / LocalBlock / Stage / fork_feat out_indices=[0,2,4,6]） |
| `model/tvimblock.py` | TViMBlock / SS2D（Laplace mixer：低频 AvgPool 下采样 + 四向 CrossScan/CrossMerge selective scan + 高频 RepDW 残差，frequency ramp inception 通道分配） |
| `model/__init__.py` | 包导出 |

## 简化内容

- 删除 `detection/`、`segmentation/`、`data/`、训练/评测/可视化脚本与图片。
- 依赖 `selective_scan_cuda`（VMamba kernel）与 timm/einops——本课题在
  `models/model/layers/ss2d.py` 中做了无 timm/无 einops 的自包含移植，SSM 后端改用
  mamba-ssm 的 `selective_scan_fn`（Triton 内核，适配 RSML-3 的 RTX 5090 sm_120），
  并在 `models/model/tinyvim_s_slim.py` 中做了 Stage4 深度裁剪与预训练 key 重映射。
