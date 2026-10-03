# SHViT（reference code，简化提取版）

- **题名**：SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design
- **venue / 层级**：CVPR 2024 · **CCF-A**
- **一作 / 机构**：Seokju Yun（POSTECH）
- **官方论文**：https://openaccess.thecvf.com/content/CVPR2024/html/Yun_SHViT_Single-Head_Vision_Transformer_with_Memory_Efficient_Macro_Design_CVPR_2024_paper.html
- **官方代码**：https://github.com/ysj9909/SHViT
- **官方模型**：S1 = 6.3M / 241M FLOPs @224 / ImageNet-1K Top-1 72.8

## 保留内容（官方源码原样，仅作机制参考）

- `model/shvit.py` —— 核心结构：`Conv2d_BN`（含 fuse）、`BN_Linear`、`PatchMerging`、
  `Residual`、`FFN`、`SHSA`（Single-Head Self-Attention：partial-channel QKV）、
  `BasicBlock`、`SHViT`；
- `model/build.py` —— 模型族配置（`SHViT_s1 = embed_dim [128,224,320] /
  depth [2,4,5] / partial_dim [32,48,68] / types ["i","s","s"]`）与官方权重加载方式
  （checkpoint 外层 key = `"model"`）。

## 已剔除（训练/数据集/下游/导出等与本课题无关）

`data/`、`downstream/`、`engine.py`、`losses.py`、`main.py`、`export_model.py`、
`speed_test.py`、`utils.py`、`requirements.txt`、图片与 pycache。

## CASA-CD 使用方式

SHViT 版主线（`models/model/shvit_s1_trunc.py` 截断主干 + CASA-STR Run1）已于
2026-10-03 归档（代码保留在 git 历史与
`docs/temporary/models_and_metrics_CASA-STR_Run1.txt` 快照中）；当前主线已切换为
TinyViM-S-Slim（`models/model/tinyvim_s_slim.py`，见 `others/TinyViM-main/`）。
本目录仅作为官方机制参考存档，不参与训练。

- 上游许可证：见 `LICENSE`（官方仓库自带）。
- 下载日期：2026-10-02（用户上传）。
