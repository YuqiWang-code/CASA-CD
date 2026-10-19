# EdgeRefNet（参考实现，简化提取版）

- 论文：**EdgeRefNet: An Edge-Guided Refinement Network for Building Change Detection in Remote Sensing Images**
  （上游 README 标题；正文第一行又写作 "EdgeRefNet: **Edge-Guided Enhancement and Refinement Network** for Building
  Change Detection of Remote Sensing Images" —— 两个题名均出自官方 README 原文），IEEE TGRS 2026（任务给定）。
- 官方仓库：https://github.com/Wafaa-Hima/EdgeRefNet （下载自 `main` 分支，commit `ba6f872`）
- 许可证：未标注（仓库无 LICENSE 文件，仅 `requirements.txt`）。
- 下载/简化日期：2026-10-19。
- 与本项目的关系：EdgeRefNet 把「边缘监督 + 边缘特征回注变化分支」做成与主干解耦的双分支结构，
  与本项目对边界精度的关注一致；其 `forward` 返回 `[edge_map, change_map]` 的双监督形式可直接借用为
  CASA-CD 的辅助损失设计。

## 保留内容

| 文件（= 上游路径） | 说明 |
|---|---|
| `custom_models/edge_block.py` | 边缘分支本体：`Edge_Encoder`（四级 ResNet 特征 1/2–1/8 各过一个带 3×3 DWConv 的 `Mlp` 精修，反卷积对齐到同一尺度后 1×1 融合成 32 通道边缘特征，即 EDB 边缘检测/建模块）、`Edge_Decoder`（`Mlp` + 2× 双线性上采样 + 3×3 卷积出 2 通道边缘图，即 EEM 边缘增强/精化模块），另含 `DWConv`、`Mlp` |
| `custom_models/attention_block.py` | 高频增强注意：`HighFrequencyModule`（可微固定卷积核，默认各向同性 Sobel，`filtering`/`thresholding` 两种模式提高频）、`HighFrequencyEnhancementStage`、`SCSEBlock`（通道 + 空间双重激励）、`HFAB` 组合体 |
| `custom_models/networks.py` | 整网 `BASE_Transformer`（即 `--net_G EdgeRefNet` 对应的模型，见文件末尾 `__main__` 用法）：双时相 ResNet-18 + 语义 token 的 `Transformer`/`TransformerDecoder`（上下文引导交叉注意力）+ 边缘分支（`edge_encoder` → `HF` → `edge_decoder` → `HF__` → `edge_classifier`）+ 边缘特征回注与双时相绝对差分类 |

## 简化内容

- 删除 `main_cd.py`、`eval_cd.py`、`utils.py`、`data_config.py`、`requirements.txt`、`data_preparation/`、
  `datasets/`、`images/`、`samples_LEVIR/`、`samples_WHU/`、`misc/`、全部 `__pycache__/`。
- 删除非本论文的对比模型与训练件：`custom_models/BIT.py`、`ChangeFormer.py`、`ChangeFormerBaseNetworks.py`、
  `TransformerBaseNetworks.py`、`DTCDSCN.py`、`EGCTNet.py`、`pvtv2.py`、`resnet.py`、`base_model.py`、
  `basic_model.py`、`trainer.py`、`evaluator.py`、`losses.py`、`help_funcs.py`、`__init__.py`。
- **上游 import 语句原样保留，未做任何改写**：`from custom_models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d`、
  `from custom_models.edge_block import Edge_Encoder, Edge_Decoder`、`from custom_models.attention_block import HFAB`、
  `from custom_models.BIT import BIT` 等。其中 context-guided cross-attention 的实现位于未收录的
  `custom_models/help_funcs.py`（BIT 的 token 交叉注意力解码器，`Transformer`/`TransformerDecoder`），
  超出「1–3 个核心文件」的上限，故本目录代码不能直接运行，仅作机制参考；另依赖 timm / einops / torchvision。
- 未做清理的上游痕迹：`networks.py` 的 `forward` 里保留了 `print(f"x1 shape: ...")` 等调试输出，
  `edge_adjust` 一行带有阿拉伯语注释（`# تصغير الحجم إلى النصف`），均按原样保留以便与上游逐行对应。
- 数据集说明（上游 README）：`A/`、`B/`、`label/`、`label_edge/`、`list/`，其中 `label_edge` 由 Canny 算子作用于
  变化标签得到；本论文实现基于 EGENet 仓库代码改写（上游 README「References」声明）。
