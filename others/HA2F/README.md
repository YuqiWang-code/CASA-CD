# HA2F（参考实现，简化提取版）

- 论文：**HA2F**（IEEE TGRS 2026，任务给定：层级自适应聚合；DHFCM / NAFRM）。⚠️ 官方模型页**未给出论文题名**，
  仓库名为 `RemoteSensingChangeDetection-RSCD.HA2F`，发布组织为 `InPeerReview`（匿名评审式发布），
  model card 只有环境/数据/训练命令，没有标题、作者与 BibTeX。
- 官方地址：https://huggingface.co/InPeerReview/RemoteSensingChangeDetection-RSCD.HA2F
  （HuggingFace **model** repo，`main`，commit `486e9a7`；带 `.gitattributes`，实为 git 仓库而非纯模型卡）
- 许可证：未标注（仓库无 LICENSE 文件，HF model card 的 YAML metadata 为空）。
- 下载/简化日期：2026-10-19。
- 与本项目的关系：HA2F 的「ViT 语义 + CNN 细节双分支 + 高层特征逐级注入聚合」正是 CASA-CD 关心的层级聚合范式，
  `FeatureInjector` / `SimplifiedFGFM` 可作为 CASA-CD 解码器多尺度上采样的对照实现。

## 是否「仅模型卡」的核实结果

**不是**——HF API 的文件清单确认该仓库**有完整源码**：`model/{encoder,decoder,utils,resnet,metric_tool,trainer}.py`、
`model/layers/*`、`dataset/{dataset,Transforms}.py`、`main.py`、`eval.py`、`requirements.txt`。
但同时**没有任何权重文件**（无 `.pth/.bin/.safetensors`，HF API `usedStorage: 0`），且 encoder 硬编码从
`checkpoint/dinov2_vits14_pretrain.pth`（或 `checkpoint/deit_tiny_patch16_224-a1311bcf.pth`）加载预训练骨干，
所以这是一份「代码 + 训练入口」发布，权重需另行准备（checkpoint 目录上游未提供）。

## 保留内容

| 文件（= 上游路径） | 说明 |
|---|---|
| `model/decoder.py` | 核心机制全部在此：`CrossAttention` / `Block`（cross-attn + MLP）、`ContentAwareAggregation`（双时相高层特征互相内容感知聚合：投影 + 深度可分离注意图 + 残差式融合）、`FeatureInjector`（以 c5 为 query 对 c2/c3/c4 逐级做交叉注意力注入，融合成新的 c5）、`DualAttentionGate`（通道注意力 + mean/std 空间注意力门控）、`SimplifiedFGFM`（`flow_warp` 光流对齐式由粗到细特征传播）、`Decoder`（四级差异建模 `cat[x, y, abs(x-y)]` → 3 层 MLP 通道压缩，再自 c5 向 c2 逐级 FGFM 传播，4×4 转置卷积上采样出变化图） |
| `model/encoder.py` | `Encoder`：DINOv2 ViT-S/14 @256（`model_type='small'`；另有 `tiny` 用 DeiT-Tiny 结构）语义分支 + ResNet-18 细节分支，`detail_capture` 取 layer1–4，`fusion_conv` 把 layer4 与 ViT 特征拼接融合后回填到最高层，`forward(x, y)` 返回双时相各自的 `[c2, c3, c4, fused]`；文件内自带完整 `DinoVisionTransformer` 定义（含 `get_intermediate_layers`） |

## 简化内容

- 删除训练/数据/配置/度量件：`main.py`、`eval.py`、`dataset/`、`model/trainer.py`、`model/metric_tool.py`、
  `model/utils.py`（`weight_init`/`init_seed`/`BCEDiceLoss`/`BCE`/`adjust_learning_rate`）、`model/resnet.py`、
  `model/layers/`、`requirements.txt`、`.gitattributes`、全部 `__pycache__/`（含上游遗留的 `.pyc`）。
- **上游 import 语句原样保留，未做任何改写**：`from model.utils import weight_init`、`from model.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block`、
  `from model.resnet import resnet18`、`from einops import rearrange`。这些依赖文件未随本目录提供，
  因此本目录代码不能直接运行，仅作机制参考。
- 注：上游 `model/` 目录本身没有 `__init__.py`（仅 `model/layers/` 有），此处未替上游补加。

## 命名与论文机制的对应（据代码实读，不臆造）

在 `model/encoder.py` / `model/decoder.py` 中**没有出现** `DHFCM` 或 `NAFRM` 这两个名字。按代码结构，
最接近「层级自适应聚合」的是 `FeatureInjector`（高层 c5 → 低层 c2/c3/c4 的逐级交叉注意力注入）
配合 `ContentAwareAggregation`（双时相互相聚合）；「频率/差异」类机制在本仓库这两个核心文件中未见同名实现。
因此本目录只按上游代码原名记录，不臆造缩写映射。
