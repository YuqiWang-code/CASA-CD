# GCCANet（参考实现，简化提取版）

- 论文：**GCCANet**（官方 README 仅写 "Official implementation of **GCCANet** for remote sensing change detection"，
  未给出完整题名与作者列表），IEEE TGRS 2026（任务给定）。
- 官方仓库：https://github.com/E1ison/GCCANet （下载自 `main` 分支，commit `8d98d93`）
- 许可证：未标注（仓库无 LICENSE 文件，GitHub API `license` 字段为 `null`）。
- 下载/简化日期：2026-10-19。
- 与本项目的关系：GCCANet 的「DINOv2 全局语义引导 → ResNet 多尺度特征注入 + 多尺度差异编码 + 最高分辨率小目标支路」
  与本项目 CASA-CD 的多尺度变化检测主线直接可比，其 `ablate_*` 开关（`ablate_dino_fusion` / `ablate_injector` /
  `ablate_hybrid_attention_only` / `ablate_difference_encoder`）可直接借用为消融实验模板。

## 保留内容

| 文件（= 上游路径） | 说明 |
|---|---|
| `models/gcca_model.py` | 核心机制全部在此：`ALA`（4 层 DINOv2 中间层 GAP→MLP→softmax 加权自适应层聚合）、`HybridAttention`（1×1 降维 + 多头自注意力 + 残差）、`DLM` 差异学习模块（逐级 `cat[pre, post, abs(pre-post)]` → 两层卷积 → LayerNorm/GELU → HybridAttention → 通道对齐）、`BSM` 特征注入模块（DINOv2 语义特征插值到各级 ResNet 分辨率 → 1×1 adapter → 拼接 → `EfficientResidualFusionBlock` 深度可分离残差融合，`inject_alpha` 控制注入强度）、`GCCA` 整网（ResNet 多尺度 + DINOv2 引导 → 差异编码 → 解码） |
| `models/cd_modules.py` | `Decoder`：FPN 式自顶向下逐级上采样 + 残差精修，在 p2/p3/p4/p5 四级各出一个辅助预测，并在最高分辨率 p2 上单独出 `small_mask`（小目标支路） |

## 简化内容

- 删除 `train.py`、`eval_cd.py`、`utils.py`、`data_config.py`、`create_SOGT.py`、`mean_std.py`、`datasets/`、
  `misc/`、`logs/`（五个数据集的训练日志）、`vis/`、所有 `__pycache__/`。
- 删除 `models/trainer.py`、`models/evaluator.py`、`models/losses.py`、`models/networks.py`（`define_G` 构建器）、
  `models/backbone/`（DINOv2 骨干定义 `dinov2.py` + `dinov2_layers/`）、`models/semseg/dpt.py`、`models/util/blocks.py`。
- 上游 import 语句原样保留，未做任何改写：`from models.cd_modules import Decoder`、`from models.backbone.dinov2 import DINOv2`
  （文件中部导入）、`from utils import weight_init`。后两者对应的文件未随本目录提供，因此本目录代码不能直接运行，
  仅作机制参考；另依赖 timm / einops / torchvision / matplotlib（`gcca_model.py` 顶部导入了 `matplotlib.pyplot`，
  上游遗留，未删除）。

## 命名与论文机制的对应（据代码实读，不臆造）

上游源码中**没有**字面名为 `MS-SOE` 的模块。按代码可确认的对应关系：

- 「小目标敏感」→ `cd_modules.py` 的 `Decoder` 把最高分辨率 p2 单独分出 `cls_small` 支路，并对 p2/p3/p4/p5 四级
  全部做深监督（`forward` 返回 `main_mask, small_mask, mask_p3, mask_p4, mask_p5`）；`gcca_model.py` 的
  `GCCA.forward` 再把 5 个输出按需插值成元组。
- 「环境鲁棒」→ `BSM` 把 DINOv2 的全局上下文注入 ResNet 每一级特征（跨域/跨季节语义对齐），
  `ALA` 负责把 DINOv2 的 4 个中间层自适应聚合成单一引导特征。
