# CASA-CD 研究方案（ChatGPT 路线记录）

> 记录时间：项目启动日。来源：与 ChatGPT 的讨论（用户转述）。
> 状态：**记录待用**。先复现 baseline，SAT 创新点等复现完毕后再推进。

## 0. 项目定位

- 任务：高分辨率遥感图像**二值变化检测**（BCD，全监督）。
- 立场：学术创新 / 方法创新，不做工程任务；不把 loss 包装成创新点；不做防御性编程。
- 总目标：**极轻量 RSBCD（带预训练权重），在极低参数量等级上 F1/IoU 超过 SOTA**。
- Baseline：ChangeViT-Tiny（PR 2025），官方预训练权重 `deit_tiny_patch16_224-a1311bcf.pth`，约 11.68M Params / 27.15G FLOPs。
- 创新来源：SAT（Selective Aggregation Transformer for Image Super-Resolution，CVPR 2026，arXiv:2604.07994，https://github.com/PhuTran1005/SAT）——**非对称、变化感知 token：完整 Query、压缩 KV**。

## 1. 核心动机（来自 SAT 的迁移）

- 高分辨率遥感图像 token 数极大，但真正发生变化的区域通常只占少数；
- 不能粗暴下采样，否则小建筑、窄道路、边界变化容易丢失；
- 因此：**保留完整的变化查询位置（Query），压缩双时相中的冗余背景信息（Key/Value）**；
- 即不要压缩最终需要逐像素判别的 Query，而是压缩用于提供上下文的 K/V。

## 2. ChatGPT 给出的路线（采纳要点）

**路线定名：CASA-CD — Change-Aware Selective Aggregation Network for ultra-lightweight fully-supervised binary change detection in remote sensing images。**

### 2.1 创新主线一：CASAA（Change-Aware Asymmetric Token Modeling）

- 原 Attention：`O(N^2)`；CASAA：`O(NK), K≪N`。
- 不是随机压缩 / Pooling / 普通 SAA，而是：
  - 疑似变化 token → 尽量保留；
  - 稳定背景 token → 强聚合；
  - 所有 Query → 继续逐位置判别。
- 在 ChangeViT-T 的 ViT Attention 上改：`Q,K,V → Q, K̃, Ṽ`（Full Change Query + Compressed Change-Aware K/V），保留预训练 DeiT-Tiny 的 Q/K/V 能力，不推翻 pretrained 结构。
- 论文卖点不仅是降 FLOPs：**压缩冗余背景反而减少背景干扰，提高稀疏变化表征质量**。

### 2.2 创新主线二：Ultra-Light Multi-Scale Change Representation（整体极轻量化结构）

- ChangeViT 原结构：DeiT-Tiny + ResNet18 detail-capture branch + Feature Injector + Decoder。
- 目标结构：
  ```text
  Pretrained DeiT-Tiny
        │
      CASAA
        │
  Lightweight Detail Branch
        │
  Lightweight Change Decoder
        │
    Change Map
  ```
- 重点处理 ChangeViT 原来的 **ResNet18 detail-capture branch**（很重），换成轻量 detail path；decoder 同样做轻量。
- 参数量路线：`11.68M → 5M → 3M → <3M`。

### 2.3 论文定位

> 在极低参数量和计算量下，通过变化感知的非对称 token 建模，实现比现有轻量 BCD 更强的精度—复杂度权衡。

- 不做新 Loss、不靠训练 trick、不把剪枝/量化当核心创新。
- 最终目标：`ChangeViT-T + pretrained DeiT-Tiny → CASAA（Full Q + Change-aware K/V）→ Lightweight Detail Path → Lightweight Decoder → <3M Params → F1/IoU 超轻量 SOTA`。

## 3. 实施顺序（当前阶段）

1. **先把 baseline 复现跑通**（4 数据集，加载预训练权重重新 train）；
2. 复现完毕后再做 SAT / CASAA 创新点。

## 4. 与 ChatGPT 的其它约定（沿用）

- baseline 用 **Tiny**（不是 Small）——定位是极轻量，ChangeViT-T 只是起点 baseline，不是最终模型；
- 训练协议、数据集、服务器约定见 `docs/RSML-3_服务器环境与变化检测数据统一说明.md`。
