# CASAA Run1 — 创新主线一：变化感知非对称 Token 建模（唯一变量验证）

> 设计依据：`docs/temporary/CASA-CD_CASAA_Run1_修改与实验设计建议.md`。
> Run1 只回答一个问题：**「变化感知的非对称 K/V 建模本身是否有效？」**
> 不改 loss / decoder / ResNet18 / 训练协议；极轻量结构（主线二）不在本 Run。

## 方法（Paired Late-Stage CASAA）

- ChangeViT-T 的最后 4 个 ViT block（0-based **8, 9, 10, 11**）的 self-attention
  替换为 CASAAAttention（`models/model/layers/casaa.py`）；
- 前 8 层完全不变；decoder / Feature Injector / ResNet18 detail branch 完全不变；
- **Full Query**：每时相每层 N = 256（16×16 patch）query 完整保留，输出始终 256 token；
- **压缩 K/V**：K = keep_ratio × N = **64**（`--casaa_keep_ratio 0.25`）；
- **变化感知路由（A2, router=change）**：
  - change score `s_i = 1 - cos(x1_i, x2_i)`（参数自由、T1/T2 对称、无新 loss）；
  - **Kc = 32** 个 change-score TopK token **原位保留**（`--casaa_change_share 0.50`）；
  - **Kb = 32** 个稳定背景 token 用共享背景描述 `Norm((x1+x2)/2)` 做 deterministic
    density-peak 聚类，两时相按**共享 assignment** 分别做簇内均值；
  - `C_t = [X_t[Ic]; mean_cluster(X_t_bg)]`，索引/聚类拓扑共享、特征不混时相；
- **预训练原位继承**：qkv/proj 参数名与形状不变，DeiT-Tiny 权重原位加载，
  **零新增参数**；routing 在 no_grad 下计算，聚合保留梯度；
- **A1 对照（router=content, SAA-style）**：同 K=64、同层位置，全部 token 内容聚类、
  无变化感知保留。A2 > A1 才能把收益归因到 change awareness。

## 实验矩阵（Run1）

| Variant | Q | K/V | Change-aware | Layer | Mode |
|---|---:|---:|---|---|---|
| A0 baseline（已有） | 256 | 256 | × | 12 层原 attention | baseline |
| A1 SAA-style | 256 | 64 | × | blocks 8-11 | `--mode saa` |
| **A2 CASAA-v1** | **256** | **64** | **✓** | **blocks 8-11** | `--mode casaa` |

## 启动顺序

1. **筛选阶段**（先跑；一张 RTX 5090 同时最多一个任务）：
   - GPU0/GPU1 空闲时（推荐，双卡并行）：
     ```bash
     cd /home/yqwang/projects/CASA-CD/train_scripts/CASAA/Run1
     nohup bash run_screen.sh > /dev/null 2>&1 &
     ```
     GPU0：A1_SAA_LEVIR → A2_CASAA_LEVIR；GPU1：A1_SAA_SYSU → A2_CASAA_SYSU。
   - 只有一张卡可用时（GPU1 串行，RUN_GPU=1 覆盖任务内 GPU 配置）：
     ```bash
     nohup bash run_screen_gpu1_serial.sh > /dev/null 2>&1 &
     ```
     顺序：A1_LEVIR → A2_LEVIR → A1_SYSU → A2_SYSU。
2. **补全阶段**（筛选达到成功阈值后）：
   ```bash
   nohup bash run_full_casaa.sh > /dev/null 2>&1 &
   ```
   GPU0：A2_CASAA_CDD；GPU1：A2_CASAA_WHU。

## 路径

- Checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run1/{A1_SAA,A2_CASAA}/<dataset>/`
- 日志：`/home/yqwang/outputs/CASA-CD/CASAA/Run1/{A1_SAA,A2_CASAA}/<dataset>/train_log.txt`
- 与 baseline 同协议：BCE+Dice、poly LR、max_steps=80000、batch 16、256×256、
  seed 16、test-as-val（每 epoch 在 test 上挑 best）、epoch 0 后跳过一次评估。

## 成功阈值（LEVIR + SYSU，对照 baseline Run1）

- A2 相对 baseline 任一数据集 **F1 ≥ +0.30**，另一数据集下降 ≤ 0.15；
- IoU 与 F1 方向一致；A2 相对 A1 在至少一个数据集 **F1 ≥ +0.15**；
- 参数量不增加；Query/output token 数不变；FLOPs/VRAM 不显著高于 baseline。
- baseline 对照：CDD 97.75 / LEVIR 91.95 / SYSU 82.48 / WHU 94.84（F1）。

失败现象的解释路径见设计文档 §17（LEVIR↑SYSU↓ → adaptive change quota；
SYSU↑LEVIR↓ → detail cue 辅助 routing；A1≈A2 都提升 → 需重设计 change routing）。
