# CASA-CD：ViT 训练崩溃发现与 Run2 修订（2026-09-28）

> 状态：**已证实、已处理**。影响 Run1 结论的解释；Run2 已按修订方案启动。
> 证据：服务器 snapshot 实测 + Run1 全部 checkpoint 核验（原始数据在
> `/tmp/casaa_snapshot.txt`、`/tmp/casaa_snap2.txt`，服务器）。

## 1. 现象：官方协议把 ViT 训练成精确零权重

用真实 LEVIR、ChangeViT 官方协议（统一 lr=2e-4、BCE+Dice、poly、batch 16、256×256）、
baseline 模式 snapshot 实测：

| step | pos_embed | patch_embed | blocks.0.qkv | resnet.conv1 | ViT grad(norm1) |
|---|---:|---:|---:|---:|---:|
| 0 | 4.432 | 8.009 | 19.168 | 12.579 | — |
| 400 | 1.70 | 7.53 | 14.97 | 12.51 | 5.0e-5 |
| 800 | 0.055 | 5.58 | 8.79 | 12.49 | 9.7e-6 |
| 1200 | 0.012 | 1.10 | 3.26 | 12.46 | 3.0e-8 |
| 1600 | **0.0007** | **0.029** | **1.12** | 12.43 | 1.1e-11 |

- ViT 全部权重被训练动力学**主动压到零**（约 3.6 epoch 内）；ResNet/decoder 不受影响。
- 机制：Adam 对小梯度做「全步长」更新（m/sqrt(v)≈sign），一致的负反馈梯度 →
  权重持续衰减；权重→0 后 ViT 输出为 0 → 梯度消失（**吸收态**，不可自恢复）。
- 保存/加载往返无损（in-mem == file），确认是**训练动力学崩溃，不是文件损坏**。
- 对照实验：`--vit_lr_ratio 0.1` 显著减缓（400 steps 内 pos 4.43→4.30），但长期
  仍可能衰减；**冻结 ViT**（`--freeze_vit 1`）完全稳定（1200 steps 无任何变化）。

## 2. Run1 结果核验（全部 checkpoint）

| checkpoint | ViT 零权重键 | 含义 |
|---|---:|---|
| baseline/CDD、WHU、LEVIR best+last | 150/150 | ViT 全程死亡（optimizer 矩也全零=从未收梯度） |
| CASAA A1/A2 LEVIR best+last | 150/150 | 同上 |
| baseline/SYSU best | 31/150 | 部分崩溃 |
| A1/A2 SYSU best | 1/150（仅 mask_token） | **健康** |
| A1/A2 SYSU last | 106/150 | epoch 9 之后渐进崩溃 |

- 与 SYSU-A2 训练曲线吻合：best F1=0.8235 在 epoch 9（ViT 健康），此后 ViT 渐进
  崩溃、val F1 衰减到 0.8106——**崩溃对 SYSU 是有害的**。
- 用零 ViT 的 A2-LEVIR checkpoint 重新 eval：F1=0.9186，与 Run1 日志完全一致——
  即 Run1 的 LEVIR 数字全部来自「死 ViT」模型（退化为 ResNet+decoder）。
- **推论**：Run1 在 LEVIR 上的 A0/A1/A2 比较**不检验 CASAA**（router 作用在被
  模型丢弃的组件上）；SYSU 的比较仅在 best（epoch 9）附近有效。决策文档与
  README 中基于 Run1 LEVIR 的结论（如「A1≈baseline 压缩无损」）需要冻结 ViT 重做。

## 3. Run2 修订

- 全部 run 加 `--freeze_vit 1`（路由机制作用于健康的 pretrained DeiT-Tiny；
  注意力上下文选择仍改变 ViT 前向，机制照常被检验）。
- 4 run：`A1_SAA_FROZEN`（content 对照）+ `A3_ORACLE_FROZEN`（GT occupancy，
  DIAGNOSTIC-ONLY）× LEVIR/SYSU，GPU1 串行。
- 成败判据不变：Oracle 相对 A1_FROZEN 至少一个数据集 F1 ≥ +0.30 → 可部署 v2；
  否则停止 change-aware router 迭代转主线二。

## 4. 对主线二与论文的启示

- ChangeViT 的 ViT 在建筑类数据集（LEVIR/CDD/WHU）上对官方协议训练极不稳定、
  且似乎贡献有限（死 ViT 也能复现论文 LEVIR F1——上游论文模型可能同样退化）；
  SYSU 上 ViT 有真实贡献（崩溃掉 ~1.3 F1）。
- 主线二若继续用 DeiT-Tiny，必须自带 ViT 稳定性方案（冻结/低 LR/更短训练）；
  或直接采用更小、更稳定的轻量 backbone。
- 论文中「baseline 复现」需要以**健康 ViT**的运行为准（冻结 ViT 版本），并如实
  报告官方协议下的 ViT 崩溃现象。
