# CASA-TViM Run1 — CAACP-SS2D 主线（TinyViM-S-Slim + CAACP + TAR/DCR）

依据 `docs/temporary/CASA-CD_VMamba骨干与变化感知SS2D_调研分析与可执行方案_2026-10-03.md`
执行。SHViT 版 CASA-STR 已归档（`train_scripts/CASA-STR/Run1`，不再扩展）。

## 架构（CASA-TViM-STRNet）

```
[A;B] (2B) -> TinyViM-S-Slim shared（预训练 1000e EMA 权重，可训练 @0.1×）
    stem -> F1(1/4,48) -> F2(1/8,64) -> stage2 prefix(1/16,168)
    -> [CAACP-SS2D 共享 change score] -> stage2 final TViM -> F3(1/16,168)
    -> slim stage4 (Local×3 + TViM) -> F4(1/32,224)
per-time [F1,F2,F3,F4] -> MultiScaleTAR(encoder_dims=(48,64,168,224), D=96)
    -> DCRDecoder(D=96) -> head Conv1x1 -> bilinear -> sigmoid
```

- **CAACP-SS2D（创新一）**：只改 Stage3 末个 TViM 的低频路径——官方 AvgPool2×2 换为
  双时相 change score（1−cos，rank 归一化，参数自由）驱动的 **2×2 cell 加权聚合**
  （A/B 共享权重、规则 8×8 lattice、无 TopK、四向 CrossScan 几何不变）；dense 高频
  残差路径完整保留；`c = c_avg + β*(c_ca − c_avg)`，β=0 初始化 → epoch-0 逐位等于
  官方预训练模型。
- **TAR/DCR（创新二）**：沿用已验证的 STR 折叠模块，D=96。
- deploy：`switch_to_deploy()` 只折叠 TAR/DCR；CAACP 是推理期真实模块。
- Slim：Stage4 删除 2 个 LocalBlock（预训练 key 显式重映射 network.6.5→6.3），
  trunk 4,645,180；deploy 粗估 4.88M ≤5M（服务器实测为准）。

## 变体网格（caacp × rep_mode，2×2 因子设计）

| 组 | caacp | rep | 作用 |
|---|---|---|---|
| A0_TVIM_PLAIN | 0 | plain | 新 backbone baseline（先锁 floor）|
| M1_FULL | 1 | full | 完整方法（主实验）|
| A1_CAACP | 1 | plain | 创新一 |
| A2_STR | 0 | full | 创新二 |

每变体 **4 数据集完整 80K**（16 run 全跑，无单数据集 gating）。

## 执行（run_all.sh，一个脚本串完，不分步）

- 4 波：A0×4 → M1×4 → A1×4 → A2×4；每波 4 数据集并行。
- GPU0 = CDD + LEVIR、GPU1 = SYSU + WHU（每卡 2 个并发任务）；
  **batch 32**（用户指示翻倍，记录于 run_manifest.json）。
- 每脚本 retry 上限 3（崩溃自动从 last.pth 续训）。

## 训练协议（固定，不做调参）

BCE+Dice；Adam(2e-4, 0.9/0.99, wd=1e-4)；poly(0.9)+200 warmup；80000 steps；seed 16；
test-as-val；backbone ×0.1 LR（2e-5）、new ×1.0。数据契约 legacy_6ch_reverse_v1。
新增 `[BACKBONE-ADAPT]` 审计日志（每 1000 iter：grad/param 范数 + 相对预训练 L2 drift）。

## 硬门槛

- 有效推理参数 ≤5M（deploy 折叠后实测）；F1 四数据集 SYSU≥85 / LEVIR≥92.5 /
  WHU≥95 / CDD≥98（IoU 同向）；训练完成模型 `[REPARAM-REAL-ARGMAX-DISAGREE]==0`。
- SS2D kernel：mamba-ssm `selective_scan_fn`（Triton，适配 RTX 5090 sm_120；
  官方 selective_scan_cuda 在 torch 2.14/CUDA 13.2 未编译）。

## 路径

- 检查点：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run1/<VARIANT>/<DS>/`
- 日志：`/home/yqwang/outputs/CASA-CD/CASA-TViM/Run1/<VARIANT>/<DS>/train_log.txt`
- 预训练：`pretrained_weight/tinyvim_s_1000e.pth`（sha256 首16 `f63e58da8109dbb1`）
