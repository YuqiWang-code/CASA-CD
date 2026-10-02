# CASA-STR Run1 — 主线重构（SHViT-S1 truncated + CASAA@1/16 + TAR/DCR）

依据 `docs/temporary/CASA-CD_主线重构_CASAA_STR_完整调研与实验方案_2026-10-02.md`
与 `docs/temporary/CASA-CD_本地实施注意事项_导师原始思路对齐与P0-P2审查_2026-10-02.md`
的最终建议执行。旧的 TASS 路线（Run11）保留为历史资产，不再扩展。

## 架构（CASA-STRNet）

```
[A;B] (2B) -> SHViT-S1 truncated (patch_embed + blocks1 + blocks2, 预训练冻结权重可训练@0.1x)
  stem -> F1(32,64²) -> F2(64,32²) -> blocks1 -> F3(128,16²)
  -> split A/B -> [CASAA pair @1/16] -> F3'（change score 来自 F2 1/8，参数自由）
  -> concat 2B -> blocks2 -> F4(224,8²)
per-time [F1,F2,F3',F4] -> MultiScaleTAR(encoder_dims=(32,64,128,224), D=160)
  -> DCRDecoder -> head Conv1x1 -> bilinear -> sigmoid
```

- CASAA@1/16：qk_dim=16 单头、K=64（Kc=32 变化直保留 + Kb=32 背景共享 assignment 聚合）、
  keep_ratio=0.25、change_share=0.5；qkv/proj 正常初始化 + **external β=0 gate**（单一零初始化机制）。
- deploy：`switch_to_deploy()` 只折叠 TAR/DCR（FP64 折叠、单次 FP32 cast）；CASAA 是推理期真实模块。

## 变体网格（attn_mode × rep_mode）

| 组 | 变体 | attn_mode | rep_mode | 定位 |
|----|------|-----------|----------|------|
| 基线 | A0_BASE_PLAIN | none | plain | 基线（先锁定 backbone 选择）|
| 主实验 | M1_CASAA_STR | change | full | 双创新全开 |
| 消融1 | A1_CASAA_PLAIN | change | plain | 只开创新一（CASAA）|
| 消融2 | A2_STR_ONLY | none | full | 只开创新二（STR rep）|
| 对照1 | C1_FULLATTN_PLAIN | full | plain | 全注意力对照 |
| 对照2 | C2_CONTENT_SAA_PLAIN | content | plain | 内容密度 SAA 对照 |

## 执行顺序与 GPU 分工

每 Phase 双 GPU 并行；每 GPU 一个顺序队列（前一个完成/3 次 retry 耗尽后启动下一个）。
GPU0：CDD → LEVIR；GPU1：SYSU → WHU。

- Phase1：`run_queue_phase1_gpu{0,1}.sh` → A0×4 + M1×4（A0 先跑完锁定 backbone）
- Phase2：`run_queue_phase2_gpu{0,1}.sh` → A1×4 + A2×4
- Phase3：`run_queue_phase3_gpu{0,1}.sh` → C1×4 + C2×4

每个变体在 **全部 4 个数据集** 上跑（无单数据集 gating）。

## 训练协议（固定，不做调参）

- BCE+Dice loss；Adam(2e-4, 0.9/0.99, wd=1e-4)；poly(0.9) + 200 warmup；80000 steps；
  batch 16；256×256；seed 16；test-as-val。
- 双学习率组：backbone ×0.1（2e-5）、new ×1.0（2e-4），backbone **可训练不冻结**；
  scheduler 按组 `lr_scale` 生效（models/model/utils.py）。
- 数据契约：legacy_6ch_reverse_v1（沿用既有 ToTensor reverse 口径）。
- 从头训练纪律：`--resume` 禁止，崩溃恢复只用自己的 `last.pth`；
  `run_manifest.json` 逐字段严格校验（含 shvit_s1.pth SHA256、seed、lr ratio 等）。

## 硬门槛（推理期）

- 有效推理参数 ≤ 5M（deploy 折叠后实测 ~2.42M）。
- 训练完成模型的 `[REPARAM-ARGMAX-DISAGREE] == 0`（train-graph vs deploy-graph 0.5 二值化不一致为 0）。
- 数据目标：SYSU F1≥85 / LEVIR≥92.5 / WHU≥95 / CDD≥98，F1 与 IoU 同向。

## 目录

- `<VARIANT>/train_<DS>.sh`：24 个正式训练脚本（retry 上限 3）
- `run_queue_phase{p}_gpu{g}.sh`：阶段队列
- `dryrun_a0_gpu{g}.sh`：A0 短训 dry-run（2 epochs，验证 [LR-GROUPS] 0.1× 与完整 TEST 链路；
  ckpt 目录独立为 `Run1/_dryrun/A0_BASE_PLAIN/<DS>`，不污染正式 A0）
- `_gen_scripts.py`：脚本生成器（幂等，改网格后重跑）

## 服务端路径

- 代码：`/home/yqwang/projects/CASA-CD`
- 数据：`/share_datasets/CD/<DS>-CD-256/`
- 预训练：`pretrained_weight/shvit_s1.pth`（sha256 首16 `b7cf2c237ea01197`）
- 检查点：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-STR/Run1/<VARIANT>/<DS>/`
- 日志：`/home/yqwang/outputs/CASA-CD/CASA-STR/Run1/<VARIANT>/<DS>/train_log.txt`
- 诊断：`/home/yqwang/outputs/CASA-CD/diagnostics/CASA-STR/Run1/`
