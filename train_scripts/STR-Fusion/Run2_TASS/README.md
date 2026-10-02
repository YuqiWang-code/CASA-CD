# STR-Fusion Run2_TASS（Run11：Task-Adaptive Spatial Stem）

> 设计文档：`docs/temporary/CASA-CD_下一步方案_Run11_TASS_设计与预注册.md`
> TASS-D0 记录：`docs/temporary/CASA-CD_Run11_TASS-D0结果与Run11终止.md`（含用户决策附录）
> 状态：**训练执行中（用户决策：取消 gate 拦截、双卡可用、必须训练）**；
> TASS-D0 FAIL 保留为已知负证据，不再作为停止条件。

## 正式结果（只认最后完整 TEST RESULTS 区块；全部从头 80K）

| 数据集 | C0_TOKEN F1 | M1_TASS F1 | ΔF1(M1−C0) | M1 Recall/Prec/IoU | deploy(M)/FLOPs(G) | fold disagree |
|---|---|---:|---:|---|---|---|---|
| SYSU | 0.8215 | 0.8174 | **−0.41pp** | 0.7873/0.8498/0.6911 | 3.455 / 3.546 | 5.72e-6 |
| LEVIR | 0.8779 | **0.9016** | **+2.37pp** | 0.8842/0.9197/0.8208 | 3.455 / 3.546 | 0 |
| WHU | 训练中 | 训练中 | — | — | — | — |
| CDD | 队列排队 | 队列排队 | — | — | — | — |

- **LEVIR +2.37pp 且 Recall/Precision 双升**（+2.05/+2.72pp）：TASS 首个强正面
  结果，0.858M 参数换真实证据增益（α=1.48/0.89/0.51）；SYSU −0.41pp（轻微
  置信度锐化）——TASS 价值 dataset-dependent，最强在建筑小目标（LEVIR）。
- 硬目标差距：SYSU 82.15（−2.85）/ LEVIR 90.16（−2.34）；C0 token-only 本身
  低于冻结 A1 锚点（LEVIR 87.79 vs 91.84）。
- 硬条件：VIT checksum 不变 ✓、deploy 3.455M ≤5M ✓、二值化 disagreement≈0 ✓。

## 训练布局（双卡串行队列，全部从头 80K）

- `C0_TOKEN/`（spatial_mode=token，GPU1）：SYSU ✓ → LEVIR ✓ → WHU → CDD
  （`run_C0_queue.sh`）；
- `M1_TASS/`（spatial_mode=tass，GPU0）：SYSU ✓ → LEVIR ✓ → WHU → CDD
  （`run_M1_queue.sh`）；
- 正式结果只认各 `train_log.txt` 最后一个完整 `=== TEST RESULTS ===` 区块；
  deploy ≤5M 硬门槛；禁止 checkpoint 微调（崩溃恢复仅限本任务 last.pth）。
- 结果解释口径：M1 增益必须在「TASS-D0 G1=1/4（仅 SYSU 正）」背景下陈述。

## TASS-D0 裁决（零训练 raw source gate，四数据集）

- G0 审计有效性 **PASS**（B4 边界带与 SF-D0 四位小数逐位复现：0.5216/0.5369/0.6113/0.5718）；
- G1（Rfuse 边界带 lift ≥ +0.020，SYSU 必过且 ≥2/4）**1/4 FAIL**：
  SYSU +0.0309 ✓ / CDD −0.0075 / LEVIR −0.0409 / WHU −0.0297；
- G2（Rfuse 像素 ≥ B4−0.010，SYSU 必过且 ≥3/4）**2/4 FAIL**：SYSU/WHU 过，
  CDD −0.0346 / LEVIR −0.0032。
- **裁决**：未训练的 raw 像素差分证据只在 SYSU 与 B4 互补，三个建筑数据集上被
  辐射伪变化主导 → 不实现/不训练 TASS；按方案 §16 锁定 fallback：**不再开 Run12
  救援线，下一轮直接进入分析型论文收尾设计（方向 c）**。
- 审计工具存档：`analyse/run11_tass_source_gate.py` + `audit_TASS_D0.sh`。

## 方案一句话

冻结 ViT4（语义锚点）+ 固定 TAR/DCR（full rep）+ 新增共享 Siamese 的极小可训练
空间 stem（1/4–1/16 三尺度，zero-init α 残差注入到 B1↑4/B2↑2/B3），唯一变量 =
是否存在 task-adaptive spatial residual。C0（token-only）与 M1（TASS）都从头
完整 80K；deploy ≤5M 硬门槛；四数据集 F1 硬目标 85/92.5/95/98。

## 预注册 gate

- **TASS-D0（零训练，先于实现）**：G0 审计有效性（B4 边界带复现 SF-D0 ±0.01 /
  ViT4 checksum / 样本数）；G1 PRbnd(Rfuse) ≥ PRbnd(B4)+0.020（SYSU 必过且 ≥2/4）；
  G2 PRpix(Rfuse) ≥ PRpix(B4)−0.010（SYSU 必过且 ≥3/4）。任一 FAIL → 转方向 (c)。
- **T-S11-1..6（smoke，D0 PASS 后才实现）**：shape / C0/M1 epoch-0 逐位一致 /
  α 两步梯度链 / 冻结 checksum / rep 折叠回归 / 预算 ≤5M。
- **SYSU 决策（C0 80K → M1 80K，串行，不并行）**：
  PASS = M1 F1≥85.00 且 M1−C0≥+0.30pp 且 IoU 提升且 deploy≤5M 且 argmax-disagree=0；
  WEAK-A = 84.50≤F1<85.00 且 Δ≥+0.30pp；WEAK-B = F1≥85.00 但 0≤Δ<+0.30pp；
  FAIL = 其余（含 confidence-sharpening 签名）。非 PASS → 停，不救。
- **扩展（仅 SYSU PASS）**：LEVIR C0+M1（≥92.5，Δ≥+0.20pp）→ WHU C0+M1
  （≥95，Δ≥+0.15pp）→ CDD C0+M1（≥98，Δ≥+0.15pp）；每数据集两个完整 80K。
- 正式 80K 上限 = 8 个（四数据集 × C0/M1）；禁止 checkpoint 微调（崩溃恢复仅限
  本任务自身 last.pth）；阈值执行后不改。

## 目录

```
train_scripts/STR-Fusion/Run2_TASS/
├── README.md
├── audit_TASS_D0.sh
├── dryrun_SYSU.sh            # D0 PASS 后
├── C0_TOKEN/{SYSU,LEVIR,WHU,CDD}/train.sh   # D0 PASS 后
└── M1_TASS/{SYSU,LEVIR,WHU,CDD}/train.sh    # D0 PASS 后
```

- checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/STR-Fusion/Run2_TASS/<group>/<dataset>/`
- 日志：`/home/yqwang/outputs/CASA-CD/STR-Fusion/Run2_TASS/<group>/<dataset>/train_log.txt`
- 诊断：`/home/yqwang/outputs/CASA-CD/diagnostics/STR-Fusion/Run2_TASS/`
- 结果纪律：只读最后完整 `=== TEST RESULTS ===` 区块 + arch.json/run_manifest.json 校验。
