# STR-Fusion Run2_TASS（Run11：Task-Adaptive Spatial Stem）

> 设计文档：`docs/temporary/CASA-CD_下一步方案_Run11_TASS_设计与预注册.md`
> 状态：**TASS-D0 零训练 raw source gate 阶段**（gate 不过不实现/不训练任何 80K）

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
