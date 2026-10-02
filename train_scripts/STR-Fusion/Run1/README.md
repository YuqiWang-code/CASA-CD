# STRFusion Run1（CASA-CD × STR-RepNet 融合主线）

> 设计文档：`docs/temporary/CASA-CD_STR融合_Run1_设计与预注册方案.md`
> 终止记录：`docs/temporary/CASA-CD_STR融合_Run1_SF-D0结果与融合主线终止.md`
> 状态：**SF-D0 gate FAIL → 已按预注册终止（0 个 80K，未跑 dry run）**

## SF-D0 裁决（零训练接口 gate，四数据集）

- G0 审计有效性 **PASS**（R4-1 对照精确复现；B4 边界带与 R8-D0 四位小数逐位一致）；
- G1（fuse4 边界带 lift ≥ +0.02，SYSU 必过且 ≥2/4）**0/4 FAIL**：
  CDD +0.0100 / SYSU +0.0089 / LEVIR −0.0061 / WHU −0.0090；
- G2（fuse2 ≥ B4−0.02，≥3/4）3/4（仅 WHU −0.0273）。
- **裁决：多深度 token 金字塔携带严格多于 B4-only 边界带证据的假设被四数据集证伪 →
  不启动任何 80K**。冻结 plain ViT 的 token 流无论取多少深度都受 inner-patch 局限
  （R8-D0「ranking ≠ dense 重建」的多深度推广）。
- 补救方向（可训练 stem / 受限解冻 / 放弃融合线）需**重新预注册**，本轮不自动执行。

## 留存资产（可复用于后续预注册）

- 全套实现：`models/model/str_{reparam,tar,dcr,encoder,fusion}.py`；
- 等价性：T0/T1/T2/T2b ALL PASSED（活分支全模型折叠 5.1e-7/5.4e-7、disagree=0）；
- smoke：T-SF-1..8 全过（含 C0/M1 epoch-0 逐位一致、RNG 纪律落地）；
- 预算 G4 PASS：deploy **2,596,353（2.596M < 3M）** / FLOPs 2.2136G，C0/M1 deploy 相等；
- 审计工具：`analyse/run1_strfusion_{budget,interface_audit}.py`。

## 方案一句话

冻结 ViT4 的四个 block 状态 B1/B2/B3/B4 经固定无参数重采样组成四尺度二时相金字塔
（64/32/16/8），接 STR-RepNet 的 TAR 二时相 bridge（concat+sum+signed-diff → 单 1×1）
与 DCR 解码器（DW3/PW/fuse 全部可折叠），训练协议完全沿用 CASA-CD（BCE+Dice、
Adam 2e-4、poly 0.9、80000 steps、batch 16、256×256、seed 16、test-as-val、GPU1）。
部署折叠为单路径静态卷积（FP64 组合、一次 FP32 cast、二值化 disagreement = 0），
deploy EFFECTIVE ≈ 2.60M < 3M（机器审计为准）。

## 消融

| 组 | 定义 | deploy 图 |
|---|---|---|
| C0_Plain | `--str_rep_mode plain`（无 aux 分支，BN-FR α 保留） | 单路径 convs |
| M1_TAR-DCR | `--str_rep_mode full`（TAR+DCR aux 全开，零初始化） | 与 C0 同函数类（折叠后） |

归因：Rep = M1−C0（受控唯一变量）；System = M1−R4-1（历史 82.77，上下文参考）。

## 预注册判据

- **SF-D0（零训练，先于一切训练）**：G0 审计有效性（ViT4 checksum / SYSU R4-1 对照
  0.6535/0.5948±0.005 / B4 边界带 PR-AUC 复现 R8-D0 ±0.01）；G1 PRbnd(fuse4) ≥
  PRbnd(B4)+0.02（SYSU 必过且 ≥2/4）；G2 PRbnd(fuse2) ≥ PRbnd(B4)−0.02（≥3/4）；
  G4 预算 deploy ≤3M。G1/G2 任一不过 → **FAIL → 0 个 80K，主线终止**。
- **SF-1（SYSU 决策实验，C0+M1 两个 80K）**：
  - PASS：M1 F1 ≥ 0.8500 且 M1−C0 ≥ +0.10pp 且 deploy ≤3M 且二值化 disagreement=0；
  - WEAK：M1 ≥ 0.8500 但 rep 不可辨识（0 ≤ M1−C0 < +0.10pp）→ 不作 rep 创新主张，不扩展；
  - FAIL：M1 < 0.8500 或 M1−C0 < 0 或「置信度锐化」签名 → **停止，不救机制**。
- **SF-2（仅 SF-1 PASS）**：LEVIR M1（≥92.5，<91.84 停）→ WHU M1（≥95，<94.84 停）
  → CDD M1（≥98，<97.75 停；第 5 个 80K，前三阶段全 PASS 才启动）。
- 正式 80K 预算 = SYSU×2 + LEVIR×1 + WHU×1 = **4 个**。
- 禁止：D/深度/branch sweep、改 loss、threshold tuning、seed 重跑、增广改动。

## 目录

```
train_scripts/STR-Fusion/Run1/
├── README.md
├── audit_SF_D0.sh                 # 零训练接口 gate（四数据集）
├── dryrun_SYSU.sh                 # 60-step scratch dry run
├── C0_Plain/    train_{SYSU,LEVIR,WHU,CDD}-CD-256.sh
└── M1_TAR-DCR/  train_{SYSU,LEVIR,WHU,CDD}-CD-256.sh
```

- checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/STR-Fusion/Run1/<group>/<dataset>/`
- 日志：`/home/yqwang/outputs/CASA-CD/STR-Fusion/Run1/<group>/<dataset>/train_log.txt`
- 诊断：`/home/yqwang/outputs/CASA-CD/diagnostics/STR-Fusion/Run1/`
- 正式结果只认同一 train_log.txt 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块；arch.json sidecar 校验。
