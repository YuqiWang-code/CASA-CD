# STRFusion Run1（CASA-CD × STR-RepNet 融合主线）

> 设计文档：`docs/temporary/CASA-CD_STR融合_Run1_设计与预注册方案.md`
> 状态：**代码实现 + 审计阶段**（SF-D0 gate 未裁决前不允许启动任何 80K）

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
