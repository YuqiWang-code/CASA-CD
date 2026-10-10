# CASA-TViM Run3 结果对比（自动生成 + D2 复测裁决）

基线：Run1 M1_FULL（CAACP rank + STR，deploy 4.880M）。E4 deploy 4.880M；E5 deploy 4.954M。

## D2 复测（SYSU small component，预注册机制条件裁决）

| 变体 | small P/R/F1 | medium F1 | large F1 | band2 | 预注册 small | 裁决 |
|---|---|---|---|---|---|---|
| M1_FULL | — / — / 0.1796 | 0.4620 | 0.8273 | 0.6710 | — | 基线 |
| E4_RA_CAACP | 0.2561 / 0.1663 / **0.1842** | 0.4647 | 0.8058 | 0.6506 | ≥0.195 | **✗（+0.0046，方向微弱正确但幅度不足）** |
| E5_FS_TAR | 0.2378 / 0.1477 / **0.1652** | 0.4227 | 0.7919 | 0.6480 | ≥0.205 | **✗（−0.0144，small Recall 反而更低）** |

## 最终裁决（§13 全条件）

**E4_RA_CAACP = FAIL**：CDD 97.13（✗）、WHU 95.02（✗）、SYSU 83.16（✗）、Recall 81.44（✗）、small 0.1842（✗）→ 按决策树**停 RA 线**，不再调 Stage3 residual / β / 叠 CP。
**E5_FS_TAR = FAIL**：CDD 97.13（✗）、WHU 94.91（✗）、SYSU 83.24（✗）、Recall 81.26（✗）、small 0.1652（✗）、LEVIR 91.18（✗ 差 0.02）→ 按决策树**不再扩大 spatial kernel**，转 SP-DCR / S2-HCAACP。

关键负结果：
- RA 确实把 SYSU small 从 0.1796 微抬到 0.1842（所有改动中最高），但以整体 Recall −3pp 为代价——
  推翻"residual cancellation 是 small 主因"假设：change-aware c 进入 residual 恰恰是 CAACP 高 Recall 的来源。
- FS-TAR 的 1/4 尺度 3×3 signed-diff 未恢复 tiny evidence（small R 0.1477 反而最低）——
  空间核扩大本身不是解，问题在更早的语义编码层。
- 五轮结构尝试（E1~E5）稳定模式：LEVIR 0~+0.22（上限 +0.25 内）、SYSU 全部 −0.23~−0.62；M1 仍是 SYSU 最优。

## 下一候选（决策树指向）

- **S2-HCAACP**：CAACP 从 Stage3(1/16) 前移到 Stage2(1/8)（32²→CA 2×2→16²→uniform 2×2→8²，保持 8² context lattice 与 β=0 预训练等价）；small 在 1/8 有 <4 cells 仍可解析。先验 SYSU +0.25~+0.65。
- **SP-DCR**：删除 DCR 末端第二个 DW3（Pointwise-only refine），减少末端 spatial mixing 对弱正响应的压制。先验 SYSU +0.15~+0.45。
- 两者都在"保护小目标正证据"主线上；不叠模块、各自单变量。

- M1 CDD-CD-256: F1=97.18 IoU=94.51 R=97.03 P=97.33
- M1 LEVIR-CD-256: F1=91.07 IoU=83.60 R=90.33 P=91.82
- M1 SYSU-CD-256: F1=83.47 IoU=71.63 R=84.41 P=82.55
- M1 WHU-CD-256: F1=95.07 IoU=90.60 R=93.56 P=96.62

## E4_RA_CAACP

| 数据集 | M1 F1/IoU | E4_RA_CAACP F1/IoU | ΔF1 | R/P | 硬条件 | β | disagree | deploy(M) |
|---|---|---|---|---|---|---|---|---|---|
| CDD-CD-256 | 97.18 / 94.51 | 97.13 / 94.42 | -0.05 | 97.02 / 97.25 | ✗ 97.13>=97.18 | 1.76e-02 | 0.00e+00 | 4.880 |
| LEVIR-CD-256 | 91.07 / 83.60 | 91.07 / 83.60 | +0.00 | 90.05 / 92.12 |  | 2.06e-02 | 0.00e+00 | 4.880 |
| SYSU-CD-256 | 83.47 / 71.63 | 83.16 / 71.17 | -0.31 | 81.44 / 84.94 |  | 6.74e-03 | 0.00e+00 | 4.880 |
| WHU-CD-256 | 95.07 / 90.60 | 95.02 / 90.50 | -0.05 | 93.42 / 96.66 | ✗ 95.02>=95.07 | 6.54e-03 | 0.00e+00 | 4.880 |

预注册机制条件（SYSU F1>83.47、SYSU Recall≥84.41、SYSU small F1≥0.195（D2 复测））→ 需 D2 复测（python analyse/run2_zero_cost_diag.py --ckpt_run Run3 --variants E4_RA_CAACP --datasets SYSU-CD-256）裁决；LEVIR 不显著低于 91.07。

## E5_FS_TAR

| 数据集 | M1 F1/IoU | E5_FS_TAR F1/IoU | ΔF1 | R/P | 硬条件 | β | disagree | deploy(M) |
|---|---|---|---|---|---|---|---|---|---|
| CDD-CD-256 | 97.18 / 94.51 | 97.13 / 94.43 | -0.05 | 97.03 / 97.24 | ✗ 97.13>=97.18 | 1.04e-02 | 9.54e-07 | 4.954 |
| LEVIR-CD-256 | 91.07 / 83.60 | 91.18 / 83.78 | +0.11 | 90.25 / 92.12 |  | 1.79e-02 | 0.00e+00 | 4.954 |
| SYSU-CD-256 | 83.47 / 71.63 | 83.24 / 71.29 | -0.23 | 81.26 / 85.31 |  | 1.69e-02 | 0.00e+00 | 4.954 |
| WHU-CD-256 | 95.07 / 90.60 | 94.91 / 90.31 | -0.16 | 93.11 / 96.78 | ✗ 94.91>=95.07 | 9.79e-03 | 0.00e+00 | 4.954 |

预注册机制条件（SYSU F1≥83.80、SYSU Recall>84.41、SYSU small F1≥0.205（D2 复测））→ 需 D2 复测（python analyse/run2_zero_cost_diag.py --ckpt_run Run3 --variants E5_FS_TAR --datasets SYSU-CD-256）裁决；LEVIR 不显著低于 91.2。
