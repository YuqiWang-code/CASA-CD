# CASA-TViM Run2 — LEVIR/SYSU 定向提升（CP-CAACP + FRH）

Run1（16/16 完成）基线（F1/IoU，best-pth 测试集）：

| 变体 | CDD | LEVIR | SYSU | WHU |
|---|---|---|---|---|
| M1 (CAACP+STR) | 97.18 / 94.51 | 91.07 / 83.60 | 83.47 / 71.63 | 95.07 / 90.60 |

目标：保持 CDD ≥ 97.18、WHU ≥ 95.07，同时把 LEVIR 91.07 → ≥ 92、SYSU 83.47 → ≥ 84。

## 变体（调研文档《CASA-CD_Run2_LEVIR_SYSU定向提升…》执行顺序 E1 → E2 → E3）

| 变体 | caacp | score_mode | frh | 说明 |
|---|---|---|---|---|
| E1_CP_CAACP | 1 | cp | 0 | **首选一**：CAACP 权重从 `w∝ε+rank` 改为 `w∝1+s·r`（置信度保留）。零变化像素 q=0 → 全零变化 cell 严格退化为均匀池化，修复 rank-only 在零变化图上的强制不均匀（LEVIR Precision 拖低主因）。**0 新参数**。 |
| E2_FRH | 1 | rank | 1 | **首选二**：输出头从 64² 1×1 升为 128² 重参数化细粒度头（up2 → base 1×1 + γ·[RepDW3→1×1]，γ=0 初始化），deploy 折叠为单 3×3 Conv(96→1)，**+768 参数**（4,880,958 总） |
| E3_CP_FRH | 1 | cp | 1 | 组合（仅当 E1/E2 各自有效） |

全部：rep_mode=full（TAR/DCR）、backbone_lr_ratio 0.1、batch 32、seed 16、80K steps、
BCE+Dice、Adam(2e-4)、poly(0.9)+200warmup、threshold 0.5、test-as-val。与 Run1 协议逐项一致。

## 波次与并行

- Wave1 = E1_CP_CAACP ×4 → Wave2 = E2_FRH ×4 → Wave3 = E3_CP_FRH ×4
- 每波 4 数据集并行：GPU0=CDD+LEVIR、GPU1=SYSU+WHU（每卡 2 并发，batch 32，~19GB/卡）
- `run_all.sh` 一个脚本串完全部 12 个 80K；任一 run 3 次 retry 后仍失败 → 整波 WAVE-ABORT

## 产物位置

- 代码：`models/model/layers/caacp_ss2d.py`（CP 公式）、`models/model/str_fine_head.py`（FRH）、
  `models/model/casa_tvim_str_net.py`（`--caacp_score_mode` / `--frh` 接线）
- checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run2/<VARIANT>/<DS>/`
- 日志：`/home/yqwang/outputs/CASA-CD/CASA-TViM/Run2/<VARIANT>/<DS>/train_log.txt`

## smoke（上服务器后、开训前）

```
cd models && python smoke_test.py --pretrained_weight_path <deit pth> \
  --tinyvim_pretrained_weight_path <tinyvim_s_1000e.pth> --mode casa_tvim_str
```
T-CA-7（CP 公式：β=0 rank/cp 逐位一致、零变化退化均匀池、手算权重对照）、
T-CA-8（FRH：γ=0 修正精确 0、折叠单 3×3 +768、init fold <1e-4、disagree==0）必须全过。

## 注

- 80000 步的"精确步数"论证（调研文档 P0）不严格要求：best-pth 即有效最终结果。
- E1 与 M1 仅差 score 公式（head/decoder 初始化逐位相同）；E2 与 M1 仅差 head 结构。
