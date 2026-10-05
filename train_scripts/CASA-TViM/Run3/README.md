# CASA-TViM Run3 — SYSU 小目标瓶颈定向（RA-CAACP + FS-TAR）

Run2 复盘（[`docs/temporary/CASA-CD_Run2复盘_SYSU小目标瓶颈与下一轮结构实验设计_2026-10-05.md`](../../../docs/temporary/CASA-CD_Run2复盘_SYSU小目标瓶颈与下一轮结构实验设计_2026-10-05.md)）核心结论：

1. Run2 的 E1/E2/E3 在 SYSU 上都是 **Recall 被压掉 3.5~4.3pp**（Precision 反升）——不是优化冲突（交互项全非负），而是**机制/尺度错配**。
2. SYSU 最大单一缺口 = **small components（<256px）F1 0.1796**；这类目标在 1/16 已是 **sub-token**（<1 feature cell），Stage3 CAACP 对 medium/large 有效（+0.047/+0.036）但 small 微负（−0.004）——教科书式 context scale mismatch。
3. FRH 在末端 128² 只能"修图"不能"恢复 1/16 之前丢掉的极小目标语义"。

## 变体（第一批，各改一个创新，单变量）

| 变体 | 改动 | deploy 参数 | 预注册条件（复盘文档 §13） |
|---|---|---|---|
| E4_RA_CAACP | CAACP residual 从 `x−Up(c)` 改为 **`x−Up(c_avg)`**（change-aware c 只进 SS2D、不再从 dense residual 扣除高频证据） | **0** | CDD≥97.18、WHU≥95.07、SYSU F1>83.47、SYSU Recall≥84.41、SYSU small≥0.195、LEVIR 不明显低于 91.07 |
| E5_FS_TAR | stage1（1/4）TemporalRep1x1 → **TemporalRepFine3x3**（signed-diff 分支变 3×3，在 tiny target 仍可解析的 1/4 尺度先提取双时相空间邻域差异） | **+73,728 ≈ 4.954M** | CDD≥97.18、WHU≥95.07、SYSU F1≥83.80、Recall>84.41、small≥0.205、LEVIR≥91.20 |

两变体均：β/aux zero-init → epoch-0 与 M1 逐位一致；deploy 折叠为单卷积；不改 loss/增广/阈值。

## 波次与并行

- Wave1 = E4_RA_CAACP ×4 → Wave2 = E5_FS_TAR ×4
- 每波 4 数据集并行：GPU0=CDD+LEVIR、GPU1=SYSU+WHU（每卡 2 并发，batch 32）
- `run_all.sh` 串完 8 个 80K；任一 run 3 次 retry 失败 → WAVE-ABORT

## 失败裁决（复盘文档 §13）

- E4：SYSU small<0.190 或 SYSU F1≤83.47 → 停 RA，不再调 β/叠 CP → 转 S2-HCAACP
- E5：SYSU small 提升 <+0.015 或 Recall 不升 → 不再扩大 spatial kernel → SP-DCR / S2-HCAACP
- 只有 E4/E5 各自单变量成立才允许组合（不复刻 Run2 的"单项弱仍组合"）

## 产物位置

- 代码：`models/model/layers/caacp_ss2d.py`（residual_mode）、`models/model/str_tar.py`（TemporalRepFine3x3）、`models/model/casa_tvim_str_net.py`（接线）
- checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run3/<VARIANT>/<DS>/`
- 日志：`/home/yqwang/outputs/CASA-CD/CASA-TViM/Run3/<VARIANT>/<DS>/train_log.txt`

## smoke

```
cd models && python smoke_test.py --pretrained_weight_path <deit pth> \
  --tinyvim_pretrained_weight_path <tinyvim_s_1000e.pth> --mode casa_tvim_str
```
T-CA-9（RA：β=0 avg_anchor==current 逐位 + β 梯度链）、
T-CA-10（FS-TAR：zero-init 位同、fold 单 3×3、deploy 4.954M、init fold<1e-4、disagree==0）必须全过。
