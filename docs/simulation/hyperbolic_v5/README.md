# hyperbolic_v5 —— 本轮（V4 复核 → S7c）产物索引

对应任务书：`docs/temporary/CASA-CD_V4仿真复核与下一轮实验执行方案.md`（基准 main@`38c7342`）。
本目录只放**已跑过、可复现**的结果；未做的条款在 `CORRECTIONS_V5.md` 里标"未做"。

## 文件 ↔ 任务书条款

| 文件 | 对应条款 | 状态 |
|---|---|---|
| `CORRECTIONS_V5.md` | A5 纠错表 + A0/A1/A2 状态 + 历史表述改写 | ✅ |
| `S7C_REPORT.md` | S7c-A（探针）、S7c-B（半径/池化）、对旧结论影响 | ✅ |
| `results/runtime_inventory.md` | A0 冻结来源核查 | ✅ |
| `results/s7c_parity_note.md` | A2 parity 补齐说明 | ✅ |
| `results/s7c_probe.csv` | 探针 AUC/AP（4 checkpoint × 4 尺度 × 7 探针） | ✅ |
| `results/s7c_paired.csv` | 配对 ΔAUC + 图像级 bootstrap CI | ✅ |
| `results/s7c_radius.csv` | 每图每尺度半径分布（Q/support） | ✅ |
| `results/s7c_pool.csv` | 每样本每区域每类的池化统计（K/V） | ✅ |
| `tools/simulation/s7c_capture.py` | S7c 采集脚本（服务器运行） | ✅ |
| `tools/simulation/s7c_aggregate.py` | 把服务器 CSV 汇总到本目录 | ✅ |
| S9a / S8b / S9b / S13a-0b / S13a-1 / A3 / A4 | 见 `CORRECTIONS_V5.md` §0 | ❌ 未做 |

## 复现

```bash
# 1) 服务器（env casacd），每个 (arm, dataset) 约 5 分钟
cd /home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/runtime/tools_simulation
S7B_THREADS=6 python -u s7c_capture.py --arm R_FULL-2334 --dataset SYSU-CD-256 \
  --val-limit 60 --train-limit 32 --parity 0 --probe-cap 40000
# 2) 本地：把 analysis/simulation/S7c/<arm>_<dataset>/*.csv 拉到 .tmp/s7c/ 后
python tools/simulation/s7c_aggregate.py
```

## 三条口径提醒（读数字前先看）

1. **半径只有一种正确写法**：`rho = asinh(‖z‖)`，z 是真正送入 `lift` 的坐标（Q/K 已除 √d
   一次，V 不除）。`asinh(‖a‖/√d)` 只能用于**未缩放**的原始投影。
2. **Q 与 K 的半径不能相除**：两者来自不同流（T_s 与 LN_diff(D_s)），"K/Q 比"没有物理意义。
   能比的是同一区域的"原型半径 / 成员加权半径"。
3. **本目录的数字都带 sample_id/scale 粒度**，可以重算任何子集；`S7C_REPORT.md` 里的表
   是两 seed 的平均，逐 seed 的 CI 在 `s7c_paired.csv`。
