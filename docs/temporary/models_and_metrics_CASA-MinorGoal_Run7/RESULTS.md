# CASA-MinorGoal / Run7 结果

协议 `casa_minor_run7_manifold_pooling_v1`，seed 2334，val 选模（`--selection-split val`），test 阈值固定 0.5，GPU1 两槽，8 项约 10.8 h。

## 两个改动

1. **流形上压缩原型**：`--prototype-pooling lorentz`。K/V 不再先在切空间做加权平均再 `lift`，而是把每像素特征先 `lift` 到 Lorentz 模型，再按类条件权重求加权重心（Klein 重心），得到严格落在双曲面上的原型。零参数增量。
2. **删掉 TAR 的 `β`**：差异特征以单位权重进入融合（`fused = block(content*(1+gate) + delta)`），不再有"差异→共有"之间的可学习标量。每个 `cg` 臂少 4 个参数。

## 2×2（两库）

| arm | module | tar | pooling | SYSU F1 | Δ | LEVIR F1 | Δ | train / deploy (M) |
|---|---|---|---|---:|---:|---:|---:|---|
| `CTRL` | none | reparam | euclid | 81.392 | — | 90.885 | — | 2.0471 / 1.9358 |
| `TAR_ONLY` | none | cg | euclid | 81.995 | +0.603 | 91.095 | +0.209 | 2.0935 / 2.0438 |
| `HRA_ONLY` | manifold | diff | lorentz | 82.533 | +1.141 | 90.851 | −0.035 | 1.9618 / 1.9234 |
| `TAR_HRA` | manifold | cg | lorentz | **83.016** | **+1.625** | 90.713 | −0.172 | 2.1113 / 2.0617 |

逐臂 R / P 与选模落点：

| arm | SYSU R / P | ep | val | LEVIR R / P | ep | val |
|---|---|---:|---:|---|---:|---:|
| `CTRL` | 77.88 / 85.24 | 25 | 80.68 | 89.86 / 91.93 | 134 | 91.38 |
| `TAR_ONLY` | 77.50 / 87.05 | 62 | 81.41 | 90.70 / 91.49 | 172 | 91.49 |
| `HRA_ONLY` | 79.18 / 86.19 | 21 | 81.12 | 90.00 / 91.71 | 175 | 91.46 |
| `TAR_HRA` | 78.85 / 87.65 | 25 | 81.92 | 89.36 / 92.10 | 112 | 91.36 |

## 读法

- SYSU 三条全过预登记门（TAR 0.36 / HRA 0.36 / 联合 0.47，噪声 0.32 pp）。HRA 最干净（3.6 倍噪声），TAR 只比噪声高 0.28 pp。
- **不可加**：0.603 + 1.141 = 1.744，实测 1.625，少 0.12 pp。机制也不同——TAR 是纯精度（P +1.81、R −0.38），HRA 是纯召回（R +1.30、P +0.95）。
- LEVIR 三条全在噪声内，与 Run6 一致：有效域只有 SYSU。
- 本轮 `CTRL` 比 Run2 同 seed 的 `CTRL_S2334`（82.092）低 0.70 pp，因此 `83.016` 不能与历史上 83.37（Run3 `ARM_C`）直接比；四个 Δ 是同 run 配对差。

## 文件

- `models_and_metrics_CASA-MinorGoal_Run7.txt` —— 42 个代码文件 + 386 行指标的快照（`tools/models_to_txt.py`）。
- `CASA-MinorGoal_Run7_总体模型结构.pptx` —— 结构化框图。
- 原始日志与权重在服务器：`/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run7/<arm>/<dataset>/`，`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run7/`。
