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

- `models_and_metrics_CASA-MinorGoal_Run7.txt` —— 42 个代码文件 + 388 行指标的快照（含 wave 2 两条）。
- `CASA-MinorGoal_Run7_总体模型结构.pptx` —— 结构化框图。
- 原始日志与权重在服务器：`/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run7/<arm>/<dataset>/`，`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run7/`。

## 补跑（wave 2）：池化轴单因子

Run7 的 2×2 里两条带检索的臂同时也带着"流形上池化"，所以 2×2 回答不了"重心本身值不值"。补跑固定架构（`manifold + cg + lorentz aggregate`、边缘项关闭、seed 2334、val 选模、test τ=0.5），只切 `--prototype-pooling` 一根轴，SYSU 单库，协议 `casa_minor_run8_pooling_axis_v1`，2.77 h：

| arm | prototype pooling | F1 | Δ | R / P | IoU | best ep | val | 训练/部署 (M) |
|---|---|---:|---:|---|---:|---:|---:|---|
| `POOL_EUCLID` | 切空间均值 → lift | **83.212** | — | 80.50 / 86.11 | 71.25 | 17 | 81.59 | 2.1113 / 2.0617 |
| `POOL_LORENTZ` | 先 lift → Klein 重心 | 82.824 | **−0.387** | 79.46 / 86.48 | 70.68 | 25 | 81.72 | 2.1113 / 2.0617 |

- 两臂参数逐位相同（池化是纯几何操作），差异只能来自几何；裁决 **worse（≤ −0.36，噪声 0.32）**。
- 与 `EUCLID_ALL` 82.504 vs `EDGE_ALL` 82.258（聚合换欧氏 +0.246）方向一致：这套网络里欧氏均值不比 Klein 重心差。
- 复现噪声：`POOL_LORENTZ` 与 `TAR_HRA`（等价配置）差 0.19 pp、`POOL_EUCLID` 高 0.20 pp，故结论应写"无正效应、可能略负"。
- 机制同向：重心偏保守（P 86.48 > 86.11、R 79.46 < 80.50），召回掉 1.04 pp。

合起来这一轮的三句话：**模块有用（TAR +0.60 / HRA +1.14 / 联合 +1.63），但把压缩搬进双曲面没用（−0.39）；双曲几何的作用仍在打分，不在内容压缩。**
