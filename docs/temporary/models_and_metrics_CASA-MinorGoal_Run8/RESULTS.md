# CASA-MinorGoal / Run8 结果

协议 `casa_minor_run8_pooling_axis_v1`，只切**一根轴**：原型在哪里做加权平均。两臂 `module=manifold`、`tar=cg`、`aggregate=lorentz`、边缘项关闭、seed 2334、val 选模、test τ=0.5，SYSU 单库，GPU1 两槽，2.77 h。

| arm | prototype pooling | F1 | Δ | R / P | IoU | best ep | val | 训练/部署 (M) |
|---|---|---:|---:|---|---:|---:|---:|---|
| `POOL_EUCLID` | 切空间加权平均 → 再 lift | **83.212** | — | 80.50 / 86.11 | 71.25 | 17 | 81.59 | 2.1113 / 2.0617 |
| `POOL_LORENTZ` | 每个成员点先 lift → 加权 Klein 重心 | 82.824 | **−0.387** | 79.46 / 86.48 | 70.68 | 25 | 81.72 | 2.1113 / 2.0617 |

`compare.py` 裁决：**worse（≤ −0.36）**，即"把压缩搬进双曲空间"没有收益，反而比切空间均值低 0.39 pp（门限 0.36、噪声 0.32，刚好越过）。

## 三条读法

1. **压缩位置不是杠杆**。两臂参数逐位相同、只有池化算子不同，差 −0.387 pp 落在"更差"一侧。结合 Run7 的 `EUCLID_ALL` 82.504 vs `EDGE_ALL` 82.258（+0.246，当时是把**聚合**换成欧氏，噪声内），两次实验方向一致：在这套网络里，欧氏均值至少不比 Klein 重心差。
2. **复现检查同时给出噪声尺度**。`POOL_LORENTZ` 与 Run7 的 `TAR_HRA` 配置等价：82.824 vs 83.016，差 0.19 pp；而 `POOL_EUCLID`（83.212）比 Run7 的 lorentz 高 0.20 pp。也就是说跨 run 的漂移约 0.2 pp，本轮的 −0.387 有约一半处在这个漂移里，只能说"没有正效应、可能略负"。
3. **机制线索与 Run6 一致**。重心把判定推向更保守（P 86.48 > 86.11，R 79.46 < 80.50），和 `EUCLID_ALL` 那次"均值高召回、重心高精度"的换位完全同向。代价是召回掉了 1.04 pp，F1 净负。

## 结论（可直接写进文档）

> 在压缩阶段把原型平均搬到双曲面上，不带来精度收益；SYSU 上比切空间均值低 0.39 pp，且与 Run7 的 2×2 里"模块有用、几何中性"一致。双曲几何在这套结构里的作用仍只限于打分，把几何推进到内容压缩并没有改变这一点。

## 文件

- `models_and_metrics_CASA-MinorGoal_Run8.txt` —— 42 个源码文件 + 388 行指标快照。
- 原始日志/权重在服务器：`/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run8/<arm>/SYSU-CD-256/`、`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run8/`。
