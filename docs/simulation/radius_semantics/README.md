# radius_semantics —— "双曲原点近/远"仿真套件

本目录回答一个问题：**我们模型里双曲模块的点，离原点近/远到底意味着什么？**
答案写在 [`results/SUMMARY.md`](results/SUMMARY.md)（先看这个），逐条数值见 `results/log_s*.txt`。

与上一套 `../hyperbolic/` 的关系：那套是"机制是否可行"的可行性仿真（H1–H5）；
这套是**语义 + 归因 + 判定**：几何量在我们自己的实现和真实数据上究竟承载了什么信息，
以及"想改码的话，哪条路还有钱可赚"。

## 目录结构

```
radius_semantics/
├─ run_all.m                入口：依次跑 S1–S6，写 results/log_all.txt
├─ rs_setup.m               工具集（曲率 -1、行向量、零工具箱）：几何/池化/统计/绘图
├─ s1_geometry.m            纯几何：近/远意味着什么，能分辨什么
├─ s2_our_radius.m          我们模块的半径由什么构成（LN、qk 奇异值、池化）
├─ s3_datasets.m            四个真实数据集上的半径语义（LEVIR/SYSU/WHU/CDD）
├─ s4_hierarchy_pooling.m   层级/具体性、五种池化算子、检索效果、径向重锚
├─ s5_decision_gate.m       改码判定门：翻转率↔半径 cv、收益上限、噪声地板
├─ s6_radius_knobs.m        半径归因：四个旋钮各能换来什么、代价多少
└─ results/
   ├─ SUMMARY.md            ★ 结论汇总（带表）
   ├─ decision_table.md     改码决策表
   ├─ log_s1..s6.txt / log_all.txt   每个脚本的完整数值（UTF-8，可直接看）
   ├─ res_s1..s6.mat        中间变量
   └─ figures/fig_s1..s6_*.png      10 张图
```

## 依赖与约定

- MATLAB R2022a 实测通过；只用到 base MATLAB（无 Image Processing / Statistics 工具箱）。
  盒滤波、AUC、Spearman、bootstrap 全部在 `rs_setup.m` 里自己实现。
- 曲率固定 −1；所有点用**行向量**（N×d）；半径 `ρ = asinh(‖z‖)`，Lorentz 点 `x = (√(1+t²), z)`；
  精确交叉测距 `cosh d = √(1+tq²)√(1+tk²) − zq·zkᵀ`。
- 随机种子在每个脚本里固定（`rng(n,'twister')`），结果可复现。
- 中文字体用 Microsoft YaHei（`R.zh()`）；MATLAB 控制台若显示乱码是 Windows 代码页问题，
  以 `results/log_s*.txt`（UTF-8）为准。

## 三个必须知道的实现事实（后面所有结论都建立在它们上）

1. `HyperbolicContext.geometric_point` 里 `lift(projected/sqrt(head_dim))` 直接提升，
   **不做任何径向裁剪**，所以半径 = `asinh(‖qk(x)‖/√32)`，与"范数"是同一件事。
2. LayerNorm 在 qk 之前，于是半径**几乎完全免疫**输入强度/对比度（eps=1e-5 带来的残留 ~1e-3）。
3. 键是 cell 内像素的 Lorentz 均值（`manifold_pool`），方向被平均掉 ⇒ 实测 K 半径 ≈ 0.7×Q 半径，
   这一缩水由"池内方向相关 c"控制（`A = √(c+(1−c)/n)`），与池大小 n 几乎无关。

## 一句话复现

```matlab
cd docs/simulation/radius_semantics
run_all
```

预计 4 分钟（其中 s3 占 3.5 分钟，要读 240 张真实影像）。
