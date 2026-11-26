# hyperbolic_v2 / radius_semantics_v2 —— 阶段 0 修正版

依据 `docs/temporary/CASA-CD_双曲几何与原型压缩_仿真实验完整执行方案.md`
（2026-11-26，rev dce3cdae…，基于 main@`1a42f6e`）的阶段 0 要求：
**先修正、再诊断、后设计**；不删不覆盖旧目录；不要求双曲必胜，负面结论同样有效。

- 旧版：`docs/simulation/hyperbolic/`（H1–H5）、`docs/simulation/radius_semantics/`（S1–S6），原样保留。
- 新版：本目录 `docs/simulation/hyperbolic_v2/`、`docs/simulation/radius_semantics_v2/`。
- **结论状态与新旧对照看 [`hyperbolic_v2/CORRECTIONS.md`](hyperbolic_v2/CORRECTIONS.md)**。

---

## 1. 目录

```
hyperbolic_v2/
  hyper_setup2.m        工具箱：距离/映射/池化/统计分开命名（不复用 v1 的 hyper_setup.m）
  c1_math_checks.m      16 项数学与几何单元检查（A1–A8, B1–B2, C1–C6）
  c2_retrieval.m        修正版 H3/H5：独立查询、四种归一化、同心壳分辨力
  c3_content_geometry.m 修正版 H2：内容 vs 几何的 2x2 公平对照（无监督原型）
  CORRECTIONS.md        v1 逐条结论状态 + 本次六类实现修正
  results/              log_c*.txt（UTF-8）、res_c*.mat
  figures/              fig_c*.png

radius_semantics_v2/
  r1_radius_anatomy.m   修正版 S1/S2：仿射 LN 上界、atanh 解析式、四种 CV、增益作用点
  r2_dataset_radius.m   修正版 S3：val 口径、固定抽样清单、总体方差、真·向量池化、A/B 对称校核
  results/              log_r*.txt、res_r*.mat、r2_sample_list.txt（固定抽样清单）
  figures/              fig_r*.png
```

约定：MATLAB R2022a，**不依赖任何工具箱**（`ttest`/`corr`/`boxplot` 均未使用；
AUC 用秩公式、Spearman 用平均秩、bootstrap 手写）。所有图用
`exportgraphics(...,'Resolution',150)`；console 抓回的中文可能是 GBK 乱码，
**以 `results/log_*.txt`（UTF-8）为准**。

## 2. 复现

```matlab
cd docs/simulation/hyperbolic_v2
c1_math_checks          % ~4 s
c2_retrieval            % ~10 s
c3_content_geometry     % ~60 s

cd ../radius_semantics_v2
r1_radius_anatomy       % ~3 s
r2_dataset_radius       % ~100 s（读 4 数据集 × 40 对真实影像）
```

## 3. 四件必须先知道的事

1. **两种半径口径**：`hyper_setup2.S.radius` 是内在半径 `asinh(||z||)`；
   模块里的半径是 `asinh(||z||/sqrt(D))`（D=head_dim）。两者只差一个固定的
   `sqrt(D)` 哑元缩放，但**混用会得出"采样超过理论上界"的假失败**（r1 踩过）。
2. **直接 Lorentz 提升 ≠ 标准指数映射**：`d(0,π(z))=asinh||z||` 与
   `d(0,exp_0(v))=||v||` 同时成立，且 `|z|=sinh|v|`。v1 注释把二者混同。
3. **OEM 命名以官方实现为准**：`OEM_α = Norm_L(Σ w_i t_i^α x_i)`，**α=0 就是
   标准 Einstein 中点，也就是当前 Lorentz 加权重心**（c1-C1：max|Δ|=0.0）。
   v1 的 `pool_einstein`（多乘一次 t）和 `pool_oem(p)`（多乘 p+1 次幂）实际对应
   官方 α=1/2/3。
4. **`atanh` 解析式只对等半径成员成立**：`ρ_pool = atanh(A tanh ρ)`，
   `A=||Σ w_i u_i||`；等权时 `A²=c+(1-c)/n`。要点是 **A 必须实测**——
   `c`（单方向与主轴的平均余弦）与"平均两两内积"在有限 n 下不是同一个量，
   用解析 c 代入会差 10–20 个百分点（r1 踩过，修正后偏差 1.4e-15）。

## 4. 本轮得到的三条硬结论

1. **几何自由度在真实半径区间里几乎不存在。** 独立查询下测地与欧氏的
   Spearman 恒为 1（`1−Spearman ≤ 1.7e-5`），JS 散度 ≤1.5e-3 且只在半径发散度
   被推到 cv≈1.3（远超模块实测 0.15–0.3）时才明显（c2）。
2. **半径大时是欧氏更会分辨半径，不是双曲。** 同心壳 AUC：a=3 时测地 0.507、
   z 空间欧氏 1.000、Poincaré 坐标欧氏 0.527；a=0.2 三者都≈0.88（c2 表 5）。
   原因是对 z 坐标
   `cosh d−1 = 2sinh²(Δρ/2) + sinhρ₁sinhρ₂(1−cosθ)`、
   `|Δz|² = (sinhρ₁−sinhρ₂)² + 2sinhρ₁sinhρ₂(1−cosθ)`：
   **角度项完全相同，只有径向项不同**（测地 ≈(Δρ)²/2，欧氏 ≈(Δρ·coshρ)²）。
   所以"测地适合表达层级"等价于"自动做半径归一化"——半径是噪声时占优，
   半径是信号时失效。
3. **内容效应是真实的，但比 v1 小一个量级。** 公平 2×2（无监督原型、共用场景、
   GT 只在算指标时用）：内容 D−T2 **+7.6～+9.4 pp**（CI 不含 0），
   几何 geo−euc **−1.0～+0.8 pp**（CI 含 0）。v1 的 +22.84 pp 里约 +12.9 pp
   来自"用 GT 造原型 + 同场景评估"（c3 表 4）。

## 5. 数据集侧（r2，val 口径）

- 半径与"是否变化"无关：每图 AUC = LEVIR 0.467 / SYSU 0.441 / WHU 0.360 / CDD 0.529，
  CI 都很宽（有效图分别 18/40/12/40，多数 val 图无变化像元）。
- 背景 vs 变化的半径中位数差 **|Δρ| ≤ 0.02**。
- `Spearman(ρ, |A−B|)` 跨数据集变号（−0.57…+0.15），没有稳定关系。
- 32×32 cell 池化保留率 **80–88%（硬分簇）/ 77–86%（软混合）**，而模块实测
  K/Q≈0.70–0.73；成员 CV 只有 0.035–0.059（原型 CV 0.09–0.17）。
  说明手工影像统计特征在 cell 内**高度平行**，池化压不动半径；模块那 30% 的
  收缩来自骨干特征比原始影像统计量"更独立"，手工替身无法复现——这条只能由
  S7 的真实 checkpoint 来回答。
- A/B 交换一致性 = **1.0000**（特征基全部对称，校核通过）。

## 6. 没做的（如实标注，不用替身补齐）

S7（真实特征诊断）、S8（原型压缩信息损失）、S9（软注意力/下游）、S10（几何适用
条件正交构造）、S11/S12 均未开始：S7 需要真实 checkpoint，执行方案明确禁止用
替身前端冒充；S8/S9 依赖 S7 的真实特征统计。H2 的 `hyp_vs_scalar` 等预算 probe
矩阵也未重做，标 **证据不足**。
