# CASA-CD 工作模型源码

共享 MobileNetV2 四尺度特征（24/32/96/320 通道，64²/32²/16²/8²）→ 四级 TAR（64 通道）→ 跨尺度检索/上下文 → DCR → 共享 1×1 预测头。检索始终保留**完整 Q（5,440 个位置）**，只把 KV 压缩成**224 个类条件原型**（128/32/32/32）。

**本树对应 CASA-MinorGoal / Run7 的口径**（协议 `casa_minor_run7_manifold_pooling_v1`）：TAR 用 `--tar-variant cg` 的门控融合，检索用 `--module manifold`，原型池化轴为 `--prototype-pooling {euclidean,lorentz}`。更早的 `spp` / `difference`（Run1–Run3）、H/E 几何对照（CASA-Pyramid Run12–Run16）与 `qkv` 遗留臂都还在这棵树里、可复现。Run8 的尺寸无损分解（CD²-TAR）、Run9 的换骨干、Run10 的双曲层次原型**已从本树移除**，只在 `docs/`、`train_scripts/` 与 git 历史里留档。

## 组件与开关

### TAR：时相融合与通道投影（`model/str_tar.py`）

| `--tar-variant` | 每尺度结构 | 训练/部署参数* | 用途 |
|---|---|---|---|
| `reparam`（默认） | `TemporalRep1x1`（concat / P+Q / Q−P 三支 1×1+BN，可折叠）→ SiLU → `RepLocalBlock` | 2.047125 / 1.935810 M | 主线基线 |
| `plain` | 单 1×1+BN → SiLU → `PlainLocalBlock` | 1.974157 / 1.935810 M | 去可折叠冗余的对照 |
| `diff` | `(P−Q).abs()` → 1×1+BN → SiLU → `PlainLocalBlock` | 1.943949 / 1.905602 M | 去掉学习式时相混合 |
| `spp` | 融合支 + `\|P−Q\|`→1×1→`RepSPPBlock`（1/3/5/7 DW 并行，可折叠成单个 7×7 DW + 1×1）→ 相加 → `RepLocalBlock` | 2.118293 / 1.995714 M | 多尺度差分池化 |
| `spp1` | 同上，但 SPP 只有 1×1 窗口 | 2.095509 / 1.983426 M | 分离"多尺度窗口"本身 |
| `cg` | `ChangeGateTARStage`：内容支 `SiLU(1×1[P,Q])`；差分支先把两期各除以 7×7 局部均值模长（曝光漂移抵消），再 `1×1[d⁺,d⁻]`；`σ` 门乘性调制内容、差分以单位权重相加（无 β 标量）→ `RepLocalBlock` | 2.093461 / 2.043842 M | Run4–Run7 主线 |

\* 指 `--module none` + `--decoder-variant reparam` 时的整网参数。`spp`/`spp1` 额外输出每尺度差分特征 `D_s`，供 `--module difference` 使用；`diff` 在 `return_difference=True` 时也直接给出 `D_s`。

### 上下文：检索与写回（`model/layers/`）

| `--module` | 类 | 几何 | KV 来源 | context 参数 |
|---|---|---|---|---|
| `none` | `AuxiliaryContext` | — | 无检索，只出 guide | 193 |
| `hyperbolic` | `HyperbolicContext` | 曲率 −1 Lorentz 测地平方距离 | `T_s` 的类池化原型 | 11,477 |
| `matched` | `MatchedEuclideanContext` | 同一空间坐标的欧氏平方距离 | 同 H（逐位同构） | 11,477 |
| `difference` | `DifferenceContext` | 同 H | `\|P−Q\|` 的类池化原型（多一层 `LayerNorm(D_s)`） | 11,605 |
| `manifold` | `ManifoldContext` | 同 H，聚合轴可切 `--aggregate {lorentz,euclidean}` | 网格原型，KV 内容跟随 TAR 变体（`cg`/`diff` 时是 `D_s`），另有独立 support 头与 level 通路 | 见速查表（整网） |
| `qkv` | `EuclideanContext` | 欧氏 | Run5 遗留：压缩方式、读出、门控都不同 | 11,477 |

共同实现（`pyramid_context.py` / `hyperbolic_context.py`）：`X=LN(T_s)`，`p=σ(guide(X))`；按 `p`、`1−p` 分别归一化池化得到 `P⁺/P⁻`；`K=L(W_qk P/√d)`、`V=W_v P`（K、V 共用同一内容统计，`V` 留在欧氏空间），`Q=L(W_qk X/√d)`。打分 `−d²/(2τ²)+log(support)+scale_bias−log(M)`，τ 可学习（初值 0.5）。读出 `U=W_out[gate·(AV−mean_hw(AV))]`，残差 `T'_s=T_s+γ_s·U_s+local_s`（`local` 只在最细两层）。`γ` 零初始化，因此**推理时模块与 CTRL 等价**；训练开始后自行打开。

- `--support {mass,cell,pixel,none}`：类先验 `log(support)` 的来源；`cell` 是 `mass` 的别名。
- `--kv-hierarchy {flat,nested}`：`flat` 为 8/4/4/4（224 KV）；`nested` 为 8/4/2/1 并做父子 2×2 聚合（170 KV）。
- `--freeze-tau`：把 τ 冻结在实测中位数 0.4696。
- `--prototype-pooling {euclidean,lorentz}`（默认 `euclidean`）：`lorentz` 先把每个像素特征 lift 到双曲面再做类条件加权重心（Klein 重心），原型严格落在双曲面上、零参数增量。Run7 单因子实测 `lorentz` 比切空间均值低 **0.387 pp**（SYSU 82.824 vs 83.212），故默认仍是欧氏池化。
- `--aggregate {lorentz,euclidean}`（默认 `lorentz`）：`ManifoldContext` 的加权读出在哪个空间做；`euclidean` 只改聚合，打分仍在流形上。
- `--edge-weight` / `--edge-dilate` / `--edge-detach {none,retrieval}`：GT Sobel 边缘监督的权重、膨胀半径与梯度落点。`retrieval`（**默认**）让检索回写的边缘头输入先 `detach()`，边缘梯度只经 TAR 的 `D_s` 回传；Run6 在 SYSU 上实测 `none` 82.01 → `retrieval` 83.09（+1.09 pp，噪声 0.32），因此默认切断。
- `--context-heads {1,2,4}`：1 头 32 维（默认）；多头时 `head_dim=16`。

### 解码器与预测头（`model/str_dcr.py`、`model/casa_net.py`）

| `--decoder-variant` | 结构 | 部署参数增量 |
|---|---|---|
| `reparam`（默认） | 三级 `RepPairFuse1x1`（concat/sum/diff，可折叠）+ `RepLocalBlock`，再 `refine` | 基线 |
| `plain` | 融合换单 1×1+BN，局部块换普通 DW3+PW1 | −0.037 M |
| `none` | `SingleLevelDecoder`：只在最粗尺度上采样，细尺度只做残差相加 | −0.072 M |

预测头为 1×1（64→1），四级各算 BCE+Dice 深监督。**128² 细节头**（`--detail-head`）：两时相 128² 特征拼接 → 1×1(64) → DW3×3(replicate) → SiLU → 1×1(1，零初始化)，与上采样的 coarse 相加，新增 2,689 参数；`--detail-target {area,nearest-exact}` 决定 GT 下采样方式。guide 头只在训练期提供 0.1 系数（默认）的面积 BCE。

## 参数速查（训练 / 部署，M）

| module | tar | decoder | train / deploy |
|---|---|---|---:|
| none | reparam | reparam | 2.047125 / 1.935810 |
| none | spp | reparam | 2.118293 / 1.995714 |
| none | spp1 | reparam | 2.095509 / 1.983426 |
| none | diff | reparam | 1.943949 / 1.905602 |
| none | reparam | none | 1.974730 / 1.900802 |
| hyperbolic / matched | reparam | reparam | 2.058409 / 1.947094 |
| hyperbolic | spp | reparam | 2.129577 / 2.006998 |
| difference | spp | reparam | 2.129705 / 2.007126 |
| difference | diff | reparam | 1.955361 / 1.917014 |
| none | cg | reparam | 2.093461 / 2.043842 |
| manifold | diff | reparam | 1.961771 / 1.923424 |
| manifold | cg | reparam | 2.111283 / 2.061664 |

`--prototype-pooling`、`--aggregate`、`--support`、`--kv-hierarchy` 都是纯几何/统计选择，不改参数。

所有组合的部署折叠 `max|Δlogit| ≤ 5.96e-8`（CPU）/ `1.19e-7`（GPU），二值预测零像素差。

## 训练协议（`train_CASA.py`）

`--module`、`--tar-variant`、`--decoder-variant` 之外，还有 `--selection-split {test,val}`、`--detail-head`、`--detail-target`、`--support`、`--kv-hierarchy`、`--freeze-tau`、`--edge-weight`、`--edge-dilate`、`--edge-detach`、`--per-image-metrics`、`--interior-positive-weight`、`--guide-weight`、`--retrieval-weight`、`--encoder-lr-mult`、`--context-heads`、`--resume`。

- 每 epoch 在 `select_loader`（test 或 val，由 `--selection-split` 决定）上按阈值 0.5 的 F1 选权重，tie 时更新；manifest 记 `selection_split` / `independent_test`，val 选模才可称独立。
- 训练结束后在 val 上做 121 点阈值校准（`operating_point.py`），再在 test 上复测主阈值 0.5 与校准阈值。
- `--per-image-metrics` 在 test 时写 `per_image_cm.npz`（`index/tp/fp/fn` + `dataset/split/batch_size/threshold/list_sha256/protocol/seed/selection_split`），供图像级 bootstrap；`.dup-*` 中断目录没有 TEST 块，汇表时会被跳过。
- `strict_fp32_fold_v1`：关 TF32/开最高精度的上下文里做部署折叠对比，写 `deploy_check.json`；`verify_deployment` 首级容差 2e-4。
- forward 返回四级概率；`return_aux=True` 时返回 logits 与辅助量；`predict` 只出第一级；`diagnostics()` 给出半径、熵、尺度路由、门控与局部残差比例。

## 文件

- `model/casa_net.py`：整网、module/变体装配、128² 细节分支与部署折叠。
- `model/str_tar.py`：`TemporalRep1x1`、`DifferenceTemporal1x1`、`RepSPPBlock`、`SPPTARStage`、`MultiScaleTAR`。
- `model/str_dcr.py`：`RepPairFuse1x1`、`RepLocalBlock`、`PlainFuse1x1`、`PlainLocalBlock`、`SingleLevelDecoder`、`build_decoder`。
- `model/str_reparam.py`：DW3/PW1x1/PairFuse 的 FP64 折叠实现。
- `model/layers/pyramid_context.py`：池化、先验、尺度偏置、局部高通、诊断；`hyperbolic_context.py`：Lorentz 检索与门控写回；`difference_context.py`：差分 KV；`manifold_context.py`：独立 support 头、level 通路与流形聚合；`lorentz_ops.py`：提升与距离；`class_pooling.py`：一阶类池化。
- `dataset/`：列表读取、BGR→RGB、同步增广、ImageNet 归一化；`model/metric_tool.py`：混淆矩阵与指标。
- `train_CASA.py` / `operating_point.py` / `profile_CASA.py`：训练、阈值校准、显存与吞吐测量。
- 测试：`test_CASA.py`、`test_ablation_variants.py`、`test_attribution.py`、`test_centered_readout.py`、`test_context_geometry.py`、`test_direct_geometry.py`、`test_hyperbolic_context.py`、`test_minor_goal.py`、`test_run4_modules.py`、`test_run7_modules.py`、`test_pyramid_context.py`、`test_interior_loss.py`、`test_data_pipeline.py`、`test_CASA_resume.py`（直接 `python <file>.py` 运行，全部通过；`check_run3_switches.py`、`check_per_image_metrics.py` 为在线检查脚本）。
- 诊断脚本：`analyze_hyperbolic.py`（冻结干预）、`analyze_arm_diagnostics.py`、`analyze_geometry_degeneracy.py`、`analyze_selectivity_ablation.py`、`analyze_evidence_postproc.py`（后两个依赖 scipy）。

实验矩阵与结论见根目录 [README.md](../README.md)、[SOURCES.md](SOURCES.md)、[HYPERBOLIC_DESIGN.md](HYPERBOLIC_DESIGN.md)，以及 `train_scripts/CASA-MinorGoal/Run{1,2,3,4,5,6,7}/` 与 `train_scripts/CASA-Pyramid/Run13–Run16/`。
