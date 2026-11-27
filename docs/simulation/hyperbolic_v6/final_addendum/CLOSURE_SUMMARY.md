# V6 仿真封存：修正版附录（Final addendum）

本目录由 `tools/simulation/finalize_fix.py` 生成。**所有数值来自本目录的 CSV**，
没有手抄单元格；历史报告保留原文，但顶部已加撤回横幅。

## 1. 修正了哪些代码

| # | 文件 | 原问题 | 修正 |
|---|---|---|---|
| 1 | `s9a_fix.py` | `SCALE-OFF` 写成 `bias - source_bias(0)`，实际只留学习偏置，与 `COUNT-OFF` 完全相同 | 三项先验拆成 `log_support / learned / count`：`SCALE-OFF = base + log_support + count`，`COUNT-OFF = base + log_support + learned`，`SUPPORT-OFF = base + learned + count`，`BASE` 仍走冻结函数并用显式项复算校验 |
| 2 | `s9a_fix.py` | `GUIDE` 每尺度 `cat([change..., background...])`，每区域两类不相邻 | 改为 `(region, class)` 交替：`[r0-change, r0-background, r1-change, r1-background, ...]` |
| 3 | `s9a_fix.py` | `UNIFORM` 只是 lambda 返回零、未进入注意力记录，旧 JS 来自别的路径 | 统一在 `make_logits` 内生成全零 logits 并记录，JS 用 `sqrt(JS)`（nats，上界 `sqrt(ln2)=0.833`） |
| 4 | `s8b_fix.py` | 双曲距离把 `z·c` 多除了一次 `t_z t_c`；方向余弦用含时间范数 | `<L(z),L(c)>_L = t_z t_c - z·c`，`hyp2 = acosh(<L,L>)^2`，`chord^2 = 2(<L,L>-1)`；方向余弦取原点切向（纯空间）`z·c/(norm_z norm_c)`，旧口径保留为 `*_cos_ambient` 审计列 |
| 5 | `s8b_fix.py` | greedy/DP 用未归一化 `J`，`objective` 用 `J/M_{s,c}`，三者目标不一致 | 三者统一为 `sum_{R,c} J(R,c)/M_{s,c}`（`M_{s,c}` 分区前固定） |
| 6 | `s8b_fix.py` | 采样按 `images//4` 抓取且截断，SYSU 只得 9 张，报告误称 12 张 | 按**实际存在的分层**等配额+余数+top-up，记录实际张数、文件名与 list SHA-256；临时名单写在输出目录而非数据根目录 |
| 7 | `s13_fix.py` | 每臂重建 head 且 `base` 来自全局 RNG，各臂基础 logits 不同 | 特征/head 初值/base logits/target 只生成一次并深拷贝给各臂，记录 base SHA-256 |
| 8 | `s13_fix.py` | `LOGSHARE` 用 `logsigmoid(mean sigmoid)`（把已压缩的 share 再压一次）；一步更新只改 weight 不改 bias | `LOGSHARE = logsigmoid(mean logit)`；一步 SGD 同时更新 weight 与 bias，并报告两者梯度范数 |

## 2. 是否重新训练

**仿真纠错本身不需要重新训练。** 这些实验是冻结 Run10 权重的反事实（同权重、同数据流，只换检索算子/先验/几何），
重跑即恢复可比性；错误在脚本的数学与统计，不在训练。
但**冻结权重结论不能代替重新训练结果**：Run11（ImageNet 初始化、6 臂×2 seed×2 库）正在 GPU0 上重训 HRA 三轴矩阵，
论文里任何「模块有效」的主张只能用 Run11 的 test 指标支撑；本附录只用于解释/撤回旧的仿真结论。

## 2.5 被撤回的旧结论（旧值 → 修正值）

旧值抄自历史报告，仅用于说明撤回点；修正值全部来自本目录 CSV，没有手抄。

| 旧报告说法 | 旧值（作废） | 修正值（CSV） |
|---|---|---|
| S9a `SCALE-OFF` 与 `COUNT-OFF` 数值相同，说明学到的尺度先验与计数先验不可区分 | +0.29 / +0.29（仅 seed2335 的 CI 排除 0） | LEVIR：SCALE-OFF +0.013 pp [-0.005, +0.032]、COUNT-OFF -0.037 pp [-0.081, -0.000]；SYSU：SCALE-OFF -0.229 pp [-0.536, -0.032]、COUNT-OFF +0.265 pp [+0.067, +0.548] |
| S9a `UNIFORM` 注意力统计来自未记录 uniform 的旧路径 | JS 0.27 / 0.10（口径不明） | uniform 已写入记录；`sqrt(JS)` 0.63–0.72（0.39–0.52 nats，上界 0.833） |
| S8b 双曲距离/方向余弦 | FIX `K_hyp2` 1.3306、`K_cos` 0.4836，且仅 8 臂入表 | FIX `K_hyp2` 0.4345、`K_cos` 0.8429，10 臂全部入表（DP-V-* 不再丢失） |
| S8b 采样 | “每 checkpoint 12 张 val”，SYSU 实际只有 9 张 | 按实际分层等配额：SYSU 12 张（4/4/4）、LEVIR 12 张（3/3/3/3），名单与 SHA-256 已记录 |
| S13 `LOGSHARE` 定义与一步更新 | prior −0.474，grad 0.0344，且只更新 weight | `logsigmoid(mean logit)`：prior -0.693，W grad 0.332，b grad 0.328；mid-init 下与 SHARE 区分（prior -0.432 vs -0.461，W grad 0.280 vs 0.242） |
| S9a `NOCTX` 两个 LEVIR seed 的数值 | −16.68 / −4.26（顺序与 SYSU 的 −9.83/−1.63 对调） | LEVIR -5.320 pp [-8.769, -2.566]（48 图）；SYSU -9.784 pp [-14.020, -5.899]（45 图） |
| Run10 `confidence` 先验确实按置信度加权 | 报告把 `R_COMPACT`/`R_LEAN` 记为 confidence prior | 训练 40k 步的 `R_COMPACT-2334` 仍 `share≡0.5`、`support≡1e-6`、`support_head` 梯度为 0：`abs(2s-1)` 在 0.5 处梯度为 0，零初始化后永久锁死，实际是**与输入无关的常数先验**；因此 `R_COMPACT`/`R_LEAN` 与 `R_FULL` 的差异不能归因于该先验 |

## 3. S9a 修正结果

输入 2976 行 per-image、11160 行注意力记录、744 行预算扫描。
选模口径：test 参与选模的下游结论不在此附录；所有 Delta 均为同一批图像上的配对差，
先按图像在 checkpoint（seed）上取平均，再对图像做 2000 次配对 bootstrap 得到 95% CI。

### 3.1 各臂绝对指标（%）

**LEVIR-CD-256**（均值 over checkpoints）

| arm | F1 | IoU | OA | Recall | Precision | Kappa |
|---|---|---|---|---|---|---|
| AGG | 91.013 | 83.509 | 98.486 | 87.774 | 94.507 | 90.188 |
| BASE | 91.993 | 85.174 | 98.605 | 91.755 | 92.235 | 91.229 |
| COUNT-OFF | 91.983 | 85.155 | 98.602 | 91.812 | 92.156 | 91.217 |
| DE | 92.000 | 85.185 | 98.605 | 91.801 | 92.205 | 91.236 |
| DEAGG | 91.184 | 83.797 | 98.508 | 88.294 | 94.279 | 90.370 |
| DECAL | 91.976 | 85.145 | 98.602 | 91.724 | 92.231 | 91.211 |
| DISTONLY | 92.022 | 85.223 | 98.604 | 92.157 | 91.888 | 91.257 |
| GUIDE | 91.872 | 84.967 | 98.589 | 91.305 | 92.450 | 91.100 |
| KE | 92.010 | 85.203 | 98.604 | 91.965 | 92.057 | 91.245 |
| KVE | 91.930 | 85.065 | 98.584 | 92.336 | 91.530 | 91.153 |
| NOCTX | 81.699 | 69.533 | 96.192 | 94.535 | 72.396 | 79.641 |
| PRIORONLY | 90.325 | 82.359 | 98.361 | 87.603 | 93.243 | 89.431 |
| SCALE-OFF | 91.995 | 85.176 | 98.604 | 91.827 | 92.164 | 91.230 |
| SUPPORT-OFF | 92.042 | 85.258 | 98.609 | 92.087 | 91.999 | 91.280 |
| UNIFORM | 90.753 | 83.073 | 98.420 | 88.782 | 92.834 | 89.889 |
| VE | 91.918 | 85.045 | 98.585 | 92.113 | 91.727 | 91.143 |

**SYSU-CD-256**（均值 over checkpoints）

| arm | F1 | IoU | OA | Recall | Precision | Kappa |
|---|---|---|---|---|---|---|
| AGG | 78.908 | 65.178 | 93.105 | 78.051 | 79.860 | 74.789 |
| BASE | 77.959 | 63.888 | 93.136 | 73.470 | 83.183 | 73.915 |
| COUNT-OFF | 78.129 | 64.118 | 93.167 | 73.864 | 83.044 | 74.098 |
| DE | 78.035 | 63.988 | 93.151 | 73.636 | 83.162 | 73.997 |
| DEAGG | 79.047 | 65.368 | 93.184 | 77.791 | 80.458 | 74.981 |
| DECAL | 78.086 | 64.059 | 93.199 | 73.324 | 83.700 | 74.084 |
| DISTONLY | 77.835 | 63.717 | 93.145 | 72.850 | 83.764 | 73.806 |
| GUIDE | 77.713 | 63.554 | 93.078 | 73.053 | 83.213 | 73.639 |
| KE | 78.019 | 63.970 | 93.137 | 73.715 | 82.990 | 73.972 |
| KVE | 78.035 | 63.992 | 93.077 | 74.416 | 82.172 | 73.940 |
| NOCTX | 71.733 | 56.047 | 92.071 | 61.734 | 87.009 | 67.314 |
| PRIORONLY | 76.692 | 62.196 | 93.004 | 69.784 | 85.363 | 72.629 |
| SCALE-OFF | 77.854 | 63.744 | 93.121 | 73.190 | 83.335 | 73.804 |
| SUPPORT-OFF | 77.764 | 63.623 | 93.137 | 72.644 | 83.873 | 73.734 |
| UNIFORM | 75.198 | 60.285 | 92.716 | 67.226 | 85.889 | 71.016 |
| VE | 77.882 | 63.785 | 93.051 | 74.038 | 82.309 | 73.777 |

### 3.2 配对 Delta（百分点，vs BASE）

**LEVIR-CD-256**

| arm | Delta pp | 95% CI | verdict（bootstrap） | 同号 seed 单元 | n 图 |
|---|---|---|---|---|---|
| SUPPORT-OFF | +0.159 | [+0.008, +0.360] | positive (CI excludes 0) | 2/2 | 48 |
| DISTONLY | +0.122 | [+0.004, +0.284] | positive (CI excludes 0) | 2/2 | 48 |
| KE | +0.105 | [-0.007, +0.256] | inconclusive (CI spans 0) | 2/2 | 48 |
| DE | +0.086 | [+0.001, +0.192] | positive (CI excludes 0) | 2/2 | 48 |
| SCALE-OFF | +0.013 | [-0.005, +0.032] | inconclusive (CI spans 0) | 2/2 | 48 |
| COUNT-OFF | -0.037 | [-0.081, -0.000] | negative (CI excludes 0) | 0/2 | 48 |
| KVE | -0.047 | [-0.152, +0.050] | inconclusive (CI spans 0) | 0/2 | 48 |
| DECAL | -0.068 | [-0.179, -0.003] | negative (CI excludes 0) | 0/2 | 48 |
| VE | -0.162 | [-0.352, -0.010] | negative (CI excludes 0) | 0/2 | 48 |
| GUIDE | -0.458 | [-1.267, +0.009] | inconclusive (CI spans 0) | 0/2 | 48 |
| DEAGG | -1.749 | [-3.704, -0.436] | negative (CI excludes 0) | 0/2 | 48 |
| AGG | -2.402 | [-4.476, -0.770] | negative (CI excludes 0) | 0/2 | 48 |
| NOCTX | -5.320 | [-8.769, -2.566] | negative (CI excludes 0) | 0/2 | 48 |
| UNIFORM | -10.138 | [-16.687, -4.369] | negative (CI excludes 0) | 0/2 | 48 |
| PRIORONLY | -12.348 | [-19.287, -6.024] | negative (CI excludes 0) | 0/2 | 48 |

**SYSU-CD-256**

| arm | Delta pp | 95% CI | verdict（bootstrap） | 同号 seed 单元 | n 图 |
|---|---|---|---|---|---|
| AGG | +1.693 | [-0.198, +4.035] | inconclusive (CI spans 0) | 2/2 | 45 |
| DEAGG | +1.523 | [-0.047, +3.466] | inconclusive (CI spans 0) | 2/2 | 45 |
| KVE | +0.483 | [-0.079, +1.165] | inconclusive (CI spans 0) | 2/2 | 45 |
| VE | +0.453 | [-0.042, +1.057] | inconclusive (CI spans 0) | 2/2 | 45 |
| GUIDE | +0.364 | [-0.033, +0.831] | inconclusive (CI spans 0) | 2/2 | 45 |
| COUNT-OFF | +0.265 | [+0.067, +0.548] | positive (CI excludes 0) | 1/2 | 45 |
| KE | +0.067 | [-0.084, +0.243] | inconclusive (CI spans 0) | 1/2 | 45 |
| DISTONLY | -0.090 | [-0.460, +0.261] | inconclusive (CI spans 0) | 1/2 | 45 |
| SCALE-OFF | -0.229 | [-0.536, -0.032] | negative (CI excludes 0) | 0/2 | 45 |
| DECAL | -0.454 | [-0.867, -0.121] | negative (CI excludes 0) | 1/2 | 45 |
| SUPPORT-OFF | -0.502 | [-1.234, +0.038] | inconclusive (CI spans 0) | 1/2 | 45 |
| DE | -0.892 | [-1.830, -0.116] | negative (CI excludes 0) | 1/2 | 45 |
| PRIORONLY | -2.817 | [-4.790, -0.998] | negative (CI excludes 0) | 0/2 | 45 |
| UNIFORM | -2.855 | [-4.745, -1.176] | negative (CI excludes 0) | 0/2 | 45 |
| NOCTX | -9.784 | [-14.020, -5.899] | negative (CI excludes 0) | 0/2 | 45 |

### 3.3 UNIFORM 注意力偏离

| checkpoint | dataset | arm | JS (nats) | sqrt(JS) | 上界 |
|---|---|---|---|---|---|
| R_FULL-2334 | LEVIR-CD-256 | AGG | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | COUNT-OFF | 0.2331 | 0.4828 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | DE | 0.1162 | 0.3409 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | DEAGG | 0.1162 | 0.3409 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | DECAL | 0.0885 | 0.2974 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | DISTONLY | 0.2350 | 0.4847 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | GUIDE | 0.2645 | 0.5143 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | KE | 0.0809 | 0.2844 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | KVE | 0.0809 | 0.2844 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | NOCTX | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | PRIORONLY | 0.3619 | 0.6016 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | SCALE-OFF | 0.0919 | 0.3031 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | SUPPORT-OFF | 0.1592 | 0.3990 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | UNIFORM | 0.4374 | 0.6614 | 0.8326 |
| R_FULL-2334 | LEVIR-CD-256 | VE | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | AGG | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | COUNT-OFF | 0.2310 | 0.4806 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | DE | 0.1026 | 0.3202 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | DEAGG | 0.1026 | 0.3202 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | DECAL | 0.0528 | 0.2297 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | DISTONLY | 0.3393 | 0.5825 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | GUIDE | 0.1941 | 0.4405 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | KE | 0.0598 | 0.2446 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | KVE | 0.0598 | 0.2446 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | NOCTX | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | PRIORONLY | 0.3757 | 0.6130 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | SCALE-OFF | 0.1133 | 0.3366 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | SUPPORT-OFF | 0.2864 | 0.5351 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | UNIFORM | 0.5235 | 0.7235 | 0.8326 |
| R_FULL-2334 | SYSU-CD-256 | VE | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | AGG | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | COUNT-OFF | 0.2343 | 0.4841 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | DE | 0.0864 | 0.2940 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | DEAGG | 0.0864 | 0.2940 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | DECAL | 0.0612 | 0.2475 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | DISTONLY | 0.1989 | 0.4460 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | GUIDE | 0.3409 | 0.5839 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | KE | 0.0700 | 0.2645 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | KVE | 0.0700 | 0.2645 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | NOCTX | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | PRIORONLY | 0.3433 | 0.5859 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | SCALE-OFF | 0.0653 | 0.2555 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | SUPPORT-OFF | 0.0754 | 0.2747 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | UNIFORM | 0.3926 | 0.6266 | 0.8326 |
| R_FULL-2335 | LEVIR-CD-256 | VE | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | AGG | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | COUNT-OFF | 0.2302 | 0.4798 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | DE | 0.0781 | 0.2795 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | DEAGG | 0.0781 | 0.2795 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | DECAL | 0.0506 | 0.2250 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | DISTONLY | 0.3687 | 0.6072 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | GUIDE | 0.1210 | 0.3479 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | KE | 0.0609 | 0.2469 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | KVE | 0.0609 | 0.2469 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | NOCTX | 0.0000 | 0.0000 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | PRIORONLY | 0.3524 | 0.5936 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | SCALE-OFF | 0.0275 | 0.1659 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | SUPPORT-OFF | 0.2842 | 0.5331 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | UNIFORM | 0.4961 | 0.7043 | 0.8326 |
| R_FULL-2335 | SYSU-CD-256 | VE | 0.0000 | 0.0000 | 0.8326 |

### 3.4 预算曲线（F1 %，BASE 臂）

| dataset | gridset | checkpoint | F1 |
|---|---|---|---|
| LEVIR-CD-256 | b128 | R_FULL-2334 | 92.137 |
| LEVIR-CD-256 | b224 | R_FULL-2334 | 92.114 |
| LEVIR-CD-256 | b32 | R_FULL-2334 | 92.197 |
| LEVIR-CD-256 | b64 | R_FULL-2334 | 92.185 |
| SYSU-CD-256 | b128 | R_FULL-2334 | 78.676 |
| SYSU-CD-256 | b224 | R_FULL-2334 | 78.815 |
| SYSU-CD-256 | b32 | R_FULL-2334 | 78.801 |
| SYSU-CD-256 | b64 | R_FULL-2334 | 78.727 |
| LEVIR-CD-256 | b128 | R_FULL-2335 | 91.883 |
| LEVIR-CD-256 | b224 | R_FULL-2335 | 91.873 |
| LEVIR-CD-256 | b32 | R_FULL-2335 | 91.886 |
| LEVIR-CD-256 | b64 | R_FULL-2335 | 91.895 |
| SYSU-CD-256 | b128 | R_FULL-2335 | 77.158 |
| SYSU-CD-256 | b224 | R_FULL-2335 | 77.104 |
| SYSU-CD-256 | b32 | R_FULL-2335 | 77.124 |
| SYSU-CD-256 | b64 | R_FULL-2335 | 77.121 |

## 4. S8b 修正结果

样本：等配额分层采样，实际张数与分层计数（来自各 summary JSON）：

| checkpoint/dataset | images | sampled | leaves | DP 违反 | greedy 违反 |
|---|---|---|---|---|---|
| R_FULL-2334/LEVIR-CD-256 | 12 | bg:3, large:3, mid:3, small:3 | 4 | 0 | 118 |
| R_FULL-2334/SYSU-CD-256 | 12 | large:4, mid:4, small:4 | 4 | 0 | 174 |
| R_FULL-2335/LEVIR-CD-256 | 12 | bg:3, large:3, mid:3, small:3 | 4 | 0 | 125 |
| R_FULL-2335/SYSU-CD-256 | 12 | large:4, mid:4, small:4 | 4 | 0 | 177 |

`sse` = 欧氏环境平方误差；`hyp2` = Lorentz 测地距离平方；`chord` = 环境弦长平方；
`cos` = 原点切向方向余弦（纯空间），`cos_ambient` = 旧口径（含 `tanh|z|tanh|c|` 阻尼）仅供审计。
括号内为按图像 bootstrap 的 95% CI（n 见 CSV 的 `_n` 列）。

| arm | K_sse | K_hyp2 | K_cos | V_sse | V_hyp2 | V_cos | J |
|---|---|---|---|---|---|---|---|
| ADA-K-E | 0.6392 | 0.4473 | 0.8346 | 159.3002 | 2.4395 | 0.9130 | 0.9981 |
| ADA-K-H | 0.6399 | 0.4477 | 0.8344 | 159.0217 | 2.4382 | 0.9130 | 0.8336 |
| ADA-V-E | 0.6402 | 0.4479 | 0.8352 | 158.1719 | 2.4349 | 0.9128 | 78.8007 |
| ADA-V-H | 0.6416 | 0.4487 | 0.8345 | 157.8978 | 2.4332 | 0.9127 | 10.2067 |
| DP-K-E | 0.5736 | 0.4093 | 0.8562 | 144.8011 | 2.2561 | 0.9183 | 0.8702 |
| DP-K-H | 0.5747 | 0.4097 | 0.8562 | 145.3277 | 2.2599 | 0.9183 | 0.7363 |
| DP-V-E | 0.5968 | 0.4230 | 0.8490 | 144.3505 | 2.2199 | 0.9205 | 68.9584 |
| DP-V-H | 0.6030 | 0.4263 | 0.8478 | 147.0948 | 2.2239 | 0.9211 | 8.7980 |
| FIX | 0.6167 | 0.4345 | 0.8429 | 153.6918 | 2.3667 | 0.9145 | nan |
| RAND | 0.6390 | 0.4483 | 0.8367 | 158.8185 | 2.4529 | 0.9118 | nan |

相对 FIX 的 `hyp2` 比值（<1 表示比固定网格更接近成员）：

| arm | K_hyp2/FIX | V_hyp2/FIX |
|---|---|---|
| ADA-K-E | 1.0296 | 1.0308 |
| ADA-K-H | 1.0305 | 1.0302 |
| ADA-V-E | 1.0310 | 1.0288 |
| ADA-V-H | 1.0327 | 1.0281 |
| DP-K-E | 0.9421 | 0.9532 |
| DP-K-H | 0.9430 | 0.9549 |
| DP-V-E | 0.9736 | 0.9380 |
| DP-V-H | 0.9813 | 0.9396 |
| RAND | 1.0318 | 1.0364 |

自目标 `J`（greedy 与 DP 在同一归一化目标 `sum_RC J(R,c)/M_{s,c}` 上竞争；

FIX 为同目标参照，`J_rel < 1` 表示自适应分区确实优于固定网格）：

| arm | objective | J | J_FIX | J_rel |
|---|---|---|---|---|
| ADA-K-E | J_KE | 0.9981 | 0.9549 | 1.0453 |
| ADA-K-H | J_KH | 0.8336 | 0.8002 | 1.0416 |
| ADA-V-E | J_VE | 78.8007 | 76.3064 | 1.0327 |
| ADA-V-H | J_VH | 10.2067 | 9.7812 | 1.0435 |
| DP-K-E | J_KE | 0.8702 | 0.9549 | 0.9113 |
| DP-K-H | J_KH | 0.7363 | 0.8002 | 0.9201 |
| DP-V-E | J_VE | 68.9584 | 76.3064 | 0.9037 |
| DP-V-H | J_VH | 8.7980 | 9.7812 | 0.8995 |

注意：旧报告把 `sse`、测地距离与方向余弦混在同一张表里比较，本版分列并给出各自定义；
旧表里 DP-V-* 被静默丢弃，本版 10 臂全部输出。

## 5. S13 修正结果

### 5.1 合成 prior-梯度（各臂共享同一 base logits）

| arm | init | log prior min | W grad norm | b grad norm | attn L1 / step | 定义 |
|---|---|---|---|---|---|---|
| OLD | zero | -1.382e+01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | /2*share-1/ |
| SHARE | zero | -6.931e-01 | 3.324e-01 | 3.281e-01 | 2.213e-02 | log(mean_cell sigmoid) |
| NONZERO | kaiming | -3.926e+00 | 1.561e+01 | 1.590e+01 | 2.681e-02 | /2*share-1/ |
| LOGSHARE | zero | -6.931e-01 | 3.324e-01 | 3.281e-01 | 2.213e-02 | logsigmoid(mean_cell logit) |
| OFF | zero | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | none |
| SHARE-mid | mid | -8.116e-01 | 2.417e-01 | 2.173e-01 | 1.231e-02 | log(mean_cell sigmoid) |
| LOGSHARE-mid | mid | -8.105e-01 | 2.802e-01 | 2.477e-01 | 1.653e-02 | logsigmoid(mean_cell logit) |

零初始化下 `OLD` 的 prior 恒为 `log(1e-6)` 且梯度恰为 0（`|2*0.5-1|` 在 kink 处），
这正是 confidence 分支在部署初始化时的梯度阻断；`SHARE` 与 `LOGSHARE` 在零初始化时数值相同是可预期的
（所有 cell 的 logit 相同时 `log(mean σ) = log σ(mean)`），mid-init 两行才区分二者定义。

### 5.2 真实 CASANet 反向检查

| arm | prior | support_head 归零 | W grad norm | b grad norm | share min | share max | support min |
|---|---|---|---|---|---|---|---|
| R_FULL-2334 | mass | False | 4.196e-04 | 5.522e-05 | 0.000454 | 0.917343 | 4.542e-04 |
| R_FULL-2334 | mass | True | 4.705e-04 | 4.404e-05 | 0.500000 | 0.500000 | 5.000e-01 |
| R_COMPACT-2334 | confidence | False | 0.000e+00 | 0.000e+00 | 0.500000 | 0.500000 | 1.000e-06 |
| R_COMPACT-2334 | confidence | True | 0.000e+00 | 0.000e+00 | 0.500000 | 0.500000 | 1.000e-06 |

`mass` 路径在零初始化下梯度非零；`confidence` 路径在 `s=0.5` 处 `abs(2s-1)` 的导数为 0，
又被 `clamp_min(1e-6)` 截成常数，因此梯度恒为 0、`s` 永远停在 0.5。
表 5.2 的真实 checkpoint 证实这不是初始化窗口问题：训练 40k 步后的 `R_COMPACT-2334` 仍然
`share≡0.5`、`support≡1e-6`、梯度 0，即 Run10 的 confidence 先验从头到尾都是常数先验。

## 6. 引用规则

- 旧报告中的 SCALE-OFF / GUIDE / UNIFORM / S8b 距离与方向余弦 / S13 LOGSHARE 数值一律作废；
- 需要数值时引用本目录的 `table_*_fix.csv` 与对应报告；
- 冻结权重反事实只回答「同一权重下算子差异」，任何模块收益主张引用 Run11 的 test 指标。
