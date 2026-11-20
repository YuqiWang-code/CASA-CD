# SYSU-CD 四行结果（TAR / HRA 拆解，256×256）

参数量为部署折叠后参数量（M）；GFLOPs 为部署折叠形态、输入 256×256、2×MACs。全部 `--selection-split val`、阈值 0.5、MobileNetV2 共享编码。

| # | 组 | TAR | HRA | 参数量 (M) | GFLOPs | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ① | 基础（无 TAR、无 HRA） | — | — | 1.936 | 1.734 | 80.21 | 83.65 | 91.64 | 81.90 | 69.34 | 76.46 |
| ② | TAR + HRA | SPP 多尺度差分池化 | hyperbolic 类条件检索 | 2.007 | 1.950 | 81.16 | 85.71 | 92.36 | 83.37 | 71.48 | 78.42 |
| ③ | 仅 TAR | SPP 多尺度差分池化 | — | 1.996 | 1.835 | 80.20 | 85.56 | 92.14 | 82.79 | 70.64 | 77.70 |
| ④ | 仅 HRA | —（1×1 差分） | difference 差分原型 KV | 1.917 | 1.827 | 81.81 | 85.14 | 92.34 | 83.44 | 71.59 | 78.47 |

相对 ① 的 F1 增量：② +1.47、③ +0.89、④ +1.54；IoU 增量：② +2.14、③ +1.30、④ +2.25。

## 每行对应的实验

| # | 实验臂 | 轮次 | seed | 训练/部署参数 | 报告编号 |
|---|---|---|---|---|---|
| ① | CTRL_S2335（`module=none, tar=reparam`） | Run2 | 2335 | 2,047,125 / 1,935,810 | SYSU 81.90（该配置最差 seed） |
| ② | ARM_C（`module=hyperbolic, tar=spp, support=mass`） | Run3 | 2334 | 2,129,577 / 2,006,998 | SYSU 83.37 |
| ③ | SPP（`module=none, tar=spp`） | Run1 | 2333 | 2,118,293 / 1,995,714 | SYSU 82.79 |
| ④ | DIFF_NOSPP（`module=difference, tar=diff`） | Run2 | 2333 | 1,955,361 / 1,917,014 | SYSU 83.44 |

## 引用注意（写进图注即可）

1. 行①按"最差一组"取 seed 2335；若换同 seed 控制（2334，F1 82.09），②③④ 的 Δ 分别变成 +1.28 / +0.70 / +1.35。
2. ② 与 ④ 相差 0.07（轮内噪声 0.32），即 **TAR 的差分池化叠加在 HRA 之上没有额外收益**：SPP 与差分原型读的是同一份变化证据，`DIFF−SPP = SYSU −0.13`。表里不要写成"两者级联增益"。
3. ② 的 HRA 是 hyperbolic 版（Run3，无重心读出/level 路径）；④ 是 difference 版。若坚持用结构图第 5–6 页画的 manifold 版，②应改为 Run6 `EDGE_TAP_CUT`：部署 2,062 / 1.950 GFLOPs，F1 83.09（+1.19 vs ①）。
4. 四行分属三轮，seed 不同；单 seed 格差 0.2–0.3 不宜过度解读，只有行内配对（②③④ 各自对同 seed 控制）的 Δ 有统计意义。

数据来源：`docs/experiment_metrics.xlsx`（Recall/Precision/OA/F1/IoU/Kappa）、服务器 `per_image_cm.npz` 复算一致；参数取自 `train_scripts/CASA-MinorGoal/Run{1,2,3}/config.py` 的 TRAINABLE/DEPLOY_PARAMS，GFLOPs 由 `outputs/pptx/compute_flops.py` 实测（四行参数与 config 完全一致，`params_match=True`）。
