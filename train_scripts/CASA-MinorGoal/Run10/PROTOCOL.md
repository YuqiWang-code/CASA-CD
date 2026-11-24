# CASA-MinorGoal / Run10：把 TAR、HRA、DCR 各自做扎实（`casa_minor_run10_lean_v1`）

## 这一轮要回答什么

Run7 的三件工作同时上，收益无法归因，且各自都带着可以直接指出的毛病（见
`outputs/CASA-MinorGoal/Run7/analysis/`）。Run10 不改主干拓扑，只做三组"把话说清楚"的
修改，然后一次只放开一个因素：

1. **DCR**：主输出从 64² 提到 128²（`FineRefine` 用编码器 stem 的 32 通道在 128² 上做
   `1×1 → BN → SiLU → DW3×3 → BN → SiLU → 1×1`，末层零初始化即恒等），深监督四级
   权重改 `1 / .5 / .25 / .125`。这是主干，所有臂都带。
2. **TAR**：`ChangeGateTARStage` 从"增强器"改成"竞争器"
   `g = σ(1×1(positive − negative))`、`fused = block(g · n(content) + (1 − g) · n(delta))`，
   两路各做一次逐像素跨通道归一（`_channel_norm`），门零初始化 ⇒ 第 0 步 g = 0.5；
   局部对比核随尺度走 `(3, 5, 7, 7)`（64²→3×3、8²→7×7）。
3. **HRA**：`−mean` 变成可学习门 `mean_gate`（零初始化 ⇒ 开局仍去均值），检索可只在
   16²/8² 跑（`retrieval_scales=(2,3)`），原型可从 224 压到 32（每尺度 2×2 格 ×2 类），
   support 先验可从"参与质量"换成"置信"`|2p − 1|`。

## 臂与门限

| 臂 | TAR | 检索 | 说明 |
|---|---|---|---|
| `D_CTRL` | reparam | 无 | 新主干本身 |
| `D_CG` | cg（竞争门） | 无 | 只放开 TAR |
| `R_FULL` | cg | 0–3 尺度，224 原型，mass | 只放开 HRA（对照 Run7 口径） |
| `R_RES` | cg | 2–3 尺度，224 原型，mass | 检索只在粗两级 |
| `R_COMPACT` | cg | 0–3 尺度，32 原型，confidence | 压缩 + 置信先验 |
| `R_LEAN` | cg | 2–3 尺度，32 原型，confidence | 三项简化叠加 |

* 种子 2334 / 2335，`--selection-split val`，主阈值 0.5，bs 32，40k step。
* 数据：SYSU 先、LEVIR 后（按用户要求）。
* 门限（预登记，跑前写死）：`max(0.36, 1.5σ) = 0.59 pp` 表示"该模块带来收益"；
  `−0.36 pp`（一个噪声单位）表示"该简化不掉点"。
  `D_CG` vs `D_CTRL`、`R_FULL` vs `D_CG` 用 0.59；`R_RES`/`R_COMPACT`/`R_LEAN` vs
  `R_FULL` 用 −0.36。

## 与 Run7 的可比性

Run7 的读数（SYSU：CTRL 81.392 / TAR_ONLY 81.995 / HRA_ONLY 82.533 / TAR_HRA 83.022）
是旧主干（主输出 64²、四级等权、单向门）。Run10 的 `D_CTRL` 已经换主干，因此**跨轮
只做方向参考**，同轮比较以 `D_CTRL`/`D_CG`/`R_FULL` 为准。

## 产物

* `outputs/CASA-CD/CASA-MinorGoal/Run10/<arm>/<dataset>/train_log.txt`（含 TEST RESULTS）
* `params.json`（smoke 回填的 trainable/deploy）、`comparison.md/json`
* 收尾：`docs/experiment_metrics.xlsx`、`docs/temporary/models_and_metrics_CASA-MinorGoal_Run10/`
