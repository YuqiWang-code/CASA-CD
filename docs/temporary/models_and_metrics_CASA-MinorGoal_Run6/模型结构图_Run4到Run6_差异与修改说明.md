# 总体模型结构图：Run4 → Run6 核对与修改说明

对象：`CASA-MinorGoal_Run6_总体模型结构.pptx`（9 页，13.333 × 7.5 in）
对照代码：`models/`（HEAD `65b8a2d`，含 `--edge-detach` / `--edge-scales` / `--aggregate`）
备份：`CASA-MinorGoal_Run6_总体模型结构.orig-20261118.pptx` 之前一版 → 本次改动前另存为
`CASA-MinorGoal_Run6_总体模型结构.orig-20261120.pptx`

## 一句话结论

图里画的本来就不是 `--module difference` 的论文主线，而是 **manifold 读出 + TAR-CG 门控**这一臂
（Run4 的 `CG_HRA`，训练 2,111,287 / 部署 2,061,668）。逐层核对后，第 2–7 页的模块细节与当前
`models/` 代码**完全一致**；Run6 相对 Run4 只有一处架构级改动——**检索回写 `Uₛ` 在喂给 HRA 边缘头前
`detach()`**——另有两个只影响训练/对照的开关。所以本次只改文字与一根箭头，不重画任何框。

## 逐页核对：一致（未改）

| 页 | 图中内容 | 代码位置 | 结论 |
| --- | --- | --- | --- |
| 2 | 时相门 `gate=σ(Conv1×1(64+Cₛ→64)([δₛ, d⁺+d⁻]))`；`fused=RepLocal(content·(1+gate)+β·δₛ)`，β 零初始化；`δₛ=BN(1×1([cat(d⁺,d⁻)]))` | `models/model/str_tar.py` `ChangeGateTARStage` | 一致 |
| 3 | `RepDW3` 四路（DW3×3 / DW1×3 / DW3×1 / α₁X，各带 BN）、`RepPW1x1` 四路（1×1 64→64 / 低秩 64→16→64 / 逐通道缩放 / α₂h）；部署折叠为 DW3×3→SiLU→PW1×1→SiLU | `models/model/str_reparam.py` | 一致 |
| 4 | `Qₛ`＝LN(Tₛ)→共享 W_qk 64→32→÷√32→洛伦兹提升→HₛWₛ×33；`K/V`＝LN(Dₛ)→引导头 64→1→σ→格内归一化→变化/背景原型→四尺度拼接 224→K 224×33、V 224×32；原型 128/32/32/32；`D₄=δ₄, Dₛ=δₛ+↑(Dₛ₊₁)` | `models/model/layers/manifold_context.py`、`str_tar.py` `MultiScaleTAR` | 一致 |
| 5 | `−d²(Q,K)/(2τₛ²)+log 支持度（64→2→σ→网格平均）+ bₛⱼ−log Nⱼ → Softmax → Aₛ`；`L(V)` 224×33→`AₛL(V)`→Klein 重心 `x/√max(1,t²−‖x‖²)`→Rₛ 32×HₛWₛ；5440 行查询 | `manifold_context.py` `support_head` / `aggregate` | 一致 |
| 6 | 门控 69 通道（本地值 32 + ΔRₛ 32 + pₛ 1 + 检索变化概率 1 + x + y + s/4）；level 路径 32→64→SiLU→64→64→tanh(λ)；`γₛ⊙Uₛ`；局部残差仅 S1/S2 | `manifold_context.py` `gate=Conv2d(69,1,1)`、`readout` | 一致 |
| 7 | 融合块 [L,H] 四路（concat 128→64 / L+H / H−L / αL）→SiLU→RepLocal；`d4=t4` 逐级 fuse+block，末级 refine→1×1 64→1→↑×4→256²→σ→阈值 0.5 | `models/model/str_dcr.py` `DCRDecoder` | 一致 |
| 8 | CTRL 2,047,125/1,935,810；TAR_CG 2,093,465/2,043,846；HRA_SPP 2,136,115/2,013,536；CG_HRA 2,111,287/2,061,668 | `train_scripts/CASA-MinorGoal/Run2|Run4/config.py` | 数值一致（口径为 Run4 选型，已加注） |
| 9 | 双分支边缘头：TAR 侧读跨尺度累加后的 `Dₛ`，HRA 侧读 `Uₛ`（乘 γₛ 之前）；Sobel 软标签链；8 项取均值 ×`--edge-weight` | `str_tar.py` `edge_head`、`manifold_context.py` `EdgeHead`/`edge_logits` | 一致（只缺 detach 标注） |

## 差异与改进（Run6 相对 Run4 图）

1. **梯度截断（唯一的架构级差异，也是 Run6 唯一要验证的假设）**
   代码：`edge_detach="retrieval"` 为默认，`HyperbolicContext.edge_logits` 里 `update = update.detach()`，
   `ManifoldContext` 继承。两个边缘头、`--edge-weight 0.1`、8 项平均全都不变，边缘梯度只经 TAR 的
   `Dₛ` 回传，不再经 HRA 回写链路扰动检索。
   结果：SYSU EDGE_FULL 82.01（R/P 78.97/85.28）→ EDGE_TAP_CUT 83.09（79.66/86.83），**+1.085 pp**，
   已大于 0.32 的噪声；`train_scripts/CASA-MinorGoal/Run6/config.py` 三个臂（EDGE_ALL / EDGE_FINE /
   EUCLID_ALL）均带该默认值。
2. **`--edge-scales {all,fine}`**：边缘监督只挂在最细两个尺度上的对照臂，参数与部署开销不变。
3. **`--aggregate {lorentz,euclidean}`**：把第 5 页的洛伦兹提升 + Klein 重心换成欧氏 `AₛV`，
   用来隔离"双曲几何本身"的贡献（结论：收益来自"KV 含变化证据"，不是几何）。
4. **代码有、图未画（可选项，运行时默认关闭，故不改图）**
   - `--tar-variant spp / spp1 / diff` 与 `--module difference`：论文主线的 `DifferenceContext`
     只有一处差别——KV 原型多一层 `LayerNorm(Dₛ)`，V 保持欧氏 32 维（无 `L(V)`、无重心），
     门控 66 通道（无 x/y/s 三个位置通道、无 level 路径）；差分**不做**跨尺度累加，每尺度独立 `Dₛ`。
     这与第 2、4、5、6 页全部不同，属于另一条臂的图，不在"最小改动"范围内。
   - `--detail-head`：128² 细节头 `Conv1×1(32→64)→DW3×3→SiLU→Conv1×1(64→1)`（末层零初始化，
     +2,689 参数），主输出改为 `↑(coarse,256)+↑(detail,256)`。默认关闭。
   - `--support {cell,mass}`、`--kv-hierarchy {flat,nested}`、`--freeze-tau`：图按 flat/默认画，未提。

## 本次实际改动（最小幅度）

| 页 | 形状 | 改动 |
| --- | --- | --- |
| 1–9 | `TextBox 1`（页眉） | `Run4` → `Run6`；第 9 页 `当前训练扩展` → `Run6 · 训练扩展` |
| 1 | `TextBox 7`（左上注释） | 补 `HRA 侧读 detach(Uₛ)`，与第 9 页呼应 |
| 8 | `TextBox 2`（标题） | `四组实验的时相融合、HRA 与解码器` → `Run4 四组选型与 Run6 增量` |
| 8 | `TextBox 6`（注释） | 补 Run6 增量：边缘 0.1、`--edge-scales`、`--aggregate`，实测指向第 9 页 |
| 9 | `TextBox 2`（标题） | `训练时的双分支边缘监督` → `…与 Run6 梯度截断` |
| 9 | `TextBox 6`（注释） | 写明两个头/权重不变、HRA 头改读 `detach(Uₛ)`、梯度只经 `Dₛ`、82.01→83.09（+1.085 pp） |
| 9 | `Connector 20`（Uₛ → HRA 边缘头） | 改**红色 C00000 虚线 1.5 pt**（原 222222 实线 1.05 pt），标出截断点 |
| 9 | 新增文本框 | 红色 9.4 pt `detach(Uₛ) · 梯度不回传到回写`，置于该箭头下方（2.30, 4.84, 2.40×0.22） |

未动的：所有框的位置/尺寸/填充/边框、连线走向、字号与配色、页脚、页码、第 2–7 页全部内容。
第 8 页页脚仍为 `Run4 · 四组实验配置`——那四列确实是 Run4 选型，标题已说明。

## 复核方式

`office_render` 以 150 dpi 导出第 1、8、9 页 PNG 目视：无文字溢出、无遮挡、红色虚线箭头压在
`HRA 回写 Uₛ → HRA 边缘头` 之间，注释与既有 9–11.5 pt 注释同级。
PyPowerPoint COM 在本机不可用（`0x80080005`），渲染改用 `office_render`。
