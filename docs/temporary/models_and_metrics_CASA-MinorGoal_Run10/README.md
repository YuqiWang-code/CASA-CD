# CASA-MinorGoal / Run10 快照

「把三个工作（TAR / HRA / DCR）做扎实」这一轮的收尾材料。协议 `casa_minor_run10_lean_v1`，
24 项 = 6 臂 × SYSU/LEVIR × seed 2334/2335，40k step、bs32、val 选模、阈值 0.5、只在 GPU1 上两槽跑。

- `models_and_metrics.txt` — 生成时的 `models/` 代码快照（42 个文件）。**指标一节目前留空**，
  按 config 的 `JOBS` 列出 24 个待跑项；收齐后重跑同一条命令覆盖即可：
  `python tools/models_to_txt.py --tag CASA-MinorGoal --run Run10 --out docs/temporary/models_and_metrics_CASA-MinorGoal_Run10/models_and_metrics.txt`
  （带 `--metrics placeholder` 就是本次的空表版本）。训练实跑用的是 `runtime/` 冻结副本，源码身份以 `source_code.tar.gz` 为准。
- `CASA-MinorGoal_Run10_总体模型结构.pptx` — 用户自己维护的结构图，本轮不由脚本重画。

## 六条臂与门限

| 臂 | TAR | 检索 | 说明 |
|---|---|---|---|
| `D_CTRL` | reparam | 无 | 新主干对照 |
| `D_CG` | cg（竞争门 + 尺度自适应局部对比） | 无 | 单看 TAR 改动 |
| `R_FULL` | cg | manifold，检索 0–3 层、224 原型、support mass | 单看检索改动 |
| `R_RES` | cg | 同上，但只在 16²/8² 检索 | 减派发开销 |
| `R_COMPACT` | cg | 0–3 层、32 原型、support confidence | 减原型与先验口径 |
| `R_LEAN` | cg | 2–3 层、32 原型、confidence | 两者合并 |

门限取 `max(0.36, 1.5σ)`，σ=0.39（Run10 前的同配置重跑噪声）：`D_CG` vs `D_CTRL` ≥ +0.59、
`R_FULL` vs `D_CG` ≥ +0.59，三个减法臂 vs `R_FULL` 只要求不低于 −0.36。

## 与 Run7 的差别（都在既有模块内，不新增模块）

- **DCR**：主输出从 64² 抬到 128²（`FineRefine` 把 `d1` 与编码器 stem 16ch 拼接后过 `1×1→BN→SiLU→DW3×3→BN→SiLU→1×1`），
  深监督权重由四级等权改成 `1/.5/.25/.125`，`DCRDecoder.refine` 并入 `FineRefine`。
  `FineRefine` 在部署形态保留两个 BN：128² 上有死通道（`running_var` 到 1e-45），折 BN 会把 FP32 舍入放大成 4.8e-2 的概率差。
- **TAR**：门从「只放大不抑制」改成正负两路的竞争，`g=σ(1×1(pos−neg))`、`fused=block(g·n(content)+(1−g)·n(delta))`，
  局部对比核按尺度取 `(3,5,7,7)`。
- **HRA**：`−mean` 改成可学的 `mean_gate`（初值 0 保留原行为）、`--retrieval-scales` 只跑粗两级、
  `--compact-prototypes` 把 224 个原型压到 32、`--support confidence` 把先验从「数量」换成「置信」。
