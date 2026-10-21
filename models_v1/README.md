# models_v1/ — 探索性实验 / 额外任务区

**这里的东西不属主线**。主线是 `../models/`（当前 Run9：HSAM）。
本目录是一个**自包含的 v1 快照 + 探索线**：可以随便改、随便试，只要不把结果混进主线结论。

## 当前在做什么

| 文件 | 用途 |
|---|---|
| `train_stgeo.py` | **探索线 Exp1**（STG 系列）：把局部结构张量当 SPD(2)/H² 上的点，用黎曼测地距离做变化证据 |
| `test_st_geodesic.py` | 上述几何的数值验收（拉回度量 / 闭式测地距离 / 退化检查） |
| `eval_object_metrics.py` | 对象级指标评估（按 GT 面积分组） |
| `eval_stgeo_tb.py` | Exp1 的逐样本读数落盘 |
| `check_sup_wiring.py` / `epoch0_check.py` | 接线与 epoch-0 自检 |

## 为什么这里还有一份完整代码快照

`train.py` / `eval.py` / `main.py` / `model/` / `dataset/` 是 v1 的**冻结快照**，
上面的探索脚本要 import 它们。所以本目录里与 `../models/` 同名的文件**是刻意的副本**——
不要为了"去重"删掉，那会让探索线直接跑不起来。

> 2026-10-21 整理时确认：本目录与 `../models/` 的同名文件（`train.py` 除外）逐字节一致。
> 探索线若要跟进主线改动，请显式同步并记录，不要指望自动一致。

## 纪律

- 本目录的实验**不影响主线结论**。要进论文/主线，必须按
  `../train_scripts/CASA-TViM/Run9/` 那种「预注册 + 同期 CTRL + 唯一变量」的方式重开一轮。
- Exp1 的历史设计与结果已归档到 `../docs/temporary/过去的想法/CASA-CD_Exp1_*.md`。
- 许可证：与 `../models/` 同源（ChangeViT），文本在
  `../docs/参考文献/baseline/ChangeViT_LICENSE_v1.txt`。
