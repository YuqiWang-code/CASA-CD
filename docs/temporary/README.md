# 源码快照与实验资料

| 内容 | 位置 |
|---|---|
| 当前模型源码与全部正式指标 | `models_and_metrics_CASA_current.txt` |
| Run7实际训练源码与159条正式指标 | `models_and_metrics_CASA_Run7.txt` |
| Run6训练源码与指标 | `models_and_metrics_CASA_Run6.txt` |
| 其他轮次冻结快照 | 对应的 `models_and_metrics_<tag>_<run>.txt` |
| 历史方案、复盘、诊断与旧说明 | [过去的想法](过去的想法/) |
| 完整正式指标表 | [experiment_metrics.xlsx](../experiment_metrics.xlsx) |

当前模型与训练入口见 [models](../../models/README.md) 和 [CASA/Run7](../../train_scripts/CASA/Run7/README.md)。

Run6的运行源码归档位于 `outputs/CASA/Run6/source_code.tar.gz`，基线为 `a946963`。本轮结果收集从该归档生成Run6快照；`current`快照记录当前工作区源码。历史轮次的源码归档、日志和权重按各自轮次保存。

Run6的四库诊断与Run7设计见[分析报告](CASA_Run6_analysis_Run7_design.md)。Run7实际运行源码基线为`2e60998`，对应归档为`outputs/CASA/Run7/source_code.tar.gz`。
