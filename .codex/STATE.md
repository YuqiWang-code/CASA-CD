# 当前状态

更新：2026-10-24 02:28:18 CST。

## CASA / Run7 正式训练

- 协议`casa_40k_relativecontext_v7`；实际模型/训练源码`2e60998`，日志根目录与本地outputs/CASA/Run7均有source_code.tar.gz。
- 2026-10-24 02:21:37 CST启动，supervisor PID1240766。唯一队列train_scripts/CASA/Run7/pipeline.py，入口models/train_CASA.py。
- LEVIR→SYSU→WHU→CDD优先，共12项；两卡按实测预算补位。当前五路均完成至少三个完整epoch，损失下降、无NaN/Inf/OOM或退出。
- 显存预留：双曲14986MiB、普通11582MiB、CTRL9534MiB。具体活跃进程以python .codex/status.py --run Run7为准。
- 四个真实健康探针通过，正式训练从MobileNetV2预训练新初始化，不使用探针或Run6权重。
- 两模块各一个机制改动：普通局部相对变化分配；双曲公共质心参照的相对几何上下文。模型可学习参数不增加。

## 已完成资料

Run6十二项正式结果已收集，指标表147条，原135条逐单元格保留。Run6冻结TXT为a946963实际运行源码，不能用Run7覆盖。验证样本与特征在outputs/CASA/Run6/analysis，分析报告在docs/temporary/CASA_Run6_analysis_Run7_design.md。

Run7启动、探针和首轮损失见train_scripts/CASA/Run7/README.md；当前源码与Run7冻结快照在docs/temporary。部署延迟和参数已记录，双曲模块相对Run6有额外延迟成本。

本次定时任务已完成Run7正式启动与初期观察，并已删除，不再周期监控。后续按用户指令收集Run7结果。不要重复启动、恢复历史队列、自动重启新失败项或覆盖运行源码；失败保留日志和checkpoint并报告具体原因。
