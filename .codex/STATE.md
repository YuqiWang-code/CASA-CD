# 当前状态

更新：2026-10-24。

## CASA / Run7 已完成

- 协议`casa_40k_relativecontext_v7`，实际模型与训练源码`2e60998`。
- 2026-10-24 02:21:37 CST启动，11:01:54 CST队列全部完成，共12项正式结果，失败项为空，两张GPU空闲。
- 所有实验达到完整训练预算；部署折叠检查通过，二值预测差异均为0。
- 日志与JSON已下载至`outputs/CASA/Run7`，源码归档为同目录`source_code.tar.gz`。
- 指标表159条，原147条历史记录逐单元格保留。Run7冻结TXT使用归档中的实际运行源码；current TXT使用当前工作区源码。
- LEVIR和WHU两模块均高于同轮CTRL，CDD增益较小；SYSU两模块分别低于CTRL约1.504和1.151个百分点。正式指标和结果说明见根README。

## 资料与后续

Run6冻结TXT保留a946963实际运行源码与对应历史指标，分析产物在`outputs/CASA/Run6/analysis`，设计报告在`docs/temporary/CASA_Run6_analysis_Run7_design.md`。

本次定时自动化已删除，不再周期监控。当前没有获准启动的新一轮训练；后续按用户指令推进，不恢复历史队列。连接与服务器操作见SERVER.md。
