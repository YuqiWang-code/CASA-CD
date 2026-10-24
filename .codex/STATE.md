# 当前状态

更新：2026-10-24。

## CASA-Pyramid / Run1

协议`casa_pyramid_40k_guide02_v1`。用户授权去HFEA、原生四尺度经TAR融合后做联合跨尺度QKV。新脚本`train_scripts/CASA-Pyramid/Run1`，入口`models/train_CASA.py`。

三组均有预测评分辅助监督；每尺度32个KV，联合128个，完整Q、四头16维，普通欧氏/双曲几何匹配预算。双曲与普通训练2,063,581、部署1,952,266参数，CTRL训练2,047,125、部署1,935,810参数。

结构、几何、梯度、共同初始化、优化器/RNG、部署折叠与显存检查通过。四项真实探针均通过，各模块LEVIR3epoch、SYSU4epoch。正式已从MobileNetV2预训练全新初始化启动，12项，LEVIR→SYSU→WHU→CDD优先，两卡按实测显存补位。不得重复启动。

2026-10-24 15:03:59 CST正式启动，源码基线`5730d13`，supervisor PID1278526。观察时间2026-10-24T15:09:03+08:00，五项均超过200 step并完成至少三个完整epoch；主损失、评分损失下降，日志未见NaN/Inf/OOM或异常退出。两GPU按预算补位，实际活动分配以实时状态为准。

| 实验 | GPU | PID | 完整epoch | step | 首三轮总损失 |
|---|---:|---:|---:|---:|---|
| M1_HYPER_QKV/LEVIR-CD-256 | 0 | 1278528 | 6 | 1332 | 2.1771 → 1.3086 → 1.1589 |
| M2_CA_QKV/LEVIR-CD-256 | 1 | 1278559 | 5 | 1200 | 2.1809 → 1.3001 → 1.1561 |
| M0_CTRL/LEVIR-CD-256 | 1 | 1278589 | 5 | 1200 | 2.1860 → 1.3181 → 1.1563 |
| M1_HYPER_QKV/SYSU-CD-256 | 0 | 1278621 | 3 | 1125 | 2.7225 → 2.0634 → 1.8491 |
| M2_CA_QKV/SYSU-CD-256 | 1 | 1278651 | 3 | 1125 | 2.7276 → 2.0697 → 1.8466 |

显存预算GPU0/1为30914/31072MiB；双曲/普通/CTRL预留12392/11008/8832MiB。后续自动补位，不固定每卡两路。当前没有正式TEST结果，探针不计指标。源码归档、environment.json、source_baseline.json与health_summary.json保存在本轮outputs。

本轮日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run1`，本地`outputs/CASA-Pyramid/Run1`。状态用`python .codex/status.py --tag CASA-Pyramid --run Run1`。

## 历史与指标

旧CASA/Run7十二项完成并收集，指标159条，旧147条逐单元格保留。旧源码基线2e60998，原服务器models保存在`/home/yqwang/outputs/CASA-CD/CASA/Run7/archive/server_models_before_Pyramid_Run1`；对应source_code.tar.gz和冻结TXT保留。旧队列不恢复。

本轮收齐后应171条正式记录，其他新增一并保留。新冻结TXT用本轮归档，当前TXT用工作区。没有活动自动化，不自动创建监控。用户后续指令优先。
