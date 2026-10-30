# CASA-MoE Run2：GPU1接续记录

2026-10-30 20:35:03 CST，接续进程PID3116514启动，状态`waiting_for_Run8`。先等待CASA-Pyramid Run8四项完整TEST及训练/调度进程退出，再使用物理GPU1；不操作其他任务。

实际运行副本：`/home/yqwang/outputs/CASA-CD/CASA-MoE/Run2/runtime/20261030_203501`。冻结源码`source_code.tar.gz`已下载至本地`outputs/CASA-MoE/Run2`。协议`casa_moe_40k_candidates_v2`，Python3.10.21、PyTorch2.14.0+cu132。

## 已完成检查

| 配置 | 训练参数M | 部署参数M | CPU折叠最大误差 |
|---|---:|---:|---:|
| TAR-only | 2.110613 | 1.935617 | 1.1921e-7 |
| 可学习常量融合 | 2.212023 | 2.100452 | 5.9605e-8 |
| 动态空间融合 | 2.220151 | 2.108580 | 5.9605e-8 |

Haar重建、LL幅度、零时差W、浅层零LL、224KV、T-only同权重前向精确等价、三配置主干初始化一致、三次更新后专家梯度、真实探针检查入口均已通过CPU验证。结果见[cpu_checks.json](cpu_checks.json)。

## 接续流程

1. Run8完成且相关进程退出，GPU1至少18GiB空闲。
2. 使用冻结副本完成GPU profile，记录包括梯度诊断的batch32训练峰值与部署batch1延迟。
3. LEVIR/SYSU × TAR-only/常量/动态，六项3个完整epoch的train/val健康探针。
4. 全部通过后从MobileNetV2预训练重新初始化，启动相同六配置的40K正式训练，按实测容量并行。

正式数据优先LEVIR再SYSU，使用test逐epoch选模，非独立盲测。初期三轮训练损失、评估、任务梯度与模块诊断写入`initial_health.json`。失败保留日志与权重，不自动重启。WHU/CDD扩展等待首阶段结果。

**当前GPU预检、真实数据探针和正式训练尚未开始。** 状态以服务器`handoff_status.json`和`pipeline_status.json`为准。[完整设计与脚本](../../../../train_scripts/CASA-MoE/Run2/README.md)。
