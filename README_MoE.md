# CASA-MoE 探索线

普通RGB双时相、全监督遥感二值变化检测。探索双曲检索、欧式检索、时相频率证据和TAR的互补性，兼顾精度与实际推理开销。工作源码为`models_v1/`，与主线`models/`独立。

## 当前版本：Run2

共享MobileNetV2 → 四尺度公共时相表示 → H/E/W/T四候选 → 粗到细路由与单一DCR → 共享预测头。

| 路径 | Run2设计 |
|---|---|
| H 双曲检索 | 完整Q，224个跨尺度KV，32维自由投影的Lorentz距离，欧式V |
| E 欧式检索 | 与H共享原型、QK/V/out、尺度偏置与增益，仅距离与温度不同 |
| W 小波 | 对线性投影的带符号时相差做Haar；浅层高频零LL重建，深层使用低频差异 |
| T 时相融合 | 原生编码特征上的完整TAR，独立候选 |
| 路由 | 读取本层X与已计算的粗层解码特征，逐位置四权重；四路稠密执行 |

### 首阶段实验

LEVIR、SYSU各三组，共 **6项**：匹配TAR-only、每尺度可学习常量融合、动态空间融合。共同训练协议`casa_moe_40k_candidates_v2`：batch32、seed2333、40K按完整epoch上取整、四级BCE+Dice＋0.2共享guide。正式每epoch在test选模，结果不是独立盲测。WHU/CDD及删除专家实验在首阶段结果后决定。

| 配置 | 训练参数M | 部署参数M | CPU折叠最大误差 |
|---|---:|---:|---:|
| TAR-only | 2.110613 | 1.935617 | 1.1921e-7 |
| 可学习常量融合 | 2.212023 | 2.100452 | 5.9605e-8 |
| 动态空间融合 | 2.220151 | 2.108580 | 5.9605e-8 |

CPU检查已覆盖Haar重建与频带幅值、零时差W、浅层零LL、TAR-only精确等价、共享初始化、任务梯度及部署折叠。参数量不代替GPU延迟和显存实测。

### GPU1接续安排

2026-10-30 20:35:03 CST登记接续，PID3116514；20:38复核进程存活，状态`waiting_for_Run8`。仅使用物理GPU1，等待CASA-Pyramid Run8首阶段4项完整测试完成、相关进程退出后再运行：

1. GPU训练峰值显存与部署速度实测。
2. 三配置×两数据集，共六项3个完整epoch的train/val健康探针。
3. 全部通过后从预训练骨干全新初始化，按实测容量启动六项正式训练。

失败时停止后续启动并保留产物，不自动重试。正式前三轮损失、评估、梯度与模块诊断写入`initial_health.json`。上述登记时GPU预检、真实探针及正式训练尚未开始；实时阶段见服务器`handoff_status.json`，正式进度见`pipeline_status.json`。

日志：`/home/yqwang/outputs/CASA-CD/CASA-MoE/Run2`。实际运行副本：`runtime/20261030_203501`；源码归档已保存到本地`outputs/CASA-MoE/Run2/source_code.tar.gz`。

[模型说明](models_v1/README.md) · [Run2脚本与操作](train_scripts/CASA-MoE/Run2/README.md) · [检查与接续记录](docs/experiments/CASA-MoE/Run2/README.md)

Run1实际源码保留在冻结TXT与源码归档，当前`models_v1`不用于复现Run1。

## Run1结果与版本记录

独立探索线 **CASA-MoE / Run1（GPU1，24/24完成）** 使用[models_v1](models_v1/README.md)：共享TAR底座，双曲QKV、欧式QKV、TAR和频率四条路径进行像素动态融合。LEVIR/SYSU各9组消融，WHU/CDD各3组；部署参数1.954M，稠密执行。动态MoE相对同轮CTRL的F1差值为−0.2054/−0.6801/−0.1958/−0.0430个百分点，尚未形成净收益。[完整日志与曲线](docs/experiments/CASA-MoE/Run1/README.md) · [结果分析](docs/experiments/CASA-MoE/Run1/analysis.md) · [冻结快照](docs/temporary/models_and_metrics_CASA-MoE_Run1.txt) · [实验矩阵](train_scripts/CASA-MoE/Run1/README.md)。

MoE Run1收集后指标表共253条，既有229条记录逐单元格保留；冻结TXT使用本轮实际训练源码归档。


## 研究判断

Run1动态MoE在四库均未超过同轮CTRL及固定等权融合。Run2以匹配TAR-only和可学习常量融合检验新增结构及动态路由的价值；低参数量、路由熵或专家权重本身不证明有效性。若动态配置没有额外收益，优先简化；所有启用专家仍稠密计算，不宣称稀疏加速。

目标数据集为LEVIR、SYSU、WHU、CDD；F1努力目标分别为92%、84%、95%、97.5%，不作为统一SOTA声明。允许为有效表征小幅增参，部署参数目标≤5M，同时实测延迟和显存。完整Q、压缩KV、结构简洁、收益可归因是固定原则。

## 结果收集

使用项目工具下载日志、逐单元格保留历史指标、从实际运行源码生成冻结TXT：

```powershell
python train_scripts/CASA-MoE/Run2/collect_results.py
```

正式指标只取完整TEST RESULTS，不将探针、验证数值或停止项入表。数据、权重、凭据和`outputs/`不提交Git；公开训练记录位于`docs/experiments/CASA-MoE/`。
