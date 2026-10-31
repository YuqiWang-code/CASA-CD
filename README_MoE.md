# CASA-MoE 探索线

普通RGB双时相、全监督遥感二值变化检测。探索双曲检索、欧式检索、时相频率证据和TAR的互补性，兼顾精度与实际推理开销。工作源码为`models_v1/`，与主线`models/`独立。

## 当前版本：Run3

共享MobileNetV2 → 四尺度公共时相表示 → H/E/W/T四候选 → 粗到细路由与单一DCR → 共享预测头。

| 路径 | 设计 |
|---|---|
| H 双曲检索 | 完整Q、224个跨尺度KV、32维Lorentz距离、欧式V |
| E 欧式检索 | 与H共享原型、QK/V/out、尺度偏置与增益，仅距离与温度不同 |
| W 小波 | 对线性投影的带符号时相差做Haar；浅层高频零LL重建，深层低频差异 |
| T 时相融合 | 原生编码特征上的完整TAR，独立候选 |
| 路由 | 读取本层LN(X)、LN(TAR)及已计算的粗层LN(D)，逐位置四权重 |

Run3为路由补充正在参与竞争的TAR特征，新增4096参数。精度假设是更完整的候选输入能改善选择；效果由同轮对照检验。H/E的加权注意力合并后执行一次AV/out，保留完整梯度；诊断时分开读取两个专家。小波、几何及训练损失保持原设置，四路仍稠密执行。

### 训练与运行

LEVIR、SYSU各动态融合、TAR-only、可学习常量融合，共 **6项**。仅物理 **GPU0，单路串行**，先LEVIR三项，再SYSU三项；至少保留1GiB空闲显存。其他进程及GPU1主线分别运行。

协议`casa_moe_40k_candidate_evidence_v3`：batch32、seed2333、40K按完整epoch上取整、四级BCE+Dice＋0.2共同guide。Adam全网lr5e-4、warmup200、poly0.9，context卷积/线性权重解耦WD1e-4。正式训练从MobileNetV2预训练全新初始化，每epoch在test选模并复测，结果不是独立盲测。

| 配置 | 训练参数M | 部署参数M |
|---|---:|---:|
| TAR-only | 2.110613 | 1.935617 |
| 可学习常量融合 | 2.212023 | 2.100452 |
| 动态空间融合 | 2.224247 | 2.112676 |

结构检查已覆盖合并/分离读出的输出与全部输入、参数、路由梯度等价，Haar重建、零时差W、公共初始化、TAR-only精确等价和部署折叠。动态版batch32 GPU进程峰值实测13156MiB，容量足够单路训练。验证使用train/val，六项三轮探针通过后正式启动。

日志：`/home/yqwang/outputs/CASA-CD/CASA-MoE/Run3`；权重：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MoE/Run3`。冻结源码为`runtime/20261031_125233`及本轮`source_code.tar.gz`。当前工作源码不用于直接复现旧轮次。

[模型说明](models_v1/README.md) · [Run3脚本](train_scripts/CASA-MoE/Run3/README.md) · [审查与启动记录](docs/experiments/CASA-MoE/Run3/README.md)

<!-- CASA_MOE_RUN3_RESULTS_START -->

2026-10-31 13:21:47 CST正式启动，GPU0单路串行。六项三轮train/val探针全部通过，当前运行LEVIR动态融合，正式结果0/6。已观察正式前四轮：主损失2.145→1.193→1.075→1.030，选模F1为80.20%→83.78%→85.90%→83.79%；梯度有限且各分支有效，保留最佳checkpoint，继续训练。

<!-- CASA_MOE_RUN3_RESULTS_END -->

## Run2结果

2026-10-31 10:27:23 CST完成6/6项，无失败。

| 数据集 | TAR-only F1 (%) | 静态融合 F1 (%) | 动态融合 F1 (%) | 动态相对TAR (pp) |
|---|---:|---:|---:|---:|
| LEVIR | 91.0245 | 91.1507 | 91.0649 | +0.0405 |
| SYSU | 82.5655 | 82.7655 | 83.5636 | +0.9981 |

动态融合的收益集中于SYSU；LEVIR的1px边界F1相对TAR提高0.5383pp，但召回降低0.5497pp。SYSU动态版最佳至末轮训练损失继续下降，选模损失上升，说明后期泛化仍是共同薄弱项。H/E在SYSU最佳权重的四层残差余弦为0.9982–0.9995，几何互补仍需进一步证据。

[完整日志与曲线](docs/experiments/CASA-MoE/Run2/README.md) · [冻结TXT](docs/temporary/models_and_metrics_CASA-MoE_Run2.txt) · [脚本](train_scripts/CASA-MoE/Run2/README.md)。本轮源码归档与指标保持独立。

## Run1结果

24/24项完成。动态MoE相对同轮CTRL的F1差值在LEVIR/SYSU/WHU/CDD为−0.2054/−0.6801/−0.1958/−0.0430pp。完整记录及当时四专家结构保存在[Run1日志](docs/experiments/CASA-MoE/Run1/README.md)、[分析](docs/experiments/CASA-MoE/Run1/analysis.md)和[冻结TXT](docs/temporary/models_and_metrics_CASA-MoE_Run1.txt)。

## 研究原则与限制

目标数据集为LEVIR、SYSU、WHU、CDD；F1努力目标分别为92%、84%、95%、97.5%，不作为统一SOTA声明。允许为有效表征小幅增参，部署参数目标≤5M，同时实测延迟和显存。完整Q、压缩KV、结构简洁、收益可归因是固定原则。

单seed结果不足以证明稳定提升；test参与选模须明确标识。低参数量、路由熵或专家权重本身不证明有效性，整条MoE收益也不能全部归为双曲几何。所有专家稠密计算，不宣称稀疏加速。GPU0与其他任务共享，当前延迟测量用于容量和运行观察，不与独占GPU结果直接比较。WHU/CDD和删除专家实验根据首阶段证据安排。

## 结果收集

```powershell
python train_scripts/CASA-MoE/Run3/collect_results.py
python train_scripts/CASA-MoE/Run3/export_records.py
```

使用项目tools下载日志、汇表和生成实际运行源码的冻结TXT。只取完整TEST RESULTS，逐单元格保留全部历史指标；本轮启动前指标表共263条。探针不作为正式结果。数据、权重、凭据和`outputs/`不提交Git，公开训练记录位于`docs/experiments/CASA-MoE/`。
