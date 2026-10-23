# CASA-CD

遥感二值变化检测。当前主线 **CASA-STRNet**：WDMF-Net 的 MobileNetV2＋HFEA 编码器，接四尺度处理模块、TAR 时空重参数化跳层连接和 DCR 重参数化解码器。

**当前代码基线：CASA/Run6，协议 `casa_40k_changevalues_v6`。** 普通 QKV 同时保留有符号变化与变化幅度；双曲 QKV 先计算逐位置变化位移，再做几何压缩。历史正式结果共135条，Run6已通过四项健康检查，正在两卡进行正式训练。

| 编码器 | 跳层与解码器 | 当前代码基线 | 状态 |
|---|---|---|---|
| MobileNetV2 + HFEA，四级各64通道 | MultiScaleTAR + DCR，dim=64 | CASA/Run6 变化 Value | 两卡五路训练中 |

## 模型

```text
A / B → shared MobileNetV2
    → 五级特征 [16,24,32,96,320]
    → shared HFEA → 四级双时相特征 [64,64,64,64]，分辨率 64/32/16/8
    → 每层：无模块 / 双曲原型 QKV / 普通变化感知原型 QKV
    → MultiScaleTAR 时空重参数化跳层融合
    → DCR 自顶向下解码，输出四级特征
    → 共享 1×1 预测头，上采样到 256×256，四级 BCE+Dice 深监督
```

两个处理模块的输出均进入 **TAR 跳层连接**：保留双时相特征，由 TAR 融合，再通过 DCR 逐级传递到解码结果。

| 实验 | 四级处理模块 | 训练参数量 |
|---|---|---:|
| `M0_CTRL` | 两个模块都关闭 | 3,100,404 |
| `M1_HYPER_QKV` | 模块一：完整双曲 Q、压缩 K、逐位置位移后压缩的 V | 3,134,332 |
| `M2_CA_QKV` | 模块二：欧氏双原型压缩、有符号与绝对差分 Value | 3,151,228 |

**普通压缩 QKV**：逐 token LayerNorm 后计算时相差异，每个4×4 cell产生变化、稳定两个对齐原型。完整 Q 以余弦相似度和可学习温度查询另一时相的压缩 K。全局 V 拼接投影后的有符号时相差与绝对差，分别保留方向和幅度；时相交换只对有符号部分取反。全局结果与原有局部有符号差分拼接，经输出投影和零初始化γ加回输入。

**双曲压缩 QKV**：32维嵌入经裁剪和指数映射进入c=1的庞加莱球。Q/K使用共享的Möbius度量投影，K按同位置双曲距离产生的权重做几何压缩。V使用独立的Möbius投影，在全部对应位置计算双向变化位移，再对位移做几何压缩。负双曲距离注意力检索上下文，在流形内聚合变化 V，经Log映射与局部几何差分融合后回接残差。

两者均保持 Q 的 `N=HW` 个位置；每个区域两个原型使 KV 为 `N/8`。四层模块参数互不共享，同层双时相共享参数。Run2 使用旧设计，其 KV 为 `N/16`，两个模块的训练参数均为 3,133,176。

MobileNetV2、HFEA 和训练设置来自 [WDMF-Net 官方仓库](https://github.com/LYT-Works/WDMF-Net)，版本 `453c5493b2aab88c908ced9135b677285059ec70`；参考源码在 [`others/WDMF-Net/`](others/WDMF-Net/REFERENCE.md)。新版普通模块保留 SAT 的完整 Q / 压缩 KV 结构；双曲模块结合 Deep Change Monitoring 的流形表征与裁剪方法，采用几何原型压缩和变化位移聚合。旧 VMamba HSAM/HQKV 不参与当前模型前向。

## 训练协议

训练采用 WDMF-Net 的策略。当前模型与训练入口使用 **`casa_40k_changevalues_v6`**。Run3的 **`casa_40k_separateqkv_v3`**、Run4 的 **`casa_40k_supervisedkv_v4`**、Run5 的 **`casa_40k_pairnorm_v5`** 属于历史实验协议，复现时使用相应轮次的源码归档。

| 设置 | 值 |
|---|---|
| 输入 / batch / seed | 256×256 / 32 / 2333 |
| 优化器 | Adam，lr=5e-4，betas=(0.9,0.99)，eps=1e-8，主体 weight decay=1e-4；四层处理模块的全部参数 weight decay=0 |
| 学习率 | poly^0.9，沿用上游前 200 step warmup |
| 增广 | resize、随机裁剪后恢复、水平/垂直翻转、时相交换 |
| 损失 | 四级输出 BCE+Dice 等权求和 |
| 选模与测试 | epoch 0 后逐 epoch 验证，以验证集最高 F1 选模，训练结束测试该模型 |
| 预算 | 目标 40,000 step，按上游实现向上取整到完整 epoch |
| 标签 / 预测阈值 | `gray>=128` / `prob>0.5` |

| 数据集 | Train / Val / Test | 实际 optimizer step |
|---|---:|---:|
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | 40,125 |
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | 40,182 |
| WHU-CD-256 | 5,947 / 743 / 744 | 40,145 |
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | 40,248 |

CDD 按相同预算扩展到本轮；`gray>=128` 保持 JPG 标签的正确二值化。结构、健康检查与两卡调度见 [`train_scripts/CASA/Run6/`](train_scripts/CASA/Run6/README.md)。

当前模型及全部历史指标快照为 `docs/temporary/models_and_metrics_CASA_current.txt`；Run6快照为 `docs/temporary/models_and_metrics_CASA_Run6.txt`。各历史轮次保留匹配的源码归档与冻结快照。Run3与Run5的对照见 [模块分析](docs/temporary/CASA_Run3_Run5_analysis.md)。

## Run6 训练

四个数据集各运行双曲、普通QKV与CTRL，共12项。优先级为LEVIR→SYSU→WHU→CDD，两卡按实测显存预算填充空闲容量。正式训练从全新初始化开始。

训练前，两模块各完成3个LEVIR epoch与4个SYSU epoch，记录逐轮训练损失、梯度范数、分支/残差幅度、几何分布及完整验证结果。健康检查使用完整40K学习率计划，产物与正式实验分开。

<!-- CASA_RUN6_HEALTH_START -->

| 数据集 / 模块 | epoch / step | 逐轮平均训练损失 | 短训验证F1 (%) |
|---|---:|---|---:|
| LEVIR / 双曲 | 3 / 666 | 2.110 → 1.172 → 1.068 | 87.87 |
| LEVIR / 普通 | 3 / 666 | 2.109 → 1.175 → 1.061 | 88.41 |
| SYSU / 双曲 | 4 / 1500 | 2.592 → 1.959 → 1.723 → 1.582 | 79.50 |
| SYSU / 普通 | 4 / 1500 | 2.594 → 1.945 → 1.710 → 1.567 | 78.27 |

四项均通过真实数据健康检查，四层分支、残差及任务梯度均非零。双曲查询半径CV最低0.167，查询裁剪比例最高0.66%。短训验证结果不作为正式测试指标。

正式训练初期损失下降，未见NaN、OOM或进程异常退出。双曲查询半径CV最低0.162；部分层注意力仍接近均匀，验证诊断采样中的位移裁剪比例最高28.45%。初期数值状态与最终精度收益分别记录。

<!-- CASA_RUN6_HEALTH_END -->

<!-- CASA_RUN6_RESULTS_START -->

状态更新：2026-10-23 12:02:42 CST；队列 `running`；正式结果 **0/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 训练中：2100/40182 | — | — |
| LEVIR | `M1_HYPER_QKV` | 训练中：1776/40182 | — | — |
| LEVIR | `M2_CA_QKV` | 训练中：1998/40182 | — | — |
| SYSU | `M0_CTRL` | 训练中：1875/40125 | — | — |
| SYSU | `M1_HYPER_QKV` | 训练中：1600/40125 | — | — |
| SYSU | `M2_CA_QKV` | 待启动 | — | — |
| WHU | `M0_CTRL` | 待启动 | — | — |
| WHU | `M1_HYPER_QKV` | 待启动 | — | — |
| WHU | `M2_CA_QKV` | 待启动 | — | — |
| CDD | `M0_CTRL` | 待启动 | — | — |
| CDD | `M1_HYPER_QKV` | 待启动 | — | — |
| CDD | `M2_CA_QKV` | 待启动 | — | — |

<!-- CASA_RUN6_RESULTS_END -->

日志与JSON：`outputs/CASA/Run6`；服务器权重：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA/Run6`。完整TEST RESULTS区块进入指标表，原有135条正式记录全部保留。

## Run5 历史结果

Run5 于 **2026-10-23 09:21 CST** 完成全部 **12/12** 项训练与正式测试。运行源码基线为 `2675fd9`，十二项均达到完整 40K 预算并通过部署折叠检查，最大误差 **3.13e-06**，二值预测差异均为 0。

<!-- CASA_RUN5_RESULTS_START -->

状态更新：2026-10-23 09:35:51 CST；队列 `complete`；正式结果 **12/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 完成 | 91.42 | 84.19 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.68 | 84.64 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.55 | 84.42 |
| SYSU | `M0_CTRL` | 完成 | 82.73 | 70.55 |
| SYSU | `M1_HYPER_QKV` | 完成 | 82.06 | 69.58 |
| SYSU | `M2_CA_QKV` | 完成 | 82.17 | 69.74 |
| WHU | `M0_CTRL` | 完成 | 94.43 | 89.44 |
| WHU | `M1_HYPER_QKV` | 完成 | 93.88 | 88.46 |
| WHU | `M2_CA_QKV` | 完成 | 93.41 | 87.63 |
| CDD | `M0_CTRL` | 完成 | 97.42 | 94.98 |
| CDD | `M1_HYPER_QKV` | 完成 | 97.47 | 95.07 |
| CDD | `M2_CA_QKV` | 完成 | 97.47 | 95.06 |

<!-- CASA_RUN5_RESULTS_END -->

### Run5 结果解读

双曲 QKV 在 LEVIR 的 F1 为 **91.68%**、IoU 为 **84.64%**，相对同期 CTRL 提升 **0.26 / 0.45 个百分点**；普通 QKV 提升 **0.14 / 0.23 个百分点**。CDD 两个模块的 F1 均约 **97.47%**，相对 CTRL 分别提高 **0.05 / 0.04 个百分点**。

SYSU 双曲、普通 QKV 的 F1 分别为 **82.06% / 82.17%**，相对 CTRL 低 **0.67 / 0.56 个百分点**。WHU 分别为 **93.88% / 93.41%**，相对 CTRL 低 **0.55 / 1.02 个百分点**。

四库双曲训练均未重现半径整体饱和。全部逐轮诊断中，半径 CV 最低约 **0.160**，查询裁剪比例最高约 **5.27%**，方向集中指标最高约 **0.369**。归一化控制了本轮几何退化，精度收益因数据集而异。

正式日志和 JSON 已下载至 `outputs/CASA/Run5`。指标表完整保留原有 **123** 条记录，并加入本轮 **12** 条，共 **135** 条；模型代码与全部指标快照为 `docs/temporary/models_and_metrics_CASA_Run5.txt`。Run3、Run4 冻结快照及服务器权重继续保留。

### Run5 已知问题

SYSU 三组最佳验证模型分别出现在第 9 / 4 / 11 轮（CTRL / 双曲 / 普通），后期验证指标下降，泛化退化仍然存在。本轮为 seed 2333 的单次实验，尚未进行多种子重复。


## Run4 历史记录

Run4 于 **2026-10-23 00:01 CST** 停止，未产生完整正式测试结果。日志和 JSON 已下载至 `outputs/CASA/Run4`，服务器 checkpoint 全部保留；代码和指标快照为 `docs/temporary/models_and_metrics_CASA_Run4.txt`。指标表仍保留 **123 条**正式记录。

<!-- CASA_RUN4_RESULTS_START -->

状态更新：2026-10-23 00:03:55 CST；队列 `stopped`；正式结果 **0/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 已停止 | — | — |
| LEVIR | `M1_HYPER_QKV` | 已停止 | — | — |
| LEVIR | `M2_CA_QKV` | 已停止 | — | — |
| SYSU | `M0_CTRL` | 已停止 | — | — |
| SYSU | `M1_HYPER_QKV` | 已停止 | — | — |
| SYSU | `M2_CA_QKV` | 待启动 | — | — |
| WHU | `M0_CTRL` | 待启动 | — | — |
| WHU | `M1_HYPER_QKV` | 待启动 | — | — |
| WHU | `M2_CA_QKV` | 待启动 | — | — |
| CDD | `M0_CTRL` | 待启动 | — | — |
| CDD | `M1_HYPER_QKV` | 待启动 | — | — |
| CDD | `M2_CA_QKV` | 待启动 | — | — |

<!-- CASA_RUN4_RESULTS_END -->

## Run3 历史结果

普通与双曲模块均已通过两轮 LEVIR 真实数据短训（各 444 step），四层分支与任务梯度均非零。正式训练使用全新初始化和完整 40K 预算。三组实验为 CTRL、双曲 QKV、普通压缩 QKV。

启动时 GPU0 运行 LEVIR/SYSU 双曲模块，GPU1 运行 LEVIR 普通模块与 LEVIR/SYSU CTRL。两卡显存分别约 **29.2 / 29.9 GiB**，利用率采样为 **95% / 96%**；空闲容量按 LEVIR → SYSU → WHU → CDD 优先级自动补充。

<!-- CASA_RESULTS_START -->

状态更新：2026-10-22 22:46:15 CST；队列 `complete`；正式结果 **12/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 完成 | 91.11 | 83.67 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.48 | 84.30 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.17 | 83.77 |
| SYSU | `M0_CTRL` | 完成 | 81.78 | 69.18 |
| SYSU | `M1_HYPER_QKV` | 完成 | 81.47 | 68.73 |
| SYSU | `M2_CA_QKV` | 完成 | 81.01 | 68.07 |
| WHU | `M0_CTRL` | 完成 | 93.95 | 88.59 |
| WHU | `M1_HYPER_QKV` | 完成 | 94.32 | 89.25 |
| WHU | `M2_CA_QKV` | 完成 | 93.94 | 88.58 |
| CDD | `M0_CTRL` | 完成 | 97.45 | 95.02 |
| CDD | `M1_HYPER_QKV` | 完成 | 97.47 | 95.06 |
| CDD | `M2_CA_QKV` | 完成 | 97.48 | 95.09 |

<!-- CASA_RESULTS_END -->

### Run3 LEVIR 三组结果

三组均完成 **40,182 step**。双曲 QKV 的 F1 **91.48%**、IoU **84.30%**，相对同期 CTRL 提升 **0.37 / 0.63 个百分点**；普通压缩 QKV 为 **91.17% / 83.77%**，相对 CTRL 提升 **0.06 / 0.10 个百分点**。CTRL 为 **91.11% / 83.67%**。这组结果使用 seed 2333，尚未进行多种子重复。

三组部署折叠最大误差均不超过 **1.23e-6**，二值预测差异为 0。

### Run3 四库结果

| 数据集 | CTRL F1 / IoU (%) | 双曲 QKV F1 / IoU (%) | 普通 QKV F1 / IoU (%) |
|---|---:|---:|---:|
| LEVIR | 91.11 / 83.67 | 91.48 / 84.30 | 91.17 / 83.77 |
| SYSU | 81.78 / 69.18 | 81.47 / 68.73 | 81.01 / 68.07 |
| WHU | 93.95 / 88.59 | 94.32 / 89.25 | 93.94 / 88.58 |
| CDD | 97.45 / 95.02 | 97.47 / 95.06 | 97.48 / 95.09 |

双曲模块在 LEVIR、WHU 上的 F1 均高于同期 CTRL 约 0.37 个百分点。SYSU 上双曲、普通模块的精确率分别提高 1.96、1.97 个百分点，召回率下降 2.27、3.11 个百分点，F1 分别下降 0.31、0.78 个百分点。CDD 三组接近。结果为 seed 2333 的单次实验。

十二项均达到完整 epoch 的 40K 预算并通过部署折叠检查，二值预测差异均为 0。正式日志和 JSON 已收集至 `outputs/CASA/Run3`；指标表保留原有 111 条并纳入本轮 12 条，共 123 条。Run3 的模型代码和指标快照保存在 `docs/temporary/models_and_metrics_CASA_Run3.txt`，运行源码保存在 `outputs/CASA/Run3/source_code.tar.gz`。

## Run2 历史结果

Run2 于 **2026-10-22 11:41 CST** 停止。SYSU、LEVIR、WHU 三组均已完成正式测试，CDD 三项已停训并保留产物。Run2 代码和权重已归档保留；新版模块使用独立的 Run3 目录。

<!-- WDMF_RESULTS_START -->

状态更新：2026-10-22 11:46:54 CST；Run2 队列 `stopped`；正式结果 **9/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| SYSU | `M0_CTRL` | 完成 | 81.62 | 68.95 |
| SYSU | `M1_HYPER_QKV` | 完成 | 81.75 | 69.13 |
| SYSU | `M2_CA_QKV` | 完成 | 82.85 | 70.72 |
| LEVIR | `M0_CTRL` | 完成 | 91.46 | 84.26 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.28 | 83.96 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.39 | 84.14 |
| WHU | `M0_CTRL` | 完成 | 93.92 | 88.53 |
| WHU | `M1_HYPER_QKV` | 完成 | 94.07 | 88.80 |
| WHU | `M2_CA_QKV` | 完成 | 94.53 | 89.63 |
| CDD | `M0_CTRL` | 已停止 | — | — |
| CDD | `M1_HYPER_QKV` | 已停止 | — | — |
| CDD | `M2_CA_QKV` | 已停止 | — | — |

Run1 SYSU CTRL 已完成：F1 **82.65%**，IoU **70.44%**；作为独立历史对照保留。

<!-- WDMF_RESULTS_END -->

正式指标来自每条 `train_log.txt` 最后一个完整的 `=== TEST RESULTS ===` 区块。验证行用于选模，不作为最终测试结果。

### SYSU 三组结果

纯变化感知压缩 QKV 取得本轮 SYSU 最高 F1 **82.85%**、IoU **70.72%**，相对同期 CTRL 提升 **1.22 / 1.77 个百分点**；双曲 QKV 的 F1 **81.75%**、IoU **69.13%**，相对同期 CTRL 提升 **0.13 / 0.18 个百分点**。三组均完成 40,125 step。

### LEVIR 三组结果与部署校验

LEVIR 三组均完成 **40,182 step**。CTRL 的 F1 **91.46%**、IoU **84.26%**；双曲 QKV 为 **91.28% / 83.96%**，相对 CTRL 为 **−0.18 / −0.30 个百分点**；普通压缩 QKV 为 **91.39% / 84.14%**，相对 CTRL 为 **−0.07 / −0.12 个百分点**。

部署等价校验采用 **`strict_fp32_fold_v1`**：关闭 cuDNN benchmark 和 TF32，主输出最大绝对误差阈值为 **2e-4**，记录四尺度误差与二值预测差异，随后恢复原后端设置用于正式测试。

LEVIR 模块二从已完成 40,182 step 的权重通过校验并完成测试；模块一从 **39,294 step** 的模型、优化器及选模状态补完剩余 **888 step**，达到 **40,182 step**，也已完成测试。两项折叠误差分别为 **2.80e-6 / 7.15e-7**，二值预测均无差异。CTRL 的折叠误差为 **1.28e-6**，二值预测无差异。恢复前产物保留于日志 `archive/20261022_075845` 和 checkpoint `recovery_backups/20261022_075845` 目录。[折叠诊断记录](docs/temporary/CASA-WDMF_Run2_LEVIR_fold_diagnosis_2026-10-22.txt)。

### WHU 三组结果

普通压缩 QKV 的 F1 **94.53%**、IoU **89.63%**，相对同期 CTRL 提升 **0.62 / 1.10 个百分点**；双曲 QKV 为 **94.07% / 88.80%**，相对 CTRL 提升 **0.15 / 0.27 个百分点**。

## 验证与资源

### 当前基线验证

当前 Run3 结构已通过几何运算、四尺度接口、主任务梯度、公共路径初始化、部署折叠和续训检查。双曲、普通模块均能严格加载 Run3 checkpoint；CPU 四尺度最大折叠误差为 `5.96e-8`，二值预测差异为 0。

### Run4 验证

受监督分配头通过单像素变化占比保留、BCE/Dice 数值、四层损失平均、零残差系数下辅助梯度、时相交换对称性和三变体完整网络检查。CPU 四尺度部署折叠误差均为 `5.96e-8`，二值预测差异为 0。

满 batch、单进程训练 profile 的峰值 allocated 显存：CTRL **8,873 MiB**、双曲模块 **14,321 MiB**、普通模块 **10,924 MiB**；训练 step 用时约 **0.078 / 0.133 / 0.085 s**。调度预留结合真实数据探针的 peak reserved 和 CUDA 进程开销计算。

### 历史验证

新版的几何运算、四层接口、相同输入零增强、时相交换对称性、全部任务梯度、公共路径初始化和部署折叠检查通过。两个新版模块的 CPU 四尺度最大折叠误差均为 `5.96e-8`。以下满 batch 资源与 SYSU 八轮探针数据属于旧 Run2。

- MobileNetV2 预训练加载 312 个模型张量，无缺失；分类器权重不参与特征编码。
- 三个变体的参数组覆盖、四级接口、公共路径初始化、梯度与残差激活、TAR/DCR 部署折叠测试均通过，最大输出误差约 `6e-8`。
- 两个模块均通过 SYSU 八个完整 epoch、3,000 step 的健康短训，四层分支输出和全部任务梯度均非零；[逐层诊断](docs/temporary/CASA_Run1_Run2_scripts/Run2/README.md#sysu-health-results)。
- 满 batch 单进程峰值显存：CTRL **8,873 MiB**，模块一 **11,931 MiB**，模块二 **10,150 MiB**；相应训练 step 用时约 **0.084 / 0.096 / 0.080 s**。每张 32 GB 卡并发两项，两张卡合计四项。
- 数值非有限时终止当前队列并保留日志；模型不使用旧几何模块的退化硬门。

## 产物与结果收集

服务器代码：`/home/yqwang/projects/CASA-CD/`。环境：`casacd`，使用已配置的 PyTorch 2.14/CUDA 13.2 运行时。

```bash
# 最新完成轮次的记录
tail -20 /home/yqwang/outputs/CASA-CD/CASA/Run5/pipeline.log
cat /home/yqwang/outputs/CASA-CD/CASA/Run5/pipeline_status.json

# 本地下载历史日志，以及保存当前模型快照
python tools/download_training_results.py --tag CASA --run Run5
python tools/models_to_txt.py --tag CASA --run current
```

| 产物 | 位置 |
|---|---|
| 预训练权重 | `/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth` |
| 数据 | `/share_datasets/CD/<dataset>/{A,B,label,list}/` |
| 最新完成轮次日志 | `/home/yqwang/outputs/CASA-CD/CASA/Run5/<variant>/<dataset>/` |
| 最新完成轮次 checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA/Run5/<variant>/<dataset>/` |
| 本地日志与结果 | `outputs/CASA/Run5/` |
| 指标表 | [`docs/experiment_metrics.xlsx`](docs/experiment_metrics.xlsx) |
| 当前模型与指标快照 | `docs/temporary/models_and_metrics_CASA_current.txt` |
| Run5 冻结快照 | `docs/temporary/models_and_metrics_CASA_Run5.txt` |

历史实验使用对应轮次的源码归档与冻结快照；`current` 快照记录当前 `models/` 与全部历史正式指标。汇表使用 `tools/extract_metrics_to_excel.py`，历史指标缺少对应日志时拒绝覆盖现有工作簿。

## 目录

```text
models/
  train_CASA.py / profile_CASA.py / test_CASA.py
  model/casa_net.py
  model/mobilenet_v2.py / encoder_fusion.py
  model/layers/change_context_qkv.py / hyperbolic_context_qkv.py
  model/layers/context_compression.py / poincare_ops.py
  model/str_tar.py / str_dcr.py / str_reparam.py
  dataset/casa_transforms.py
train_scripts/CASA/Run3/     三组×四库的训练与收集
others/WDMF-Net/                 上游参考源码
tools/                          通用收集、汇表、快照和对象指标工具
outputs/                        本地实验产物，不进入 Git
docs/temporary/                 实验记录与 txt 快照
```

## 已知限制与历史

Run1 SYSU 模块一的 stage1 投影权重与 γ 衰减到约 `1e-41`，分支输出和任务梯度均为零；stage2–4 仍有效。该实验已停止，剩余旧队列取消，SYSU CTRL 已完成测试，结果作为独立历史对照保留，日志和检查点独立保存。Run1 代码归档于服务器 `code_backups/CASA-WDMF_Run1_20261021_2305.tar.gz`。

Run2 SYSU 的两个处理模块均完成 **40,125 step**，各 106 次逐 epoch 诊断中四层输出均有限。模块一各层半径 CV 始终非零，末轮 stage1 `gamma=0.236`、半径 `CV=0.172`；模块二各层 gamma 始终非零，末轮 stage1 `gamma=-0.269`、注意力熵 `0.914`。正式测试结果见上表。

HES-Net 原文报告浅层 HSAM 可能放大噪声；本轮检验四层均加入压缩几何 QKV 的结果。Run2 在正式队列前分别对两个模块运行八个 SYSU epoch 的健康探针，检查四层分支输出和任务梯度。长期稳定性和准确率由完整训练判定。

LEVIR 模块一的旧 checkpoint 未保存随机数状态，本次续训保留优化器和总步数，并重新以 seed=2333 初始化随机流；训练轨迹与不中断运行不同。新 checkpoint 已保存完整随机状态。

SYSU 的 Run1 CTRL F1 为 **82.65%**，Run2 CTRL 为 **81.62%**，两次相差 **1.03 个百分点**。Run2 三个变体各运行一次，当前差值未包含重复试验方差。

本轮采用 MobileNet、40K 目标步数和独立验证集选模。此前 VMamba 主线的 80K/test-as-val 结果属于不同协议，应以每轮同期 CTRL 衡量模块贡献。Run10 已于 2026-10-21 停止，SYSU CTRL 最后记录 **39,375/80,000 step**，无正式 TEST 结果。

- [VMamba Run9 / Run10 记录](docs/temporary/CASA-VMamba_Run9_Run10_2026-10-21.md)
- [Run1–Run8c 历史结论与指标](docs/temporary/历史版本结论与指标_2026-10-21.md)
- [服务器与数据统一说明](docs/RSML-3_服务器环境与变化检测数据统一说明.md)

Run2 的原始日志和检查点保留，正式实验各自使用独立目录。权重与 `outputs/` 不进入 Git；参考论文随 `docs/` 保存。
