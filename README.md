# CASA-CD

遥感二值变化检测。当前主线 **CASA-Pyramid / Run2**，训练协议`casa_pyramid_40k_qk_euclideanv224_v2`。

## 项目定位与研究原则

面向普通RGB双时相影像的全监督遥感二值变化检测，在LEVIR、SYSU、WHU和CDD上研究有竞争力的检测精度与轻量推理，进一步探索极轻量部署和较少的可学习参数。研究贡献聚焦双曲QKV与变化感知的上下文压缩QKV，形成结构简洁、问题明确、证据充分的方法。

- **模块有明确作用。** 双曲模块需要证明几何表示能改善变化与非变化特征的区分；普通模块需要证明变化感知压缩能保留有效上下文和细节。两者均保持Q的空间位置，仅压缩KV，并分别验证设计依据与实际收益。
- **复杂度服务于问题。** 优先采用常见、可解释的运算，新增结构应对应已观察到的具体缺陷。不依赖特殊激活函数或无关新元件制造指标增益，使提升能够归因于核心机制。
- **比较使用一致协议。** 在相同骨干、训练预算、数据划分和选模规则下比较CTRL与两个模块，通过消融确认每项改动的作用；关键结论关注重复实验的稳定性和各数据集的差异。
- **效率按实际部署衡量。** 分别报告可学习参数量、部署折叠后的参数量与计算量，以及注明输入尺寸、batch和硬件的推理速度、显存占用。准确率与成本共同评价，较少参数本身不等同于更快推理。
- **设计与评价保持独立。** 使用训练集、验证集及其中的样本和中间特征分析问题、调整结构；测试集用于方案确定后的最终评价，不用于反复选择改动。领先性结论须由相同评价口径的实验支持。

## 模型

```text
双时相RGB A / B
    → 共享MobileNetV2原生四尺度：24/32/96/320通道
    → 四级TAR融合为64通道变化特征
    → 跨尺度QKV / CTRL直通
        完整Q：5440个位置，四头16维
        网格8/4/4/4 → KV数量128/32/32/32 → 联合224个KV
        欧氏Q/K 或 双曲Q/K；两版均用欧式V
        logits增加-log(源尺度KV数)，保留可学习尺度偏置
        四尺度输出投影与零初始化gamma残差相加
    → DCR逐级空间融合 → 共享预测头四级输出
```

普通版先对LayerNorm后的特征按变化/背景概率池化，再投影K/V，保持分配权重梯度。双曲版Q/K仍在各头庞加莱球内匹配，K经Lorentz质心压缩，V改为先池化再欧式投影，直接用注意力加权读出，不对V做Exp/Log或几何质心。双曲曲率-1、切空间截断2.3。CTRL保留相同评分头及辅助监督，直接将TAR特征送入DCR。

| 变体 | 训练参数 | 折叠部署参数 |
|---|---:|---:|
| M0_CTRL | 2,047,125 | 1,935,810 |
| M1_HYPER_QKV | 2,063,581 | 1,952,266 |
| M2_CA_QKV | 2,063,581 | 1,952,266 |

不含HFEA和骨干1280通道尾层。推理保留QKV与评分头，仅TAR/DCR折叠。实现和诊断定义见[models](models/README.md)，来源见[SOURCES](models/SOURCES.md)。

## 训练与测试

| 设置 | 值 |
|---|---|
| 输入 / batch / seed | 256×256 / 32 / 2333 |
| 初始化 | MobileNetV2预训练；三组共同路径初始化一致；不使用探针或旧轮权重 |
| 优化器 | Adam，lr5e-4，betas(0.9,0.99)，eps1e-8 |
| Weight decay | context.q/k/v/out四个weight为1e-4，其余context参数0，网络其余参数1e-4 |
| 学习率 | poly^0.9；首epoch且step<200时warmup |
| 主损失 | 四级BCE+Dice等权求和 |
| 评分监督 | 四尺度面积平均标签的BCEWithLogits求和，权重0.2；三组相同 |
| 增广 | resize、随机裁剪后恢复、翻转、时相交换 |
| 选模 | 独立val最高F1，相同F1更新；结束后在test评价best |
| 评价输出 | 第一级预测；gray>=128 / prob>0.5 |
| 部署检查 | strict_fp32_fold_v1，首级误差阈值2e-4，记录四级误差与二值差异 |

| 数据集 | Train / Val / Test | 实际步数 |
|---|---:|---:|
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | 40,182 |
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | 40,125 |
| WHU-CD-256 | 5,947 / 743 / 744 | 40,145 |
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | 40,248 |

## 当前实验：CASA-Pyramid / Run2

四库各双曲、普通、CTRL共12项；LEVIR→SYSU→WHU→CDD优先，两张GPU按实测显存留量并行，允许跨库填空。新模型的代码检查已通过，真实数据健康探针进行中。配置与脚本见[Run2说明](train_scripts/CASA-Pyramid/Run2/README.md)。

<!-- CASA_PYRAMID_RUN2_RESULTS_START -->

正式结果0/12。正式队列在四项健康探针通过后从MobileNetV2预训练重新初始化，不使用探针或旧Run1权重。

<!-- CASA_PYRAMID_RUN2_RESULTS_END -->

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run2
python train_scripts/CASA-Pyramid/Run2/collect_results.py
python tools/export_training_records.py --tag CASA-Pyramid --run Run2
```

本轮日志与实际源码归档位于`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run2`，checkpoint位于`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run2`，本地为`outputs/CASA-Pyramid/Run2`。正式指标目前171条，本轮十二项收齐后正常为183条，其他新增一并保留。Run2冻结TXT使用本轮实际源码，历史Run1快照保持128-token版本。

## 历史 CASA-Pyramid / Run1

原128-token、双曲V版本的完整结果与源码已冻结。



四库各双曲、普通和CTRL，共12项。LEVIR→SYSU→WHU→CDD优先，两张GPU按实测显存与留量补位，允许跨库填空。正式训练于2026-10-24 15:03:59 CST启动，源码基线`5730d13`，全部12项训练与测试于当天22:30:52 CST完成，总耗时7小时26分53秒。模型和训练脚本见[Run1说明](train_scripts/CASA-Pyramid/Run1/README.md)。

<!-- CASA_PYRAMID_RUN1_RESULTS_START -->

状态更新：2026-10-24 23:37:37 CST；队列 `complete`；正式结果 **12/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 完成 | 90.96 | 83.41 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 90.99 | 83.46 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.15 | 83.73 |
| SYSU | `M0_CTRL` | 完成 | 81.22 | 68.37 |
| SYSU | `M1_HYPER_QKV` | 完成 | 81.58 | 68.88 |
| SYSU | `M2_CA_QKV` | 完成 | 81.93 | 69.39 |
| WHU | `M0_CTRL` | 完成 | 94.08 | 88.82 |
| WHU | `M1_HYPER_QKV` | 完成 | 93.97 | 88.62 |
| WHU | `M2_CA_QKV` | 完成 | 93.78 | 88.30 |
| CDD | `M0_CTRL` | 完成 | 96.80 | 93.80 |
| CDD | `M1_HYPER_QKV` | 完成 | 96.75 | 93.70 |
| CDD | `M2_CA_QKV` | 完成 | 96.75 | 93.71 |

<!-- CASA_PYRAMID_RUN1_RESULTS_END -->

本轮相对CTRL，双曲/普通模块在LEVIR分别提高0.031/0.191个百分点，在SYSU提高0.359/0.714个百分点；WHU分别降低0.114/0.297个百分点，CDD降低0.052/0.045个百分点。普通版在LEVIR与SYSU高于双曲版，本轮尚未证明双曲几何具有稳定精度优势。两个模块部署参数均为1.952266M，精度与开销需共同评价。全部实验部署折叠检查通过，二值预测差异均为0。日志与JSON已下载至outputs/CASA-Pyramid/Run1，冻结TXT使用实际运行源码归档，当前源码TXT已同步171条正式指标。

三种模型折叠检查的首级误差均约5.96e-8，二值差异为0。探针不计入正式指标。本轮同时改变编码器融合、注意力位置和评分监督，与历史轮次的总体比较不能单独归因于几何；主要机制比较使用本轮三组共同协议。

## Run1 推理开销

RTX5090，256×256双时相输入、batch1、四级输出；折叠后30次预热、100次同步测量，随机权重。输入/权重为FP32，cuDNN允许TF32，矩阵乘TF32关闭。FLOPs仅含profiler支持的卷积/矩阵乘，双曲逐元素开销未计入。

| 变体 | 训练参数 | 部署参数 | 已统计GFLOPs | 中位延迟(ms) |
|---|---:|---:|---:|---:|
| none | 2,047,125 | 1,935,810 | 1.734 | 3.81 |
| hyperbolic | 2,063,581 | 1,952,266 | 2.113 | 7.54 |
| qkv | 2,063,581 | 1,952,266 | 2.100 | 5.53 |

## 完整训练记录

[CASA-Pyramid/Run1公开记录](docs/experiments/CASA-Pyramid/Run1/README.md)包含12项完整train_log.txt、逐epoch训练/验证CSV、实际最佳epoch、模块诊断CSV/JSONL、四库完整曲线、四项成功探针及运行环境。诊断为当轮验证首个batch的实际数值；epoch从0开始。源码输出语句与实际训练记录分别保留。

```powershell
python tools/export_training_records.py --tag CASA-Pyramid --run Run1
```

## 操作与产物

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run1
python train_scripts/CASA-Pyramid/Run1/collect_results.py
python tools/models_to_txt.py --tag CASA-Pyramid --run current
```

| 内容 | 位置 |
|---|---|
| 当前模型与入口 | models/，models/train_CASA.py |
| 本轮脚本 | train_scripts/CASA-Pyramid/Run1/ |
| 本轮日志与源码归档 | /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1/ |
| 本轮checkpoint | /share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run1/ |
| 本地产物 | outputs/CASA-Pyramid/Run1/ |
| 数据 | /share_datasets/CD/<dataset>/{A,B,label,list}/ |
| 预训练 | /home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth |
| 正式指标 | docs/experiment_metrics.xlsx |
| 当前/冻结快照 | docs/temporary/models_and_metrics_CASA-Pyramid_current.txt、models_and_metrics_CASA-Pyramid_Run1.txt |

指标表目前171条，本轮新增12项，原159条历史记录逐单元格保留。仅完整TEST RESULTS区块可入表，另有新增一并保留。冻结快照使用本轮实际运行源码；历史CASA/Run6与Run7快照不覆盖。Git提交源码、说明、指标、规定快照与docs/experiments中的公开训练记录。连接和状态见[.codex](.codex/README.md)。

## 历史 CASA / Run7

原MobileNetV2+HFEA、编码器四层独立QKV的12项已完成并收集，模型源码基线2e60998，结果提交09e45ae。其延迟为CTRL5.86ms、双曲17.35ms、普通9.02ms（RTX5090，256×256，batch1；FP32输入/权重，cuDNN允许TF32，30预热/100同步调用），仅适用于旧结构。

<!-- CASA_RUN7_RESULTS_START -->

状态更新：2026-10-24 12:45:34 CST；队列 `complete`；正式结果 **12/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 完成 | 91.01 | 83.51 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.54 | 84.40 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.33 | 84.04 |
| SYSU | `M0_CTRL` | 完成 | 82.29 | 69.91 |
| SYSU | `M1_HYPER_QKV` | 完成 | 80.79 | 67.77 |
| SYSU | `M2_CA_QKV` | 完成 | 81.14 | 68.27 |
| WHU | `M0_CTRL` | 完成 | 93.99 | 88.66 |
| WHU | `M1_HYPER_QKV` | 完成 | 94.57 | 89.70 |
| WHU | `M2_CA_QKV` | 完成 | 94.55 | 89.67 |
| CDD | `M0_CTRL` | 完成 | 97.43 | 94.99 |
| CDD | `M1_HYPER_QKV` | 完成 | 97.47 | 95.06 |
| CDD | `M2_CA_QKV` | 完成 | 97.47 | 95.06 |

<!-- CASA_RUN7_RESULTS_END -->

CASA/Run7相对CTRL，双曲/普通模块在LEVIR分别提高0.527/0.316个百分点，在WHU提高0.581/0.561个百分点；CDD均提高约0.038个百分点。SYSU分别降低1.504/1.151个百分点，仍需解决该数据集上的性能损失。上述为单seed结果，CDD的小幅差异尚需重复实验验证。

启动、健康探针与初期损失见[Run7说明](train_scripts/CASA/Run7/README.md)。Run6四库特征、完整验证支路诊断和Run7机制依据见[分析报告](docs/temporary/CASA_Run6_analysis_Run7_design.md)。

## 历史 CASA / Run6

四库各运行双曲、普通QKV和CTRL，共12项。优先级为 **LEVIR→SYSU→WHU→CDD**，两张GPU按实测显存预算并行补位。启动时GPU0运行两项双曲实验，GPU1运行普通模块和两项CTRL，共五路。

两模块各通过3个LEVIR epoch、4个SYSU epoch的真实数据健康训练。正式训练从全新初始化开始，使用MobileNetV2预训练权重。运行源码基线为 `a946963`，本轮源码保存在日志根目录的 `source_code.tar.gz`。

<!-- CASA_RUN6_RESULTS_START -->

状态更新：2026-10-24 02:02:52 CST；队列 `complete`；正式结果 **12/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 完成 | 91.55 | 84.42 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.05 | 83.57 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.62 | 84.53 |
| SYSU | `M0_CTRL` | 完成 | 82.42 | 70.09 |
| SYSU | `M1_HYPER_QKV` | 完成 | 80.83 | 67.83 |
| SYSU | `M2_CA_QKV` | 完成 | 82.39 | 70.05 |
| WHU | `M0_CTRL` | 完成 | 93.44 | 87.69 |
| WHU | `M1_HYPER_QKV` | 完成 | 94.73 | 89.99 |
| WHU | `M2_CA_QKV` | 完成 | 94.16 | 88.97 |
| CDD | `M0_CTRL` | 完成 | 97.43 | 94.99 |
| CDD | `M1_HYPER_QKV` | 完成 | 97.45 | 95.03 |
| CDD | `M2_CA_QKV` | 完成 | 97.47 | 95.07 |

<!-- CASA_RUN6_RESULTS_END -->

## 已知限制

当前实验为seed2333，尚不能主张稳定普适提升。变化评分的预测误差会影响原型，两个原型可在无变化区域接近；注意力熵或gamma单项不能作为失活定论，需结合任务梯度、支路贡献和验证数据。TAR是有序时相融合，整网不保证严格时相交换不变或相同输入恒零。超参数、选样和设计使用训练/验证数据，测试集保留最终评价。
