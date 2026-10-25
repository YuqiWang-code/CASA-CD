# CASA-CD

普通RGB双时相影像的全监督遥感二值变化检测。当前主线 **CASA-Pyramid / Run2**，协议`casa_pyramid_40k_reliableqk_testselect_v2`。

## 项目定位与研究原则

在LEVIR、SYSU、WHU、CDD四库研究检测精度与轻量推理，贡献聚焦双曲QKV和变化感知压缩QKV。保持完整Q、压缩KV；结构简洁、问题明确，新增机制应能通过对照验证。避免无关组件和特殊激活函数堆叠，准确率、部署参数、计算、实测延迟与显存共同评价。单seed的小差异不作为稳定领先证据。

同轮三组使用相同骨干、增强、损失、预算和选模规则。当前按每epoch的test F1选模，最终test复测属于**test参与选模的结果**，不能称独立盲测，也不能将与历史val选模的差异全部归因于模块。结构诊断与短训使用train/val，测试阈值固定0.5。

## 模型

```text
双时相RGB A / B
  → 共享MobileNetV2原生四尺度24/32/96/320通道
  → TAR：四级时相融合，统一64通道
  → 跨尺度QKV（或CTRL直通）
      LN → 变化/背景评分 → 区域原型及其可靠度
      完整Q：5440个位置；网格8/4/4/4 → 224个KV
      共享Q/K投影：欧氏距离或双曲距离；V为欧式
      注意力 = softmax(-距离/温度 + 尺度偏置 - log(KV数量) + log(可靠度))
      输出投影 → 每尺度gamma残差相加
  → DCR逐级解码 → 共享预测头
      训练：四级logits深监督；推理：只输出最终一级概率图
```

原型由预测类别权重`m=p`或`1-p`归一化池化，可靠度`r=Σm²/Σm`保留其类别支持信息。全背景格的变化原型会被降权；区域内少量高置信变化仍可获得高可靠度，不按面积直接压低小目标。共享Q/K减少4,096参数，并使距离比较使用同一投影。双曲版对Q/K做Exp、K用Lorentz质心压缩，曲率-1、切空间截断2.3；两版V均先池化再线性投影，直接`A@V`读出。普通版K也先池化再投影。CTRL保留相同评分监督。

| 数据集特征 | 对应设计目的 |
|---|---|
| [LEVIR](https://justchenhao.github.io/LEVIR/)：建筑变化稀疏，小建筑与边界重要 | 细尺度128个KV、完整Q；可靠度保留少量高置信变化 |
| [WHU](https://gpcv.whu.edu.cn/data/building_dataset.html)：密集建筑、大小尺度差异及背景纹理 | 联合四尺度检索；TAR/DCR保留局部细节 |
| [SYSU](https://github.com/liumency/SYSU-CD)：变化类型丰富，植被、阴影和时相差异明显 | 变化/背景监督约束语义原型，降低低支持原型影响 |
| [CDD](https://isprs-archives.copernicus.org/articles/XLII-2/565/2018/)：季节和外观差异显著 | 同步几何增强、时相交换与有监督原型；跨尺度关联 |

这些是机制目标，四库精度收益由正式对照验证。双曲映射本身不保证类别可分性提高。

| 变体 | 训练参数 | 折叠部署参数 | context参数 |
|---|---:|---:|---:|
| M0_CTRL | 2,047,125 | 1,935,810 | 193 |
| M1_HYPER_QKV | 2,059,485 | 1,948,170 | 12,553 |
| M2_CA_QKV | 2,059,485 | 1,948,170 | 12,553 |

实现见[models](models/README.md)，来源见[SOURCES](models/SOURCES.md)。

## 固定训练与评价设置

| 设置 | 值 |
|---|---|
| 输入 / batch / seed | RGB 256×256 / 32 / 2333 |
| 初始化 | MobileNetV2预训练；每项全新初始化；三组共同路径初值一致 |
| 读取与归一化 | A/B各自BGR→RGB，保持时相顺序；除255；ImageNet均值(.485,.456,.406)、标准差(.229,.224,.225) |
| resize / mask | 图像双线性，标签最近邻；标签gray≥128 |
| 裁剪 | 训练以0.5概率四边各裁0～8像素，再恢复256；A/B/标签同步 |
| 翻转 / 旋转 | 水平、垂直各0.5；均匀选择0/90/180/270度，A/B/标签同步 |
| 时相交换 | 训练以0.5概率交换A/B；标签保持对应空间位置 |
| 光度增强 / 评估增强 | 无独立时相颜色扰动；评估无随机增强、无TTA |
| 优化器 | Adam，lr5e-4，betas(.9,.99)，eps1e-8 |
| Weight decay | context.qk/v/out.weight为1e-4，其余context为0；网络其余为1e-4 |
| 学习率 / 预算 | 全局前200步warmup，之后poly^0.9；40K按完整epoch上取整 |
| 主损失 | 四级BCEWithLogits + sigmoid Dice，等权求和 |
| 评分损失 | 四尺度面积平均标签的BCEWithLogits求和×0.2，三组一致 |
| 精度 | FP32，AMP关闭；cuDNN TF32开启，matmul TF32关闭，benchmark开启 |
| 评估与选模 | 每epoch（包括0）在test评估，第一级F1最高选best，相同值更新 |
| 最终结果 | 折叠best后在同一test复测；明确标记test参与选模、非独立测试 |
| 部署检查 | strict_fp32_fold_v1；关闭TF32检查，首级最大差阈值2e-4并记录二值差 |

参考[ChangeMamba的2024官方训练脚本](https://github.com/ChenHongruixuan/ChangeMamba/blob/72597fa4a1fde0bffa94c94302a6dc848898be13/changedetection/script/train_MambaBCD.py)使用test参与选模；其原实现每500步评估并按Kappa选模，本项目采用每epoch与F1。该设置不代表所有RSCD工作的统一规范。

| 数据集 | Train / Val / Test | epoch / 实际步数 |
|---|---:|---:|
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | 181 / 40,182 |
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | 107 / 40,125 |
| WHU-CD-256 | 5,947 / 743 / 744 | 217 / 40,145 |
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | 129 / 40,248 |

## 当前实验：CASA-Pyramid / Run2

Run2于2026-10-25 13:58:47 CST启动，21:35:15 CST完成，历时7小时36分28秒；源码基线`9f515ef`。12/12项正式训练及测试完成，无失败，调度器已退出。2026-10-26核查两张GPU空闲。全部部署折叠检查通过，最大绝对差3.934e-6，二值差均为0。 配置与初期观察见[Run2说明](train_scripts/CASA-Pyramid/Run2/README.md)。

相对同轮CTRL，双曲/普通模块在LEVIR的F1分别提高0.185/0.148个百分点，WHU提高0.212/0.345个百分点；SYSU分别降低0.529/0.832个百分点；CDD双曲提高0.021个百分点、普通降低0.044个百分点。本轮呈现数据集相关的收益，SYSU是共同薄弱项；单seed微小差异不作为稳定提升证据。所有结果均为test参与选模后的复测。

<!-- CASA_PYRAMID_RUN2_RESULTS_START -->

状态更新：2026-10-26 00:19:21 CST；队列 `complete`；正式结果 **12/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 完成 | 90.98 | 83.45 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.16 | 83.76 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.13 | 83.70 |
| SYSU | `M0_CTRL` | 完成 | 83.37 | 71.48 |
| SYSU | `M1_HYPER_QKV` | 完成 | 82.84 | 70.70 |
| SYSU | `M2_CA_QKV` | 完成 | 82.54 | 70.26 |
| WHU | `M0_CTRL` | 完成 | 94.35 | 89.31 |
| WHU | `M1_HYPER_QKV` | 完成 | 94.56 | 89.69 |
| WHU | `M2_CA_QKV` | 完成 | 94.70 | 89.93 |
| CDD | `M0_CTRL` | 完成 | 96.91 | 94.00 |
| CDD | `M1_HYPER_QKV` | 完成 | 96.93 | 94.04 |
| CDD | `M2_CA_QKV` | 完成 | 96.87 | 93.92 |

<!-- CASA_PYRAMID_RUN2_RESULTS_END -->

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run2
python train_scripts/CASA-Pyramid/Run2/collect_results.py
python tools/export_training_records.py --tag CASA-Pyramid --run Run2
```

源码、日志和JSON：`outputs/CASA-Pyramid/Run2`；服务器日志根`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run2`，checkpoint根`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run2`。指标表共185条正式记录，原173条逐单元格保留，本轮新增12条。独立val选模的两项LEVIR结果另存`Run2_val`（CTRL F1 90.8261%、普通90.8842%）；历史冻结TXT与其运行源码对应。

## 当前推理开销

RTX5090，256×256，FP32；训练batch32实测6步、前2步预热。推理为折叠后predict()单输出、batch1、30预热/100同步计时、随机权重；cuDNN TF32开、matmul TF32关。FLOPs仅含支持的卷积和矩阵乘，未计双曲逐元素运算。

| 模式 | batch32峰值保留显存(MiB) | 同步训练步(ms) | 单输出推理(ms) | 已统计GFLOPs |
|---|---:|---:|---:|---:|
| none | 8200 | 54.01 | 3.68 | 1.734 |
| hyperbolic | 13518 | 103.80 | 6.42 | 2.165 |
| qkv | 11966 | 81.53 | 5.47 | 2.156 |

## 诊断依据与适用范围

固定验证样本的原型分析见[Run2设计说明](train_scripts/CASA-Pyramid/Run2/README.md)。低类别支持原型占用注意力是本轮针对的问题；SYSU的CTRL同样出现后期泛化回落，不能单独归因于QKV。可靠度先验依赖评分质量，仍可能接受高置信伪变化；共享Q/K约束也会减少映射自由度。当前结构、增强和选模协议同时变化，跨轮提升不能单独归因于某一项。

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

本轮相对CTRL，双曲/普通模块在LEVIR分别提高0.031/0.191个百分点，在SYSU提高0.359/0.714个百分点；WHU分别降低0.114/0.297个百分点，CDD降低0.052/0.045个百分点。普通版在LEVIR与SYSU高于双曲版，本轮尚未证明双曲几何具有稳定精度优势。两个模块部署参数均为1.952266M，精度与开销需共同评价。全部实验部署折叠检查通过，二值预测差异均为0。日志与JSON已下载至outputs/CASA-Pyramid/Run1，冻结TXT使用实际运行源码归档，Run1收集完成时指标表共有171条正式记录。

三种模型折叠检查的首级误差均约5.96e-8，二值差异为0。探针不计入正式指标。本轮同时改变编码器融合、注意力位置和评分监督，与历史轮次的总体比较不能单独归因于几何；主要机制比较使用本轮三组共同协议。

## Run1 推理开销

RTX5090，256×256双时相输入、batch1、四级输出；折叠后30次预热、100次同步测量，随机权重。输入/权重为FP32，cuDNN允许TF32，矩阵乘TF32关闭。FLOPs仅含profiler支持的卷积/矩阵乘，双曲逐元素开销未计入。

| 变体 | 训练参数 | 部署参数 | 已统计GFLOPs | 中位延迟(ms) |
|---|---:|---:|---:|---:|
| none | 2,047,125 | 1,935,810 | 1.734 | 3.81 |
| hyperbolic | 2,063,581 | 1,952,266 | 2.113 | 7.54 |
| qkv | 2,063,581 | 1,952,266 | 2.100 | 5.53 |

## 完整训练记录

[CASA-Pyramid/Run2公开记录](docs/experiments/CASA-Pyramid/Run2/README.md)包含12项完整日志、逐epoch训练/test选模曲线、实际最佳epoch、模块诊断、正式结果、探针及环境。Run2冻结TXT来自实际训练源码归档；current TXT对应当前models。

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

指标表目前173条，全部既有记录逐单元格保留。仅完整TEST RESULTS区块可入表，另有新增一并保留。冻结快照使用本轮实际运行源码；历史CASA/Run6与Run7快照不覆盖。Git提交源码、说明、指标、规定快照与docs/experiments中的公开训练记录。连接和状态见[.codex](.codex/README.md)。

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

当前实验为seed2333，尚不能主张稳定普适提升。变化评分的预测误差会影响原型，两个原型可在无变化区域接近；注意力熵或gamma单项不能作为失活定论，需结合任务梯度、支路贡献和验证数据。TAR是有序时相融合，整网不保证严格时相交换不变或相同输入恒零。选样与结构诊断使用训练/验证数据；当前Run2的test参与checkpoint选择，最终复测不是独立盲测。
