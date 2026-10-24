# CASA-CD

遥感二值变化检测。当前主线 **CASA-Pyramid / Run1**，训练协议`casa_pyramid_40k_guide02_v1`。

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
    → 四级TAR时相融合与局部细化：各64通道
    → 联合跨尺度QKV / CTRL直通
        各尺度完整Q ───────────────┐
        预测变化评分 → 每尺度32个KV → 联合128个KV
                                 ↓
                       四头检索（欧氏 / 双曲）
                                 ↓
                       各尺度输出投影与残差相加
    → DCR逐级空间融合与细化
    → 共享预测头，四级变化预测
```

去掉HFEA，TAR直接将原生通道投影至64；骨干结束于320通道。QKV位于TAR之后，处理已经融合的变化表征。每尺度划分4×4个区域，每区域产生变化/背景两个原型，四尺度联合128个KV；Q保留64²+32²+16²+8²=5440个位置。四头各16维，共享Q/K/V和输出投影，每个头读取全部尺度。

共享65参数评分头预测变化概率，正/负权重分别按区域归一化。普通版使用负欧氏距离与欧氏加权平均；双曲版逐头映射到庞加莱球，用负双曲距离/2和Lorentz质心压缩、聚合，Log映射后直接读出。两个版本均不含公共质心相减或额外DWConv分支。CTRL保留同样的归一化、评分头及辅助监督，但直接将TAR特征送给DCR。

| 变体 | 训练参数 | 折叠部署参数 |
|---|---:|---:|
| M0_CTRL | 2,047,125 | 1,935,810 |
| M1_HYPER_QKV | 2,063,581 | 1,952,266 |
| M2_CA_QKV | 2,063,581 | 1,952,266 |

参数和接口已通过检查。推理期保留QKV与评分头，TAR/DCR执行折叠；参数量不等同于延迟。实现和来源见[models](models/README.md)、[SOURCES](models/SOURCES.md)。

## 训练与测试

| 设置 | 值 |
|---|---|
| 输入 / batch / seed | 256×256 / 32 / 2333 |
| 初始化 | MobileNetV2预训练；三组共同路径初始化一致；不使用探针或旧轮权重 |
| 优化器 | Adam，lr5e-4，betas(0.9,0.99)，eps1e-8 |
| Weight decay | context全部参数0，其余1e-4 |
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

## 当前实验：CASA-Pyramid / Run1

四库各双曲、普通和CTRL，共12项。LEVIR→SYSU→WHU→CDD优先，两张GPU按实测显存与留量补位，允许跨库填空。模型和训练脚本见[Run1说明](train_scripts/CASA-Pyramid/Run1/README.md)。

<!-- CASA_PYRAMID_RUN1_RESULTS_START -->

正式结果0/12。结构、几何、梯度、共同初始化、部署折叠和显存检查通过；真实LEVIR/SYSU健康短训进行中。

<!-- CASA_PYRAMID_RUN1_RESULTS_END -->

三种模型折叠检查的首级误差均约5.96e-8，二值差异为0。探针不计入正式指标。本轮同时改变编码器融合、注意力位置和评分监督，与历史轮次的总体比较不能单独归因于几何；主要机制比较使用本轮三组共同协议。

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

指标表目前159条，保留全部历史，仅完整TEST RESULTS区块可入表。新一轮收齐后正常为171条，另有新增一并保留。冻结快照使用本轮实际运行源码；历史CASA/Run6与Run7快照不覆盖。Git只提交源码、说明、指标与规定快照。连接和状态见[.codex](.codex/README.md)。

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
