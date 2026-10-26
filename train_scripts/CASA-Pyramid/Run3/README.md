# CASA-Pyramid / Run3

协议 `casa_pyramid_40k_pairbanks_contrastwd0_v3`。四库各CTRL、双曲H、普通E，共12项；LEVIR→SYSU→WHU→CDD优先，两张RTX5090按实测峰值与1GiB留量并行补位。设计依据见[Run3设计文档](../../../docs/temporary/CASA-Pyramid_Run3_design.md)。

## 完成结果

Run3于2026-10-26 02:57:17 CST启动，11:53:44 CST完成，历时8小时56分27秒。12/12项正式训练及测试完成，无失败；调度器已退出，两张GPU空闲。全部部署折叠检查通过，最大绝对差5.513e-6，二值差均为0。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) | 双曲/普通相对CTRL (pp) |
|---|---:|---:|---:|---:|
| LEVIR | 90.5037 | 90.4977 | 90.5365 | -0.006 / +0.033 |
| SYSU | 81.9320 | 82.2920 | 82.1898 | +0.360 / +0.258 |
| WHU | 94.6343 | 94.3497 | 94.3585 | -0.285 / -0.276 |
| CDD | 95.3094 | 95.4613 | 95.5150 | +0.152 / +0.206 |

相对同轮CTRL，双曲/普通模块在SYSU的F1提高0.360/0.258个百分点，CDD提高0.152/0.206个百分点；LEVIR为−0.006/+0.033个百分点，WHU为−0.285/−0.276个百分点。两个模块相对Run2在四库均降低，CTRL在LEVIR、SYSU、CDD也降低；共同训练设置与模块变化需要分别归因。

[完整日志与曲线](../../../docs/experiments/CASA-Pyramid/Run3/README.md) · [结果分析](../../../docs/experiments/CASA-Pyramid/Run3/analysis.md)

## 模块

共享MobileNetV2原生四尺度→TAR→成对上下文→DCR→共享预测头。上下文额外接收四层A/B，保留完整Q和224个KV。

- E：同层共享投影得到32维绝对差D，X=LN(T+WD·D)。每尺度区域均值与变化/背景平滑均值之差组成两库，各112个token、分别softmax；独立Q/K，4头×16维，每库V为4头×8维。类别均值包含1个区域均值伪计数。S1/S2写回D及边界门调制的HP(D)。
- H：A/B投影为4头×8维，Exp后构造精确两点Fréchet中点及Log残差，平移至共同原点得到v；检索使用中点m与u=abs(v)。uniform/change同权重压缩m、u、V为224个配对token，联合双曲中点距离和差异幅度距离；V保持欧式。S1/S2写回u及边界门调制的abs(HP(v))。
- 固定3×3二项式滤波，reflect padding，不降采样。context gamma4个、detail gamma2个均零初始化，投影正常初始化。
- CTRL保留LN/guide和S1/S2共享边界头，使用相同辅助损失；不含成对投影与检索。CTRL边界头为64→1，两模块边界头为128→1。

## 设计诊断

固定SYSU val前256张的旧Run2检查：CTRL/普通/双曲F1分别87.60/87.03/87.53%，内部Recall89.72/85.57/85.07%，边界F1@1px31.27/37.47/36.65%。旧模块已改善边界，内部召回仍弱于CTRL；去掉可靠度先验没有改善该子集，关闭context则下降约1.5～1.6个百分点。设计动机是分开保留共同上下文与成对差异，不能将去先验单项称为已验证改进。数据与样例在`outputs/CASA-Pyramid/Run3/analysis`。

子集采用固定顺序，面积1～64/65～256/257～1024/>1024的变化连通域数量分别6/10/49/257；最小目标样本少，不能据此推断完整四库的细目标表现。诊断不计正式指标。

## 共同训练协议

RGB256、batch32、seed2333，MobileNetV2预训练全新初始化。40K按整epoch上取整，LEVIR/SYSU/WHU/CDD为181/107/217/129轮。每epoch在test以最高F1选best，再在同一test复测；test参与选模，非独立盲测。

Adam betas(.9,.99)、eps1e-8；骨干lr5e-5，新层lr5e-4，每步warmup/poly保留0.1倍率。普通对比K/V weight设WD0；其余context卷积/线性weight及主体WD1e-4，其他context参数WD0。四级BCEWithLogits+Dice主损失；四级guide BCE权重0.2；S1/S2边界BCEWithLogits+Dice权重0.05。边界标签取原mask3×3膨胀减腐蚀，replicate padding，再maxpool到对应feature范围。

沿用RGB ImageNet归一化；同步裁剪、HV翻转、90度旋转和时相交换。无新增光度扰动。FP32，AMP关、cuDNN TF32开、matmul TF32关；部署验证strict_fp32_fold_v1。

## 验证与启动

`validate.sh`检查几何恒等式与梯度、双库与配对压缩、局部通路、模型接口/部署、优化器倍率、边界标签和数据同步，再测batch32显存。`probe.sh`执行LEVIR双曲3轮、普通8轮和SYSU两版各4轮短训。SYSU CTRL骨干倍率1/0.1的12轮val对照已完成；均从预训练新初始化，探针不计正式指标。短训比较不能证明完整40K的后期泛化。

```bash
bash train_scripts/CASA-Pyramid/Run3/validate.sh
bash train_scripts/CASA-Pyramid/Run3/probe.sh
bash train_scripts/CASA-Pyramid/Run3/launch.sh
```

`pipeline.py`是唯一调度器；实际显存见profile.json、scheduling.json和placement.json。正式前保存source_code.tar.gz、环境与源码基线。正式权重不使用任何探针或Run2权重。新日志在`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run3`，权重在`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run3`。

## Run3 实测开销

RTX5090，256×256、FP32；训练batch32测6步，前2步预热。部署为折叠后单输出predict、batch1、30预热/100同步计时，随机权重。cuDNN TF32开、matmul TF32关。FLOPs只统计支持的卷积和矩阵乘；双曲逐元素几何与直接距离运算未完整计入。

| 变体 | batch32峰值保留显存(MiB) | 同步训练步(ms) | 单输出推理(ms) | 已统计GFLOPs |
|---|---:|---:|---:|---:|
| none | 8202 | 54.51 | 3.69 | 1.735 |
| hyperbolic | 14870 | 148.28 | 8.67 | 2.037 |
| qkv | 9426 | 69.72 | 5.95 | 2.136 |

双曲版以更高的实际计算开销换取成对几何表示；参数少于普通版，但推理更慢。是否值得由同轮精度与开销共同判断。

## 结果管理

只取完整TEST区块。本轮12项已收齐，指标表197条，原185条历史单元格保留。Run3冻结TXT使用实际训练源码归档，current TXT使用工作源码；Run2冻结快照不变。

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run3
python train_scripts/CASA-Pyramid/Run3/collect_results.py
python tools/models_to_txt.py --tag CASA-Pyramid --run current
```

## 启动检查记录

几何恒等式与FP32/FP64梯度、相同输入、时相表示对称、双库与配对压缩、任务梯度、LR倍率、对比投影免衰减及数据变换检查通过。三组未训练模型的strict FP32部署折叠首级最大差为5.96e-8，二值差为0。

普通版的对比K/V采用WD0，避免弱梯度阶段的耦合衰减把对比投影压向零。均值库、其他投影及H/CTRL优化器设置保持固定。健康检查记录投影权重RMS、两库读出RMS及主任务梯度，并对LEVIR普通版观察8个完整epoch。

### 骨干学习率短训对照

SYSU CTRL从相同初始化、seed及增强开始，比较骨干LR倍率1.0与0.1，各12epoch/4500步，评估使用val。两组均通过数值检查。

| val指标 (%) | 倍率1.0 | 倍率0.1 |
|---|---:|---:|
| 最佳F1（均第8轮） | 80.0503 | 80.5987 |
| 末轮F1 | 79.3846 | 80.2468 |
| 后四轮平均F1 | 79.1474 | 79.2011 |
| 末轮内部Recall | 75.9526 | 79.0679 |
| 末轮Boundary F1@1px | 24.2549 | 25.0307 |
| 末轮Precision | 85.8453 | 83.9450 |

采用0.1作为本轮共同设置。短训呈现召回/精度取舍，后四轮平均F1差距很小；不能据12轮单seed证明四库后期泛化已改善。完整记录位于本轮`probes/sysu_ctrl_lr{1,01}`与`analysis/lr_comparison.json`。

### 健康短训

所有短训使用train/val，正式训练从预训练重新初始化。

| 数据集 | 模块 | epoch / step | 末轮主损失 | guide损失 | boundary损失 | val F1 (%) | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| LEVIR | hyperbolic | 3 / 666 | 1.2574 | 0.4111 | 1.6348 | 83.673 | 通过 |
| LEVIR | qkv | 8 / 1776 | 0.9999 | 0.3323 | 1.3422 | 87.357 | 通过 |
| SYSU | hyperbolic | 4 / 1500 | 1.4872 | 1.1130 | 2.0173 | 78.887 | 通过 |
| SYSU | qkv | 4 / 1500 | 1.4820 | 1.1209 | 2.0209 | 78.417 | 通过 |

LEVIR普通版第8轮的对比K/V权重RMS为0.1213/0.1023，四层对比检索的query attention TV为0.1877～0.2661、读出RMS为1.0037～1.2146，任务梯度有限且非零。均值库和对比库均参与学习。对应数据见`analysis/contrast_wd_comparison.json`。

### 正式启动

2026-10-26 02:57:17 CST启动，源码基线`0f714f5`，supervisor PID1376272。12项从MobileNetV2预训练全新初始化；两卡按容量运行5项，空闲后自动补位。

| GPU | 首批实验 | PID |
|---|---|---:|
| 0 | LEVIR M1_HYPER_QKV | 1376274 |
| 0 | SYSU M2_CA_QKV | 1376366 |
| 1 | LEVIR M2_CA_QKV | 1376300 |
| 1 | LEVIR M0_CTRL | 1376333 |
| 1 | SYSU M0_CTRL | 1376397 |

实际调度预留H/E/CTRL为15774/10508/8922MiB，GPU0/1预算30914/31072MiB，保留额外1GiB。运行源码、环境、短训与显存记录已保存在本轮日志根目录。实时状态见`.codex/STATE.md`。

### 正式初期观察

观察时间：2026-10-26T03:02:17+08:00。首批5项均完成至少3个epoch，主损失下降；无NaN、OOM或异常退出，模块诊断均为有限值。

| 实验 | 已完成epoch | 首个记录主损失 | 最新主损失 |
|---|---:|---:|---:|
| M1_HYPER_QKV/LEVIR-CD-256 | 4 | 2.2786 | 1.1576 |
| M2_CA_QKV/LEVIR-CD-256 | 5 | 2.2768 | 1.1002 |
| M0_CTRL/LEVIR-CD-256 | 5 | 2.2813 | 1.1098 |
| M2_CA_QKV/SYSU-CD-256 | 3 | 2.5632 | 1.6530 |
| M0_CTRL/SYSU-CD-256 | 3 | 2.5672 | 1.6940 |

普通对比库投影、检索差异和读出幅度保持有效；双曲四层上下文及局部残差持续参与学习。检查数据见`startup_health.json`，完整日志保存在本轮outputs。首批初期观察通过；全队列完成状态见本页完成结果。

## 已知限制

本轮只有seed2333，test参与选模。双曲在LEVIR/WHU最佳点的最粗层约75%的成对输入触发切空间截断，约99%的注意力留在本尺度；半径差异和查询检索仍有效。普通上下文分支保持有效幅度，但建筑库未取得边界优势。双曲/普通推理延迟分别约为CTRL的2.35/1.61倍。SYSU后期训练损失下降、test损失上升，存在明确泛化不足；完整证据见结果分析。
