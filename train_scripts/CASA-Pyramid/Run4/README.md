# CASA-Pyramid / Run4

协议`casa_pyramid_40k_relative_adamwctx_v4`。四库各CTRL、双曲QKV、普通QKV，共12项；LEVIR→SYSU→WHU→CDD优先，两张GPU按实测显存与1GiB留量并行补位，允许跨数据集填充空闲容量。

## 模块

共享MobileNetV2原生四尺度→TAR→变化引导的相对上下文→DCR→共享预测头。Q保持完整空间分辨率，四尺度KV为128/32/32/32，共224个；4头、每头16维。两个模块共享压缩与读出机制，分别使用欧式和双曲关系检索。

- 每尺度TAR特征先LN；共享guide生成变化概率，按变化/背景权重压缩区域特征。类别可靠性为Σp²/Σp，校正模糊原型，而不直接按变化面积惩罚小目标。
- 普通版先池化LN特征再投影K/V；Q/K共享线性映射，以负平方欧氏距离检索。
- 双曲版将共享Q/K空间坐标直接提升至单位Lorentz双曲面，以Lorentz质心压缩K；以单一平方测地距离检索，V保持欧式。
- 读出为`out(A@V−V_query)`，按尺度及通道gamma残差写回TAR。S1/S2由`local_out(local_dw(X−avg3(X)))`补充局部高频信息，局部gamma同样按尺度及通道零初始化。
- CTRL保留相同LN/guide辅助监督，TAR特征直接送DCR。

## 共同训练协议

RGB256、batch32、seed2333，MobileNetV2预训练全新初始化。40K按整epoch上取整；LEVIR/SYSU/WHU/CDD为181/107/217/129轮。每epoch在test以F1选best，再在同一test复测；test参与选模，结果不是独立盲测。

Adam betas(.9,.99)、eps1e-8；骨干与新层学习率均5e-4，200步warmup后poly衰减。主体保持Adam耦合L2衰减1e-4；context全部卷积/线性weight采用独立参数组的解耦衰减1e-4，按`weight *= 1−lr·wd`更新，不进入Adam梯度矩估计。context的norm、bias、gamma、temperature、scale_bias为WD0。四级BCEWithLogits+Dice主损失加0.2倍四级guide BCE。边界与内部区域指标用于诊断。

RGB ImageNet归一化；训练同步裁剪、水平/垂直翻转、90度旋转与时相交换；评估只resize及归一化。无独立光度扰动。FP32，AMP关、cuDNN TF32开、matmul TF32关；部署验证`strict_fp32_fold_v1`。

## 验证与启动

`validate.sh`覆盖Lorentz几何与梯度、完整Q与224KV、共享投影与压缩、相对读出、局部支路、模型接口及部署折叠、context解耦衰减与主体耦合衰减的一步更新、优化器/RNG续训、数据变换，并测量batch32显存。

`probe.sh`在LEVIR执行H/E各6个完整epoch，在SYSU执行H/E各4个完整epoch，每epoch使用val评估。两卡各一项，LEVIR两项完成后执行SYSU两项；探针损失、分支输出、主任务梯度、main与0.2·guide梯度RMS/比例/余弦逐轮保存。健康通过要求数值有限且模块接收主任务梯度，精度与损失趋势同时人工核对；短训不能证明完整训练的泛化收益。正式训练从预训练全新初始化，不使用探针权重。

```bash
bash train_scripts/CASA-Pyramid/Run4/validate.sh
bash train_scripts/CASA-Pyramid/Run4/probe.sh
bash train_scripts/CASA-Pyramid/Run4/launch.sh
```

`pipeline.py`为唯一正式调度器；它确认四个探针均通过且对应本轮协议后，按`profile.json`、探针峰值及进程开销计算预约显存。`scheduling.json`记录两卡预算，`placement.json`记录物理GPU。单项异常退出会停止本轮队列并保留日志和权重；处理具体原因后再决定恢复或从新初始化重训。

正式输出：`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run4`。权重：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run4`。运行前保存`source_code.tar.gz`、源码基线、环境和配置。

## 结果管理

只采纳完整TEST RESULTS区块。已有197条正式记录全部保留，12项收齐后应为209条；若有其他新增记录亦全部保留。collector直接比较历史单元格，Run4冻结TXT取自实际运行源码归档，历史快照保持原轮次源码。

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run4
python train_scripts/CASA-Pyramid/Run4/collect_results.py
python tools/models_to_txt.py --tag CASA-Pyramid --run current
```

探针不计正式指标。本轮使用一个seed，test参与选模；完整结果与同轮CTRL比较，并结合实际推理时延评价精度和开销。

## 真实数据短训结果

train/val短训，epoch从1开始；不计正式指标。Q/K权重比为末轮RMS除以初始化RMS。

| 数据集 | 模块 | epoch | 首轮→末轮主损失 | 末轮val F1 (%) | Q/K权重比 |
|---|---|---:|---:|---:|---:|
| LEVIR | hyperbolic | 6 | 1.9594 → 0.9810 | 85.85 | 0.987 |
| LEVIR | qkv | 6 | 1.9635 → 0.9855 | 87.88 | 1.001 |
| SYSU | hyperbolic | 4 | 2.4127 → 1.4956 | 79.31 | 1.033 |
| SYSU | qkv | 4 | 2.4080 → 1.4874 | 79.87 | 1.018 |

末轮query注意力TV为0.061～0.207，S1/S2局部残差相对TAR的RMS为2.9%～9.4%；两版投影、局部支路和主任务梯度保持有效。双曲平均半径0.63～0.94、CV为0.168～0.260；自尺度注意力约22%～37%，跨尺度检索正常。验证值有早期波动，短训仅支持初期优化与数值正常的判断。

完整记录：本轮`probes/*/{run_manifest,health_history,health_result}.json`与`health_summary.json`。

## 正式启动与初期观察

源码基线`eb3babe`，正式队列于2026-10-26 13:40:12 CST启动；supervisor PID 1497160。四项短训通过：LEVIR H/E各6轮，SYSU H/E各4轮；正式实验均重新初始化。源码、配置与环境保存在本轮`source_code.tar.gz`、`source_manifest.json`和`environment.json`。

截至2026-10-26T13:45:44+08:00，首批5项均完成至少3轮，主损失下降；无NaN、OOM或异常退出，模块诊断有限，Q/K投影、检索差异和局部分支保持学习。

| 活跃实验 | GPU | 已完成epoch | 首轮→最新主损失 |
|---|---:|---:|---:|
| M1_HYPER_QKV/LEVIR-CD-256 | 0 | 5 | 1.9595 → 1.0319 |
| M2_CA_QKV/LEVIR-CD-256 | 1 | 5 | 1.9613 → 1.0101 |
| M0_CTRL/LEVIR-CD-256 | 1 | 5 | 1.9595 → 1.0138 |
| M2_CA_QKV/SYSU-CD-256 | 0 | 3 | 2.4127 → 1.6612 |
| M0_CTRL/SYSU-CD-256 | 1 | 3 | 2.4149 → 1.6666 |

两卡容量与变体预留见`scheduling.json`，实际分配见各项`placement.json`。首期记录为`startup_health.json`，探针原始记录在`probes/`。初期数值与梯度正常不等于最终精度提升；四库净收益以同轮CTRL和完整测试结果判断。

## 正式结果

Run4于2026-10-26 13:40:12 CST启动，2026-10-26 22:40:40 CST完成，历时9小时00分28秒。12/12项正式训练及测试完成，无失败；调度器已退出。全部部署折叠检查通过，最大绝对差3.815e-06，二值差均为0。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) | 双曲/普通相对CTRL (pp) |
|---|---:|---:|---:|---:|
| LEVIR | 90.9482 | 91.1166 | 91.1429 | +0.168 / +0.195 |
| SYSU | 82.3165 | 82.7753 | 82.8930 | +0.459 / +0.576 |
| WHU | 94.1891 | 94.6085 | 94.7247 | +0.419 / +0.536 |
| CDD | 96.9150 | 96.9384 | 96.8998 | +0.023 / -0.015 |

相对同轮CTRL，双曲/普通模块在LEVIR提高0.168/0.195个百分点，SYSU提高0.459/0.576个百分点，WHU提高0.419/0.536个百分点；CDD为+0.023/−0.015个百分点。两个模块在前三库均有收益，CDD基本持平；普通版在前三库略高于双曲版。SYSU普通版最佳epoch仍为3，后期泛化仍是薄弱项。本轮恢复共同学习率和损失设置，跨轮提升不能全部归因于模块；单seed及test参与选模的限制保持明确。

[完整训练记录](../../../docs/experiments/CASA-Pyramid/Run4/README.md) · [结果分析](../../../docs/experiments/CASA-Pyramid/Run4/analysis.md)。
