# CASA-Pyramid / Run5 实验记录

[设计与逐项问题核实](../../../temporary/CASA-Pyramid_Run5_design.md) · [训练配置](../../../../train_scripts/CASA-Pyramid/Run5/README.md)

Run4正向基线保留。`run4_validation_analysis.json`为每库固定128张val的同权干预及原型分析，`probes/`为Run5真实train/val短训。正式队列启动信息以`startup_health.json`及训练说明为准。

## 启动前实测

模型、几何、梯度、部署及训练接口检查通过。四个完整train/val短训均通过，四层任务梯度非零；探针不作为正式指标。

| 探针 | epoch | 主损失首轮→末轮 | 健康判断 |
|---|---:|---:|---|
| levir_hyperbolic | 6 | 1.9624 → 0.9725 | 通过 |
| levir_qkv | 6 | 1.9617 → 0.9883 | 通过 |
| sysu_hyperbolic | 4 | 2.4100 → 1.4867 | 通过 |
| sysu_qkv | 4 | 2.4084 → 1.4923 | 通过 |

RTX5090、FP32、256×256、batch1、30次预热/100次中位数，CTRL/H/E为3.75/6.83/5.84ms；随机权重结构测量。H/E训练2.064537M、折叠部署1.953222M。

## 正式启动与初期观察

源码基线`b2ebe9c`，正式队列于2026-10-27 12:54:33 CST启动，supervisor PID 1912750；正式12项，首批5项运行。实际源码、配置和环境保存在`source_code.tar.gz`、`source_manifest.json`、`environment.json`，Run4归档保持原轮次。

截至2026-10-27T13:00:35+08:00，首批均完成至少3个训练及评价epoch，主损失下降，无NaN、OOM、非有限梯度或异常退出。三项活动模块的四层诊断有限；上下文采用量、半径与权重继续记录。完整train/val短训已检查任务梯度，正式训练逐步检查梯度有限性。

| 活跃实验 | GPU | 已完成训练epoch | 首轮→最新主损失 |
|---|---:|---:|---:|
| M1_HYPER_QKV/LEVIR-CD-256 | 0 | 5 | 1.9575 → 1.0171 |
| M2_CA_QKV/LEVIR-CD-256 | 1 | 6 | 1.9588 → 0.9776 |
| M0_CTRL/LEVIR-CD-256 | 1 | 6 | 1.9603 → 0.9714 |
| M2_CA_QKV/SYSU-CD-256 | 0 | 3 | 2.4107 → 1.6581 |
| M0_CTRL/SYSU-CD-256 | 1 | 3 | 2.4136 → 1.6726 |

GPU0预约29868MiB（LEVIR双曲、SYSU普通），GPU1预约30656MiB（LEVIR普通/CTRL、SYSU CTRL）。后续按LEVIR→SYSU→WHU→CDD优先补位；实测容量不允许当前再增加双曲任务。初期日志保存在`docs/experiments/CASA-Pyramid/Run5/initial_epochs`，不是完整正式结果。

初期数值与梯度正常不等于最终精度提高；最终收益以同轮CTRL和完整TEST区块判断。每epoch test参与选模，非独立盲测。
