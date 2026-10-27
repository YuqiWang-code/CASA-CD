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
