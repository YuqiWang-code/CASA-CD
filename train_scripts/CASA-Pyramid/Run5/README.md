# CASA-Pyramid / Run5

协议`casa_pyramid_40k_selective_selfref_v5`。Run4作为完整正向基线保留。四库各CTRL、双曲、普通，共12项；LEVIR→SYSU→WHU→CDD优先，两张GPU按实测显存与1GiB留量并行补位。

## 模块与共同协议

TAR后、DCR前，完整5440Q、224跨尺度KV、4头×16维。区域变化/背景概率平方归一化，二阶类别支持度；每个query的自身K/V和压缩原型联合softmax，写回`ΣA_ij(V_j−V_i)`。普通版为欧氏K与平方欧氏距离，双曲版为直接Lorentz坐标、质心K与平方测地距离，两版V均欧式。Run4细两层高通与逐通道残差保持，参数匹配。

共同设置保持Run4：RGB256、batch32、seed2333、40K整epoch，Adam全网5e−4、betas(.9,.99)、200warmup/poly.9。context卷积/线性权重解耦WD1e−4，其他context WD0，主体Adam耦合L2 WD1e−4；四级BCEWithLogits+Dice＋.2guide。每epoch test/F1选模并复测，非独立盲测。正式从MobileNetV2预训练全新初始化，不继承Run4或探针权重。

## 分析、测试与启动

`analyze_run4.py`从Run4源码归档读取模型，对每库固定128张val做同权干预，并测量原型纯度；使用Run4权重，不改历史训练。结果与逐项判断见[设计说明](../../../docs/temporary/CASA-Pyramid_Run5_design.md)。

`validate.sh`检查压缩、自参照质量/差值、常量场、跨尺度梯度、几何与同点导数、模型接口、部署折叠、优化器/RNG、同步增强与调度容量，并测量batch32显存。

`probe.sh`使用LEVIR的H/E各6epoch、SYSU各4epoch完整train/val，记录每轮主损失、guide、任务梯度与辅助梯度、上下文采用量、残差及半径。四个health_result均通过且协议对应时，pipeline才允许正式训练。探针不是正式指标。

```bash
bash train_scripts/CASA-Pyramid/Run5/validate.sh
bash train_scripts/CASA-Pyramid/Run5/probe.sh
bash train_scripts/CASA-Pyramid/Run5/launch.sh
```

唯一调度器`pipeline.py`，placement记录物理GPU，scheduling记录实测预约显存。出现本轮NaN、非有限梯度或异常退出，队列停止并保留日志/权重；按具体原因修复。正常早期val波动不单独作为停训依据。

服务器日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run5`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run5`。启动前保存实际源码、manifest、环境与协议；不覆盖活动训练源码。

## 结果收集

历史209条指标逐单元格保留；12项完整TEST收齐后应为221条，其他新增历史同样保留。冻结TXT使用Run5/source_code.tar.gz中的实际源码，current TXT使用工作源码。

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run5
python train_scripts/CASA-Pyramid/Run5/collect_results.py
python tools/models_to_txt.py --tag CASA-Pyramid --run current
python tools/export_training_records.py --tag CASA-Pyramid --run Run5
```

短训与正式启动实测记录见本轮输出的health_summary、source_manifest、startup_health和pipeline_status。

## 启动前实测

模型、几何、梯度、部署及训练接口检查通过。四个完整train/val短训均通过，四层任务梯度非零；探针不作为正式指标。

| 探针 | epoch | 主损失首轮→末轮 | 健康判断 |
|---|---:|---:|---|
| levir_hyperbolic | 6 | 1.9624 → 0.9725 | 通过 |
| levir_qkv | 6 | 1.9617 → 0.9883 | 通过 |
| sysu_hyperbolic | 4 | 2.4100 → 1.4867 | 通过 |
| sysu_qkv | 4 | 2.4084 → 1.4923 | 通过 |

RTX5090、FP32、256×256、batch1、30次预热/100次中位数，CTRL/H/E为3.75/6.83/5.84ms；随机权重结构测量。H/E训练2.064537M、折叠部署1.953222M。
