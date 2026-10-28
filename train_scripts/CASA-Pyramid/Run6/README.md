# CASA-Pyramid / Run6

协议`casa_pyramid_40k_taskgeometry_v6`。四库各双曲版与CTRL，共8项，仅使用物理GPU0；优先LEVIR→SYSU→WHU→CDD，按实测显存与1GiB留量并行补位。普通QKV保持Run5结构，本轮不训练。

## 模型与共同设置

TAR后、DCR前保留完整Q。双曲版每区变化/背景各两个外观子原型，四尺度256/64/64/64，共448KV；一个32维空间。Q/K方向共享投影，非零方向半径`asinh(1+p)`，p为面积标签监督的预测变化占比。K和欧式V来自同一压缩内容；双曲距离决定条件检索，内容门控独立决定采用量，写回`W_out[g(AV−V_query)]`。保留原TAR和细两层高通残差。[模块设计](../../../models/HYPERBOLIC_DESIGN.md)。

| 变体 | 训练参数M | 部署参数M | context参数 |
|---|---:|---:|---:|
| M1_HYPER_QKV | 2.058474 | 1.947159 | 11,542 |
| M0_CTRL | 2.047125 | 1.935810 | 193 |

共同设置保持Run4/Run5：RGB256、batch32、seed2333、Adam全网5e−4，betas(.9,.99)、200步warmup/poly^.9、40K整epoch。context卷积/线性权重解耦WD1e−4，其余context WD0；主体Adam耦合L2 WD1e−4。四级BCEWithLogits+Dice＋.2guide；同步裁剪、水平/垂直翻转、90度旋转、时相交换，RGB ImageNet归一化。每epoch test F1选模，最终同一test复测，非独立盲测。正式权重从MobileNetV2预训练全新初始化，不继承Run5或探针权重。

| 数据集 | epoch | 实际训练步数 |
|---|---:|---:|
| LEVIR-CD-256 | 181 | 40,182 |
| SYSU-CD-256 | 107 | 40,125 |
| WHU-CD-256 | 217 | 40,145 |
| CDD-CD-256 | 129 | 40,248 |

## 测试与启动

`validate.sh`检查双曲公式、压缩内容对应、独立采用门控、常量/零场、稀有匹配、任务梯度、优化器和部署折叠，并测量H/CTRL的batch32显存。`test_pipeline.py`检查8项唯一性、数据集优先级、GPU0容量与补位。

`probe.sh`在GPU0进行双曲LEVIR 6epoch和SYSU 4epoch完整train/val短训；记录训练/验证主损失、guide、各层任务梯度、门控、残差、半径、子原型与路由。两项health_result通过且协议、轮数、步数相符后，调度器才允许正式训练。探针不作为正式指标。

服务器执行：

```bash
bash train_scripts/CASA-Pyramid/Run6/validate.sh
bash train_scripts/CASA-Pyramid/Run6/probe.sh
bash train_scripts/CASA-Pyramid/Run6/launch.sh
```

`pipeline.py`为唯一调度器；`scheduling.json`只允许GPU0。`run_one.sh`拒绝其他物理GPU，`placement.json`记录实际分配。队列失败时保留产物并停止本轮其余活动项；不循环重启。停止本轮使用`stop.sh`。

每项训练完成后自动加载best权重，执行strict_fp32_fold_v1检查和完整test，输出正式TEST RESULTS。需要单项补测时：

```bash
bash train_scripts/CASA-Pyramid/Run6/test_one.sh M1_HYPER_QKV LEVIR-CD-256
```

单项补测同样仅GPU0，结果保存到本轮`retests/`，不另作正式实验记录。

## 记录与结果收集

- 日志：`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run6`
- 权重：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run6`
- 本地：`outputs/CASA-Pyramid/Run6`
- `source_code.tar.gz`、`source_manifest.json`、`environment.json`保存实际启动源码、基线与环境；运行中不覆盖模型。

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run6
python train_scripts/CASA-Pyramid/Run6/collect_results.py
python tools/export_training_records.py --tag CASA-Pyramid --run Run6
python tools/models_to_txt.py --tag CASA-Pyramid --run current
```

只汇入完整TEST RESULTS；原221条正式记录逐单元格保留，8项齐全后应至少229条。Run6冻结TXT使用实际源码归档，current TXT使用工作源码。全部完成后再导出完整日志、逐轮CSV及曲线。

## 正式启动与初期观察

Run6于2026-10-28 10:16:28 CST启动，源码基线`9bef162`。截至2026-10-28T10:22:04+08:00，队列正常，正式结果0/8；初期各活动实验至少完成3个训练及test选模epoch，未见非有限损失、异常退出或整体分支失活。

| 实验 | 物理GPU / PID | 已完成epoch | 最近主训练损失 | 最近test选模F1(%) |
|---|---|---:|---:|---:|
| M1_HYPER_QKV/LEVIR-CD-256 | 0 / 2241915 | 5 | 1.0124 | 87.31 |
| M0_CTRL/LEVIR-CD-256 | 0 / 2241946 | 6 | 0.9818 | 86.65 |
| M0_CTRL/SYSU-CD-256 | 0 / 2241973 | 3 | 1.6636 | 80.43 |

两项train/val健康探针均通过，正式实验从预训练骨干全新初始化。GPU0当前显存30460MiB，利用率83%；GPU1没有运行本轮训练。初期诊断和短训只能说明当前训练正常，四库精度与泛化结论待完整正式结果。

[启动检查与探针](../../../docs/experiments/CASA-Pyramid/Run6/README.md)。
