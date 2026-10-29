# CASA-Pyramid / Run7

协议`casa_pyramid_40k_retrieval_v7`。四库各`M1_HYPER_QKV`与`M0_CTRL`，共8项，仅使用物理GPU0。按LEVIR→SYSU→WHU→CDD优先级，以实测显存和1GiB留量并行补位，允许跨库填充可用容量。

## 模型与训练

双曲模块位于TAR后、DCR前，保留完整Q。每区域变化/背景各一个原型，共224KV（128/32/32/32），一个32维共享Q/K投影。Q/K经过`/sqrt(32)`直接Lorentz提升，K/V使用同一压缩内容P；独立内容门控决定相对修正采用量，保留原TAR及细两层局部残差。

双曲版训练参数2.058409M、折叠部署参数1.947094M。普通QKV保持Run5结构，本轮正式实验只比较双曲与CTRL。

共同设置：RGB256、batch32、seed2333、40K按完整epoch上取整；Adam全网LR5e−4、200步warmup、poly0.9。context卷积/线性权重解耦WD1e−4，其余context参数WD0，主体耦合L2 WD1e−4。训练使用同步裁剪、水平/垂直翻转、90°旋转、时相交换和ImageNet归一化。

四级主损失为BCEWithLogits＋Dice。双曲辅助监督为0.1 guide BCE＋0.1类别检索BCE，CTRL为0.2 guide BCE。类别检索以变化/背景原型的logsumexp证据差预测GT面积均值。半径不预设置信或层级含义，原型类别名不保证内容纯净。

每epoch在test上按F1选模，结束后加载最佳权重，执行`strict_fp32_fold_v1`折叠检查及完整测试；test参与选模，结果不属于独立盲测。所有正式实验从MobileNetV2预训练全新初始化，不继承筛选权重。

## 筛选、启动与测试

`selection_probe.py`使用train/val进行候选比较，筛选依据写入`selection_decision.json`。启动要求`passed:true`、实际`selected_arm`及匹配协议；`profile.json`须来自最终模型的H/CTRL测量。`probe.sh`检查已有记录，不重复健康短训。

```bash
bash train_scripts/CASA-Pyramid/Run7/validate.sh
bash train_scripts/CASA-Pyramid/Run7/probe.sh
bash train_scripts/CASA-Pyramid/Run7/launch.sh
```

启动前归档实际源码、配置、环境及源码基线。日志位于`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run7`，权重位于`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run7`。

```bash
bash train_scripts/CASA-Pyramid/Run7/test_one.sh M1_HYPER_QKV LEVIR-CD-256
```

单独复测保存到`retests/`，不新增正式实验记录。`stop.sh`停止本轮调度器及其训练子进程。

## 结果收集

```bash
python .codex/status.py --tag CASA-Pyramid --run Run7
python train_scripts/CASA-Pyramid/Run7/collect_results.py
python tools/export_training_records.py --tag CASA-Pyramid --run Run7
```

只采纳完整`TEST RESULTS`区块；221条历史记录逐单元格保留，8项收齐后至少229条，另有新增也全部保留。冻结TXT使用本轮实际源码归档，当前源码快照单独生成。根README以`CASA_PYRAMID_RUN7_RESULTS`标记更新状态与结果。

Run7于2026-10-28 16:00:08 CST启动，2026-10-29 00:38:32 CST完成，历时8小时38分24秒。8/8项正式训练与测试完成，无失败，GPU0队列已退出。全部部署折叠检查通过，最大绝对差3.397e-06，二值差均为0。源码基线`8a54fa2`；8项均已下载、汇表及生成冻结快照，指标表229条，原221条历史单元格保留。[完整记录](../../../docs/experiments/CASA-Pyramid/Run7/README.md) · [分析](../../../docs/experiments/CASA-Pyramid/Run7/analysis.md)。
