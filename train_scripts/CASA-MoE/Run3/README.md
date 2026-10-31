# CASA-MoE / Run3

候选证据改进轮。工作源码为 `models_v1/`，正式运行使用本轮冻结副本；Run2 的源码、日志与指标保持独立。

动态路由增加本层 `LN(TAR)`，与 `LN(X)` 和粗层解码特征拼接，仅新增4096参数。H/E共享加权注意力后只执行一次AV/out，分支诊断仍分开计算；小波、几何、候选融合公式及共同损失保持一致。结构假设是直接读取TAR能改善候选选择，精度效果由本轮同协议对照检验。

## 实验与训练

| 顺序 | 数据集 | 实验 | 配置 |
|---|---|---|---|
| 1 | LEVIR | M_D_DYNAMIC | 动态空间融合 |
| 2 | LEVIR | M_T_TAR | 匹配 TAR-only |
| 3 | LEVIR | M_S_STATIC | 每尺度可学习常量融合 |
| 4 | SYSU | M_D_DYNAMIC | 动态空间融合 |
| 5 | SYSU | M_T_TAR | 匹配 TAR-only |
| 6 | SYSU | M_S_STATIC | 每尺度可学习常量融合 |

协议 `casa_moe_40k_candidate_evidence_v3`。batch32、seed2333、40K目标按完整epoch上取整，MobileNetV2预训练全新初始化。共同四级BCE+Dice和0.2 guide；Adam、学习率5e-4、warmup200及poly0.9保持一致。正式每epoch在test选模并复测，结果不是独立盲测。探针使用train/val，探针权重不进入正式训练。

仅使用物理GPU0，**同时最多一项训练**。严格按表内顺序串行；每次启动按实测峰值和当前空闲显存检查，至少保留1GiB。其他进程、GPU1及主线队列由各自任务管理。

## 预检与启动

```bash
source train_scripts/CASA-MoE/Run3/environment.sh
python train_scripts/CASA-MoE/Run3/prepare_runtime.py
bash train_scripts/CASA-MoE/Run3/validate.sh
bash train_scripts/CASA-MoE/Run3/probe.sh
bash train_scripts/CASA-MoE/Run3/launch.sh
```

`prepare_runtime.py` 将模型与本轮脚本保存到日志目录的 `runtime/<时间>` 和 `source_code.tar.gz`。预检覆盖结构、梯度、部署折叠及三种配置的batch32训练峰值。六项真实数据探针各训练3个完整epoch，全部通过后才允许正式队列启动。失败保留日志与权重，由具体原因决定修复，不自动重试。

`pipeline_status.json` 记录当前实验、物理GPU、完成和失败项。`initial_health.json` 保存每项正式训练前三轮损失、选模数据、任务梯度与模块诊断。使用 `stop.sh` 停止本轮正式调度器和其子进程。

日志：`/home/yqwang/outputs/CASA-CD/CASA-MoE/Run3`。

权重：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MoE/Run3`。

## 结果收集

```powershell
python train_scripts/CASA-MoE/Run3/collect_results.py
python train_scripts/CASA-MoE/Run3/export_records.py
```

collector使用项目tools下载日志、汇入Excel并生成本轮冻结TXT，逐单元格保留历史指标，更新本页及根目录 `README_MoE.md`。只有完整TEST RESULTS进入指标表。全部六项完成后export发布完整日志、逐轮CSV、模块诊断及曲线，差值由本轮结果计算。当前工作源码与历史冻结TXT分别维护。

<!-- CASA_MOE_RUN3_RESULTS_START -->

2026-10-31 13:21:47 CST正式启动，supervisor PID3230536，首项LEVIR动态融合PID3230540，固定GPU0单路串行。六项三轮train/val探针全部通过，正式结果0/6。

<!-- CASA_MOE_RUN3_RESULTS_END -->
