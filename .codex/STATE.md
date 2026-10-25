# 当前状态

更新：2026-10-25。

## CASA-Pyramid / Run2

协议`casa_pyramid_40k_reliableqk_testselect_v2`：224-token、可靠度先验、共享Q/K、双曲Q/K+欧式V；qk/v/out权重WD1e-4，其余context WD0。RGB读取与标准化，训练同步裁剪/HV翻转/90度旋转/时相交换；每epoch从0在test以F1选模，明确标注非独立测试。

四库各三组共12项，LEVIR→SYSU→WHU→CDD优先，两GPU按实测峰值并行补位。模型与数据检查已通过，LEVIR/SYSU真实短训完成后由唯一调度器`train_scripts/CASA-Pyramid/Run2/pipeline.py`启动；入口`models/train_CASA.py`。正式模型从预训练重新初始化，探针不作为正式指标。

日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run2`；checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run2`；本地`outputs/CASA-Pyramid/Run2`。实际训练源码归档`source_code.tar.gz`；当前没有活动自动化。

独立val选模的两个完整LEVIR结果保存在Run2_val（CTRL90.8261%、普通90.8842%），日志、源码和checkpoint分别归档；指标表173条，原171条逐单元格保留。当前Run2收齐正常185条，其他新增保留。冻结Run2_val TXT与其运行源码对应；原型支持度分析在Run2/analysis。

<!-- STARTUP -->
正式队列于2026-10-25 13:58:47 CST启动，源码基线`9f515ef`，supervisor PID1334437。截至2026-10-25T14:04:46+08:00，首发五项均完成至少3个训练及test评估epoch，主损失和评分损失下降，未见NaN/Inf/OOM或异常退出。两模块四项真实短训均通过，四层分支及qk/v/out任务梯度非零。正式权重从MobileNetV2预训练全新初始化，未使用探针或历史权重。

| 实验 | GPU | PID | 已完成epoch | 首三轮总损失 |
|---|---:|---:|---:|---|
| M1_HYPER_QKV/LEVIR-CD-256 | 0 | 1334439 | 5 | 2.2231 → 1.3654 → 1.1898 |
| M2_CA_QKV/LEVIR-CD-256 | 1 | 1334470 | 6 | 2.2181 → 1.3658 → 1.1979 |
| M0_CTRL/LEVIR-CD-256 | 1 | 1334499 | 6 | 2.2215 → 1.3699 → 1.2030 |
| M1_HYPER_QKV/SYSU-CD-256 | 0 | 1334531 | 3 | 2.7216 → 2.1003 → 1.8660 |
| M0_CTRL/SYSU-CD-256 | 1 | 1334561 | 3 | 2.7246 → 2.1221 → 1.8952 |

<!-- END_STARTUP -->

## CASA-Pyramid / Run1 已完成

协议`casa_pyramid_40k_guide02_v1`，模型与训练源码基线`5730d13`。2026-10-24 15:03:59 CST启动，22:30:52 CST完成，12/12正式测试齐全，无失败。最后核查两GPU均空闲，队列与子进程结束。

原生MobileNetV2四尺度经TAR后做联合跨尺度QKV，不含HFEA。每尺度32个KV、联合128个、四头16维；三组有相同评分辅助监督。QKV训练2.063581M/部署1.952266M，CTRL训练2.047125M/部署1.935810M。

全部日志与JSON已下载至`outputs/CASA-Pyramid/Run1`。指标表171条，原159条历史记录逐单元格保留；本轮冻结TXT使用source_code.tar.gz内实际模型源码，current TXT使用当前工作区。旧CASA/Run6和Run7冻结快照保持原轮次。

本轮相对CTRL，双曲/普通模块在LEVIR分别提高0.031/0.191个百分点，在SYSU提高0.359/0.714个百分点；WHU分别降低0.114/0.297个百分点，CDD降低0.052/0.045个百分点。普通版在LEVIR与SYSU高于双曲版，本轮尚未证明双曲几何具有稳定精度优势。两个模块部署参数均为1.952266M，精度与开销需共同评价。

全部部署折叠检查通过，首级最大绝对差不超过3.25e-6，二值差异均为0。Run1已归档，当前活动轮次见Run2；不恢复历史队列。连接与操作见SERVER.md。

## 产物

- 日志：`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1`
- 权重：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run1`
- 本地结果：`outputs/CASA-Pyramid/Run1`
- 当前/冻结TXT：`docs/temporary/models_and_metrics_CASA-Pyramid_current.txt`、`models_and_metrics_CASA-Pyramid_Run1.txt`
- 旧CASA/Run7服务器模型归档：`/home/yqwang/outputs/CASA-CD/CASA/Run7/archive/server_models_before_Pyramid_Run1`

## GitHub公开训练记录

docs/experiments/CASA-Pyramid/Run1保存12项完整日志、逐轮CSV、实际最佳epoch、完整训练/验证曲线、逐轮模块诊断CSV/JSONL、成功探针及环境。导出入口tools/export_training_records.py。模块诊断是每轮验证首个batch的实际测量值，epoch从0开始。
