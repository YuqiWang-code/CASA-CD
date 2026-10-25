# 当前状态

更新：2026-10-24。

## CASA-Pyramid / Run2 正式训练

协议`casa_pyramid_40k_qk_euclideanv224_v2`，224-token布局，双曲Q/K+欧式V，投影WD1e-4、其余context WD0。四库各三组共12项，优先LEVIR→SYSU→WHU→CDD，两GPU按profile和探针峰值并行补位，允许跨库填空。

正式队列于2026-10-25 11:07:35 CST启动，实际模型与训练源码基线`986fe75`，supervisor PID1320738。观察于2026-10-25T11:13:15+08:00，首发5项均完成至少三个完整epoch，主损失和评分损失下降；日志未见NaN/Inf/OOM或异常退出。正式模型从MobileNetV2预训练重新初始化，不使用探针或Run1权重。

| 实验 | GPU | PID | 完整epoch | step | 首三轮总损失 |
|---|---:|---:|---:|---:|---|
| M1_HYPER_QKV/LEVIR-CD-256 | 0 | 1320740 | 6 | 1332 | 2.1786 → 1.3012 → 1.1443 |
| M2_CA_QKV/LEVIR-CD-256 | 1 | 1320773 | 6 | 1400 | 2.1831 → 1.3028 → 1.1398 |
| M0_CTRL/LEVIR-CD-256 | 1 | 1320803 | 6 | 1500 | 2.1820 → 1.3040 → 1.1549 |
| M1_HYPER_QKV/SYSU-CD-256 | 0 | 1320835 | 3 | 1200 | 2.7263 → 2.0737 → 1.8518 |
| M0_CTRL/SYSU-CD-256 | 1 | 1320865 | 3 | 1400 | 2.7273 → 2.0825 → 1.8697 |

唯一调度器train_scripts/CASA-Pyramid/Run2/pipeline.py，入口models/train_CASA.py。日志/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run2，checkpoint/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run2，本地outputs/CASA-Pyramid/Run2。source_code.tar.gz冻结实际源码，source_baseline.json和environment.json记录基线与环境。

四项真实探针已通过，LEVIR各3epoch、SYSU各4epoch，不计正式指标。当前正式结果0/12，指标表171条历史保持不变，收齐后正常183条，其他新增保留。不要重复启动，不覆盖运行模型或训练脚本；新失败先读取原因并保留现场，不自动重启。没有活动自动化，不自行创建。旧Run1服务器模型已归档在Run1/archive/server_models_before_Run2。

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
