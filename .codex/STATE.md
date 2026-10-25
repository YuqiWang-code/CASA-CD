# 当前状态

更新：2026-10-26。

## CASA-Pyramid / Run3 训练中

源码基线`0f714f5`，协议`casa_pyramid_40k_pairbanks_contrastwd0_v3`：普通均值/对比双库、双曲中点/差异配对检索、224KV、细两层边界细节。普通k_contrast/v_contrast权重WD0；骨干LR倍率0.1，其余训练设置见README。

LEVIR普通8epoch、双曲3epoch，SYSU两版各4epoch健康探针全部通过；数值、任务梯度、对比投影与检索正常。CTRL骨干倍率1/0.1的12epoch对照已完成，采用0.1；短训不证明长期泛化改善。

正式队列2026-10-26 02:57:17 CST启动，初始supervisor PID1376272。GPU0运行LEVIR H(1376274)、SYSU E(1376366)；GPU1运行LEVIR E(1376300)、LEVIR CTRL(1376333)、SYSU CTRL(1376397)，以实时pipeline_status.json为准。优先LEVIR→SYSU→WHU→CDD，共12项，两卡按容量补位，不重复启动。

本轮日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run3`，权重`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run3`。正式训练使用全新预训练初始化，不使用探针权重。运行源码、manifest、环境已归档。未完成记录保存在日志与checkpoint根的`archive/20261026_024558`，不作为正式指标。

初期观察截至2026-10-26T03:02:17+08:00，首批5项均完成至少3轮，损失下降、诊断有限，无数值或进程错误。详细记录见Run3 README与startup_health.json。

指标表185条历史记录保持不变，Run3正式结果0/12。没有本轮自动化任务。模型和训练源码保持运行版本，最新授权仅要求完成启动与初期观察。

## CASA-Pyramid / Run2 已完成

Run2于2026-10-25 13:58:47 CST启动，21:35:15 CST完成，历时7小时36分28秒；源码基线`9f515ef`。12/12项正式训练及测试完成，无失败，调度器已退出。2026-10-26核查两张GPU空闲。全部部署折叠检查通过，最大绝对差3.934e-6，二值差均为0。

协议`casa_pyramid_40k_reliableqk_testselect_v2`；224-token、共享Q/K、类别支持度先验、双曲Q/K+欧式V。每epoch使用test F1选模，非独立测试。本轮已归档，不恢复已完成队列。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) |
|---|---:|---:|---:|
| LEVIR | 90.9779 | 91.1631 | 91.1259 |
| SYSU | 83.3667 | 82.8377 | 82.5352 |
| WHU | 94.3527 | 94.5648 | 94.6981 |
| CDD | 96.9090 | 96.9303 | 96.8654 |

相对同轮CTRL，双曲/普通模块在LEVIR的F1分别提高0.185/0.148个百分点，WHU提高0.212/0.345个百分点；SYSU分别降低0.529/0.832个百分点；CDD双曲提高0.021个百分点、普通降低0.044个百分点。本轮呈现数据集相关的收益，SYSU是共同薄弱项；单seed微小差异不作为稳定提升证据。所有结果均为test参与选模后的复测。

日志与JSON已下载至`outputs/CASA-Pyramid/Run2`。指标表185条，原173条历史单元格保留；Run2冻结TXT来自source_code.tar.gz，current TXT来自工作源码。完整训练日志、逐轮曲线、实际best_epoch与诊断见`docs/experiments/CASA-Pyramid/Run2`。旧Run2_val两项完整结果保留，历史冻结TXT保持对应源码。

服务器日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run2`；checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run2`。

## CASA-Pyramid / Run1 已完成

协议`casa_pyramid_40k_guide02_v1`，模型与训练源码基线`5730d13`。2026-10-24 15:03:59 CST启动，22:30:52 CST完成，12/12正式测试齐全，无失败。最后核查两GPU均空闲，队列与子进程结束。

原生MobileNetV2四尺度经TAR后做联合跨尺度QKV，不含HFEA。每尺度32个KV、联合128个、四头16维；三组有相同评分辅助监督。QKV训练2.063581M/部署1.952266M，CTRL训练2.047125M/部署1.935810M。

全部日志与JSON已下载至`outputs/CASA-Pyramid/Run1`。指标表171条，原159条历史记录逐单元格保留；本轮冻结TXT使用source_code.tar.gz内实际模型源码，current TXT使用当前工作区。旧CASA/Run6和Run7冻结快照保持原轮次。

本轮相对CTRL，双曲/普通模块在LEVIR分别提高0.031/0.191个百分点，在SYSU提高0.359/0.714个百分点；WHU分别降低0.114/0.297个百分点，CDD降低0.052/0.045个百分点。普通版在LEVIR与SYSU高于双曲版，本轮尚未证明双曲几何具有稳定精度优势。两个模块部署参数均为1.952266M，精度与开销需共同评价。

全部部署折叠检查通过，首级最大绝对差不超过3.25e-6，二值差异均为0。Run1已归档，当前活动轮次见Run3；不恢复历史队列。连接与操作见SERVER.md。

## 产物

- 日志：`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1`
- 权重：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run1`
- 本地结果：`outputs/CASA-Pyramid/Run1`
- 当前/冻结TXT：`docs/temporary/models_and_metrics_CASA-Pyramid_current.txt`、`models_and_metrics_CASA-Pyramid_Run1.txt`
- 旧CASA/Run7服务器模型归档：`/home/yqwang/outputs/CASA-CD/CASA/Run7/archive/server_models_before_Pyramid_Run1`

## GitHub公开训练记录

docs/experiments/CASA-Pyramid/Run1保存12项完整日志、逐轮CSV、实际最佳epoch、完整训练/验证曲线、逐轮模块诊断CSV/JSONL、成功探针及环境。导出入口tools/export_training_records.py。模块诊断是每轮验证首个batch的实际测量值，epoch从0开始。
