# 当前状态

更新：2026-10-27。

## CASA-Pyramid / Run5

协议`casa_pyramid_40k_selective_selfref_v5`；平方类别选择与二阶支持度、自身候选和224原型联合softmax，回写`ΣA(Vj−Vi)`。Run4主体、几何、局部分支与共同训练设置保持；H/E参数训练2.064537M、部署1.953222M。

模型/几何/梯度/部署、同步增强、优化器与调度检查通过。LEVIR H/E各6epoch、SYSU H/E各4epoch真实train/val短训通过，正式队列于2026-10-27 12:54:33 CST启动，源码基线`b2ebe9c`，supervisor PID 1912750。两卡首批5项已完成至少3个训练及评价epoch，主损失下降，未见NaN、OOM或非有限梯度。正式结果0/12，历史指标209条保持不变。不得重复启动队列。四库各三组共12项，优先LEVIR→SYSU→WHU→CDD，两GPU按实测显存补位；正式从MobileNetV2预训练重新初始化。每epoch test F1选模，非独立盲测。

脚本`train_scripts/CASA-Pyramid/Run5`，日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run5`，权重`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run5`。运行源码与环境已归档，首期观察见startup_health.json。Run4已收集并保留，指标209条；Run5完成应221条。原型和同权干预见本地`outputs/CASA-Pyramid/Run5/analysis`及`docs/temporary/CASA-Pyramid_Run5_design.md`。当前无本轮自动化。

## CASA-Pyramid / Run4 已完成

Run4于2026-10-26 13:40:12 CST启动，2026-10-26 22:40:40 CST完成，历时9小时00分28秒。12/12项正式训练及测试完成，无失败；调度器已退出。全部部署折叠检查通过，最大绝对差3.815e-06，二值差均为0。

源码基线`eb3babe`，协议`casa_pyramid_40k_relative_adamwctx_v4`。共享TAR Q/K、可靠性压缩、相对回写和细层高通，双曲直接Lorentz坐标；context权重解耦衰减。全网lr5e-4、main+0.2guide、40K，test逐epoch选模。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) | 双曲/普通相对CTRL (pp) |
|---|---:|---:|---:|---:|
| LEVIR | 90.9482 | 91.1166 | 91.1429 | +0.168 / +0.195 |
| SYSU | 82.3165 | 82.7753 | 82.8930 | +0.459 / +0.576 |
| WHU | 94.1891 | 94.6085 | 94.7247 | +0.419 / +0.536 |
| CDD | 96.9150 | 96.9384 | 96.8998 | +0.023 / -0.015 |

相对同轮CTRL，双曲/普通模块在LEVIR提高0.168/0.195个百分点，SYSU提高0.459/0.576个百分点，WHU提高0.419/0.536个百分点；CDD为+0.023/−0.015个百分点。两个模块在前三库均有收益，CDD基本持平；普通版在前三库略高于双曲版。SYSU普通版最佳epoch仍为3，后期泛化仍是薄弱项。本轮恢复共同学习率和损失设置，跨轮提升不能全部归因于模块；单seed及test参与选模的限制保持明确。

125份日志与JSON已下载至`outputs/CASA-Pyramid/Run4`；指标表共209条正式记录，原197条历史记录逐单元格保留。Run4冻结TXT使用实际`source_code.tar.gz`中的源码，current TXT使用当前工作源码。完整日志、逐轮CSV、曲线及模块诊断已发布至`docs/experiments/CASA-Pyramid/Run4`。

日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run4`，权重`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run4`。本轮队列已完成，本次核查两卡无活动训练进程；不重复启动。当前无本轮自动化。

## CASA-Pyramid / Run3 已完成

Run3于2026-10-26 02:57:17 CST启动，11:53:44 CST完成，历时8小时56分27秒。12/12项正式训练及测试完成，无失败；调度器已退出，两张GPU空闲。全部部署折叠检查通过，最大绝对差5.513e-6，二值差均为0。

源码基线`0f714f5`，协议`casa_pyramid_40k_pairbanks_contrastwd0_v3`。普通均值/对比双库、双曲中点/差异配对检索、224KV、细两层边界细节；普通对比K/V权重WD0，骨干LR倍率0.1。每epoch test F1选模并复测，非独立盲测。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) | 双曲/普通相对CTRL (pp) |
|---|---:|---:|---:|---:|
| LEVIR | 90.5037 | 90.4977 | 90.5365 | -0.006 / +0.033 |
| SYSU | 81.9320 | 82.2920 | 82.1898 | +0.360 / +0.258 |
| WHU | 94.6343 | 94.3497 | 94.3585 | -0.285 / -0.276 |
| CDD | 95.3094 | 95.4613 | 95.5150 | +0.152 / +0.206 |

相对同轮CTRL，双曲/普通模块在SYSU的F1提高0.360/0.258个百分点，CDD提高0.152/0.206个百分点；LEVIR为−0.006/+0.033个百分点，WHU为−0.285/−0.276个百分点。两个模块相对Run2在四库均降低，CTRL在LEVIR、SYSU、CDD也降低；共同训练设置与模块变化需要分别归因。

SYSU出现明确的后期泛化不足：CTRL与普通版最佳epoch为3、双曲为33，训练损失继续下降而test选模损失上升。普通对比K/V保持有效幅度和随query变化的检索，未复现归零；双曲最粗层在LEVIR/WHU的最佳checkpoint诊断中约75%输入对触发裁剪，但半径差异和检索仍有效。该层约99%注意力留在本尺度，跨尺度利用有限。CDD三组最佳均为最后一轮，当前40K预算尚不能证明已达到平台。

日志与JSON已下载至`outputs/CASA-Pyramid/Run3`；指标表197条正式记录，原185条历史单元格保留。Run3冻结TXT使用`source_code.tar.gz`内实际模型源码，current TXT使用当前工作源码。完整日志、逐轮表格、曲线、最佳epoch、模块诊断及环境已导出至`docs/experiments/CASA-Pyramid/Run3`。

服务器日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run3`，权重`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run3`。未完成记录保存在日志与checkpoint根的`archive/20261026_024558`，不作为正式指标。当前无活动训练及本轮自动化，不重启已完成队列。

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
