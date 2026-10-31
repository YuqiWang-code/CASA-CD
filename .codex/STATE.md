# 当前状态

更新：2026-10-31。

## CASA-Pyramid / Run9 训练中

2026-10-31 12:43:09 CST正式启动，源码基线`3a04aca`，supervisor PID3226168。仅GPU1；首批LEVIR的H_REFINE、E_REFINE、M0_CTRL三路并行，显存约29.6GiB。已观察各组三个以上完整epoch，主损失由约1.93下降到1.04–1.06；两种模块的精炼权重、原型与残差均发生变化，数值有限。E_REFINE第三轮选模F1回落，主训练损失仍下降，当前无数值或任务梯度故障，继续完整预算观察。

预计全部14项还需约14–18小时，完成窗口为北京时间**11月1日03:00–07:00**。估算基于首批并行训练速度，随跨库数据加载、评价开销和并发组合变化。正式结果0/14，现有指标仍263条。

协议casa_pyramid_40k_refinement_v9；用户授权仅GPU1，GPU0有无关任务，不操作。14项：四库H_REFINE(hyperbolic,β1)/E_REFINE(matched,β1)/M0_CTRL(none,β0)，SYSU/WHU各补E_BASE(matched,β0)。Run8 H_NONE两项作β0双曲外部参照，不重复入表。全部main+.1guide、retrieval0，seed/context seed2333，fresh MobileNetV2预训练、40K、batch32，逐epoch test选模。

唯一调度器train_scripts/CASA-Pyramid/Run9/pipeline.py；初始PID3226168。首批GPU1 LEVIR H_REFINE PID3226170、E_REFINE PID3226198、CTRL PID3226228，以实时pipeline_status.json为准。GPU预算31072MiB，H/E/CTRL/E_BASE预留11270/10188/8852/10098MiB；动态按容量补位，允许跨库填空。优先LEVIR→SYSU→WHU→CDD。不得重复启动或覆盖活动源码，不恢复历史队列。

日志/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9；checkpoint/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run9。源码以本轮source_code.tar.gz为准，本地已下载，基线3a04aca。六项两轮train/val探针全部通过，正式没有使用探针权重。记录见docs/experiments/CASA-Pyramid/Run9，快照Run9/current均已生成。完整TEST出现后用Run9/collect_results.py和tools/export_training_records.py收集，原263条逐单元格保留，14新项全收后至少277条。当前无定时自动化。

## CASA-MoE / Run3 GPU0串行训练中

用户于2026-10-31授权审查优化MoE后改用GPU0单路串行。协议casa_moe_40k_candidate_evidence_v3，源码提交734940a。动态router增加LN(TAR)输入，共4096参数；H/E加权attention合并一次AV/out，保留诊断分支；W及共同训练协议不变。六项：LEVIR DYNAMIC/TAR/STATIC→SYSU DYNAMIC/TAR/STATIC，最多一个活动实验，固定物理GPU0并保留至少1GiB显存，不停止GPU0无关任务，不操作GPU1主线。

实际运行副本/home/yqwang/outputs/CASA-CD/CASA-MoE/Run3/runtime/20261031_125233，source_code.tar.gz已下载。本轮代码已冻结，运行时不覆盖。CPU输出/梯度等价、各分支任务梯度和部署折叠通过，动态部署2.112676M；batch32实测进程13156MiB。六项三轮train/val探针全部通过，动态版LEVIR/SYSU主损失2.140→1.083及2.344→1.524，新增TAR输入各层梯度非零。2026-10-31 13:21:47 CST正式启动，唯一supervisor PID3230536，当前LEVIR M_D_DYNAMIC PID3230540，GPU0预算16012MiB，动态/TAR/静态预留13156/9164/12966MiB。正式从预训练骨干全新初始化，前三轮健康记录写入initial_health.json；不得重复启动探针或队列。失败保留产物并读原因，不自动重试。

日志/home/yqwang/outputs/CASA-CD/CASA-MoE/Run3；checkpoint/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MoE/Run3。正式epoch0–3主损失2.145→1.193→1.075→1.030，test选模F1为80.20%→83.78%→85.90%→83.79%，早期有波动，梯度有限且各分支有效，未见NaN/OOM，继续训练。初期健康、探针、日志和冻结TXT已下载并记录到docs/experiments/CASA-MoE/Run3。每epoch test F1选模，非独立盲测。现有正式指标263条，无本轮新TEST结果。收集使用Run3/collect_results.py及export_records.py，更新README_MoE.md，保留历史记录。当前无MoE定时自动化。

## CASA-MoE / Run2 已完成并收集

2026-10-31 10:27:23 CST，GPU1接续六项正式实验完成，无失败，无活动子进程。TAR/static/dynamic在LEVIR及SYSU各一项。协议casa_moe_40k_candidates_v2；实际runtime=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run2/runtime/20261030_203501。源码以冻结归档和TXT为准，工作源码models_v1与主线models独立。全部日志/JSON、Excel及冻结TXT已收集，实际记录见docs/experiments/CASA-MoE/Run2。不得重复启动接续或队列。

## CASA-Pyramid / Run8 首阶段已完成并收集

2026-10-30 19:06:54 CST启动，2026-10-31 01:03:56 CST完成，历时5小时57分02秒。仅GPU1，SYSU/WHU各H_CLASS/H_NONE共4项正式40K实验，全部TEST完整，无失败。协议casa_pyramid_40k_attribution_v8，源码基线dea8421；run/context seed2333，β=0，guide固定0.1，仅retrieval系数0.1/0不同。逐epoch test选模，结果非独立盲测。

SYSU F1为82.7738/82.6149%，WHU为94.3619/94.5690%；H_CLASS相对H_NONE分别+0.1589/−0.2072pp。全部折叠通过，最大误差1.3709e−6，二值差0。完整日志、曲线、auxiliary_validation、分析与冻结TXT见docs/experiments/CASA-Pyramid/Run8。后续共同监督、匹配欧氏与β1精炼对照尚未启动，不自动扩展或恢复已完成队列。

日志/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8，checkpoint/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run8；实际source_code.tar.gz和下载产物在outputs/CASA-Pyramid/Run8。当前指标263条（253历史＋Run8四项＋MoE Run2六项），历史单元格均保留。2026-10-31 12:08探查GPU1空闲，GPU0有无关训练，不操作。

## CASA-Pyramid / Run7 已完成

Run7于2026-10-28 16:00:08 CST启动，2026-10-29 00:38:32 CST完成，历时8小时38分24秒。8/8项正式训练与测试完成，无失败，GPU0队列已退出。全部部署折叠检查通过，最大绝对差3.397e-06，二值差均为0。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 双曲相对CTRL (pp) |
|---|---:|---:|---:|
| LEVIR | 91.1058 | 91.1316 | +0.026 |
| SYSU | 82.7919 | 82.8851 | +0.093 |
| WHU | 94.7471 | 94.5274 | -0.220 |
| CDD | 96.9528 | 96.9054 | -0.047 |

双曲相对同轮CTRL在LEVIR、SYSU提高0.026、0.093个百分点，WHU、CDD降低0.220、0.047个百分点；四库等权平均差值为-0.037个百分点。SYSU精确率提高、召回降低，WHU的1px边界F1降低约0.370个百分点。当前尚未形成四库一致的净收益。

协议`casa_pyramid_40k_retrieval_v7`，实际训练源码基线`8a54fa2`。224KV、1×32直接Lorentz投影、同P K/V、独立内容门控；双曲辅助0.1guide+0.1类别检索，CTRL为0.2guide。仅GPU0；每epoch test F1选模，结果非独立盲测。普通QKV保持Run5结构。

全部日志与JSON已下载至outputs/CASA-Pyramid/Run7。指标表229条，221条历史单元格全部保留。Run7冻结TXT使用实际source_code.tar.gz；完整训练曲线、逐轮CSV、best_epoch、模块诊断及结果分析见docs/experiments/CASA-Pyramid/Run7。16项LEVIR/SYSU配对短训保存在该目录selection，非正式指标。

服务器日志/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run7，checkpoint/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run7。队列和子进程已结束，不恢复已完成训练。2026-10-29探查GPU0为166MiB/0%，GPU1仍有独立CASA-MoE任务，不操作该任务。

Run6于2026-10-28 12:46 CST停止，0/8完整结果，源码、日志及checkpoint保留，不恢复。当前无本轮自动化。

## CASA-Pyramid / Run5 已完成

Run5于2026-10-27 12:54:33 CST启动，2026-10-27 22:10:22 CST完成，历时9小时15分49秒。12/12项正式训练及测试完成，无失败；调度器已退出，两张GPU空闲。全部部署折叠检查通过，最大绝对差3.457e-06，二值差均为0。

源码基线`b2ebe9c`，协议`casa_pyramid_40k_selective_selfref_v5`。平方类别选择、二阶支持度、自身候选和224原型联合softmax，回写`ΣA(Vj−Vi)`；共同协议保持Run4。每epoch test F1选模，非独立盲测。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) | 双曲/普通相对CTRL (pp) |
|---|---:|---:|---:|---:|
| LEVIR | 91.0562 | 91.0931 | 91.2324 | +0.037 / +0.176 |
| SYSU | 82.6402 | 82.3319 | 82.9401 | -0.308 / +0.300 |
| WHU | 94.3515 | 94.4315 | 94.4922 | +0.080 / +0.141 |
| CDD | 96.9524 | 96.9051 | 96.8727 | -0.047 / -0.080 |

普通版在LEVIR、SYSU、WHU相对同轮CTRL提高0.176、0.300、0.141个百分点，CDD降低0.080个百分点；双曲版在LEVIR、WHU提高0.037、0.080个百分点，在SYSU、CDD降低0.308、0.047个百分点。普通版四库均高于双曲版。相对Run4，普通版LEVIR/SYSU绝对F1略升、WHU/CDD下降；双曲版四库均降低，同轮净收益也减弱。Run5数值稳定，但尚未证明平方选择与自身竞争优于Run4的组合。Run4继续保留为正向基线。

117份日志与JSON已下载至`outputs/CASA-Pyramid/Run5`；指标表共221条正式记录，原209条历史单元格完整保留。Run5冻结TXT来自该轮`source_code.tar.gz`内实际运行源码，current TXT来自工作源码。完整日志、逐轮CSV、曲线、实际best_epoch及模块诊断见`docs/experiments/CASA-Pyramid/Run5`。

服务器日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run5`；权重`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run5`。本轮已结束，不重复启动；当前无本轮自动化。

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
