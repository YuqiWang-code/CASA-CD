# 当前状态

更新：2026-11-17。

## CASA-MinorGoal / Run1 多尺度差分池化与差分条件检索 已完成并收集

2026-11-16 12:22 启动、当日完成 9/9 项（`CTRL`/`SPP`/`DIFF` × WHU/SYSU/LEVIR），仅用物理 GPU1 两槽；`SPP` 三臂在训练结束后崩于部署折叠的 GPU 设备不一致（`torch.zeros` 未继承 `device`），修好后用 `--resume` 只重做评测（24–108 s/项），权重取自 `last.pth` 的 `best_state_dict`。协议 `casa_minor_spp_diff_v1`，全部**验证集选模**、阈值 0.5、seed 2333。

改动：①`--tar-variant spp` 在时相融合之外并行 `|P−Q|` 支路，1×1 升维后接 `RepSPPBlock`（1/3/5/7 并行 DW，宽支 BN γ/β 从 0 开始），部署折叠成单个 7×7 DW + 1×1（`max_diff 1.19e-7`）；②新增 `DifferenceContext`（`HyperbolicContext` + `LN(D_s)`），Q 仍是 `T_s` 全位置 5,440，KV 换成 `|P−Q|` 的 224 个类条件原型。

结果（F1%，Δ 对同轮 CTRL）：CTRL 93.6234 / 81.7290 / 90.8488（2.047125 M 训练、1.935810 M 部署）；SPP 93.6925(+0.069) / **82.7915(+1.062)** / 91.0612(+0.212)（2.118293/1.995714）；DIFF **94.3468(+0.723)** / 82.6630(+0.934) / 90.8535(+0.005)（2.129705/2.007126）。延迟 5.880 / 5.987 / 8.883 ms（同会话 benchmark，batch1 256² FP32 deploy 形态）。

结论：①多尺度空间池化有效且增益在精度侧（SYSU P 83.46→85.56，R 不动），是本项目首次单库拉开控制臂 1 pp 以上；②双曲不缺几何、缺的是 KV 内容——KV 换成差分原型后 WHU +0.723 pp（P 94.10→95.57），此前"双曲无收益"测的其实是"检索内容没有差分"；③两处改动不叠加（DIFF−SPP：WHU +0.654、SYSU −0.129、LEVIR −0.208），SPP 单独性价比更好，DIFF 的 51% 延迟只换 WHU 的 0.65 pp；④单 seed，SYSU/WHU 的 +0.7~+1.1 pp 超出噪声（WHU 0.18、SYSU 0.32），SPP/WHU +0.07、DIFF/LEVIR +0.005 不能当信号；SPP/WHU 的 val 峰值 94.709 对应 test 93.693（−1.02 pp），是按 val 选模踩在噪声点上的典型。

数据：`docs/experiment_metrics.xlsx` 346 行（其中 Run1 9 行，身份为 `CASA-MinorGoal` + `Run1` + `CTRL/SPP/DIFF`）；TXT 快照 docs/temporary/models_and_metrics_CASA-MinorGoal_Run1.txt；对照表 outputs/CASA-MinorGoal/Run1/comparison.{json,md}；延迟 outputs/CASA-MinorGoal/Run1/inference_benchmark.json；训练记录 docs/experiments/CASA-MinorGoal/Run1/（本地产物，未入库）。

## CASA-Pyramid / Run16 模块去留（TAR/DCR × 三个历史版本）已完成并收集

2026-11-15 15:12 启动、2026-11-15 23:56 CST 完成 12/12 项，无失败，双卡各两槽，运行根 /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run16。协议 `casa_pyramid_40k_valselect_module_removal_v1`，全部验证集选模，控制值复用 Run15 同版本 `none`。两条臂只换单阶段内容：`TAR_DIFF` 把学习式时相混合换成固定 `|P−Q|`+1×1（训练 1.943949 M、部署 1.905602 M），`DCR_SINGLE` 把多级解码换成单层上采样（1.974730 M / 1.900802 M）。

结果（Δ 对同版本控制）：Run1 TAR +0.126/+0.410、DCR −0.337/−1.034；Run4 TAR +0.807/−0.379、DCR +0.596/−3.159；Run10 TAR −0.005/+1.468、DCR −0.547/+0.347。

结论：①TAR 的学习式时相混合不承载精度——六项平均 +0.405 pp，无一掉出 0.18–0.32 pp 噪声，叠加 Run14（−0.297/+0.181/−0.242）仍为正，换固定绝对差还能省 103,176 个训练参数、30,208 个部署参数且延迟不变。②DCR 是唯一平均为正贡献的模块——六项平均 −0.689 pp，四负两正，最差 Run4/SYSU −3.159，最好 Run4/WHU +0.596；连同 Run14（WHU −0.212/SYSU −0.611/LEVIR −0.876），多级解码在小目标密、边界碎的 SYSU 上最不能少，结论应写成"平均有效、逐库逐基线报告"。③两模块符号均随库翻转（TAR：WHU +0.126/+0.807/−0.005、SYSU +0.410/−0.379/+1.468），跨版本只能比版本内 Δ。

数据：`docs/experiment_metrics.xlsx` 337 行（其中 Run16 12 行，身份为 `Run16` + `Run1/TAR_DIFF` 三级目录）；TXT 快照 docs/temporary/models_and_metrics_CASA-Pyramid_Run16.txt；对照表 outputs/CASA-Pyramid/Run16/comparison.{json,md}；训练记录 docs/experiments/CASA-Pyramid/Run16/（本地产物，未入库）。三个版本树已加 `--tar-variant/--decoder-variant` 超集开关（默认 reparam 与原行为一致）。

## CASA-Pyramid / Run15 历史版本统一口径消融已完成并收集

2026-11-14 21:09 启动、2026-11-15 13:59 CST 完成 20/20 项，无失败（中途 `Run4/hyperbolic` 首次前向 OOM，把该臂预留从 12500 提到 18500 MiB 后重跑成功），两卡各两槽，运行根 /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run15。三个历史版本源码树已落到仓库根 `models_Run1/`、`models_Run4/`、`models_Run10/`（含 SOURCE.md 记 sha256 与补丁说明）。Run1 为 Poincaré 早期实现且归一化均值为 BGR 顺序，只在本轮内部可比；Run4/Run10 仅新增 `--selection-split {test,val}`（默认 test 保持原行为）。

结果（val 选模、阈值 0.5，WHU/SYSU）：Run1 控制 93.6175/81.7839、hyperbolic 94.1792(+0.562)/83.1153(+1.331)、qkv 94.4367(+0.819)/81.6452(−0.139)；Run4 控制 93.3574/83.0623、hyperbolic 93.9368(+0.579)/83.1173(+0.055)、qkv 94.1704(+0.813)/81.9697(−1.093)；Run10 控制 94.0877/81.0951、hyperbolic 93.9819(−0.106)/82.6912(+1.596)、matched 93.6482(−0.440)/82.7001(+1.605)、qkv 94.0871(−0.001)/82.6776(+1.583)。

结论：①几何本身无收益——Run10 的 `matched` 与 `hyperbolic` 逐位同构，仅换距离，差 +0.334/−0.009 pp；连同 Run14 12 对均值 −0.167、Run12 +0.118/+0.330，三个独立实现的配对差全部跨零且在 0.18–0.32 pp 噪声内。②Run10 在 SYSU 的三条臂（含欧氏双胞胎）同为 +1.58~1.61 pp，收益来自"加 context 模块"而非双曲，且其控制臂 81.0951 是三个版本里最弱。③WHU 上 qkv 臂三版本均不低于控制（+0.819/+0.813/−0.001），hyperbolic 在 Run1/Run4 为正、Run10 为负，两库排序不稳定。④跨版本比较必须用版本内 Δ：三个控制臂本身相差 0.73 pp（WHU）与 1.97 pp（SYSU）。

数据：`docs/experiment_metrics.xlsx` 325 行（其中 Run15 20 行，身份为 `Run15` + `Run1/none` 三级目录）；TXT 快照 docs/temporary/models_and_metrics_CASA-Pyramid_Run15.txt；对照表 outputs/CASA-Pyramid/Run15/comparison.{json,md} 与 version_table.md；训练记录 docs/experiments/CASA-Pyramid/Run15/（本地产物，未入库）。

## CASA-Pyramid / Run14 配对 seed 与去模块消融已完成并收集

2026-11-14 14:32 CST 完成 30/30 项，无失败、无停止，墙钟约 37.6 h，仅物理 GPU1 两槽（GPU0 全程被他人进程占用）。协议沿用 v12r3 训练配置；Part A 的 H/E 配对臂改为 val 选模（`--selection-split val`，test 仅末次复测，`independent_test=true`），Part B 的 TAR_DIFF/DCR_SINGLE 仍为 test 选模，用于与 Run13 构成 full→plain→removed 阶梯。

配对结论：WHU/SYSU/LEVIR/CDD 四库 H−E 配对差为 −0.692/+0.057/−0.023/−0.009 pp，12 对均值 −0.167 pp（sd 0.64，se 0.19，t=−0.90），双曲几何在读模口径下无精度收益。去模块（相对同库 test 选模控制臂）：TAR 换固定 |P−Q| 后 WHU −0.297、SYSU +0.181、LEVIR −0.242 pp；DCR 换单层解码后 WHU −0.212、SYSU −0.611、LEVIR −0.876 pp，是唯一在两库以上稳定为负的移除。val 选模下 H/E 的 WHU 93.19/93.88 比 Run12 test 选模的 94.6–94.8 低 1.4–1.5 pp，跨轮不可直接比。

## CASA-Pyramid / Run13 结构消融已完成并收集

2026-11-12 22:00 CST 完成 6/6 项，队列无失败，两张 RTX 5090 各并发 2 路。协议沿用 casa_pyramid_40k_testselect_heads_detail_v12r3，臂为 `TAR_PLAIN`（TAR 的 concat/P+Q/Q−P 换单支 1×1+BN、RepLocalBlock 换普通 DW3×3+PW1×1）与 `DCR_PLAIN`（RepPairFuse1x1 换单支 1×1+BN），各跑 WHU/SYSU/LEVIR，控制值复用 Run12 同配置结果。

结果：WHU TAR/DCR 94.7843/94.7339（控制 94.6403，噪声 0.177pp）；SYSU 82.3698/82.3239（控制 82.5512，噪声 0.319pp）；LEVIR 91.0016/91.1427（控制 91.0455，单次）。全部差值在复跑跨度内，部署参数三臂同为 1,935,810，延迟 3.678/3.680/3.672 ms。结论：可折叠多分支不承载可测精度收益，只能作为实现选择表述；H/E 与细节头对照仍取 Run12。

运行根 /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run13，checkpoint 根 /share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run13，冻结源码见运行目录 source_code.tar.gz。6 条完整 TEST 已入表，共 305 条；TXT 快照 docs/temporary/models_and_metrics_CASA-Pyramid_Run13.txt，消融表 outputs/CASA-Pyramid/Run13/ablation_table.{json,md}。

## CASA-Pyramid / Run12 test 选模 GPU1 正式训练中

2026-11-10 12:46:57 CST 从预训练骨干重新启动 26 项，唯一 supervisor 177805。仅物理 GPU1，首发 A_H/WHU PID178183、CTRL_ANCHOR/WHU PID178319，实时以 pipeline_status.json 为准。WHU 18、SYSU 6、LEVIR 2，模型、seed 2333/2334、batch32、40K 和显存调度沿用上一尝试。

协议 casa_pyramid_40k_testselect_heads_detail_v12r3。每轮在 test 按阈值 0.5 的 F1 选权重，最终在同一 test 复测。阈值在 val 的 0.10–0.70 网格校准，作为次指标。沿用同结构 GPU 资源和探针结果，test 选模、val 校准及 checkpoint 恢复专项通过。

此前 val 选模尝试按用户要求停止，完成 0 项。日志和权重归档为 CASA-Pyramid/archive/Run12_valselect_20261110_124522。运行根为 /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run12，checkpoint 根为 /share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run12。实际训练源码见运行目录 source_code.tar.gz。正式收集仅使用本协议完整 TEST 区块。

## CASA-Pyramid / Run11 已完成并收集

2026-11-10 10:36:14 CST 完成 24/24 项，队列无失败。24 项部署折叠通过，最大绝对差 3.994e-6，首级二值差均为 0。仅 GPU1，协议 casa_pyramid_40k_linearpool_interior_v11，模型与训练源码基线 832d17d。每 epoch test F1 选模，同集复测。

2026-11-09 12:13:36 CST 恢复队列，SYSU E_SIMPLE/E_INTERIOR 分别从 23,250/19,500 步接续，恢复优化器与随机状态。恢复前记录见服务器 Run11/archive/resume_20261109_121333。日志与源码根为 /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run11，权重根为 /share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run11。

24 条完整 TEST 已入表，共 299 条，原 275 条历史单元格保留。冻结 TXT 来自 source_code.tar.gz，current TXT 来自工作源码。完整日志、逐轮 CSV、诊断、曲线、续训和结果分析见 docs/experiments/CASA-Pyramid/Run11。

SYSU H-simple 相对 CTRL 提高 0.4971pp，主要来自误报减少。SYSU 三种结构的内部正像素加权均降低内部召回，后期评估回落仍在。WHU E-simple 的 1px 边界 F1 提高 0.6226pp，H-simple 下降 0.8986pp。

## CASA-Pyramid / Run10 已完成并收集

2026-11-01 03:44:17 CST完成12/12项，无失败、无活动子进程，supervisor已退出。源码8e8e559，协议casa_pyramid_40k_supervisedposterior_v10，仅GPU1。Run9停止产物保留，不恢复旧队列。

2026-11-02下载334项产物；12条完整TEST结果已入Excel，共275条，原263条逐单元格保留。冻结TXT来自实际source_code.tar.gz，current TXT同步更新；完整日志/逐轮CSV/诊断/曲线/结果在docs/experiments/CASA-Pyramid/Run10，分析见RESULT_ANALYSIS.md。本轮H/E四库平均同轮F1净增益−0.2199/−0.0291pp，WHU H−0.6195pp，SYSU后期泛化问题仍在；数值和折叠检查正常。保留Run4正向基线，未启动新训练，无监控自动化。后续新方案和训练按用户指令。

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


