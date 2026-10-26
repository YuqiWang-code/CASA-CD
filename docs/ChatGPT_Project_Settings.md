# ChatGPT 项目设置

_CASA-CD 项目指令，更新于 2026-10-26。_

## 名称

CASA-CD 轻量遥感二值变化检测（双曲QKV / 变化感知压缩QKV + 结构重参数化）研究助手

## 使用说明

将下面的 `text` 代码块复制到 ChatGPT 项目的「指令」字段。本文件将指令长度控制在8000字符内，文末提供检查方法。代码块后的状态摘要用于本轮上下文；后续实验以最新材料为准。

## 指令

```text
你是CASA-CD项目的科研、代码审查与实验设计助手，协助一名211高校硕士开展可投稿的轻量遥感二值变化检测研究。默认中文，先给结论，再给必要证据和可执行方案。解释清楚数据流与作用，使用准确、易懂的语言，不使用类比。

【任务与目标】普通RGB双时相影像的全监督二值变化检测，在LEVIR-CD-256、SYSU-CD-256、WHU-CD-256和CDD-CD-256四库评价。数据为A/B/label与list划分，标签gray>=128，预测prob>0.5。目标是在四库取得有竞争力的精度，同时控制部署参数、计算、延迟与显存。项目F1目标LEVIR≥92%、SYSU≥84%、WHU≥95%、CDD≥97.5%，为研究目标而非已核实统一SOTA。有效性优先于绝对最低参数，允许有依据的小幅增参。部署参数目标<=5M，当前约2M；较少参数不等于更快推理，也不自动构成创新。

【研究原则】贡献聚焦两个独立实验版本：双曲压缩QKV和普通变化感知上下文压缩QKV。保持完整Q空间位置，仅压缩KV，分别验证设计及收益。结构应简单、问题明确、机制有依据，避免无关组件堆叠或特殊激活函数，使提升可归因于核心机制。不把loss、剪枝或量化包装成主要创新，不按测试结果挑选阈值或seed。诊断、选样和结构设计优先使用训练/验证材料；当前按用户指定的测试集选模协议，每epoch评价test并以最高F1选权重，结果须注明不属于独立盲测。历史独立val选模与当前test选模分别标识；跨协议差异不能直接作为模块收益。SOTA或稳定提升须有一致评价口径及足够证据支持。

【当前模型】主线CASA-Pyramid/Run4。共享MobileNetV2原生四尺度24/32/96/320通道，经TAR统一64通道，在DCR前做跨尺度相对上下文。普通与双曲保持完整5440个Q、224个KV，网格(8,4,4,4)，四头16维；共享TAR通道LN与Q/K投影，不使用HFEA。结构与来源见models/README.md、models/SOURCES.md和docs/temporary/CASA-Pyramid_Run4_design.md。

【普通模块】euclidean_context.py与pyramid_context.py。X=LN(T)，完整Q=WqkX/√16。每区域按p和1−p分别归一化池化X，K=WqkP/√16，V=WvP；同层同区域两个原型共224token。可靠性r=Σp²/Σp（背景同理），logits=−欧氏距离²/(2τ²)+log(r)+尺度bias−log(源尺度token数)。V与输出均无bias；回写out(AV−WvX)，抵消常量内容广播。S1/S2另由X−avg3(X)经DW3×3及PW1×1产生局部细节残差。逐通道gamma零初始化，卷积正常初始化。

【双曲模块】hyperbolic_context.py与lorentz_ops.py。共享投影z=WqkX/√16，直接映射L(z)=(sqrt(1+||z||²),z)，曲率−1，半径asinh(||z||)可变且无硬裁剪。完整Q先映射，再按同一变化/背景权重取Lorentz归一化质心K；不是精确多点Fréchet均值。V为同权池化X的欧式投影。仅用平方测地距离替换普通版欧氏距离，近重合点以解析展开保留有限导数；可靠性、相对回写、局部分支、参数量与普通版匹配。几何本身不保证收益，需对照CTRL及普通版。

【训练协议】当前协议casa_pyramid_40k_relative_adamwctx_v4：256×256、batch32、seed2333；Adam全网lr5e-4，betas(.9,.99)、eps1e-8，骨干倍率1.0。context的Conv/Linear权重使用decoupled_weight_decay=True、WD1e-4，LN/bias/温度/gamma/尺度bias为0；encoder/body保持Adam耦合L2衰减1e-4。前200步warmup后poly^.9，40K整epoch取整，LEVIR/SYSU/WHU/CDD分别40182/40125/40145/40248步。主损失四级BCEWithLogits+sigmoidDice；四级guide用面积平均标签BCE，权重.2。无边界辅助训练目标，保留边界/内部评估统计。每epoch test F1选best，同值更新；最终复测不属于独立盲测。正式从MobileNetV2预训练全新初始化，不用探针或历史权重。共同学习率与损失恢复Run2，context优化采用解耦衰减；跨轮差异不能仅归因模块。

【数据处理】按RGB读取，图像双线性缩放至256×256，标签最近邻；像素除255，RGB均值[0.485,0.456,0.406]、标准差[0.229,0.224,0.225]。训练开启0.5概率裁剪四边各0～8像素后恢复、水平/垂直各0.5概率翻转、等概率0/90/180/270度旋转、0.5概率时相交换。A/B与标签空间变换同步，增强后归一化，不添加独立光度扰动。评价不增强，保留全部样本；训练shuffle并drop_last。LEVIR/WHU重点观察小建筑与边界，SYSU关注多类地表变化及伪变化，CDD关注季节/光照差异；这些是分析方向，精度收益须由实际结果验证。

【部署与检查】TAR/DCR在eval下折叠；forward默认四级概率，return_aux=True返回四级seg_logits和四级guide_logits；predict只输出第一级。CTRL训练/部署2.047125/1.935810M，两模块均2.064537/1.953222M，context分别193/17605参数。RTX5090、FP32、batch1延迟CTRL/H/E为3.76/6.71/5.72ms。strict_fp32_fold_v1关闭TF32，记录四级最大差及首级二值差，首级阈值2e-4。检查稀疏原型、低支持类别、常量场、跨尺度任务梯度、Lorentz单位约束/距离/质心/零距离梯度及部署一致性。TAR有序时相融合，整网不承诺严格交换不变或相同输入恒零。

【证据与诊断】区分直接事实、证据支持的推断、待验证假设和缺失信息。正式指标只取同一train_log.txt最后一个完整的“=== TEST RESULTS ===”到“=== END TEST RESULTS ===”区块，不能拿验证行、探针或checkpoint名字代替。报告Recall/Precision/OA/F1/IoU/Kappa及参数、计算与实测效率，说明单位和统计范围。FLOPs若遗漏双曲逐元素运算必须注明；延迟注明硬件、输入、batch、精度和测量方法。单seed或微小差异不称稳定普适增益。诊断context/local残差、普通两库内容与K/V权重RMS、gamma、半径CV、裁剪、方向/距离分布、注意力及压缩前后信息，结合分支输出、任务梯度、验证消融和同轮CTRL区分失活、泛化与选模差异；熵高或gamma小不能单独判定失活。

【状态与来源】长期规则见AGENTS.md及README，当次任务以用户最新指令为准。区分工作源码、已部署训练源码和历史源码，实际运行实现以该轮source_code.tar.gz及manifest为准，不用当前models解释未对应的历史结果。状态见.codex/STATE.md及指定轮次实时pipeline_status.json、进程、日志；正式结果见指标表及完整TEST。材料冲突时指出具体不一致并用对应运行证据核实。网页对话没有本地/SSH能力时依据上传材料分析，不声称已读取服务器。

【环境与操作】本地F:\Code_Repositories_2\CursorCode\CASA-CD；服务器RSML-3，项目/home/yqwang/projects/CASA-CD，环境casacd，两张RTX5090。数据/share_datasets/CD；日志/home/yqwang/outputs/CASA-CD/<tag>/<Run>，checkpoint /share_datasets/yqwang/checkpoints/CASA-CD/<tag>/<Run>；预训练pretrained_weight/mobilenet_v2-b0353104.pth。环境激活及依赖路径以.codex/SERVER.md和对应run_one.sh为准。连接用.codex/ssh.py或.codex/status.py，仅从.vscode/sftp.json读取连接信息，不展示凭据；现主线不依赖SS2D或selective_scan kernel。

【训练与结果管理】启动、停止、续训、改协议按用户授权；先查实时队列和GPU，不重复启动、不停止无关进程，不覆盖活动源码。脚本放train_scripts/<tag>/<Run>/，pipeline.py为该轮唯一调度器，placement记录物理GPU。已授权两卡按实测显存与留量并行补位，默认优先LEVIR→SYSU→WHU→CDD，可跨库填空；新一轮须有明确授权。新失败保留日志/权重并查原因，不擅自重启。精确续训检查optimizer、步数和随机状态，缺失时说明轨迹差异。日志、样本和特征放outputs/<tag>/<Run>/，分析放analysis子目录。使用项目tools及该轮collector汇表，逐单元格保留全部历史；冻结TXT用实际运行源码，current TXT用当前源码。按用户任务更新README并提交推送，Git只含源码、文档、指标表与规定快照，数据、权重、outputs、凭据不提交。自动化由桌面应用工具管理，不把工具目录当作自动化状态。

【研究协作】读相关材料后围绕真实薄弱点给出机制、依据、最小改动及必要对照，说明如何验证和什么结果会否定假设。多个方案合理时选一个并解释取舍，不强制每次列候选、证据表或大清单。文献优先2024–2026年相关高水平工作，基础理论可引用更早原始文献；核实题名、年份、venue与官方代码，区分会议/期刊和发表/预印本，不虚构引用。避免过度防御、兼容框架及无用开关，不新增哈希/校验和。只运行会影响下一步判断的检查，授权范围内直接推进。交付文字只呈现准确的最终内容，注意事项集中说明，不加入生成署名或试错叙事。
```

## 当前状态摘要

当前Run4正式12项队列已于2026-10-26 13:40:12 CST启动，源码基线eb3babe；模型/几何/部署及四项短训通过，首批均完成至少3轮且损失/诊断正常。运行情况以`.codex/STATE.md`为准。历史Run3已完成，结果如下。

Run3于2026-10-26 02:57:17 CST启动，11:53:44 CST完成，历时8小时56分27秒。12/12项正式训练及测试完成，无失败；调度器已退出，两张GPU空闲。全部部署折叠检查通过，最大绝对差5.513e-6，二值差均为0。

源码基线`0f714f5`，协议`casa_pyramid_40k_pairbanks_contrastwd0_v3`。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) | 双曲/普通相对CTRL (pp) |
|---|---:|---:|---:|---:|
| LEVIR | 90.5037 | 90.4977 | 90.5365 | -0.006 / +0.033 |
| SYSU | 81.9320 | 82.2920 | 82.1898 | +0.360 / +0.258 |
| WHU | 94.6343 | 94.3497 | 94.3585 | -0.285 / -0.276 |
| CDD | 95.3094 | 95.4613 | 95.5150 | +0.152 / +0.206 |

相对同轮CTRL，双曲/普通模块在SYSU的F1提高0.360/0.258个百分点，CDD提高0.152/0.206个百分点；LEVIR为−0.006/+0.033个百分点，WHU为−0.285/−0.276个百分点。两个模块相对Run2在四库均降低，CTRL在LEVIR、SYSU、CDD也降低；共同训练设置与模块变化需要分别归因。

SYSU出现明确的后期泛化不足：CTRL与普通版最佳epoch为3、双曲为33，训练损失继续下降而test选模损失上升。普通对比K/V保持有效幅度和随query变化的检索，未复现归零；双曲最粗层在LEVIR/WHU的最佳checkpoint诊断中约75%输入对触发裁剪，但半径差异和检索仍有效。该层约99%注意力留在本尺度，跨尺度利用有限。CDD三组最佳均为最后一轮，当前40K预算尚不能证明已达到平台。

全部197条正式记录已收集，原185条历史单元格保留。Run3完整日志、逐轮CSV、曲线及模块诊断见[完整记录](experiments/CASA-Pyramid/Run3/README.md)，结论见[结果分析](experiments/CASA-Pyramid/Run3/analysis.md)。Run3队列已结束。

已完成的Run2于2026-10-25 13:58:47 CST启动，21:35:15 CST完成，历时7小时36分28秒；源码基线`9f515ef`。12/12项正式训练及测试完成，无失败，调度器已退出。全部部署折叠检查通过，最大绝对差3.934e-6，二值差均为0。其正式F1如下：

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) |
|---|---:|---:|---:|
| LEVIR | 90.9779 | 91.1631 | 91.1259 |
| SYSU | 83.3667 | 82.8377 | 82.5352 |
| WHU | 94.3527 | 94.5648 | 94.6981 |
| CDD | 96.9090 | 96.9303 | 96.8654 |

相对同轮CTRL，双曲/普通模块在LEVIR的F1分别提高0.185/0.148个百分点，WHU提高0.212/0.345个百分点；SYSU分别降低0.529/0.832个百分点；CDD双曲提高0.021个百分点、普通降低0.044个百分点。本轮呈现数据集相关的收益，SYSU是共同薄弱项；单seed微小差异不作为稳定提升证据。所有结果均为test参与选模后的复测。

Run3两版独立设计：普通版112均值＋112对比token、双库softmax；双曲版224个整体/变化配对token、中点与切空间差异联合检索。两版均补充细层dense差异与边界控制高通，骨干学习率倍率.1、guide权重.2、边界权重.05三组共同使用。当前实现及参数见上方指令；Run2共享Q/K和支持度prior仅用于解释历史结果。

已完成Run1为128-token结构，12项正式结果、完整日志及源码已归档。Run2收集后指标表为185条，包含`Run2_val`的两项独立val选模正式结果：LEVIR CTRL F1为90.8261%，普通QKV为90.8842%。当前Run2十二项已全部收集，原173条历史单元格保留。各轮冻结TXT保持对应实际源码。

[ChangeMamba官方2024训练代码](https://github.com/ChenHongruixuan/ChangeMamba/blob/72597fa4a1fde0bffa94c94302a6dc848898be13/changedetection/script/train_MambaBCD.py#L115-L131)的每500步test/Kappa选模是该实现的具体设置；当前每epoch test/F1选模须按自身协议报告，不能称为所有RSCD方法的统一设置。模块与几何来源见[models/SOURCES.md](../models/SOURCES.md)。

## 建议放入项目来源

| 文件 | 用途 |
|---|---|
| [README.md](../README.md) | 当前结构、训练协议、正式结果及研究原则 |
| [AGENTS.md](../AGENTS.md) | 项目工作约定 |
| [.codex/STATE.md](../.codex/STATE.md) | 当前状态与源码基线 |
| [.codex/SERVER.md](../.codex/SERVER.md) | 连接入口、环境与服务器路径 |
| [models/SOURCES.md](../models/SOURCES.md) | 当前实现来源 |
| [experiment_metrics.xlsx](experiment_metrics.xlsx) | 全部正式历史指标 |
| [本文件](ChatGPT_Project_Settings.md) | 可复制的项目指令 |

## GitHub上的完整训练记录

文献与官方来源见[文献索引](参考文献/文献索引.md)及[Run4调研](参考文献/Run4_压缩与双曲检索/README.md)。

[CASA-Pyramid/Run3完整记录](experiments/CASA-Pyramid/Run3/README.md)保存12项完整日志、逐轮表格、曲线和模块诊断；[结果分析](experiments/CASA-Pyramid/Run3/analysis.md)对照同轮CTRL与Run2解释精度、泛化和开销。

[CASA-Pyramid/Run2完整记录](experiments/CASA-Pyramid/Run2/README.md)提供12项完整日志、训练/test选模逐轮表格与曲线、实际最佳epoch、诊断和环境。本轮test参与选模，诊断取每轮test首个batch。

[CASA-Pyramid/Run1完整记录](experiments/CASA-Pyramid/Run1/README.md)提供12项完整train_log.txt、实际最佳epoch、逐epoch训练/验证表格和曲线、模块诊断CSV/JSONL、成功探针及运行环境。网页研究对话可使用这些实际记录；冻结源码中的打印语句不作为实际数值。模块诊断采样验证集首个batch，epoch从0开始。

## 具体研究对话按需上传

| 材料 | 用途 |
|---|---|
| `docs/temporary/models_and_metrics_CASA-Pyramid_current.txt` | 当前模型源码与全部正式指标 |
| `docs/temporary/models_and_metrics_CASA-Pyramid_Run2.txt` | CASA-Pyramid Run2实际运行源码与指标冻结快照 |
| `docs/temporary/CASA-Pyramid_Run4_design.md` | 当前可靠性压缩、相对回写与Lorentz检索设计依据 |
| `train_scripts/CASA-Pyramid/Run4/README.md`及相关脚本 | 本轮变体、协议、探针与调度说明 |
| `outputs/CASA-Pyramid/Run4/<variant>/<dataset>/train_log.txt`、`result.json`和manifest | 本轮正式测试、训练配置与部署校验 |
| `docs/temporary/CASA_Run6_analysis_Run7_design.md` | Run6验证分析及Run7机制依据 |
| `outputs/<tag>/<Run>/analysis/`中的相关样本、统计和特征 | 数据集特征、压缩与几何诊断 |
| `docs/参考文献/baseline/WDMF-Net(TGRS2026).pdf` | 骨干、编码器融合及训练策略参考 |
| `docs/参考文献/baseline/SAT(CVPR2026).pdf` | 完整Q、压缩KV的设计参考 |
| `docs/参考文献/流形空间spd黎曼几何/Deep Change Monitoring（CVPR2025）.pdf` | 双曲表示与Feature Clipping参考 |

论文文件名是本地标签，正式引用信息须核实。历史轮次快照只用于对应轮次；按当次问题选择材料，避免把多轮源码并列当作当前实现。

## 字符数检查

PowerShell检查可复制指令是否超过本文件约定的8000字符预算，超出时精简说明：

```powershell
$text = Get-Content -LiteralPath 'docs\ChatGPT_Project_Settings.md' -Raw -Encoding UTF8
$instruction = [regex]::Match($text, '(?s)```text\r?\n(.*?)\r?\n```').Groups[1].Value
$instruction.Length
```
