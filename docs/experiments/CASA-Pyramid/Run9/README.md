# CASA-Pyramid / Run9：区域精炼与匹配几何

源码基线`3a04aca`，协议`casa_pyramid_40k_refinement_v9`。14项新训练，仅GPU1。正式使用test逐epoch选模，val用于训练健康和机制诊断。

## 机制与对照

区域内先得到变化/背景各一个原型P0，再使用原类别权重乘exp(cos(X,P0))，归一化后得到P1。K、V和support采用同一组精炼权重；β=1固定，224KV、完整Q、单头32维。精炼不新增参数，局部和全局写回结构保持一致。

四库均训练H_REFINE、E_REFINE和M0_CTRL；SYSU/WHU另训练E_BASE，并引用Run8 H_NONE。H/E仅替换距离，E_BASE与E_REFINE仅β不同。两库构成几何×精炼对照，四库比较精炼模块的整体净收益。所有组统一main+.1guide，retrieval=0；Run8对检索监督的两库结论方向不同，此选择用于固定共同训练条件。

| 数据集 | Run9新训练 | 外部参照 |
|---|---|---|
| LEVIR | H_REFINE、E_REFINE、M0_CTRL | — |
| SYSU | H_REFINE、E_REFINE、M0_CTRL、E_BASE | Run8 H_NONE |
| WHU | H_REFINE、E_REFINE、M0_CTRL、E_BASE | Run8 H_NONE |
| CDD | H_REFINE、E_REFINE、M0_CTRL | — |

从MobileNetV2预训练全新初始化，run/context seed2333、batch32、40K整epoch取整；Adam全网lr5e-4、前200步warmup和poly^.9。context的Conv/Linear权重解耦WD1e-4，其他context参数WD0，主体耦合L2 WD1e-4。同步空间增强、随机裁切、水平/垂直翻转、90度旋转及随机时相交换与Run8保持一致。

## 检验重点

F1、Precision/Recall及IoU比较使用同轮CTRL；SYSU/WHU的精炼差值由各自β0/β1对得到，H/E差值分别在同一β比较。关注建筑边界/小目标、SYSU漏检与后期泛化、CDD季节差异下的误报。训练诊断记录权重TV、有效像素数、原型变化量、检索路由、写回幅度及任务梯度。

完整val可使用`analyze_frozen.py`检查关全局/局部、替换半径及路由干预，并用`content_diagnostics.py`按实际精炼权重测量名义类别与实际检索内容偏差。分析须显式传入module和β；GT仅在模型前向结束后用于评价，不进入模型。

[短训与推理成本](startup_health.md) · [正式初期记录](startup_health.json)

## 启动与产物

服务器日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9`；checkpoint位于`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run9`。六项两轮train/val探针全部通过后才启动正式队列；正式不用探针权重。

<!-- RUN9_STARTUP_START -->
2026-10-31 12:43:09 CST正式启动，源码基线`3a04aca`，supervisor PID3226168。仅GPU1；首批LEVIR的H_REFINE、E_REFINE、M0_CTRL三路并行，显存约29.6GiB。已观察各组三个以上完整epoch，主损失由约1.93下降到1.04–1.06；两种模块的精炼权重、原型与残差均发生变化，数值有限。E_REFINE第三轮选模F1回落，主训练损失仍下降，当前无数值或任务梯度故障，继续完整预算观察。

预计全部14项还需约14–18小时，完成窗口为北京时间**11月1日03:00–07:00**。估算基于首批并行训练速度，随跨库数据加载、评价开销和并发组合变化。正式结果0/14，现有指标仍263条。
<!-- RUN9_STARTUP_END -->

[训练脚本](../../../../train_scripts/CASA-Pyramid/Run9/README.md) · [模型与公式](../../../../models/HYPERBOLIC_DESIGN.md) · [Run8结果](../Run8/analysis.md)。正式指标只收完整TEST RESULTS；263条历史保留，新14项收齐应为277条。Run8两条参照保留原身份，不重复计数。

## 局限

单seed与test参与选模的口径保持明确。LEVIR/CDD没有独立β0臂，因此不能分别量化它们的精炼净贡献。短训证明可训练性，不能证明最终精度提升；精炼的计算成本需结合实测延迟比较。
