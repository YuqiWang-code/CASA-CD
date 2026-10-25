# CASA-Pyramid / Run3

协议 `casa_pyramid_40k_pairbanks_lr01_v3`。四库各CTRL、双曲H、普通E，共12项；LEVIR→SYSU→WHU→CDD优先，两张RTX5090按实测峰值与1GiB留量并行补位。设计依据见[Run3设计文档](../../../docs/temporary/CASA-Pyramid_Run3_design.md)。

## 模块

共享MobileNetV2原生四尺度→TAR→成对上下文→DCR→共享预测头。上下文额外接收四层A/B，保留完整Q和224个KV。

- E：同层共享投影得到32维绝对差D，X=LN(T+WD·D)。每尺度区域均值与变化/背景平滑均值之差组成两库，各112个token、分别softmax；独立Q/K，4头×16维，每库V为4头×8维。类别均值包含1个区域均值伪计数。S1/S2写回D及边界门调制的HP(D)。
- H：A/B投影为4头×8维，Exp后构造精确两点Fréchet中点及Log残差，平移至共同原点得到v；检索使用中点m与u=abs(v)。uniform/change同权重压缩m、u、V为224个配对token，联合双曲中点距离和差异幅度距离；V保持欧式。S1/S2写回u及边界门调制的abs(HP(v))。
- 固定3×3二项式滤波，reflect padding，不降采样。context gamma4个、detail gamma2个均零初始化，投影正常初始化。
- CTRL保留LN/guide和S1/S2共享边界头，使用相同辅助损失；不含成对投影与检索。CTRL边界头为64→1，两模块边界头为128→1。

## 共同训练协议

RGB256、batch32、seed2333，MobileNetV2预训练全新初始化。40K按整epoch上取整，LEVIR/SYSU/WHU/CDD为181/107/217/129轮。每epoch在test以最高F1选best，再在同一test复测；test参与选模，非独立盲测。

Adam betas(.9,.99)、eps1e-8；骨干lr5e-5，新层lr5e-4，每步warmup/poly保留0.1倍率。context卷积/线性weight及主体WD1e-4，其余context参数WD0。四级BCEWithLogits+Dice主损失；四级guide BCE权重0.2；S1/S2边界BCEWithLogits+Dice权重0.05。边界标签取原mask3×3膨胀减腐蚀，replicate padding，再maxpool到对应feature范围。

沿用RGB ImageNet归一化；同步裁剪、HV翻转、90度旋转和时相交换。无新增光度扰动。FP32，AMP关、cuDNN TF32开、matmul TF32关；部署验证strict_fp32_fold_v1。

## 验证与启动

`validate.sh`检查几何恒等式与梯度、双库与配对压缩、局部通路、模型接口/部署、优化器倍率、边界标签和数据同步，再测batch32显存。`probe.sh`执行LEVIR各3轮、SYSU各4轮两模块短训，及SYSU CTRL骨干倍率1/0.1的12轮val对照；均从预训练新初始化，探针不计正式指标。短训比较不能证明完整40K的后期泛化。

```bash
bash train_scripts/CASA-Pyramid/Run3/validate.sh
bash train_scripts/CASA-Pyramid/Run3/probe.sh
bash train_scripts/CASA-Pyramid/Run3/launch.sh
```

`pipeline.py`是唯一调度器；实际显存见profile.json、scheduling.json和placement.json。正式前保存source_code.tar.gz、环境与源码基线。正式权重不使用任何探针或Run2权重。新日志在`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run3`，权重在`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run3`。

## Run3 实测开销

RTX5090，256×256、FP32；训练batch32测6步，前2步预热。部署为折叠后单输出predict、batch1、30预热/100同步计时，随机权重。cuDNN TF32开、matmul TF32关。FLOPs只统计支持的卷积和矩阵乘；双曲逐元素几何与直接距离运算未完整计入。

| 变体 | batch32峰值保留显存(MiB) | 同步训练步(ms) | 单输出推理(ms) | 已统计GFLOPs |
|---|---:|---:|---:|---:|
| none | 8202 | 54.52 | 3.69 | 1.735 |
| hyperbolic | 14870 | 148.15 | 8.67 | 2.037 |
| qkv | 9446 | 69.42 | 5.95 | 2.136 |

双曲版以更高的实际计算开销换取成对几何表示；参数少于普通版，但推理更慢。是否值得由同轮精度与开销共同判断。

## 结果管理

只取完整TEST区块。原185条历史指标保留，本轮12项收齐后应197条（另有新增继续保留）。Run3冻结TXT使用实际训练源码归档，current TXT使用工作源码；Run2冻结快照不变。

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run3
python train_scripts/CASA-Pyramid/Run3/collect_results.py
python tools/models_to_txt.py --tag CASA-Pyramid --run current
```
