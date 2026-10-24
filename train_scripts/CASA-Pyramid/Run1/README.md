# CASA-Pyramid / Run1

协议`casa_pyramid_40k_guide02_v1`。原生四尺度 → TAR → 联合跨尺度QKV → DCR，独立于历史CASA/Run7保存。

## 结构与对照

共享MobileNetV2输出24/32/96/320通道，空间64/32/16/8；TAR统一为64通道。不使用HFEA，也不构造320→1280尾层。

每尺度分为4×4个空间区域，每区域用预测变化概率及其补数生成两个KV原型。四尺度联合128个原型；四头、每头16维，各尺度Q保留全部位置。Q/K/V、LayerNorm、评分头和输出投影跨尺度共享，含每头4×4尺度偏置和四个零初始化gamma。没有额外局部分支或公共质心相减。

| 变体 | 上下文 | 训练参数 | 部署参数 |
|---|---|---:|---:|
| M0_CTRL | 不检索，保留评分头及监督 | 2,047,125 | 1,935,810 |
| M1_HYPER_QKV | 每头双曲距离/2，Lorentz质心压缩与聚合 | 2,063,581 | 1,952,266 |
| M2_CA_QKV | 欧氏距离，欧氏加权平均 | 2,063,581 | 1,952,266 |

三者评分辅助监督相同，控制监督因素。评分logits以面积平均标签监督，损失为四尺度BCEWithLogits之和；总目标为原四级BCE+Dice之和，加0.2倍评分损失。标签不进入注意力或原型权重，推理保留评分头。

## 共同协议

MobileNetV2预训练全新初始化，batch32、seed2333、Adam5e-4/betas(0.9,0.99)/eps1e-8，context参数WD0、其余WD1e-4，poly^0.9，首epoch且step<200时warmup。40K目标向完整epoch上取整；独立val最高F1选best，再以test正式评价。评分头的监督改动已纳入三组共同协议，不能与旧轮直接归因比较。

12项：四库各三组，LEVIR→SYSU→WHU→CDD优先，两GPU按实测显存留量并行补位，可跨库填空。健康探针为每模块LEVIR3epoch、SYSU4epoch，采用正式调度，不作为正式指标。探针检查主任务对四级上下文、Q/K/V和输出的梯度，并分开记录主损失、评分损失、gamma及几何诊断。

## 执行

```bash
bash train_scripts/CASA-Pyramid/Run1/validate.sh
bash train_scripts/CASA-Pyramid/Run1/probe.sh
# 四项健康探针通过后，按profile与探针峰值配置调度
bash train_scripts/CASA-Pyramid/Run1/launch.sh
bash train_scripts/CASA-Pyramid/Run1/stop.sh
```

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run1
python train_scripts/CASA-Pyramid/Run1/collect_results.py
python tools/models_to_txt.py --tag CASA-Pyramid --run current
```

日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run1`，本地产物`outputs/CASA-Pyramid/Run1`。实际训练源码使用同目录`source_code.tar.gz`，冻结TXT从该归档提取模型；原159条正式指标逐单元格保留，本轮12项已收齐，指标表171条，另有新增一并保留。

## 已知限制

各尺度压缩比例不同：64×64特征压为32个KV是128倍，8×8特征压为32个KV是2倍；联合池固定为128个位置。预测评分并不保证所有区域都有真正变化，均匀概率会使两类原型接近。两个几何版本使用相同参数预算，实际运行成本见推理开销。TAR的有序时相融合不保证整网严格交换不变；随机时相交换增广保持共同协议。

## 正式启动与健康观察

2026-10-24 15:03:59 CST正式启动，源码基线`5730d13`，supervisor PID1278526。观察时间2026-10-24T15:09:03+08:00，五项均超过200 step并完成至少三个完整epoch；主损失、评分损失下降，日志未见NaN/Inf/OOM或异常退出。两GPU按预算补位，实际活动分配以实时状态为准。

| 探针 | epoch / step | 总损失 | 主损失末轮 | 评分损失末轮 | 验证F1(%) |
|---|---|---|---:|---:|---:|
| levir_hyperbolic | 3 / 666 | 2.1777 → 1.3040 → 1.1442 | 1.0726 | 0.3582 | 85.207 |
| levir_qkv | 3 / 666 | 2.1831 → 1.3010 → 1.1536 | 1.0815 | 0.3604 | 84.986 |
| sysu_hyperbolic | 4 / 1500 | 2.7290 → 2.0584 → 1.8242 → 1.6502 | 1.4405 | 1.0489 | 78.890 |
| sysu_qkv | 4 / 1500 | 2.7253 → 2.0647 → 1.8238 → 1.6662 | 1.4546 | 1.0578 | 78.704 |

探针只用于健康检查，不计入正式指标。四级上下文与Q/K/V/输出投影均有非零主任务梯度；评分头有直接监督梯度。探针双曲半径CV为0.094～0.179，查询裁剪比例为0。

| 实验 | GPU | PID | 完整epoch | step | 首三轮总损失 |
|---|---:|---:|---:|---:|---|
| M1_HYPER_QKV/LEVIR-CD-256 | 0 | 1278528 | 6 | 1332 | 2.1771 → 1.3086 → 1.1589 |
| M2_CA_QKV/LEVIR-CD-256 | 1 | 1278559 | 5 | 1200 | 2.1809 → 1.3001 → 1.1561 |
| M0_CTRL/LEVIR-CD-256 | 1 | 1278589 | 5 | 1200 | 2.1860 → 1.3181 → 1.1563 |
| M1_HYPER_QKV/SYSU-CD-256 | 0 | 1278621 | 3 | 1125 | 2.7225 → 2.0634 → 1.8491 |
| M2_CA_QKV/SYSU-CD-256 | 1 | 1278651 | 3 | 1125 | 2.7276 → 2.0697 → 1.8466 |

## 推理开销

RTX5090空闲GPU，256×256双时相输入、batch1、四级输出，折叠后30次预热、100次同步测量，随机权重。输入与权重为FP32，cuDNN允许TF32，矩阵乘TF32关闭。FLOPs只覆盖profiler支持的卷积/矩阵乘，不含双曲逐元素运算。

| 变体 | 训练参数 | 部署参数 | 已统计GFLOPs | 中位延迟(ms) |
|---|---:|---:|---:|---:|
| none | 2,047,125 | 1,935,810 | 1.734 | 3.81 |
| hyperbolic | 2,063,581 | 1,952,266 | 2.113 | 7.54 |
| qkv | 2,063,581 | 1,952,266 | 2.100 | 5.53 |

## 正式完成与结果收集

2026-10-24 22:30:52 CST全部完成，总耗时7小时26分53秒。12项完整TEST RESULTS均已下载、入表；部署折叠检查全部通过，二值预测差异为0。冻结TXT从本轮实际源码归档提取，含16个Python文件与171条正式指标；当前TXT同步更新。

本轮相对CTRL，双曲/普通模块在LEVIR分别提高0.031/0.191个百分点，在SYSU提高0.359/0.714个百分点；WHU分别降低0.114/0.297个百分点，CDD降低0.052/0.045个百分点。普通版在LEVIR与SYSU高于双曲版，本轮尚未证明双曲几何具有稳定精度优势。两个模块部署参数均为1.952266M，精度与开销需共同评价。
