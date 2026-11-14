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

日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run1`，本地产物`outputs/CASA-Pyramid/Run1`。实际训练源码使用同目录`source_code.tar.gz`，冻结TXT从该归档提取模型；原159条正式指标保留，本轮收齐后应为171条，另有新增一并保留。

## 已知限制

原型数量在粗尺度可能不是压缩，例如8×8特征产生32个KV；目标是控制联合池为128个位置。预测评分并不保证所有区域都有真正变化，均匀概率会使两类原型接近。两个几何版本使用相同参数预算，实际运行成本仍需测量。TAR的有序时相融合不保证整网严格交换不变；随机时相交换增广保持共同协议。
