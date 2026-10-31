# CASA-MoE / Run2

协议`casa_moe_40k_candidates_v2`。物理GPU1，接续Run8完成后执行，首阶段6/6项于2026-10-31 10:27:23 CST完成，无失败。[完整结果与曲线](../../../docs/experiments/CASA-MoE/Run2/README.md)。

## 匹配实验

| 变体 | 配置 | LEVIR | SYSU |
|---|---|---|---|
| M_T_TAR | 固定选择TAR，公共X仅参与guide监督 | ✓ | ✓ |
| M_S_STATIC | 每尺度可学习常量四权重 | ✓ | ✓ |
| M_D_DYNAMIC | 本层X及粗层DCR特征驱动空间路由 | ✓ | ✓ |

首阶段6项，LEVIR优先、随后SYSU。三组保持同一公共stem/guide、主干初始化和训练协议。WHU/CDD及删除专家消融在首阶段结果后决定，不预先扩展。

## 架构

共享MobileNetV2原生四尺度 → 公共线性投影，绝对差与时相均值构造X。

- H/E共享LN、guide、224个变化/背景原型、QK/V/out、尺度偏置及零初始化通道增益。完整Q，单头32维；H使用自由投影Lorentz平方距离，E使用相同坐标的平方欧氏距离，分别学习初始0.5的温度。两路输出均为欧式相对内容残差。
- W在原始线性投影差上做带符号Haar；S1/S2高频处理后以零LL重建送D1/D2，S2/S3的LL幅值除2处理后送D3/D4；独立零初始化增益。
- T为原生特征上的完整TAR候选。
- 从S4到S1依次计算路由、候选混合及DCR。动态router读取本层X和已经计算出的粗层解码特征，输出层零初始化；不detach粗层梯度。四路稠密执行。

匹配T-only在推理中移除仅供guide训练的stem/LN/guide。CPU检查得到部署参数：T-only 1.935617M、常量融合2.100452M、动态融合2.108580M。GPU延迟与显存以接续后的profile为准。

## 共同协议与预检

batch32，seed2333，全网lr5e-4，Adam(.9,.99)，200step warmup、poly^0.9，40K按完整epoch上取整。公共stem、router、wavelet、retrieval全部属于context：Conv/Linear权重解耦WD1e-4，其余context参数WD0；主体保持L2 WD1e-4。四级BCE+Dice＋0.2共享guide，无额外检索、边界或负载损失。RGB/ImageNet归一化，同步裁剪、翻转、90度旋转、0.5时相交换；配对loader使用独立固定seed生成器。

正式每epoch在test按F1选模并复测，结果不是独立盲测。三种配置在LEVIR/SYSU各进行3个完整train/val epoch健康探针，共6项；检查有限损失、主损失改善、分支与路由任务梯度、双曲距离、频带泄漏及部署等价。探针不作为正式指标，不续用探针权重。

CPU检查覆盖：Haar重建/常量幅值、零时差W、浅层零LL、224KV、T-only精确恢复TAR→DCR、公共参数初始化一致、3次更新的梯度、三配置部署折叠。检查结果保存`cpu_checks.json`。GPU预检、实测容量及真实探针全部通过后方可运行正式队列。失败保留日志和权重，停止接续，不自动重试。

## 接续与操作

```bash
source train_scripts/CASA-MoE/Run2/environment.sh
python train_scripts/CASA-MoE/Run2/prepare_runtime.py
bash train_scripts/CASA-MoE/Run2/start_after_run8.sh
```

唯一接续进程`handoff.py`只读Run8状态，每60秒检查。Run8停止/失败时退出并记录原因；完成后还需GPU1至少18GiB空闲，保留无关任务。顺序为CPU契约检查→GPU profile→六项真实数据探针→按实测容量调度6项正式训练。`handoff_status.json`记录等待/预检/探针/正式训练阶段，`pipeline_status.json`记录正式矩阵。源码冻结于`runtime/<时间>`及`source_code.tar.gz`，后续工作目录修改不改变当前轮次。

日志：`/home/yqwang/outputs/CASA-CD/CASA-MoE/Run2`；checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MoE/Run2`。使用`stop.sh`停止本轮接续或训练，不操作Run8及无关进程。

本地汇总使用`python train_scripts/CASA-MoE/Run2/collect_results.py`，只取完整TEST区块并逐单元格保留历史指标。冻结快照来源为实际运行源码。

<!-- CASA_MOE_RUN2_RESULTS_START -->

状态更新：2026-10-31 12:12:57 CST；队列 `complete`；正式结果 **6/6**。

| 数据集 | 实验 | 状态 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M_D_DYNAMIC` | 完成 | 91.06 | 83.60 |
| LEVIR | `M_T_TAR` | 完成 | 91.02 | 83.53 |
| LEVIR | `M_S_STATIC` | 完成 | 91.15 | 83.74 |
| SYSU | `M_D_DYNAMIC` | 完成 | 83.56 | 71.77 |
| SYSU | `M_T_TAR` | 完成 | 82.57 | 70.31 |
| SYSU | `M_S_STATIC` | 完成 | 82.77 | 70.60 |

<!-- CASA_MOE_RUN2_RESULTS_END -->
