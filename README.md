# CASA-CD

遥感二值变化检测。当前主线 **CASA-STRNet**：WDMF-Net 的 MobileNetV2＋HFEA 编码器，接四尺度处理模块、TAR 时空重参数化跳层连接和 DCR 重参数化解码器。

**当前实验：CASA/Run5，双曲 Q/K/V 采用双时相共同中心与联合 RMS 归一化，保留变化标签监督的 KV 压缩。** 四库各 CTRL、双曲 QKV、普通 QKV，共 12 项；两卡按实测显存并发，优先级 LEVIR → SYSU → WHU → CDD。

| 编码器 | 跳层与解码器 | 当前实验 | GPU |
|---|---|---|---|
| MobileNetV2 + HFEA，四级各 64 通道 | MultiScaleTAR + DCR，dim=64 | CASA/Run5，四库 × 三组，共 12 项 | GPU0/1，按实测显存并发 |

## 模型

```text
A / B → shared MobileNetV2
    → 五级特征 [16,24,32,96,320]
    → shared HFEA → 四级双时相特征 [64,64,64,64]，分辨率 64/32/16/8
    → 每层：无模块 / 双曲原型 QKV / 普通变化感知原型 QKV
    → MultiScaleTAR 时空重参数化跳层融合
    → DCR 自顶向下解码，输出四级特征
    → 共享 1×1 预测头，上采样到 256×256，四级 BCE+Dice 深监督
```

两个处理模块的输出均进入 **TAR 跳层连接**：保留双时相特征，由 TAR 融合，再通过 DCR 逐级传递到解码结果。

| 实验 | 四级处理模块 | 训练参数量 |
|---|---|---:|
| `M0_CTRL` | 两个模块都关闭 | 3,100,404 |
| `M1_HYPER_QKV` | 模块一：双曲 Q/K/V、几何原型压缩与变化位移聚合 | 3,144,192 |
| `M2_CA_QKV` | 模块二：欧氏双原型压缩、差分 Value 与局部差分 | 3,160,960 |

**普通压缩 QKV**：归一化双时相特征的均值与绝对差分进入轻量变化预测头。预测概率在每个 4×4 cell 内产生变化、稳定两个软分配原型。完整 Q 以余弦相似度和可学习温度查询另一时相的压缩 K；Value 使用对齐时相的投影差值。全局差分与局部差分拼接，经输出投影和零初始化 γ 加回输入。

**双曲压缩 QKV**：嵌入、Q/K 和 V 的切空间向量均先减去两时相共同的空间均值，再按联合 RMS 归一化，使平均平方范数为 1；随后经指数映射进入 c=1 的庞加莱球。该处理保留位置间相对幅度和时相差异，去除共同偏置与整体尺度增长，不增加参数。切向范数上限 2.3 保留用于个别离群位置。变化预测头读取 Q 的切空间均值、绝对差分与同位置双曲距离；其概率用于 K/V 的几何原型压缩。负双曲距离注意力检索上下文，以 Möbius 位移表示变化 Value，通过几何加权中心聚合，再经 Log 映射与局部几何差分融合后回接残差。

**变化权重监督**：预测头为 1×1 Conv → SiLU → 3×3 深度卷积 → SiLU → 1×1 logits。每层标签按面积平均到对应分辨率，BCE-with-logits＋soft Dice 的四层平均乘 0.1，加入原有四级预测损失。辅助监督仅更新变化预测头，主变化损失通过原有分配计算端到端训练编码器和几何分支。CTRL 无变化预测头，辅助项为零。推理仅使用影像特征。

两者均保持 Q 的 `N=HW` 个位置；每个区域两个原型使 KV 为 `N/8`。四层模块参数互不共享，同层双时相共享参数。Run2 使用旧设计，其 KV 为 `N/16`，两个模块的训练参数均为 3,133,176。

MobileNetV2、HFEA 和训练设置来自 [WDMF-Net 官方仓库](https://github.com/LYT-Works/WDMF-Net)，版本 `453c5493b2aab88c908ced9135b677285059ec70`；参考源码在 [`others/WDMF-Net/`](others/WDMF-Net/REFERENCE.md)。新版普通模块保留 SAT 的完整 Q / 压缩 KV 结构；双曲模块结合 Deep Change Monitoring 的流形表征与裁剪方法，采用几何原型压缩和变化位移聚合。旧 VMamba HSAM/HQKV 不参与当前模型前向。

## 训练协议

训练采用 WDMF-Net 的策略。Run5 协议为 **`casa_40k_pairnorm_v5`**；Run4 为 **`casa_40k_supervisedkv_v4`**；Run3 为 **`casa_40k_separateqkv_v3`**，Run2 为 **`wdmf_str_40k_modulewd0_v2`**。

| 设置 | 值 |
|---|---|
| 输入 / batch / seed | 256×256 / 32 / 2333 |
| 优化器 | Adam，lr=5e-4，betas=(0.9,0.99)，eps=1e-8，主体 weight decay=1e-4；四层处理模块的全部参数 weight decay=0 |
| 学习率 | poly^0.9，沿用上游前 200 step warmup |
| 增广 | resize、随机裁剪后恢复、水平/垂直翻转、时相交换 |
| 损失 | 四级输出 BCE+Dice 等权求和；两个模块另加 0.1 × 四层变化分配 BCE+Dice 的平均值 |
| 选模与测试 | epoch 0 后逐 epoch 验证，以验证集最高 F1 选模，训练结束测试该模型 |
| 预算 | 目标 40,000 step，按上游实现向上取整到完整 epoch |
| 标签 / 预测阈值 | `gray>=128` / `prob>0.5` |

| 数据集 | Train / Val / Test | 实际 optimizer step |
|---|---:|---:|
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | 40,125 |
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | 40,182 |
| WHU-CD-256 | 5,947 / 743 / 744 | 40,145 |
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | 40,248 |

CDD 按相同预算扩展到本轮；`gray>=128` 保持 JPG 标签的正确二值化。完整参数和执行说明见 [`train_scripts/CASA/Run5/`](train_scripts/CASA/Run5/README.md)。

## Run5 当前进度

正式训练前要求双曲模块完成 16 个完整 SYSU epoch（6,000 step），两个模块分别完成 2 个完整 LEVIR epoch（各 444 step）。检查四层分支、主任务与辅助梯度、半径 CV、裁剪比例及方向集中程度。正式训练使用全新初始化，探针不计入指标。

<!-- CASA_RUN5_RESULTS_START -->

正式结果 **0/12**；正在进行训练前验证。

<!-- CASA_RUN5_RESULTS_END -->

## Run4 历史记录

Run4 于 **2026-10-23 00:01 CST** 停止，未产生完整正式测试结果。日志和 JSON 已下载至 `outputs/CASA/Run4`，服务器 checkpoint 全部保留；代码和指标快照为 `docs/temporary/models_and_metrics_CASA_Run4.txt`。指标表仍保留 **123 条**正式记录。

<!-- CASA_RUN4_RESULTS_START -->

状态更新：2026-10-23 00:03:55 CST；队列 `stopped`；正式结果 **0/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 已停止 | — | — |
| LEVIR | `M1_HYPER_QKV` | 已停止 | — | — |
| LEVIR | `M2_CA_QKV` | 已停止 | — | — |
| SYSU | `M0_CTRL` | 已停止 | — | — |
| SYSU | `M1_HYPER_QKV` | 已停止 | — | — |
| SYSU | `M2_CA_QKV` | 待启动 | — | — |
| WHU | `M0_CTRL` | 待启动 | — | — |
| WHU | `M1_HYPER_QKV` | 待启动 | — | — |
| WHU | `M2_CA_QKV` | 待启动 | — | — |
| CDD | `M0_CTRL` | 待启动 | — | — |
| CDD | `M1_HYPER_QKV` | 待启动 | — | — |
| CDD | `M2_CA_QKV` | 待启动 | — | — |

<!-- CASA_RUN4_RESULTS_END -->

## Run3 历史结果

普通与双曲模块均已通过两轮 LEVIR 真实数据短训（各 444 step），四层分支与任务梯度均非零。正式训练使用全新初始化和完整 40K 预算。三组实验为 CTRL、双曲 QKV、普通压缩 QKV。

启动时 GPU0 运行 LEVIR/SYSU 双曲模块，GPU1 运行 LEVIR 普通模块与 LEVIR/SYSU CTRL。两卡显存分别约 **29.2 / 29.9 GiB**，利用率采样为 **95% / 96%**；空闲容量按 LEVIR → SYSU → WHU → CDD 优先级自动补充。

<!-- CASA_RESULTS_START -->

状态更新：2026-10-22 22:46:15 CST；队列 `complete`；正式结果 **12/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 完成 | 91.11 | 83.67 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.48 | 84.30 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.17 | 83.77 |
| SYSU | `M0_CTRL` | 完成 | 81.78 | 69.18 |
| SYSU | `M1_HYPER_QKV` | 完成 | 81.47 | 68.73 |
| SYSU | `M2_CA_QKV` | 完成 | 81.01 | 68.07 |
| WHU | `M0_CTRL` | 完成 | 93.95 | 88.59 |
| WHU | `M1_HYPER_QKV` | 完成 | 94.32 | 89.25 |
| WHU | `M2_CA_QKV` | 完成 | 93.94 | 88.58 |
| CDD | `M0_CTRL` | 完成 | 97.45 | 95.02 |
| CDD | `M1_HYPER_QKV` | 完成 | 97.47 | 95.06 |
| CDD | `M2_CA_QKV` | 完成 | 97.48 | 95.09 |

<!-- CASA_RESULTS_END -->

### Run3 LEVIR 三组结果

三组均完成 **40,182 step**。双曲 QKV 的 F1 **91.48%**、IoU **84.30%**，相对同期 CTRL 提升 **0.37 / 0.63 个百分点**；普通压缩 QKV 为 **91.17% / 83.77%**，相对 CTRL 提升 **0.06 / 0.10 个百分点**。CTRL 为 **91.11% / 83.67%**。这组结果使用 seed 2333，尚未进行多种子重复。

三组部署折叠最大误差均不超过 **1.23e-6**，二值预测差异为 0。

### Run3 四库结果

| 数据集 | CTRL F1 / IoU (%) | 双曲 QKV F1 / IoU (%) | 普通 QKV F1 / IoU (%) |
|---|---:|---:|---:|
| LEVIR | 91.11 / 83.67 | 91.48 / 84.30 | 91.17 / 83.77 |
| SYSU | 81.78 / 69.18 | 81.47 / 68.73 | 81.01 / 68.07 |
| WHU | 93.95 / 88.59 | 94.32 / 89.25 | 93.94 / 88.58 |
| CDD | 97.45 / 95.02 | 97.47 / 95.06 | 97.48 / 95.09 |

双曲模块在 LEVIR、WHU 上的 F1 均高于同期 CTRL 约 0.37 个百分点。SYSU 上双曲、普通模块的精确率分别提高 1.96、1.97 个百分点，召回率下降 2.27、3.11 个百分点，F1 分别下降 0.31、0.78 个百分点。CDD 三组接近。结果为 seed 2333 的单次实验。

十二项均达到完整 epoch 的 40K 预算并通过部署折叠检查，二值预测差异均为 0。正式日志和 JSON 已收集至 `outputs/CASA/Run3`；指标表保留原有 111 条并纳入本轮 12 条，共 123 条。Run3 的模型代码和指标快照保存在 `docs/temporary/models_and_metrics_CASA_Run3.txt`，运行源码保存在 `outputs/CASA/Run3/source_code.tar.gz`。

## Run2 历史结果

Run2 于 **2026-10-22 11:41 CST** 停止。SYSU、LEVIR、WHU 三组均已完成正式测试，CDD 三项已停训并保留产物。Run2 代码和权重已归档保留；新版模块使用独立的 Run3 目录。

<!-- WDMF_RESULTS_START -->

状态更新：2026-10-22 11:46:54 CST；Run2 队列 `stopped`；正式结果 **9/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| SYSU | `M0_CTRL` | 完成 | 81.62 | 68.95 |
| SYSU | `M1_HYPER_QKV` | 完成 | 81.75 | 69.13 |
| SYSU | `M2_CA_QKV` | 完成 | 82.85 | 70.72 |
| LEVIR | `M0_CTRL` | 完成 | 91.46 | 84.26 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.28 | 83.96 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.39 | 84.14 |
| WHU | `M0_CTRL` | 完成 | 93.92 | 88.53 |
| WHU | `M1_HYPER_QKV` | 完成 | 94.07 | 88.80 |
| WHU | `M2_CA_QKV` | 完成 | 94.53 | 89.63 |
| CDD | `M0_CTRL` | 已停止 | — | — |
| CDD | `M1_HYPER_QKV` | 已停止 | — | — |
| CDD | `M2_CA_QKV` | 已停止 | — | — |

Run1 SYSU CTRL 已完成：F1 **82.65%**，IoU **70.44%**；作为独立历史对照保留。

<!-- WDMF_RESULTS_END -->

正式指标来自每条 `train_log.txt` 最后一个完整的 `=== TEST RESULTS ===` 区块。验证行用于选模，不作为最终测试结果。

### SYSU 三组结果

纯变化感知压缩 QKV 取得本轮 SYSU 最高 F1 **82.85%**、IoU **70.72%**，相对同期 CTRL 提升 **1.22 / 1.77 个百分点**；双曲 QKV 的 F1 **81.75%**、IoU **69.13%**，相对同期 CTRL 提升 **0.13 / 0.18 个百分点**。三组均完成 40,125 step。

### LEVIR 三组结果与部署校验

LEVIR 三组均完成 **40,182 step**。CTRL 的 F1 **91.46%**、IoU **84.26%**；双曲 QKV 为 **91.28% / 83.96%**，相对 CTRL 为 **−0.18 / −0.30 个百分点**；普通压缩 QKV 为 **91.39% / 84.14%**，相对 CTRL 为 **−0.07 / −0.12 个百分点**。

部署等价校验采用 **`strict_fp32_fold_v1`**：关闭 cuDNN benchmark 和 TF32，主输出最大绝对误差阈值为 **2e-4**，记录四尺度误差与二值预测差异，随后恢复原后端设置用于正式测试。

LEVIR 模块二从已完成 40,182 step 的权重通过校验并完成测试；模块一从 **39,294 step** 的模型、优化器及选模状态补完剩余 **888 step**，达到 **40,182 step**，也已完成测试。两项折叠误差分别为 **2.80e-6 / 7.15e-7**，二值预测均无差异。CTRL 的折叠误差为 **1.28e-6**，二值预测无差异。恢复前产物保留于日志 `archive/20261022_075845` 和 checkpoint `recovery_backups/20261022_075845` 目录。[折叠诊断记录](docs/temporary/CASA-WDMF_Run2_LEVIR_fold_diagnosis_2026-10-22.txt)。

### WHU 三组结果

普通压缩 QKV 的 F1 **94.53%**、IoU **89.63%**，相对同期 CTRL 提升 **0.62 / 1.10 个百分点**；双曲 QKV 为 **94.07% / 88.80%**，相对 CTRL 提升 **0.15 / 0.27 个百分点**。

## 验证与资源

### Run4 验证

受监督分配头通过单像素变化占比保留、BCE/Dice 数值、四层损失平均、零残差系数下辅助梯度、时相交换对称性和三变体完整网络检查。CPU 四尺度部署折叠误差均为 `5.96e-8`，二值预测差异为 0。

满 batch、单进程训练 profile 的峰值 allocated 显存：CTRL **8,873 MiB**、双曲模块 **14,321 MiB**、普通模块 **10,924 MiB**；训练 step 用时约 **0.078 / 0.133 / 0.085 s**。调度预留结合真实数据探针的 peak reserved 和 CUDA 进程开销计算。

### 历史验证

新版的几何运算、四层接口、相同输入零增强、时相交换对称性、全部任务梯度、公共路径初始化和部署折叠检查通过。两个新版模块的 CPU 四尺度最大折叠误差均为 `5.96e-8`。以下满 batch 资源与 SYSU 八轮探针数据属于旧 Run2。

- MobileNetV2 预训练加载 312 个模型张量，无缺失；分类器权重不参与特征编码。
- 三个变体的参数组覆盖、四级接口、公共路径初始化、梯度与残差激活、TAR/DCR 部署折叠测试均通过，最大输出误差约 `6e-8`。
- 两个模块均通过 SYSU 八个完整 epoch、3,000 step 的健康短训，四层分支输出和全部任务梯度均非零；[逐层诊断](docs/temporary/CASA_Run1_Run2_scripts/Run2/README.md#sysu-health-results)。
- 满 batch 单进程峰值显存：CTRL **8,873 MiB**，模块一 **11,931 MiB**，模块二 **10,150 MiB**；相应训练 step 用时约 **0.084 / 0.096 / 0.080 s**。每张 32 GB 卡并发两项，两张卡合计四项。
- 数值非有限时终止当前队列并保留日志；模型不使用旧几何模块的退化硬门。

## 产物与结果收集

服务器代码：`/home/yqwang/projects/CASA-CD/`。环境：`casacd`，使用已配置的 PyTorch 2.14/CUDA 13.2 运行时。

```bash
# 服务器进度
tail -20 /home/yqwang/outputs/CASA-CD/CASA/Run4/pipeline.log
cat /home/yqwang/outputs/CASA-CD/CASA/Run4/pipeline_status.json

# 本地收集、汇表和快照
python train_scripts/CASA/Run4/collect_results.py
```

| 产物 | 位置 |
|---|---|
| 预训练权重 | `/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth` |
| 数据 | `/share_datasets/CD/<dataset>/{A,B,label,list}/` |
| 服务器日志 | `/home/yqwang/outputs/CASA-CD/CASA/Run4/<variant>/<dataset>/` |
| 服务器 checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA/Run4/<variant>/<dataset>/` |
| 本地日志与结果 | `outputs/CASA/Run4/` |
| 指标表 | [`docs/experiment_metrics.xlsx`](docs/experiment_metrics.xlsx) |
| 代码与指标快照 | `docs/temporary/models_and_metrics_CASA_Run4.txt` |

`collect_results.py` 使用 [`tools/download_training_results.py`](tools/download_training_results.py) 下载日志和 JSON，出现新正式结果后调用 `tools/extract_metrics_to_excel.py` 和 `tools/models_to_txt.py`。历史指标缺少对应日志时，汇表工具拒绝覆盖现有工作簿。

## 目录

```text
models/
  train_CASA.py / profile_CASA.py / test_CASA.py
  model/casa_net.py
  model/mobilenet_v2.py / encoder_fusion.py
  model/layers/change_context_qkv.py / hyperbolic_context_qkv.py
  model/layers/context_compression.py / change_assignment.py / poincare_ops.py
  model/str_tar.py / str_dcr.py / str_reparam.py
  dataset/casa_transforms.py
train_scripts/CASA/Run3/     三组×四库的训练与收集
others/WDMF-Net/                 上游参考源码
tools/                          通用收集、汇表、快照和对象指标工具
outputs/                        本地实验产物，不进入 Git
docs/temporary/                 实验记录与 txt 快照
```

## 已知限制与历史

Run1 SYSU 模块一的 stage1 投影权重与 γ 衰减到约 `1e-41`，分支输出和任务梯度均为零；stage2–4 仍有效。该实验已停止，剩余旧队列取消，SYSU CTRL 已完成测试，结果作为独立历史对照保留，日志和检查点独立保存。Run1 代码归档于服务器 `code_backups/CASA-WDMF_Run1_20261021_2305.tar.gz`。

Run2 SYSU 的两个处理模块均完成 **40,125 step**，各 106 次逐 epoch 诊断中四层输出均有限。模块一各层半径 CV 始终非零，末轮 stage1 `gamma=0.236`、半径 `CV=0.172`；模块二各层 gamma 始终非零，末轮 stage1 `gamma=-0.269`、注意力熵 `0.914`。正式测试结果见上表。

HES-Net 原文报告浅层 HSAM 可能放大噪声；本轮检验四层均加入压缩几何 QKV 的结果。Run2 在正式队列前分别对两个模块运行八个 SYSU epoch 的健康探针，检查四层分支输出和任务梯度。长期稳定性和准确率由完整训练判定。

LEVIR 模块一的旧 checkpoint 未保存随机数状态，本次续训保留优化器和总步数，并重新以 seed=2333 初始化随机流；训练轨迹与不中断运行不同。新 checkpoint 已保存完整随机状态。

SYSU 的 Run1 CTRL F1 为 **82.65%**，Run2 CTRL 为 **81.62%**，两次相差 **1.03 个百分点**。Run2 三个变体各运行一次，当前差值未包含重复试验方差。

本轮采用 MobileNet、40K 目标步数和独立验证集选模。此前 VMamba 主线的 80K/test-as-val 结果属于不同协议，应以每轮同期 CTRL 衡量模块贡献。Run10 已于 2026-10-21 停止，SYSU CTRL 最后记录 **39,375/80,000 step**，无正式 TEST 结果。

- [VMamba Run9 / Run10 记录](docs/temporary/CASA-VMamba_Run9_Run10_2026-10-21.md)
- [Run1–Run8c 历史结论与指标](docs/temporary/历史版本结论与指标_2026-10-21.md)
- [服务器与数据统一说明](docs/RSML-3_服务器环境与变化检测数据统一说明.md)

Run2 的原始日志和检查点保留，正式实验各自使用独立目录。权重与 `outputs/` 不进入 Git；参考论文随 `docs/` 保存。
