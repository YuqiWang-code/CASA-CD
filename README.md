# CASA-CD

遥感二值变化检测。当前主线 **CASA-WDMF-STRNet**：WDMF-Net 的 MobileNetV2＋HFEA 编码器，接四尺度处理模块、TAR 时空重参数化跳层连接和 DCR 重参数化解码器。

| 编码器 | 跳层与解码器 | 当前实验 | GPU |
|---|---|---|---|
| MobileNetV2 + HFEA，四级各 64 通道 | MultiScaleTAR + DCR，dim=64 | `CASA-WDMF/Run2`，四库 × 三组，共 12 项 | GPU0/1，各两路，共四路 |

## 模型

```text
A / B → shared MobileNetV2
    → 五级特征 [16,24,32,96,320]
    → shared HFEA → 四级双时相特征 [64,64,64,64]，分辨率 64/32/16/8
    → 每层：无模块 / 双曲距离压缩 QKV / 普通压缩 QKV
    → MultiScaleTAR 时空重参数化跳层融合
    → DCR 自顶向下解码，输出四级特征
    → 共享 1×1 预测头，上采样到 256×256，四级 BCE+Dice 深监督
```

两个处理模块的输出均进入 **TAR 跳层连接**：保留双时相特征，由 TAR 融合，再通过 DCR 逐级传递到解码结果。

| 实验 | 四级处理模块 | 训练参数量 |
|---|---|---:|
| `M0_CTRL` | 两个模块都关闭 | 3,100,404 |
| `M1_HYPER_QKV` | 模块一：双曲距离＋变化感知压缩 QKV | 3,133,176 |
| `M2_CA_QKV` | 模块二：变化感知压缩 QKV | 3,133,176 |

**共享压缩与残差路径**：由同位置双时相绝对差构造变化分数，在每个规则 4×4 cell 内经 softmax 得到共享权重，压缩为一个上下文 token。全分辨率 Q 查询另一时相的压缩 K/V；Q/K/V 均为 32 维。注意力 logits 标准化后聚合 V，投影回 64 通道，再乘零初始化的可学习残差系数并加回原特征。

**模块一**将 Q/K 经指数映射送入 c=1 的庞加莱球，以双曲距离替换 QK 点积。采用 HSAM 的正距离约定来强调跨时相差异，切空间范数上限为 2.3，距离用稳定的 `2 asinh(sqrt(·))` 形式。**模块二**使用普通 QK 点积。两者均在编码器四层输出上独立应用。

MobileNetV2、HFEA 和训练设置来自 [WDMF-Net 官方仓库](https://github.com/LYT-Works/WDMF-Net)，版本 `453c5493b2aab88c908ced9135b677285059ec70`；参考源码在 [`others/WDMF-Net/`](others/WDMF-Net/REFERENCE.md)。几何模块依据 HES-Net 的 HSAM 进行压缩 QKV 改造。旧 VMamba HSAM/HQKV 不参与当前模型前向。

## 训练协议

当前采用 WDMF-Net 的训练策略，协议 **`wdmf_str_40k_modulewd0_v2`**。

| 设置 | 值 |
|---|---|
| 输入 / batch / seed | 256×256 / 32 / 2333 |
| 优化器 | Adam，lr=5e-4，betas=(0.9,0.99)，eps=1e-8，主体 weight decay=1e-4；四层处理模块的 γ、Q/K/V 与输出投影 weight decay=0 |
| 学习率 | poly^0.9，沿用上游前 200 step warmup |
| 增广 | resize、随机裁剪后恢复、水平/垂直翻转、时相交换 |
| 损失 | 四级输出的 BCE+Dice 等权求和 |
| 选模与测试 | epoch 0 后逐 epoch 验证，以验证集最高 F1 选模，训练结束测试该模型 |
| 预算 | 目标 40,000 step，按上游实现向上取整到完整 epoch |
| 标签 / 预测阈值 | `gray>=128` / `prob>0.5` |

| 数据集 | Train / Val / Test | 实际 optimizer step |
|---|---:|---:|
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | 40,125 |
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | 40,182 |
| WHU-CD-256 | 5,947 / 743 / 744 | 40,145 |
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | 40,248 |

CDD 按相同预算扩展到本轮；`gray>=128` 保持 JPG 标签的正确二值化。完整参数和执行说明见 [`train_scripts/CASA-WDMF/Run2/`](train_scripts/CASA-WDMF/Run2/README.md)。

## 当前进度与结果

自 **2026-10-22 08:53 CST** 起采用 **GPU0/1 各两路，共四路**。按 SYSU → LEVIR → WHU → CDD 的优先级补充空闲槽位，允许跨数据集并行。SYSU、LEVIR 三组均已完成，当前并行运行 WHU 三组与 CDD 模块一，后续自动补入 CDD 其余两组。

<!-- WDMF_RESULTS_START -->

状态更新：2026-10-22 10:52:03 CST；Run2 队列 `running`；正式结果 **6/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| SYSU | `M0_CTRL` | 完成 | 81.62 | 68.95 |
| SYSU | `M1_HYPER_QKV` | 完成 | 81.75 | 69.13 |
| SYSU | `M2_CA_QKV` | 完成 | 82.85 | 70.72 |
| LEVIR | `M0_CTRL` | 完成 | 91.46 | 84.26 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.28 | 83.96 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.39 | 84.14 |
| WHU | `M0_CTRL` | 训练中：31100/40145 | — | — |
| WHU | `M1_HYPER_QKV` | 训练中：29800/40145 | — | — |
| WHU | `M2_CA_QKV` | 训练中：31100/40145 | — | — |
| CDD | `M0_CTRL` | 待启动 | — | — |
| CDD | `M1_HYPER_QKV` | 训练中：4700/40248 | — | — |
| CDD | `M2_CA_QKV` | 待启动 | — | — |

Run1 SYSU CTRL 已完成：F1 **82.65%**，IoU **70.44%**；作为独立历史对照保留。

<!-- WDMF_RESULTS_END -->

正式指标来自每条 `train_log.txt` 最后一个完整的 `=== TEST RESULTS ===` 区块。验证行用于选模，不作为最终测试结果。

### SYSU 三组结果

纯变化感知压缩 QKV 取得本轮 SYSU 最高 F1 **82.85%**、IoU **70.72%**，相对同期 CTRL 提升 **1.22 / 1.77 个百分点**；双曲 QKV 的 F1 **81.75%**、IoU **69.13%**，相对同期 CTRL 提升 **0.13 / 0.18 个百分点**。三组均完成 40,125 step。

### LEVIR 三组结果与部署校验

LEVIR 三组均完成 **40,182 step**。CTRL 的 F1 **91.46%**、IoU **84.26%**；双曲 QKV 为 **91.28% / 83.96%**，相对 CTRL 为 **−0.18 / −0.30 个百分点**；普通压缩 QKV 为 **91.39% / 84.14%**，相对 CTRL 为 **−0.07 / −0.12 个百分点**。

部署等价校验采用 **`strict_fp32_fold_v1`**：关闭 cuDNN benchmark 和 TF32，主输出最大绝对误差阈值为 **2e-4**，记录四尺度误差与二值预测差异，随后恢复原后端设置用于正式测试。

LEVIR 模块二从已完成 40,182 step 的权重通过校验并完成测试；模块一从 **39,294 step** 的模型、优化器及选模状态补完剩余 **888 step**，达到 **40,182 step**，也已完成测试。两项折叠误差分别为 **2.80e-6 / 7.15e-7**，二值预测均无差异。CTRL 的折叠误差为 **1.28e-6**，二值预测无差异。恢复前产物保留于日志 `archive/20261022_075845` 和 checkpoint `recovery_backups/20261022_075845` 目录。[折叠诊断记录](docs/temporary/CASA-WDMF_Run2_LEVIR_fold_diagnosis_2026-10-22.txt)。

## 验证与资源

- MobileNetV2 预训练加载 312 个模型张量，无缺失；分类器权重不参与特征编码。
- 三个变体的参数组覆盖、四级接口、公共路径初始化、梯度与残差激活、TAR/DCR 部署折叠测试均通过，最大输出误差约 `6e-8`。
- 两个模块均通过 SYSU 八个完整 epoch、3,000 step 的健康短训，四层分支输出和全部任务梯度均非零；[逐层诊断](train_scripts/CASA-WDMF/Run2/README.md#sysu-health-results)。
- 满 batch 单进程峰值显存：CTRL **8,873 MiB**，模块一 **11,931 MiB**，模块二 **10,150 MiB**；相应训练 step 用时约 **0.084 / 0.096 / 0.080 s**。每张 32 GB 卡并发两项，两张卡合计四项。
- 数值非有限时终止当前队列并保留日志；模型不使用旧几何模块的退化硬门。

## 运行与收集

服务器代码：`/home/yqwang/projects/CASA-CD/`。环境：`casacd`，使用已配置的 PyTorch 2.14/CUDA 13.2 运行时。

```bash
# 服务器启动 / 停止
cd /home/yqwang/projects/CASA-CD
bash train_scripts/CASA-WDMF/Run2/launch.sh 4
bash train_scripts/CASA-WDMF/Run2/stop.sh

# 服务器进度
tail -20 /home/yqwang/outputs/CASA-CD/CASA-WDMF/Run2/pipeline.log
cat /home/yqwang/outputs/CASA-CD/CASA-WDMF/Run2/pipeline_status.json

# 本地收集、汇表和快照
python train_scripts/CASA-WDMF/Run2/collect_results.py
```

| 产物 | 位置 |
|---|---|
| 预训练权重 | `/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth` |
| 数据 | `/share_datasets/CD/<dataset>/{A,B,label,list}/` |
| 服务器日志 | `/home/yqwang/outputs/CASA-CD/CASA-WDMF/Run2/<variant>/<dataset>/` |
| 服务器 checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-WDMF/Run2/<variant>/<dataset>/` |
| 本地日志与结果 | `outputs/CASA-WDMF/Run2/` |
| 指标表 | [`docs/experiment_metrics.xlsx`](docs/experiment_metrics.xlsx) |
| 代码与指标快照 | `docs/temporary/models_and_metrics_CASA-WDMF_Run2.txt` |

`collect_results.py` 使用 [`tools/download_training_results.py`](tools/download_training_results.py) 下载日志和 JSON，出现新正式结果后调用 `tools/extract_metrics_to_excel.py` 和 `tools/models_to_txt.py`。历史指标缺少对应日志时，汇表工具拒绝覆盖现有工作簿。

## 目录

```text
models/
  train_wdmf.py / profile_wdmf.py / test_wdmf_str.py
  model/casa_wdmf_str_net.py
  model/wdmf_mobilenet_v2.py / wdmf_encoder.py
  model/layers/change_context_qkv.py
  model/str_tar.py / str_dcr.py / str_reparam.py
  dataset/wdmf_transforms.py
train_scripts/CASA-WDMF/Run2/     三组×四库的训练与收集
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

代码、协议与文档更新后提交并推送到 GitHub。训练运行期间保持所用模型代码固定。原始日志和检查点保留，正式实验各自使用独立目录。权重与 `outputs/` 不进入 Git；参考论文随 `docs/` 保存。
