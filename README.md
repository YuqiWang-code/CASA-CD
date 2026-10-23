# CASA-CD

遥感二值变化检测。当前主线 **CASA-STRNet / Run6**，训练协议 `casa_40k_changevalues_v6`。

## 项目定位与研究原则

面向普通RGB双时相影像的全监督遥感二值变化检测，在LEVIR、SYSU、WHU和CDD上研究有竞争力的检测精度与轻量推理，进一步探索极轻量部署和较少的可学习参数。研究贡献聚焦双曲QKV与变化感知的上下文压缩QKV，形成结构简洁、问题明确、证据充分的方法。

- **模块有明确作用。** 双曲模块需要证明几何表示能改善变化与非变化特征的区分；普通模块需要证明变化感知压缩能保留有效上下文和细节。两者均保持Q的空间位置，仅压缩KV，并分别验证设计依据与实际收益。
- **复杂度服务于问题。** 优先采用常见、可解释的运算，新增结构应对应已观察到的具体缺陷。不依赖特殊激活函数或无关新元件制造指标增益，使提升能够归因于核心机制。
- **比较使用一致协议。** 在相同骨干、训练预算、数据划分和选模规则下比较CTRL与两个模块，通过消融确认每项改动的作用；关键结论关注重复实验的稳定性和各数据集的差异。
- **效率按实际部署衡量。** 分别报告可学习参数量、部署折叠后的参数量与计算量，以及注明输入尺寸、batch和硬件的推理速度、显存占用。准确率与成本共同评价，较少参数本身不等同于更快推理。
- **设计与评价保持独立。** 使用训练集、验证集及其中的样本和中间特征分析问题、调整结构；测试集用于方案确定后的最终评价，不用于反复选择改动。领先性结论须由相同评价口径的实验支持。

## 模型

```text
双时相影像 A / B
    → 共享 MobileNetV2 + HFEA 编码器
    → 四级双时相特征：64通道，分辨率64/32/16/8
    → 每层：双曲压缩QKV / 普通压缩QKV / 无模块
    → TAR 时空重参数化跳层融合
    → DCR 自顶向下解码
    → 共享预测头，四级输出上采样至256×256
```

| 实验 | 处理模块 | 训练参数量 |
|---|---|---:|
| `M0_CTRL` | 无处理模块 | 3,100,404 |
| `M1_HYPER_QKV` | 双曲距离注意力；逐位置计算变化位移，再几何压缩V | 3,134,332 |
| `M2_CA_QKV` | 欧氏注意力；全局V保留有符号差和绝对差 | 3,151,228 |

两个模块都保持完整Q，每个4×4区域压缩成两个对齐原型，使KV位置数为N/8。四层模块独立，同层双时相共享参数；输出进入TAR跳层连接。

普通模块以余弦相似度查询压缩K，将变化方向与变化幅度共同聚合，并融合局部有符号差分。双曲模块在庞加莱球内构建Q/K/V，逐位置计算Möbius位移，经几何压缩、距离注意力和几何聚合后Log映射回欧氏空间，与局部位移分支融合。两者以零初始化γ回接残差。

实现见 [`models/`](models/README.md)，来源见 [`SOURCES.md`](models/SOURCES.md)。

## 训练与测试

| 设置 | 值 |
|---|---|
| 输入 / batch / seed | 256×256 / 32 / 2333 |
| 优化器 | Adam，lr=5e-4，betas=(0.9,0.99)，eps=1e-8 |
| Weight decay | 处理模块全部参数为0，其余参数为1e-4 |
| 学习率 | poly^0.9，前200 step warmup |
| 损失 | 四级BCE+Dice等权求和 |
| 增广 | resize、随机裁剪后恢复、翻转、时相交换 |
| 选模 | 独立验证集最高F1；训练结束后测试该checkpoint |
| 预算 | 40K目标步数，向上取整到完整epoch |
| 标签 / 预测阈值 | `gray>=128` / `prob>0.5` |
| 部署检查 | 严格FP32折叠，误差阈值2e-4，记录二值预测差异 |

| 数据集 | Train / Val / Test | 实际训练步数 |
|---|---:|---:|
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | 40,182 |
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | 40,125 |
| WHU-CD-256 | 5,947 / 743 / 744 | 40,145 |
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | 40,248 |

## 当前实验：Run6

四库各运行双曲、普通QKV和CTRL，共12项。优先级为 **LEVIR→SYSU→WHU→CDD**，两张GPU按实测显存预算并行补位。启动时GPU0运行两项双曲实验，GPU1运行普通模块和两项CTRL，共五路。

两模块各通过3个LEVIR epoch、4个SYSU epoch的真实数据健康训练。正式训练从全新初始化开始，使用MobileNetV2预训练权重。运行源码基线为 `a946963`，本轮源码保存在日志根目录的 `source_code.tar.gz`。

<!-- CASA_RUN6_RESULTS_START -->

状态更新：2026-10-23 19:22:30 CST；队列 `running`；正式结果 **10/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 完成 | 91.552 | 84.420 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.052 | 83.573 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.616 | 84.529 |
| SYSU | `M0_CTRL` | 完成 | 82.417 | 70.092 |
| SYSU | `M1_HYPER_QKV` | 完成 | 80.832 | 67.831 |
| SYSU | `M2_CA_QKV` | 完成 | 82.388 | 70.051 |
| WHU | `M0_CTRL` | 完成 | 93.439 | 87.687 |
| WHU | `M1_HYPER_QKV` | 完成 | 94.731 | 89.990 |
| WHU | `M2_CA_QKV` | 完成 | 94.163 | 88.969 |
| CDD | `M0_CTRL` | 完成 | 97.428 | 94.986 |
| CDD | `M1_HYPER_QKV` | 训练中：24400/40248 | — | — |
| CDD | `M2_CA_QKV` | 训练中：22776/40248 | — | — |

<!-- CASA_RUN6_RESULTS_END -->

调度、检查、停止和结果收集说明见 [`train_scripts/CASA/Run6/`](train_scripts/CASA/Run6/README.md)。

```bash
# 本地：读取服务器实时状态
python .codex/status.py --run Run6

# 服务器：查看队列
cat /home/yqwang/outputs/CASA-CD/CASA/Run6/pipeline_status.json
tail -20 /home/yqwang/outputs/CASA-CD/CASA/Run6/pipeline.log

# 本地：下载结果、更新指标表及本轮快照
python train_scripts/CASA/Run6/collect_results.py

# 保存当前精简源码与全部正式指标
python tools/models_to_txt.py --tag CASA --run current
```

## 目录与产物

```text
.codex/                       桌面任务说明、服务器连接和只读状态工具
AGENTS.md                     项目工作约定
models/                       当前模型、数据处理、训练与测试
train_scripts/CASA/Run6/       当前轮次的检查、调度与收集脚本
tools/                        日志下载、指标汇总、源码快照等工具
others/                       方法来源的参考代码
outputs/CASA/Run6/            本地日志、JSON和运行源码归档
docs/experiment_metrics.xlsx  全部正式实验指标
docs/temporary/               当前源码快照和各轮冻结快照
docs/temporary/过去的想法/    历史方案、复盘与诊断资料
```

| 服务器产物 | 路径 |
|---|---|
| 项目 / 环境 | `/home/yqwang/projects/CASA-CD` / `casacd` |
| 数据集 | `/share_datasets/CD/<dataset>/{A,B,label,list}/` |
| 预训练权重 | `/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth` |
| 本轮日志 | `/home/yqwang/outputs/CASA-CD/CASA/Run6/` |
| 本轮checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA/Run6/` |

桌面工作目录说明见 [`.codex/README.md`](.codex/README.md)。

正式指标只采纳完整TEST RESULTS区块。指标表目前保留全部135条历史记录，Run6十二项完成后统一汇表；探针与未完成训练不计入。当前源码快照为 `docs/temporary/models_and_metrics_CASA_current.txt`；Run6快照对应本轮训练源码。权重与outputs不进入Git。

## 已知限制

当前为seed2333的单次实验，两个模块在LEVIR、SYSU与WHU的收益存在差异。双曲注意力的选择性、几何裁剪以及训练和验证之间的泛化差异需结合特征、支路贡献与同轮CTRL分析；单项诊断值不足以认定模块失活。
