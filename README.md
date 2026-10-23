# CASA-CD

遥感二值变化检测。当前主线 **CASA-STRNet / Run6**，训练协议 `casa_40k_changevalues_v6`。

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

状态更新：2026-10-23 12:02:42 CST；队列 `running`；正式结果 **0/12**。

| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M0_CTRL` | 训练中：2100/40182 | — | — |
| LEVIR | `M1_HYPER_QKV` | 训练中：1776/40182 | — | — |
| LEVIR | `M2_CA_QKV` | 训练中：1998/40182 | — | — |
| SYSU | `M0_CTRL` | 训练中：1875/40125 | — | — |
| SYSU | `M1_HYPER_QKV` | 训练中：1600/40125 | — | — |
| SYSU | `M2_CA_QKV` | 待启动 | — | — |
| WHU | `M0_CTRL` | 待启动 | — | — |
| WHU | `M1_HYPER_QKV` | 待启动 | — | — |
| WHU | `M2_CA_QKV` | 待启动 | — | — |
| CDD | `M0_CTRL` | 待启动 | — | — |
| CDD | `M1_HYPER_QKV` | 待启动 | — | — |
| CDD | `M2_CA_QKV` | 待启动 | — | — |

<!-- CASA_RUN6_RESULTS_END -->

调度、检查、停止和结果收集说明见 [`train_scripts/CASA/Run6/`](train_scripts/CASA/Run6/README.md)。

```bash
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

正式指标只采纳完整TEST RESULTS区块。指标表保留全部135条历史记录，探针与未完成训练不计入。当前源码快照为 `docs/temporary/models_and_metrics_CASA_current.txt`；Run6快照对应本轮训练源码。权重与outputs不进入Git。

## 已知限制

Run6初期部分双曲层的注意力偏平均，验证诊断采样中的位移裁剪比例最高28.45%；逐层损失、梯度及几何观察见本轮说明。最终模块收益以完整正式测试衡量，当前为seed2333的单次实验。
