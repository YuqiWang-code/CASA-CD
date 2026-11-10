# CASA-Pyramid / Run11

协议 `casa_pyramid_40k_linearpool_interior_v11`。物理GPU1，四库24项，LEVIR→SYSU→WHU→CDD，按实测显存补位；保留GPU1其他进程和至少1GiB余量。

| 变体 | 上下文 | 首级内部正像素BCE权重 |
|---|---|---:|
| H_SIMPLE | 双曲 | 1 |
| H_INTERIOR | 双曲 | 1.25 |
| E_SIMPLE | 匹配欧式 | 1 |
| E_INTERIOR | 匹配欧式 | 1.25 |
| M0_CTRL | 无上下文 | 1 |
| CTRL_INTERIOR | 无上下文 | 1.25 |

H/E保持相同结构，仅检索距离不同。SIMPLE与INTERIOR的推理结构相同，仅首级分割BCE不同；三个模型分别设加权对照，避免将通用损失收益归因给双曲几何。单seed2333，模型/上下文/数据随机流配对。

## 模型与损失

完整Q、224个KV（128/32/32/32）、单头32维。LN(TAR)→guide概率p→区域内p/1−p一阶归一化池化，K/V使用同一压缩内容；Q/K投影后直接Lorentz提升，半径可变。保留空间中心化读出、门控、四层gamma和细两层局部细节。无锚点精炼、后验或压缩温度。

辅助统一0.1 raw-guide BCE。四个输出的BCE+Dice之和作为主损失；加权组仅对首级BCE设置 `w=1+0.25*y*I[GT两像素边界带之外]`，按 `sum(w*BCE)/sum(w)` 归一化，其他输出与Dice不变。标签仅用于训练损失，不进入预测。

batch32、Adam与学习率/衰减/预处理沿用共同协议，40K按完整epoch取整；每epoch test F1选模，同集最终复测，不属于独立盲测。模型从MobileNetV2预训练全新初始化，不使用探针或历史权重。

## 检查与启动

1. 实际Run10归档和WHU冻结权重构造posterior_off输出，与新模块直接池化比较。
2. 检查BCE权重、梯度范围、几何、跨尺度梯度、部署折叠、恢复和数据变换。
3. batch32实测显存；六组SYSU及双曲两组WHU各2个完整train/val epoch，检查损失、任务梯度、分支输出和折叠。
4. 人工阅读短训记录后启动唯一 `pipeline.py`；失败时停止本队列子进程、保留产物，不循环重启。

运行源码为日志根目录的 `runtime/`，来自对应源码归档。预检使用 `source_preflight.tar.gz`，正式源为 `source_code.tar.gz`。`prepare.py` 完成检查、profile和短训；`run_one.py`只负责声明矩阵中的单项；`evaluate.py`单独复评；`collect_results.py`只采纳完整TEST区块并逐单元格保留历史，正常至少299条记录，其他新增也保留。

先在服务器查询GPU和进程，再于独立runtime脚本目录执行：

```bash
source environment.sh
python prepare.py
# 检查readiness与短训数据后：
python pipeline.py
```

经授权接续已停止的队列时，在对应 `runtime` 目录运行 `python pipeline.py --resume`。调度器优先从最后完整 epoch 接续原停止项，保留优化器、最佳权重与随机状态，跳过完整 TEST 已存在的实验。

本地收集：`python train_scripts/CASA-Pyramid/Run11/collect_results.py`。源码、配置、环境、物理GPU和初期健康记录随本轮归档。短训和验证干预不进入正式指标表。

## 评价重点

同损失下比较H/E/CTRL，同结构下比较普通/加权；SYSU关注内部FN和匹配误报量的召回，WHU关注1px边界F1及误报。历史Run10同时改变了压缩和辅助目标，跨轮差值不能单独归因于池化公式。

[正式结果与续训记录](../../../docs/experiments/CASA-Pyramid/Run11/README.md) · [结果分析](../../../docs/experiments/CASA-Pyramid/Run11/RESULT_ANALYSIS.md)。
