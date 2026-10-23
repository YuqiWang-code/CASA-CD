# CASA / Run7

协议 `casa_40k_relativecontext_v7`。骨干、TAR/DCR、训练预算和优化策略与Run6一致，两个模块各改变一个核心机制，不增加可学习参数。

## 模块

普通QKV在每个4×4区域计算局部变化对比：`z=(d-mean(d))/sqrt(var(d)+1e-4)`，使用`softmax(z)`、`softmax(-z)`生成高/低变化原型。Q保持N个位置，KV为N/8；V继续保留有符号差与绝对差，局部分支保留逐位置差分。常量区域仍退化为均匀池化，幅度信息保留在V中。

双曲QKV保留流形Q/K/V、对齐Möbius位移及几何压缩。以全部压缩V的均匀Lorentz质心作为共享参照`C0`，对每个Q检索质心`Cq`，输出`Log0((-C0)⊕Cq)`。全局分支表达查询相对公共上下文的几何偏移；逐位置位移局部分支提供绝对变化证据。均匀检索或所有V相同时，相对上下文为零。

## 训练

Adam、lr5e-4、模块WD0/主体WD1e-4、poly^0.9、首200step warmup、batch32、seed2333、四级BCE+Dice、40K向完整epoch上取整、独立验证集最佳F1选模。正式实验从MobileNetV2预训练全新初始化。

四库各M1_HYPER_QKV/M2_CA_QKV/M0_CTRL共12项，LEVIR→SYSU→WHU→CDD优先，两GPU按实测显存预算补位，允许跨库并行。日志`/home/yqwang/outputs/CASA-CD/CASA/Run7`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA/Run7`。

```bash
bash train_scripts/CASA/Run7/validate.sh
bash train_scripts/CASA/Run7/probe.sh
# 四个健康探针通过后
bash train_scripts/CASA/Run7/launch.sh
```

本地查询和收集：

```powershell
python .codex/status.py --run Run7
python train_scripts/CASA/Run7/collect_results.py
```

探针为LEVIR各3epoch、SYSU各4epoch，采用完整40K调度，独立记录全局/局部上下文及任务梯度，不能作为正式指标。原Run6冻结源码与快照保留。本轮正式源码在日志根目录source_code.tar.gz，最终TEST结果由collector汇表并生成冻结TXT。

## 已知限制

共享上下文成分可能含有有效的图像级信息。相对几何上下文是需要从头训练检验的假设，不能把冻结权重的支路替换结果当作新模型的最终收益。当前计划为单seed，重复实验稳定性需要后续验证。

## 启动与初期观察

正式队列于2026-10-24 02:21:37 CST启动，模型与训练源码基线`2e60998`，supervisor PID1240766。源码归档已保存到本地和服务器Run7日志根目录。探针与正式训练分别初始化，正式训练不使用探针权重。

四个真实数据健康探针均passed=true，所有层全局/局部分支具有有限、非零任务梯度。探针验证指标仅供健康观察，不计入正式指标：

| 探针 | epoch / step | 各epoch训练损失 | 结束验证F1(%) |
|---|---|---|---:|
| levir_hyperbolic | 3 / 666 | 2.1096 → 1.1966 → 1.0629 | 87.405 |
| levir_qkv | 3 / 666 | 2.1157 → 1.1656 → 1.0621 | 87.702 |
| sysu_hyperbolic | 4 / 1500 | 2.6001 → 1.9580 → 1.7343 → 1.5827 | 78.521 |
| sysu_qkv | 4 / 1500 | 2.6013 → 1.9554 → 1.7084 → 1.5649 | 78.955 |

正式状态观察：2026-10-24 02:28:18 CST，失败项为空，五个进程均存活且日志未出现NaN/Inf/OOM；首三轮损失下降，模块gamma、全局上下文和几何诊断保持有限。

| 实验 | GPU | PID | 完成epoch | 当前step | 首三轮训练损失 |
|---|---:|---:|---:|---|---|
| `M1_HYPER_QKV/LEVIR-CD-256` | 0 | 1240768 | 6 | 1332/40182 | 2.1158 → 1.1683 → 1.0620 |
| `M2_CA_QKV/LEVIR-CD-256` | 1 | 1240794 | 7 | 1554/40182 | 2.1071 → 1.1956 → 1.0324 |
| `M0_CTRL/LEVIR-CD-256` | 1 | 1240827 | 7 | 1600/40182 | 2.1051 → 1.1875 → 1.0535 |
| `M1_HYPER_QKV/SYSU-CD-256` | 0 | 1240868 | 3 | 1200/40125 | 2.5986 → 1.9514 → 1.7213 |
| `M0_CTRL/SYSU-CD-256` | 1 | 1240894 | 4 | 1500/40125 | 2.5973 → 1.9607 → 1.7295 |

普通模块探针原型分配TV约0.69；双曲模块探针半径CV最低约0.137，全局相对上下文有非零输出和任务梯度。初期数据用于判断训练健康，最终精度以完整训练后的正式测试评价。
