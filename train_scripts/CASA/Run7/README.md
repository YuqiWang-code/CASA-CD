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
