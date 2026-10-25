# CASA-Pyramid / Run2

## 完成结果

Run2于2026-10-25 13:58:47 CST启动，21:35:15 CST完成，历时7小时36分28秒；源码基线`9f515ef`。12/12项正式训练及测试完成，无失败，调度器已退出。2026-10-26核查两张GPU空闲。全部部署折叠检查通过，最大绝对差3.934e-6，二值差均为0。

| 数据集 | CTRL F1 (%) | 双曲 F1 (%) | 普通 F1 (%) |
|---|---:|---:|---:|
| LEVIR | 90.9779 | 91.1631 | 91.1259 |
| SYSU | 83.3667 | 82.8377 | 82.5352 |
| WHU | 94.3527 | 94.5648 | 94.6981 |
| CDD | 96.9090 | 96.9303 | 96.8654 |

相对同轮CTRL，双曲/普通模块在LEVIR的F1分别提高0.185/0.148个百分点，WHU提高0.212/0.345个百分点；SYSU分别降低0.529/0.832个百分点；CDD双曲提高0.021个百分点、普通降低0.044个百分点。本轮呈现数据集相关的收益，SYSU是共同薄弱项；单seed微小差异不作为稳定提升证据。所有结果均为test参与选模后的复测。

[完整日志、逐轮曲线与诊断](../../../docs/experiments/CASA-Pyramid/Run2/README.md)已公开归档。


协议`casa_pyramid_40k_reliableqk_testselect_v2`。共享MobileNetV2 → TAR → 可靠原型跨尺度检索 → DCR。LEVIR→SYSU→WHU→CDD，各双曲/普通/CTRL共12项，两GPU根据实测显存补位。

## 机制

区域类别权重`m`为预测变化概率`p`或背景概率`1-p`；`w=m/Σm`，`P=ΣwX`，`r=Σwm=Σm²/Σm`。其中`X=LN(T)`，完整Q保留5440位置，KV总224，网格8/4/4/4。

普通版：`Q=Wqk X`、`K=Wqk P`、`V=Wv P`；双曲版：`Q=Exp(Wqk X)`、`K=LorentzCentroid(Q,w)`、`V=Wv P`。四头各16维，投影除sqrt(16)；双曲曲率-1、切空间截断2.3。注意力logits为`-d/tau + scale_bias - log(M_source) + log(r)`，双曲距离使用`d_H/2`，普通版使用欧氏距离。两者`A@V`经输出投影，以零初始化gamma残差回接TAR特征。

共享Q/K减少4,096参数并约束同一度量空间；可靠度保留类别支持，避免纯背景格产生的低支持变化原型取得等先验，同时保留面积小但预测明确的变化。两版均不增加卷积分支或特殊激活。

| 变体 | 训练参数 | 折叠参数 |
|---|---:|---:|
| CTRL | 2,047,125 | 1,935,810 |
| 双曲 / 普通 | 2,059,485 | 1,948,170 |

## 设计证据

2026-10-25在独立val列表前8张固定图上读取已有权重，统计`r<0.05`的低支持原型及其注意力。LEVIR普通与双曲的低支持原型占比32.1%/33.3%，四层分配给它们的注意力约32.5%～33.6%/34.0%～35.8%；SYSU双曲占比32.9%，注意力约25.6%～30.9%。该小样本说明丢失类别支持信息在实际数据中可达，不代表全库平均，也不能直接推断精度收益。详细数据保存在`outputs/CASA-Pyramid/Run2/analysis/prototype_analysis.json`。

LEVIR/WHU关注稀疏小建筑与边界，使用细尺度128个KV及不按面积惩罚的可靠度；SYSU/CDD关注复杂变化类别与时相外观差异，使用监督语义原型、同步增强和时相交换。CTRL也存在SYSU后期泛化回落，模块设计不被视为该问题的唯一解释。

TAR/DCR尺度接口、共享预测头、预训练载入、混淆矩阵和折叠公式经过审查。训练使用原始logits的BCEWithLogits与sigmoid Dice；单输出`predict()`用于实际推理，避免生成三个辅助全分辨率预测。

## 固定协议

- RGB256、batch32、seed2333；从MobileNetV2预训练新建每项模型。
- Adam5e-4、betas(.9,.99)、eps1e-8；qk/v/out权重及主体WD1e-4，其他context参数WD0。
- 全局前200步warmup，poly^.9；40K取整至完整epoch，四库181/107/217/129epoch。
- 四级主损失等权，四级评分BCE权重0.2；CTRL评分监督一致。
- A/B独立读取RGB，图像双线性、mask最近邻；训练同步随机四边0～8像素裁剪后恢复(p=.5)、H/V翻转各.5、随机四分之一周旋转；时相交换.5。ImageNet RGB归一化。无独立时相颜色扰动，评估无增强/TTA。
- FP32、AMP关、cuDNN TF32开、matmul TF32关、benchmark开；部署折叠检查使用严格FP32。
- 每epoch从0开始在test选最高F1，最终在同一test复测best；manifest标注`selection_split=test, independent_test=false`。val仅用于训练前诊断。
- 此设置参考[ChangeMamba 2024官方代码](https://github.com/ChenHongruixuan/ChangeMamba/blob/72597fa4a1fde0bffa94c94302a6dc848898be13/changedetection/script/train_MambaBCD.py)的test选模，原实现为每500步/Kappa，与本轮每epoch/F1有区别。

## 检查与运行

`validate.sh`检查可靠度边界与梯度、共享Q/K坐标、池化等价性、224-token布局、模型梯度、RGB/时相与mask对齐、稳定损失、全局warmup、恢复、折叠及调度。`profile_CASA.py`测batch32峰值；`probe.sh`分别做LEVIR3epoch、SYSU4epoch的两模块真实短训，四项通过后允许正式队列。探针权重不用于正式训练。

```bash
bash train_scripts/CASA-Pyramid/Run2/validate.sh
bash train_scripts/CASA-Pyramid/Run2/probe.sh
bash train_scripts/CASA-Pyramid/Run2/launch.sh
```

`pipeline.py`是唯一正式调度器；`scheduling.json`与`placement.json`记录显存预算和物理GPU。训练启动前保存`source_code.tar.gz`、环境、配置和源码基线；活动模型不覆盖。新失败先保留日志并分析。用户已授权本次初期检查遇到真实问题时停止、修复后重训。

结果仅取完整TEST区块。原173条历史记录逐单元格保留，本轮12项已入表，共185条；独立val选模的两个完整结果归档为Run2_val。冻结TXT从实际运行源码提取，当前TXT从工作源码生成。

<!-- STARTUP -->
正式队列于2026-10-25 13:58:47 CST启动，源码基线`9f515ef`，supervisor PID1334437。截至2026-10-25T14:04:46+08:00，首发五项均完成至少3个训练及test评估epoch，主损失和评分损失下降，未见NaN/Inf/OOM或异常退出。两模块四项真实短训均通过，四层分支及qk/v/out任务梯度非零。正式权重从MobileNetV2预训练全新初始化，未使用探针或历史权重。

| 实验 | GPU | PID | 已完成epoch | 首三轮总损失 |
|---|---:|---:|---:|---|
| M1_HYPER_QKV/LEVIR-CD-256 | 0 | 1334439 | 5 | 2.2231 → 1.3654 → 1.1898 |
| M2_CA_QKV/LEVIR-CD-256 | 1 | 1334470 | 6 | 2.2181 → 1.3658 → 1.1979 |
| M0_CTRL/LEVIR-CD-256 | 1 | 1334499 | 6 | 2.2215 → 1.3699 → 1.2030 |
| M1_HYPER_QKV/SYSU-CD-256 | 0 | 1334531 | 3 | 2.7216 → 2.1003 → 1.8660 |
| M0_CTRL/SYSU-CD-256 | 1 | 1334561 | 3 | 2.7246 → 2.1221 → 1.8952 |

<!-- END_STARTUP -->

## 健康短训与实测成本

| 探针 | epoch | 总损失首→末 | 末轮主损失 | 末轮评分损失 |
|---|---:|---:|---:|---:|
| levir_hyperbolic | 3 | 2.2239 → 1.1663 | 1.0930 | 0.3664 |
| levir_qkv | 3 | 2.2229 → 1.1732 | 1.0994 | 0.3688 |
| sysu_hyperbolic | 4 | 2.7254 → 1.6954 | 1.4846 | 1.0544 |
| sysu_qkv | 4 | 2.7207 → 1.7057 | 1.4941 | 1.0580 |

RTX5090，256×256，FP32；训练batch32实测6步、前2步预热。推理为折叠后predict()单输出、batch1、30预热/100同步计时、随机权重；cuDNN TF32开、matmul TF32关。FLOPs仅含支持的卷积和矩阵乘，未计双曲逐元素运算。

| 模式 | batch32峰值保留显存(MiB) | 同步训练步(ms) | 单输出推理(ms) | 已统计GFLOPs |
|---|---:|---:|---:|---:|
| none | 8200 | 54.01 | 3.68 | 1.734 |
| hyperbolic | 13518 | 103.80 | 6.42 | 2.165 |
| qkv | 11966 | 81.53 | 5.47 | 2.156 |
