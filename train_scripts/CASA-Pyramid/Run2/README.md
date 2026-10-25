# CASA-Pyramid / Run2

协议`casa_pyramid_40k_qk_euclideanv224_v2`。原生MobileNetV2四尺度 → TAR → 224-token跨尺度QKV → DCR。

## 模型与对照

| 变体 | Q/K | V | 训练参数 | 部署参数 |
|---|---|---|---:|---:|
| M0_CTRL | 无检索 | 无检索，保留评分监督 | 2,047,125 | 1,935,810 |
| M1_HYPER_QKV | 双曲距离/2；K先Exp再Lorentz压缩 | 先池化再欧式投影，直接加权读出 | 2,063,581 | 1,952,266 |
| M2_CA_QKV | 欧氏距离；K先池化再投影 | 先池化再欧式投影，直接加权读出 | 2,063,581 | 1,952,266 |

网格(8,4,4,4)，每区域变化/背景两原型，各尺度128/32/32/32个KV，合计224；完整Q、四头16维。logits加-log(M_source)修正数量先验，保留尺度偏置。三组评分头与面积标签辅助监督一致，系数0.2。gamma零初始化，out正常初始化，双曲曲率-1、切空间截断2.3。

本轮同时改变细尺度原型数量、尺度先验、双曲V路径和投影正则。普通版池化与线性投影交换为等价计算调整。新模块的精度收益用同轮CTRL和欧氏/双曲对照评价，与Run1总体比较不能单独归因于其中一项。

## 训练与验证

40K目标向完整epoch取整；LEVIR40182、SYSU40125、WHU40145、CDD40248步。batch32、seed2333，Adam5e-4/betas(0.9,0.99)/eps1e-8，poly^0.9，首epoch且step<200时warmup。主损失为四级BCE+Dice之和，加0.2倍四级评分BCE。context.q/k/v/out四个weight采用WD1e-4，其余context参数WD0，网络其余参数WD1e-4；CTRL的归一化和评分头仍WD0。

独立val最高F1选best，相同F1更新；训练结束测试最佳权重，严格FP32折叠并记录四级误差及二值差异。三组共同路径初始化一致，正式实验从MobileNetV2预训练重新初始化，不使用探针或Run1权重。

## 启动与采集

四库×三组，共12项；LEVIR→SYSU→WHU→CDD优先，允许跨库按显存填空，两GPU按实测profile及探针峰值并行。先进行结构/梯度/等价/折叠检查，再各模块LEVIR3epoch、SYSU4epoch真实短训。四项health_result均passed且协议匹配后才允许启动正式队列。

```bash
bash train_scripts/CASA-Pyramid/Run2/validate.sh
bash train_scripts/CASA-Pyramid/Run2/probe.sh
bash train_scripts/CASA-Pyramid/Run2/launch.sh
bash train_scripts/CASA-Pyramid/Run2/stop.sh
```

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run2
python train_scripts/CASA-Pyramid/Run2/collect_results.py
python tools/export_training_records.py --tag CASA-Pyramid --run Run2
```

服务器日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run2`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run2`，本地`outputs/CASA-Pyramid/Run2`。source_code.tar.gz冻结实际训练源码，environment.json记录环境，source_baseline.json记录源码基线。旧Run1源码、日志、权重与171条正式历史记录保留；本轮十二项收齐后正常应183条，其他新增仍保留。冻结TXT用本轮归档，不用工作区覆盖历史。

诊断记录gamma、熵、温度、各尺度注意力、原型分配、RMS(gamma*update)/RMS(T)、query注意力TV及双曲半径/裁剪。正式诊断来自每轮验证首batch，探针另外检查四级上下文、Q/K/V/out的主任务梯度。探针不计入正式指标。当前为单seed，最终收益以完整测试及后续重复实验评价。

## 正式启动与初期观察

正式队列于2026-10-25 11:07:35 CST启动，实际模型与训练源码基线`986fe75`，supervisor PID1320738。观察于2026-10-25T11:13:15+08:00，首发5项均完成至少三个完整epoch，主损失和评分损失下降；日志未见NaN/Inf/OOM或异常退出。正式模型从MobileNetV2预训练重新初始化，不使用探针或Run1权重。

| 探针 | epoch / step | 各轮总损失 | 末轮主损失 | 末轮评分损失 | 验证F1(%) |
|---|---|---|---:|---:|---:|
| levir_hyperbolic | 3 / 666 | 2.1859 → 1.2926 → 1.1523 | 1.0804 | 0.3591 | 84.577 |
| levir_qkv | 3 / 666 | 2.1884 → 1.3121 → 1.1464 | 1.0744 | 0.3596 | 84.576 |
| sysu_hyperbolic | 4 / 1500 | 2.7242 → 2.0590 → 1.8205 → 1.6620 | 1.4519 | 1.0503 | 79.722 |
| sysu_qkv | 4 / 1500 | 2.7252 → 2.0705 → 1.8305 → 1.6714 | 1.4607 | 1.0535 | 78.881 |

四项passed=true，四级上下文及Q/K/V/out均有有限、非零主任务梯度；辅助评分梯度正常。探针只用于健康观察，不计正式指标。

| 实验 | GPU | PID | 完整epoch | step | 首三轮总损失 |
|---|---:|---:|---:|---:|---|
| M1_HYPER_QKV/LEVIR-CD-256 | 0 | 1320740 | 6 | 1332 | 2.1786 → 1.3012 → 1.1443 |
| M2_CA_QKV/LEVIR-CD-256 | 1 | 1320773 | 6 | 1400 | 2.1831 → 1.3028 → 1.1398 |
| M0_CTRL/LEVIR-CD-256 | 1 | 1320803 | 6 | 1500 | 2.1820 → 1.3040 → 1.1549 |
| M1_HYPER_QKV/SYSU-CD-256 | 0 | 1320835 | 3 | 1200 | 2.7263 → 2.0737 → 1.8518 |
| M0_CTRL/SYSU-CD-256 | 1 | 1320865 | 3 | 1400 | 2.7273 → 2.0825 → 1.8697 |

实测显存预算及各变体预留：

```json
{
  "protocol": "casa_pyramid_40k_qk_euclideanv224_v2",
  "gpu_budget_mib": {
    "0": 30914,
    "1": 31072
  },
  "variant_reserved_mib": {
    "M1_HYPER_QKV": 14442,
    "M2_CA_QKV": 12748,
    "M0_CTRL": 8832
  },
  "free_headroom_mib": 1024
}
```

## Run2推理开销

RTX5090空闲GPU、256×256双时相输入、batch1、四级输出；折叠后30预热、100次同步测量，随机权重。输入/权重为FP32，cuDNN允许TF32，矩阵乘TF32关闭。FLOPs仅含profiler支持的卷积和矩阵乘，不含双曲逐元素运算。

| 模式 | 训练参数 | 部署参数 | 已统计GFLOPs | 中位延迟(ms) |
|---|---:|---:|---:|---:|
| none | 2,047,125 | 1,935,810 | 1.734 | 3.84 |
| hyperbolic | 2,063,581 | 1,952,266 | 2.206 | 6.80 |
| qkv | 2,063,581 | 1,952,266 | 2.151 | 5.62 |
