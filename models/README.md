# CASA models

当前实现为CASA/Run6，协议 `casa_40k_changevalues_v6`。

```text
MobileNetV2 + HFEA → 四尺度处理模块 → TAR跳层融合 → DCR解码 → 四尺度预测
```

普通模块的全局V拼接有符号差和绝对差；双曲模块逐位置计算Möbius变化位移后再做几何压缩。两者保持完整Q与N/8的KV，四层独立，双时相共享参数。训练损失为四级BCE+Dice。

| 文件 | 用途 |
|---|---|
| `train_CASA.py` | 训练、验证选模、正式测试、部署检查和续训 |
| `profile_CASA.py` | 满batch显存与吞吐测量 |
| `test_CASA.py` | 三变体接口、梯度、公共路径初始化和部署折叠检查 |
| `test_CASA_resume.py` | 优化器、随机状态及后端设置恢复检查 |
| `test_change_context_qkv.py` | 压缩上下文、双曲几何和变化V检查 |
| `model/casa_net.py` | 整体网络 |
| `model/mobilenet_v2.py` / `encoder_fusion.py` | 骨干与编码器 |
| `model/layers/change_context_qkv.py` | 普通压缩QKV |
| `model/layers/hyperbolic_context_qkv.py` | 双曲压缩QKV |
| `model/layers/context_compression.py` / `poincare_ops.py` | 原型压缩与庞加莱球运算 |
| `model/str_tar.py` / `str_dcr.py` / `str_reparam.py` | 时空重参数化跳层、解码与折叠 |
| `model/metric_tool.py` | 分割指标 |
| `dataset/` | 双时相数据读取、归一化与训练增广 |

依赖见 `requirements.txt`，来源见 [SOURCES.md](SOURCES.md)。训练脚本与说明见 [CASA/Run6](../train_scripts/CASA/Run6/README.md)。本轮运行源码、日志与权重按轮次独立保存。
