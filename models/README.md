# CASA models

当前模型采用CASA/Run6结构，协议为 `casa_40k_changevalues_v6`，训练入口为 `train_CASA.py`。普通模块的全局Value拼接有符号差和绝对差；双曲模块逐位置计算Möbius变化位移后再做几何压缩。两者保持完整Q与N/8的KV，训练损失为四级BCE+Dice。

Run6脚本位于 `train_scripts/CASA/Run6/`，包含接口检查、真实数据健康训练、实测显存配置、两卡调度、停止和结果收集。历史轮次使用各自的 `source_code.tar.gz` 复现。

| 文件 | 用途 |
|---|---|
| `train_CASA.py` | 40K 目标步数、四尺度深监督、验证选模、正式测试与续训 |
| `profile_CASA.py` | 满 batch 显存与吞吐测量 |
| `test_CASA.py` / `test_CASA_resume.py` | 完整模型、梯度、部署折叠与续训检查 |
| `test_change_context_qkv.py` | 压缩上下文与双曲几何检查 |
| `model/casa_net.py` | CASANet：MobileNetV2/HFEA → 处理模块 → TAR → DCR |
| `model/mobilenet_v2.py` / `model/encoder_fusion.py` | 骨干与编码器 |
| `model/layers/change_context_qkv.py` | 普通变化感知压缩 QKV |
| `model/layers/hyperbolic_context_qkv.py` | 双曲变化感知压缩 QKV |
| `model/layers/poincare_ops.py` | 庞加莱球运算与几何质心 |
| `dataset/casa_transforms.py` | 数据归一化与训练增广 |

来源说明见 [SOURCES.md](SOURCES.md)。每轮训练在输出目录保存完整源码归档。

`train.py`、`eval.py` 与 VMamba/TinyViM 文件保留历史实验用途；[旧主线说明](../docs/temporary/models_VMamba_README.md)。
