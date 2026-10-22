# CASA models

当前训练入口为 `train_CASA.py`，当前实验目录为 `train_scripts/CASA/Run4`。

| 文件 | 用途 |
|---|---|
| `train_CASA.py` | 40K 目标步数、四尺度深监督、验证选模、正式测试与续训 |
| `profile_CASA.py` | 满 batch 显存与吞吐测量 |
| `test_CASA.py` / `test_CASA_resume.py` | 完整模型、梯度、部署折叠与续训检查 |
| `test_change_assignment.py` | 小变化占比、四层辅助梯度与损失平均检查 |
| `test_change_context_qkv.py` | 压缩上下文与双曲几何检查 |
| `model/casa_net.py` | CASANet：MobileNetV2/HFEA → 处理模块 → TAR → DCR |
| `model/mobilenet_v2.py` / `model/encoder_fusion.py` | 骨干与编码器 |
| `model/layers/change_context_qkv.py` | 普通变化感知压缩 QKV |
| `model/layers/hyperbolic_context_qkv.py` | 双曲变化感知压缩 QKV |
| `model/layers/change_assignment.py` | 受面积变化标签监督的软分配预测头 |
| `model/layers/poincare_ops.py` | 庞加莱球运算与几何质心 |
| `dataset/casa_transforms.py` | 数据归一化与训练增广 |

来源说明见 [SOURCES.md](SOURCES.md)。每轮训练在输出目录保存完整源码归档。

`train.py`、`eval.py` 与 VMamba/TinyViM 文件保留历史实验用途；[旧主线说明](../docs/temporary/models_VMamba_README.md)。
