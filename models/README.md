# CASA-Pyramid / Run11 工作源码

共享MobileNetV2原生四尺度特征→四级TAR（64通道）→跨尺度context→DCR→共享预测头。完整Q共5440位置，224KV按128/32/32/32分配，H/E为1头32维；匹配H/E只在检索距离上不同。

## 压缩与检索

X=LN(TAR)，p=sigmoid(guide(X))。各区域按p、1−p分别归一化池化P，K和欧式V共用P，支持度为相同权重下的类概率均值。Q/K共享投影，双曲版直接Lorentz提升，曲率−1、半径可学习，无径向裁剪。全局检索保留支持度、尺度偏置和数量先验校正。

读出为Wout[gate·(AV−空间均值AV)]，随后乘各尺度gamma；细两层保留局部高通分支。不同query检索相同时全局更新为0。gamma零初始化，out正常初始化。旧普通QKV的Run5实现独立保留，不参与本轮几何配对。

H/E训练2.058409M、部署1.947094M，context11,477参数；CTRL训练2.047125M、部署1.935810M。参数量不代表延迟，显存与吞吐按实际profile记录。

## 训练对照

协议casa_pyramid_40k_linearpool_interior_v11。六组为H/E/CTRL分别使用普通或内部正样本加权损失，四库共24项，仅GPU1。辅助统一0.1 raw-guide面积BCE。四级主损失为BCE+Dice之和；加权组仅首级BCE对GT两像素边界带之外的正像素乘1.25，并按权重总和归一化。其余三头、Dice和推理路径相同。

256RGB、batch32、40K整epoch、seed/context2333、预训练全新初始化；Adam、LR5e−4、编码器倍率1、warmup200、poly^.9及衰减分组保持共同协议。同步裁剪、水平/垂直翻转、90°旋转、时相交换和ImageNet归一化固定。训练与评估使用独立配对随机流。正式每epoch test F1选best，最终复测非独立盲测；探针train/val不算正式指标。

forward返回四级概率，return_aux=True返回logits和辅助；predict仅第一级。strict_fp32_fold_v1关闭TF32检查折叠、记录二值差。训练日志LOSS-DIAG记录优化与未加权主损失及前景内部比例，MODULE-DIAG记录半径CV、路由、写回和局部残差。

## 文件

- model/layers/class_pooling.py：一阶类归属与共享池化。
- model/layers/hyperbolic_context.py：匹配H/E检索、门控与细节分支。
- train_CASA.py：主损失、内部加权、训练与测试记录。
- test_interior_loss.py：归一化、GT边界带、梯度与首级作用范围。
- test_centered_readout.py、test_context_geometry.py、test_hyperbolic_context.py：空间读出、数学和梯度。
- train_scripts/CASA-Pyramid/Run11：冻结等价性验证、真实探针、显存准入、唯一调度与收集。

实际Run10归档与冻结权重的posterior_off用于验证池化等价性。历史运行源码以各轮归档和TXT为准。
