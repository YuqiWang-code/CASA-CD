# CASA-MoE 四专家探索线

当前工作源码对应 **Run3**，协议`casa_moe_40k_candidate_evidence_v3`，与主线`models/`独立。历史源码见各轮冻结TXT与运行归档。

```text
A/B → 共享MobileNetV2原生四尺度
 ├─ 原生特征 → 完整TAR → T候选
 └─ 共享线性投影Pa/Pb
      ├─ [|Pb−Pa|,(Pa+Pb)/2] → Conv/BN/SiLU → X
      │    └─ LN/guide → 224个变化/背景KV
      │         ├─ 共享QK/V/out + Lorentz距离 → X+RH
      │         └─ 共享QK/V/out + 欧式距离    → X+RE
      └─ DWT(Pb−Pa)
           ├─ S1/S2: 高频→处理→零LL重建 → 浅层RW
           └─ S2/S3: |LL|/2→处理       → 深层RW

S4→S1: router(LN(X), LN(TAR), LN(上一粗层D)) → 混合H/E/W/T → DCR
                                              └→ 共享四级预测头
```

完整Q、224KV（128/32/32/32）、1头32维，双曲曲率−1且无固定占比半径，H/E共享内容和投影、各有温度。小波只有浅层使用IDWT，高频重建后不加激活；低频直接注入深层。四路稠密执行。

路由直接读取两种竞争底座X与TAR，并结合粗层解码信息。各自通道归一化后拼接，由16维路由器学习选择，不把通道余弦或特征范数解释为置信度。S4无上一级解码输入。新增TAR输入共4096参数。

普通训练/推理将H/E加权注意力合并为 `A=gH·AH+gE·AE`，共享一次 `γ·out(AV−(gH+gE)·Vq)`。共享无偏置的1×1输出投影与增益使它与两路独立读出相加等价，保留对路由和检索参数的梯度。诊断及探针按专家分开读出以测量各分支。

## 实现

- `model/casa_net.py`：显式粗到细混合/DCR，匹配TAR-only与部署裁剪。
- `model/layers/candidate_context.py`：公共时相stem、配对H/E、小波分流、动态/静态路由。
- `model/layers/lorentz_ops.py`：稳定Lorentz平方距离。
- `train_CASA.py`：共同训练协议、配对loader随机流、损失/梯度/空间诊断。
- `check_candidates.py`：数值、形状、初始化、梯度、折叠契约。
- `profile_MOE.py`：包含诊断的训练峰值与实际部署推理开销。

H/E增益和小波增益零初始化；动态router末层零初始化，初始四权重均为1/4。所有新模块归入context，Conv/Linear权重解耦WD1e-4，其他context参数WD0。匹配T-only保留公共X的guide训练路径，部署时移除这一辅助路径。

部署参数：T-only **1.935617M**，可学习常量融合 **2.100452M**，动态融合 **2.112676M**。参数量不代表实际速度；Run3仅在物理GPU0单路串行，实测训练峰值后安排正式实验。

[Run3脚本与阶段矩阵](../train_scripts/CASA-MoE/Run3/README.md) · [设计规格](../docs/temporary/CASA_MoE_四专家架构设计_调研与验证方案.md) · [Run1完整结果](../docs/experiments/CASA-MoE/Run1/README.md)

[Run2完整结果与曲线](../docs/experiments/CASA-MoE/Run2/README.md)：首阶段6/6项完成，独立于CASA-Pyramid主线。
