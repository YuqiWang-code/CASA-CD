# CASA-MoE 探索线

共享MobileNetV2 → 四级TAR → 四专家动态融合 → DCR → 共享二值预测头。当前 **CASA-MoE / Run1**，协议`casa_moe_40k_dense_v1`，使用物理GPU1；源码与主线`models/`独立。复制起点`9414d79`，主线Run6运行源码`9bef162`。

## 探索方向

| 专家 | 计算 | 作用假设 |
|---|---|---|
| TAR | 共享TAR输出T，零额外专家计算 | 保留局部双时相变化证据 |
| 双曲QKV | 1头32维，完整Q检索448个跨尺度KV；Lorentz距离 | 以方向和预测变化占比组织变化上下文 |
| 普通QKV | 与双曲相同内容池及宽度，归一化方向的欧氏距离 | 提供不同度量的上下文修正 |
| 频率 | 64→16通道，单级Haar，LL与LH/HL/HH分别处理，再重建 | 结合低频区域结构与有符号高频细节 |

```text
A/B → shared MobileNetV2 → TAR → T1,T2,T3,T4
                                  |
                       shared LN / guide / pooling
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
     H: ΔH                     E: ΔE                     F: ΔF
  Lorentz Q/K               Euclidean Q/K          Haar LL + LH/HL/HH
  Euclidean V               Euclidean V            low-conditioned high
        +-------------------------+-------------------------+
                                  |
                 pixel router: πTAR, πH, πE, πF
                                  |
           Y = T + γ · (πH ΔH + πE ΔE + πF ΔF)
                                  |
                    DCR → shared prediction head
```

四个路由权重非负且和为1，等价于在T、T+γΔH、T+γΔE、T+γΔF之间加权。γ为四尺度各64通道的唯一零初始化残差系数；专家投影正常初始化。路由共享`1×1(64→16)→SiLU→1×1(16→4)→softmax`，不输入数据集编号。

H/E共享一次LN、guide及448个内容原型P：四尺度256/64/64/64，每区变化/背景各两个外观子原型。各专家有独立QK、V和输出投影；同一专家的Q/K共享投影，K与V从同一个P构造。双曲方向归一化后乘1+p，内在半径asinh(1+p)，p为guide预测占比。检索决定条件分配，外层路由控制采用量；相对读出为Wout(AV−Vquery)。

频率专家保留三个方向的有符号高频系数，低频通过sigmoid调节高频修正；Haar分析/合成无需可学习变换参数。高频也可能是纹理、阴影或配准误差，因此不作统一增强。低高频处理构成一条候选路径。

## 训练与消融

Run1共24项，优先LEVIR→SYSU→WHU→CDD：LEVIR和SYSU各9项（CTRL、单H、单E、单F、固定等权、动态MoE、动态去H/去E/去F）；WHU和CDD各3项（CTRL、固定等权、动态MoE）。单专家权重为1，等权四路各1/4，移除专家后的路由仅在剩余路径归一化。公共模块及相同专家在各变体中使用相同初始化。

共同协议为RGB256、batch32、seed2333、40K整epoch、Adam全网5e−4、200步warmup/poly^0.9；context卷积/线性权重解耦WD1e−4，其余context WD0，主体耦合L2 WD1e−4。四级BCE+Dice加0.2guide。同步裁剪、水平/垂直翻转、90度旋转和时相交换，ImageNet归一化。正式每epoch用test F1选模，最终复测属于test参与选模结果；短训使用train/val。

实现入口：[moe_context.py](model/layers/moe_context.py)、[casa_net.py](model/casa_net.py)、[train_CASA.py](train_CASA.py)。数学、梯度与折叠检查用`check_MOE.py`，显存与吞吐用`profile_MOE.py`。启动、短训、测试和收集统一见[Run1](../train_scripts/CASA-MoE/Run1/README.md)。

## 观测与边界

记录各尺度路由、按预测变化占比加权的路由分组、专家实际残差/TAR RMS、输出相似度、主分割任务梯度、检索分布与双曲半径。预测占比分组不等同于GT分组；权重大小不等同于纠错贡献。正式比较包括固定等权与逐专家移除。

本轮为**稠密动态融合**，所有启用专家均执行。TAR/DCR保留原部署折叠，输入相关路由与注意力不能折成固定卷积。单seed、test参与选模及H/E同时改变度量和学习到的投影，是本轮精度归因的限制。真实区域Top-2留待互补性成立后验证。

[调研与建议结构](research/MOE_RESEARCH.md) · [论文](../docs/参考文献/MoE动态专家探索/README.md) · [原始基线说明](BASELINE.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `model/`、`dataset/`、训练与检查脚本 | 四专家模型、共同数据设置及检查 |
| `BASELINE.md` | 复制时的原README，保留基线含义 |
| `BASELINE.json` | 起点、复制范围与旧探索归档位置 |
| `research/MOE_RESEARCH.md` | 文献核实、四专家接口、路由、频率方案与验证安排 |

探索结果使用独立目录`outputs/CASA-MoE/`。原`models_v1`内容完整保存在`outputs/tooling_archive/models_v1_before_moe_20261028_105925`。

源码记录函数以本目录为准。直接执行`python models_v1/train_CASA.py ...`导入本目录模型；正式队列使用独立冻结运行目录，源码归档与对应轮次一同保存。
