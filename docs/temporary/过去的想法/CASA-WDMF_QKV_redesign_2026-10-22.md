# CASA-WDMF：独立的欧氏与双曲压缩 QKV

## 状态

2026-10-22 11:41 CST，Run2 队列已停止，自动化已暂停。SYSU、LEVIR、WHU 共九项完成正式测试，CDD 三项保留停训日志和检查点。新增 WHU 指标已下载并入表，指标表共 111 条；已有 108 条记录逐单元格保持不变。

| Run2 / WHU | F1 (%) | IoU (%) |
|---|---:|---:|
| CTRL | 93.92 | 88.53 |
| 双曲 QKV | 94.07 | 88.80 |
| 普通压缩 QKV | 94.53 | 89.63 |

当前 `models/` 为独立模块设计，入口协议 `casa_40k_separateqkv_v3`。Run2 使用 `wdmf_str_40k_modulewd0_v2`，其代码与模型快照见 `models_and_metrics_CASA-WDMF_Run2.txt`。新版尚无数据集训练结果。

## 依据与取舍

[SAT](https://arxiv.org/abs/2604.07994) 保留完整 Query，仅聚合 Key/Value；其方法包含基于密度的聚类、范数恢复与局部/全局建模。这里保留非对称压缩，采用对齐的变化/稳定双原型，输出仍是逐位置特征。每个 4×4 区域生成两个原型，因此 `Q=N=HW`，`KV=M=N/8`。相比 Run2 的单原型，KV 数量增加一倍。

[Deep Change Monitoring](https://arxiv.org/abs/2503.00643) 研究具有层级结构的树冠变化，使用双曲表征、双曲分类器与特征裁剪。论文为采用双曲几何提供依据；本文模块采用的几何压缩和差分 Value 是面向像素级变化检测的设计。

两个模块均保留四层处理、时相共享参数、零初始化残差系数，以及 TAR/DCR 接口。处理模块参数的 weight decay 为 0，主体为 1e-4。

## 整体连接

```text
Image A / Image B
        |
        v
Shared MobileNetV2 + HFEA
        |
        +-- (A1, B1) : 64 x 64 x 64 -- Module_1 -- TAR_1 -- T1 --+
        +-- (A2, B2) : 64 x 32 x 32 -- Module_2 -- TAR_2 -- T2 --+
        +-- (A3, B3) : 64 x 16 x 16 -- Module_3 -- TAR_3 -- T3 --+--> DCR
        +-- (A4, B4) : 64 x  8 x  8 -- Module_4 -- TAR_4 -- T4 --+
                                                                   |
                                                                   v
                                              Shared 1x1 head, resize, sigmoid
                                                                   |
                                                      Four prediction maps
```

## 普通变化感知压缩 QKV

1. 对两个时相逐 token 做共享 LayerNorm，计算对齐特征差异 `d`。变化隶属度为 `s=d/(1+d)`。
2. 每个 4×4 区域内，将 `s` 与 `1-s` 分别归一化，得到变化、稳定两个原型的权重。两个时相共用权重，保证原型位置与含义对齐。
3. 原分辨率 Q 查询另一时相的压缩 K，采用余弦相似度与可学习温度，初始温度为 0.2。
4. Value 为对齐时相投影的差值。全局注意力传播压缩变化证据，同时由全分辨率深度卷积保留局部差分。

```text
A, B --> shared LayerNorm --> d = mean(abs(LN(A)-LN(B)))
                                   |
                         s = d / (1+d)
                                   |
                 4x4 cell: change / stable memberships
                                   |
                         shared weights w (2 prototypes)
                                   |
 A --> LN --> Wq -----------------> Q_A [N,32]
 B --> LN --> Wk --> pool(w) -----> K_B [M,32]
                                   |
                   P_AB = softmax(cos(Q_A,K_B) / tau)
                                   |
 A,B --> Wv --> V_A - V_B --> pool(w) --> DeltaV [M,32]
                                   |
                       G_A = P_AB @ DeltaV
                                   |
                                   +--------------------------+
                                                              |
 V_A - V_B --> DWConv3x3 --> SiLU --> local L_A ----------------+--> Concat
                                                                      |
                                                                 Wo (64->64)
                                                                      |
 A -------------------------------------------------------------> Add(gamma * update)
                                                                      |
                                                                     A'

B branch: Q_B queries K_A; Value and local difference reverse sign.
```

## 双曲变化感知压缩 QKV

双曲模块以曲率 `c=1` 的庞加莱球为工作空间。输入先经 64→32 维嵌入，除以 `sqrt(32)` 后映射入球。Möbius 线性层采用 `Exp_0(W Log_0(x))`；Q/K 共用同一度量投影，初始为单位矩阵。

变化隶属度由同位置双曲距离产生。Key 和 Value 均用 Einstein/Lorentz 质心压缩；Value 的变化量用 Möbius 位移 `(-V_B) (+) V_A` 表示。注意力以负距离检索相似上下文，再聚合其变化位移。原分辨率的双曲位移提供局部证据。聚合后的全局位移与局部位移经 Log 映射回切空间，接深度卷积、输出投影和残差。

```text
A,B --> E: 1x1 (64->32) / sqrt(32)
                      |
                tangent clipping (max norm 2.3)
                      |
                     Exp_0
                      |
             X_A, X_B in Poincare ball
                      |
       +--------------+----------------------------+
       |                                           |
   Mobius W_qk                                 Mobius W_v
       |                                           |
 Q_A, Q_B [N,32]                              V_A, V_B [N,32]
       |                                           |
 d_H(Q_A,Q_B) --> change/stable weights w            |
       |                                           |
       +--> Hyperbolic centroid(w, Q_A/Q_B)         |
       |                  |                        |
       |               K_A, K_B [M,32]             |
       |                                           v
       |                         Hyperbolic centroid(w, V_A/V_B)
       |                                           |
       |                                      CV_A, CV_B
       |                                           |
       |                              DeltaV_A = (-CV_B) (+) CV_A
       |                                           |
 Q_A + K_B --> P_AB = softmax(-d_H / tau)            |
                      |                            |
                      +----------+-----------------+
                                 |
                 Hyperbolic centroid(P_AB, DeltaV_A)
                                 |
                               Log_0 --> global G_A -----------+
                                                              |
 V_A,V_B --> (-V_B) (+) V_A --> Log_0 --> DWConv3x3 --> SiLU -----+--> Concat
                                                                      |
                                                                 Wo (64->64)
                                                                      |
 A -------------------------------------------------------------> Add(gamma * update)
                                                                      |
                                                                     A'

B branch uses reversed Mobius displacements and Q_B / K_A.
```

加权双曲质心通过等距 Lorentz 表征的归一化加权和计算，结果返回庞加莱球；该质心不等同于一般的精确 Fréchet 均值。特征和位移的最大球半径为 `tanh(2.3)`。数值距离在重合点附近采用平滑范数。各 token 的半径仍允许不同。

### 参数与代价

| 变体 | 总训练参数 | 四层模块参数 | Q / KV token 数 |
|---|---:|---:|---|
| CTRL | 3,100,404 | 0 | — |
| 普通压缩 QKV | 3,143,036 | 42,632 | N / N÷8 |
| 双曲压缩 QKV | 3,134,332 | 33,928 | N / N÷8 |

## 验证

`models/test_change_context_qkv.py` 覆盖四尺度的 Q/KV 数量、时相交换对称性、相同输入零增强、零输入与大幅值输入的有限梯度、非零任务梯度、双曲距离和 Möbius 位移、测地中点以及几何有限差分梯度。

`models/test_CASA.py` 对三个变体检查完整四尺度输出、公共路径初始化、优化器参数覆盖、三步任务优化、残差激活和部署折叠。CPU 实测两个新版模块的四尺度最大折叠误差均为 `5.96e-8`，二值预测无差异。续训状态及后端设置恢复测试通过。

## 已知限制

当前验证覆盖接口、几何运算和梯度，尚无新版数据集训练结果。双原型相对单原型增加 KV 计算量，实际吞吐与显存需在下一轮测量。双曲几何不自动保证类别解耦，也不能由有限梯度检查推断长期不塌陷；后续仍需检查半径分布、任务梯度和同期 CTRL 结果。
