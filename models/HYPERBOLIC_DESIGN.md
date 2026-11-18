# 双曲与差分检索：一阶内容压缩与跨尺度检索

本文件描述 `HyperbolicContext`、`MatchedEuclideanContext`、`DifferenceContext` 共用的数据流与公式。H/E 几何配对归档在 CASA-Pyramid / Run12–Run16；小目标主线改用差分 KV，见 CASA-MinorGoal / Run1–Run3 与根目录 [README.md](../README.md)。

## 数据流

~~~
A/B → 共享 MobileNetV2 → 四级 TAR T_s(64ch, 64²/32²/16²/8²)
                              │
                     X_s = LN(T_s)                    ← 查询侧，全位置保留
                     ├─ guide → p_s
                     ├─ 全位置 Q_s
                     └─ 区域 p/(1−p) 一阶池化 P_s
                              ├─ K_s = L(W_qk P_s/√d)
                              └─ 欧式 V_s = W_v P_s
（可选）TAR 差分支 D_s = SPP(|P−Q|) → LN_diff → 作为 P_s 的内容来源
四尺度 K/V 联合(224) → 双曲或匹配欧氏检索 → AV
→ AV − 空间均值 AV → 内容门控 → W_out → γ 残差
→ 加回 TAR（细层再加局部高通）→ DCR → 共享头
~~~

网格 8/4/4/4，每格变化/背景两个原型，KV 共 224；完整 Q 共 5,440；1 头 32 维（`--context-heads 1`）。类权重 `w_i⁺=p_i/Σp`、`w_i⁻=(1−p_i)/Σ(1−p)`，空类权重为 0；支持度 `Σw_i·m_i` 下限 1e−6。`P=Σw·src`；`L(z)=(√(1+||z||²), z)`；K 与 V 来自同一内容统计，V 留在欧氏空间。

三种 KV 内容来源：

| `--module` | 池化内容 `src` | 几何 | context 参数 |
|---|---|---|---|
| `hyperbolic` | `T_s` | 曲率 −1 Lorentz 测地平方距离 | 11,477 |
| `matched` | `T_s` | 同一空间坐标的欧氏平方距离 | 11,477 |
| `difference` | `LN_diff(D_s)`，`D_s` 来自 `--tar-variant spp\|spp1\|diff` | 同 hyperbolic | 11,605 |

`difference` 版本里 guide 与池化权重由 `LN_diff(D_s)` 计算，查询仍是 `T_s` 全位置——即"问我在哪、检索这里变了多少"。若 `D_s` 缺失（`--tar-variant reparam`）构造时直接 `ValueError`。

## 打分与读出

`logits = −d²/(2τ²) + log(support) + scale_bias − log(M_source)`，τ 可学习（初值 0.5，`--freeze-tau` 时冻结在 0.4696）。半径 `asinh(||z||)` 保留变化幅度；`--kv-hierarchy nested` 时网格换 8/4/2/1 并做父子 2×2 聚合（KV 170）。

读出 `R=softmax(logits)V`，`C=R−mean_hw(R)`，`U=W_out[gate·C]`；gate 输入本地 V、C、原 guide 与检索到的支持度。`T'_s=T_s+γ_s·U_s+local_s`，`local` 只在最细两层。`γ` 零初始化且 attention 近均匀时 `C≈0`，因此**推理态模块与 CTRL 等价**，训练中自己打开；这条性质由 `test_centered_readout.py` 断言（常量场 → 零更新、V 梯度精确为 0）。

## 证据边界

- H 与 E 只差一个距离函数（`state_dict` 同 seed 逐位相同）。Run14 的 12 对配对差均值 −0.167 pp（sd 0.64），Run12 WHU +0.118/+0.330 pp 均在 0.177 pp 复跑噪声内，Run10 同构臂差 +0.334/−0.009 pp：**几何本身没有稳定精度收益**。
- 换成差分 KV 后 Run1 的 WHU +0.723 pp、SYSU +0.934 pp，Run2 三 seed 配对差 +0.679/+0.750 pp（sd 0.13/0.16，t-LCB +0.460/+0.474）——收益来自"检索内容含变化证据"，不是双曲几何。
- 多尺度窗口本身不承载收益：`spp1`（只留 1×1）与 `diff`（无 SPP）在 Run2 都在噪声内，SYSU 上 `DIFF_NOSPP` 反而 +0.78 pp。Run3 排的四条臂（`ARM_C`、`SUPP_PIX`、`DETAIL`）正是把"模块存在"与"内容"分开归因，判据在开跑前已写进 `train_scripts/CASA-MinorGoal/Run3/README.md`。

## 实现检查

`test_hyperbolic_context.py`、`test_context_geometry.py`、`test_direct_geometry.py`、`test_centered_readout.py`、`test_attribution.py`、`test_minor_goal.py` 覆盖：提升后半径与梯度、距离参考值与小距离级数、质心恒等、类池化与线性等价、常量/零场、跨尺度梯度、H/E 配对初始化、差分流缺失报错、SPP 折叠与 nested 聚合。`check_run3_switches.py` 另做 `support`/`kv-hierarchy`/`freeze-tau` 的在线检查。
