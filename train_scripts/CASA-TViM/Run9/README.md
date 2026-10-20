# Run9：HSAM（Poincaré 球双曲相似度注意力）

> 用户指令（2026-10-20）：按论文 Fig.2 的 HSAM，把编码器 **3、4 层**的输出（或 QKV 之后的输出）
> 映射到双曲空间算双曲距离、构造注意力矩阵，再经注意力聚合 + 残差增强双时相特征。
> **就加这个模块。Run9。**
>
> 论文：Yang et al., *Hyperboloid-Embedded Siamese Network for Change Detection in
> Remote Sensing Images*, IEEE JSTARS 2024（`docs/参考文献/流形空间spd黎曼几何/`）。
> 冻结方案：`SELECTION_LOCK.json`。

## 1. 一句话

在 `outnorm` **之前**取最深两层的双时相激活，用 Poincaré 球距离构造 `HW×HW` 注意力，
双向聚合后残差加回，**零可学习参数**。唯一变量 = HSAM 开/关。

## 2. 变体（8 个 run）

| 变体 | `--hsam` | 其余 | 说明 |
|---|---:|---|---|
| `S0_CTRL` | 0 | `hqkv=0, hgsl=0` | 同期同波对照 |
| `S1_HSAM` | 1 | `hqkv=0, hgsl=0` | 主实验 |

**波次按数据集交错**：CDD → LEVIR → SYSU → WHU，每个数据集先 CTRL 再 HSAM。
这样每个库都有紧邻的同期 CTRL，且跑完前两库（≈29 h）就能做预注册的早停判定。

## 3. 冻结值（**不搜索**）

`max_steps=80000`、`batch=32`、`seed=16`、`lr=2e-4 poly`、`backbone_lr_ratio=0.1`、
`str_dim=192`、`hsam_stages=2,3`（0-based = 1/16 与 1/32 = 论文 (3,4) H-Layer）、
`curvature c=1`、`logit_norm=std`、`radius_mode=feature_energy`、`feature_source=pre_norm`、
**HSAM 参数 = 0**。

## 4. 三处必须的偏离（都在**看到任何 F1 之前**定下，依据是机制读数）

### ① 取数位置：`outnorm` 之前

`forward_features` 每个 stage 输出都过 `outnorm{i}=LayerNorm2d(dim)`。LayerNorm 在通道维
把每个空间位置标准化 ⇒ **`‖v‖² = C` 精确成立**（实测 `‖v‖/√C = 1.0000 ± 5e-7`，
`1−‖p‖²` 对所有 token 恒为 `0.4200`）。于是所有点同半径，

```
‖p−q‖² = 2r₀²(1−cosθ)，(1−‖p‖²)(1−‖q‖²)=sech⁴(1)=const
⇒ HD = arcosh(1 + K(1−cosθ))          ← 余弦距离的严格单调函数
```

实测秩相关 **`rank_corr(HD, 1−cos) = 1.000001 / 1.000000`** ⇒ 双曲性等于零，
模块等价于余弦注意力。**修法**：取 `outnorm` 之前的 `o`（`encoder.forward_pre_norm()`），
实测范数 cv = `4.9e-2 / 1.1e-1`；HSAM 在 `outnorm` 之前混合、之后照常归一化，
**CTRL 路径一个字节都不变**。

### ② 全局 RMS 标量缩放

stage3 的残差范数 `‖v‖/√C = 2.71`，不处理时 `1−‖p‖²` 最小掉到 `8.5e-6`，HD 被分母主导、
秩相关掉到 `0.0047`（"半径压倒角度"的另一种退化）。**修法**：exp 映射前只除以整张量的
**一个标量** `rms(v)`：保序、保留 token 间相对幅度、零参数、跨 stage 稳定。

### ③ logit 标准化

论文原式 `tanh(HD/√C)` 在 LayerNorm 特征上注意力熵 = **1.0000**（`prob_max` 恰为 `1/N`）、
`agg/feat = 1.9e-4` ⇒ 等于没加。**修法**：对 `HD/√C` 逐样本标准化再 softmax
（尺度无关的等价温度，不是可搜索超参）。

## 5. 硬门（每个 run 都查，任一不过 ⇒ `SystemExit(4)`，不重试、终止整轮）

| 门 | 阈值 | 挡住的失败 |
|---|---|---|
| `finite` | 必须 True | exp 饱和/NaN |
| `agg_over_feat` | `> 1e-6` | 模块零贡献 |
| `radius_cv` | `> 1e-3` | 半径恒定 |
| `attn_entropy_norm` | `< 0.995` | 注意力均匀 = 均值池化 |
| **`rank_corr_hd_vs_cos`** | **`< 0.999`** | **退化成余弦注意力（本次的核心门）** |

## 6. 预注册早停（`_r9_delta.py`）

CDD 与 LEVIR 两臂都封口后：若两库上 `ΔF1 = HSAM − CTRL` **都 ≤ 0** ⇒ **STOP**，
不再跑 SYSU/WHU；结果**按实际已完成的报，不挑选**。否则继续。

## 7. 开跑前实测（服务器，CDD 真实特征）

| 项 | `hsam=0` | `hsam=1` |
|---|---:|---:|
| 预训练加载 | `212/222, worst_diff=0.0` | 同 |
| 总参数 | 31,977,044 | **31,977,044（HSAM 0）** |
| step/s | 3.691 | 3.581 ⇒ **−3.08%** |
| 峰值显存 | 26,798 MiB | 26,941 MiB |

HSAM 读数：stage2 `r_cv=5.7e-2`、`rank_corr=0.8286`、`ent=0.909`、`agg/feat=1.64`；
stage3 `r_cv=2.3e-1`、`rank_corr=0.2902`、`ent=0.864`、`agg/feat=1.21`。

CPU 回归门：`models/test_hsam_cpu.py` **32/32 PASS**（含"论文原式必然退化"与
"post_norm 负对照 rank_corr≈1"两条固定断言）。

## 8. 命令

```bash
# 开跑（单卡串行，带预注册早停）
GPU_ID=0 NEED_MB=30000 setsid nohup bash train_scripts/CASA-TViM/Run9/_r9_pipeline_single.sh \
    > /tmp/run9/pipe.stdout 2>&1 < /dev/null &
bash train_scripts/CASA-TViM/Run9/_r9_state.sh      # 进度 + HSAM 读数 + Δ 表
tail -20 /tmp/run9/pipeline.log                     # 督导决策
bash train_scripts/CASA-TViM/Run9/_r9_stop.sh --kill-run   # 停（顺序已修正）
# 预检
bash train_scripts/CASA-TViM/Run9/_r9_smoke.sh      # 真实特征 smoke
bash train_scripts/CASA-TViM/Run9/_r9_cert.sh       # 非退化证书
bash train_scripts/CASA-TViM/Run9/_r9_radius_probe.sh  # 半径诊断
```

## 9. 保留但本轮不跑

`--hsam_radius local_dispersion`：半径只由 3×3 **局部离散度**决定、与 token 能量解耦
（代码已就绪并通过 CPU 检验）。这是更彻底的设计，离论文更远，留作后续对照臂。
