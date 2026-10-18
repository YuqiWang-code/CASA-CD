# CASA-CD Exp1-B（STG_SUP）：特征域 SPD(2)/H² 测地变化分数的**监督形态**

> 2026-10-18。用户批准（"我选 B"）。在 v1/v2 的注入形态只能"持平"之后，
> 按本项目唯一被验证有效的形态（Run7 E9 `aux_only`）重做同一条几何。

## 1. 一个关键的设计判断（必须写清，否则 B 会退化成无效对照）

v1/v2 的几何都从**原始图像**算。若在这种设计上直接加监督，梯度只能流进
「几何→7k 参数的小投影头」这条通路，**永远到不了主干** ⇒ 监督退化为
"训一个小辅助头、然后按 deploy 纪律删掉"，对模型**零影响**。

⇒ B 必须把同一套几何**搬到网络特征上**：

```
F_t = proj(stage0 特征 96→16)          # A/B 共享 proj（严格交换对称）
J_t = Σ_c [G*(∂x F)², G*(∂x F ∂y F); ·, G*(∂y F)²]      # 特征域结构张量
z_t = ((Jxx−Jyy) + 2i·Jxy)/(Jxx+Jyy)                     # 双角度坐标（H² 盘坐标）
s   = √2·arccosh(1 + 2W)（与 st_geodesic.py 完全同一闭式，参数自由）
logits = Conv1x1(s) → 上采样 256² ；L_aux = λ·(BCE+Dice)(σ(logits), GT)
```
⇒ 梯度经 s → z → J → ∂F → **proj** → **stage0 特征** ⇒ 用几何监督**重塑表征**。

## 2. 与 Run7 E9 / Run8 HGSL 的关系（novelty 边界）

| | Run7 HFAE (E9) | Run8 HGSL | **Exp1-B（本臂）** |
|---|---|---|---|
| 被监督的量 | 新增**有参**边缘头的 logits | 模块自身 Lorentz 变化分数 | **特征域结构张量的 H² 测地距离**（参数自由、尺度不变） |
| 监督信号 | GT 轮廓带 | GT 变化掩码（max 池化保底） | GT 变化掩码（max 池化保底） |
| 流形上的点 | —（欧氏） | 逐像素特征（一阶） | **局部二阶统计 J ∈ SPD(2)** |
| 需要人造量 | — | 非恒定半径 r=r0+a·q（防退化） | **无**（无半径参数） |
| deploy 代价 | 0（整头删除） | 0（只改梯度） | **0（整支删除，deploy 与 CTRL 逐位同参数）** |

## 3. 预注册（**见结果之前冻结**）

* λ = **0.2 固定，禁止搜索**（与 Run7/Run8 同值；用户已批准这 1 条新增损失）。
* 形态 = `aux_only`：训练期存在，`switch_to_deploy()` **整支置 None**。
* 协议 `exp1_stgeo_exact80k_v1`；80,000 step；batch 32；256²；seed 16；四库中**只跑 LEVIR + SYSU**。
* 对照 = `CTRL_R8`（**前向逐位相同**，已实测 `torch.equal=True, max|d|=0`）。

**机制门（必须全过）**
1. `[PRETRAIN-LOAD] worst_diff == 0`；
2. epoch-0 前向与 CTRL 逐位（已实测）；
3. deploy 参数 = **31,025,665**（= CTRL_R8 的 `[DEPLOY-PARAMS] total=31.026 M`，已实测）；
4. `[STGEO-GRAD]` 的 `proj_grad_norm > 0` **且 `encoder_stage0_grad_norm > 0`**
   （**aux-only 回传**实测 2.39e-4 / **3.21e-4** ⇒ 监督确实打到主干，这是本臂的承重条件）；
5. `[STGEO-LOSS]` 逐 epoch 下降；`[ACTUAL-OPT-STEPS] 80000` 在最后一个完整 TEST 区块内。

**接受标准（沿用冻结口径）**：LEVIR 与 SYSU 的 F1 **均 ≥ CTRL_R8**（0.9121 / 0.8188）且
**SYSU 严格提升**；守门 F1 降幅 ≤0.002、中大对象 pooled Recall 降幅 ≤0.02。

**可证伪预测（写在结果之前）**
* **B-P1**：两库像素 F1 均 ≥ CTRL（这是 B 相对 v1/v2 的核心诉求——B 不改推理图，代价应更低）；
* **B-P2**：SYSU 的 Recall 与 Precision **同时**不低于 CTRL（不再出现 v1 的 P/R 交换）；
* **B-P3**：小目标不退化——LEVIR/SYSU 的 small pooled R 均 ≥ CTRL（v2 已达 +1.8/+2.8pp）；
* **B-P4**：B 的增益若成立，应主要来自**表征改善**，因此**不依赖**任何注入支路（deploy 已无支路）。

**失败判据（预先写死）**
若 B 仍持平/微降 ⇒ 「在该主干 + 80K 下，**结构几何证据无法转化为净像素 F1**」这一结论成立，
   本项目应停止在该轴上继续加模块，转为**分析型收尾 + 负结果叙事**，或改走
   **C（Beltrami/各向异性扩散）** 作为最后一个算子级尝试。

## 4. 服务器实测（预检全绿，2026-10-18 12:26）

```
[PASS] 23/23 单测（v1/v2 全部回归）
[S1] deploy total = 31,025,665  ==  CTRL_R8 的 31.026 M ；sup 支路已删除
[S2] epoch-0 forward: torch.equal(CTRL, sup) = True, max|d| = 0
[S3] aux_only backward: proj_grad 2.39e-04, encoder_stage0_grad 3.21e-04
[S4] 3 步联合训练 loss 有限；score_mean 0.2196→0.2233
[PASS] 3 步真实 dry-run rc=0；[STGEO-GRAD] proj 6.46e-04 / encoder 6.43e+00
```

参数：aux 支路 **1,538**（proj 96×16 + head 2），训练图非主干 2,040,406；deploy 完全删除。

## 5. 执行

```bash
bash train_scripts/Exploration/Exp1/_exp1_sup_preflight.sh 1     # 预检
bash train_scripts/Exploration/Exp1/_exp1_launch_arm.sh STG_SUP 1 LEVIR-CD-256 SYSU-CD-256
```

* 启动：2026-10-18 12:27:31（pid 472910，GPU1，27.6 GB / 86%）。
* ETA：LEVIR ≈ 20:00（同日），SYSU ≈ 04:00（次日）。
* 日志 `…/Exploration/Exp1/STG_SUP/<ds>/train_log.txt`；权重 `…/checkpoints/…/STG_SUP/<ds>/`。
