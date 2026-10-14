"""Run8：HGSL — Hyperbolic Geometric Supervision Loss（本轮**唯一**新增损失，需用户审核）。

## 为什么需要它，以及它与 Run7 HFAE 的**机制差异**

Run7 的第一条可迁移经验是：**"把梯度压回表征的监督"有效，"把证据注入主特征流"无效**
（E9 `aux_only` 优于 E10 `aux_inject`）。而 Diag1 又证明：CAACP 的 β 修正通道只有
`||βΔC||/||c_avg|| ≈ 0.19%`，**任何只在 CAACP 内部改打分的方案，端到端增益上界≈0**。

⇒ 若 Run8 只做"更好的打分 + 注入"，就是第三次重做 CASAA。因此本轮引入**一条**新损失，
把双曲几何证据本身变成**训练目标**：

| | Run7 HFAE | Run8 HGSL |
|---|---|---|
| 被监督的量 | `HFEdgeHead` 的边缘 logits（**新增一个有参头**） | **模块内部的 Lorentz 变化分数 `s`（参数自由的量）** |
| 监督信号 | `E = dilate₄(gt) \\ erode₄(gt)`（GT **轮廓带**） | `gt` 的**变化掩码**在该 stage 分辨率下的占用率 |
| 作用的表征 | stage0/1 的 `local_conv` 高频分支 | **4 个 stage 的双时相配对特征**（经 Lorentz 距离的梯度） |
| deploy 代价 | 0（整头删除） | **0**（无新增参数；损失只改梯度方向） |

**这不是"换一个 loss 调参"**：HFAE 监督的是"边缘在哪里"，HGSL 监督的是
"流形几何距离是否把真实变化排得更前"——后者是 Run2–Run7 从未被优化过的量
（CAACP 的 score 一直走 `no_grad`，**从未接收过任何梯度**）。

## 数学定义

对 stage `i`（分辨率 `H_i×W_i`）、时相 A/B 特征 `x_A, x_B`：

1. 双曲变化分数（`layers/hyperbolic.py`，`no_grad` 与否由 `detach_score` 控制）：
   `s_i = 1 − exp(−d_Lorentz)`，`d` 用**非恒定半径**（局部离散度 / 全局偏离）
2. GT 在该分辨率下的变化占用率（`adaptive_max_pool2d` 保底，避免小目标被降采样吃掉）：
   `y_i = MaxPool_{H_i}(gt)`
3. `L_HGSL = (1/|S|) Σ_{i∈S} [ BCE(s_i, y_i) + (1 − Dice(s_i, y_i)) ]`
4. 总损失：`L = L_BCE+Dice + λ·L_HGSL`，`λ = 0.2`（与 Run7 同值，**预注册冻结、禁止搜索**）

## 三条必须披露的实现纪律

1. `detach_score`：**默认 True**。此时梯度不进入 scorer，损失只作用于"下游如何消费分数"；
   `detach_score=False` 才让几何量本身被优化。二者是**不同的实验臂**，不能混为一句
   "加了新损失"（否则无法归因，正是前 7 轮的教训）。
2. `y_i` 使用 `adaptive_max_pool2d`（保底）而非 `avg_pool2d`：小目标（1–15px）在 1/32 上
   平均池化后会**恒为 0**，损失将退化成"预测全 0"。
3. 该损失**没有新增任何参数**，deploy 图完全不变（`switch_to_deploy()` 不需要处理它）。
"""
import torch
import torch.nn.functional as F

from model.utils import BCEDiceLoss

HGSL_LAMBDA_DEFAULT = 0.2        # 预注册固定值；与 Run7 HFAE 同值，禁止搜索
HGSL_STAGES_DEFAULT = (0, 1, 2, 3)


def downsample_gt_occupancy(label, size, mode="max"):
    """GT → 该 stage 分辨率的占用率图 `(B,1,size,size)`。

    `mode="max"`（默认，保底）：任何被 1px 小目标覆盖的 cell 都记为 1；
    `mode="avg"`：占用**比例**，更"软"，但会把 <1 cell 的小目标抹掉（故不作默认）。
    """
    y = label.float()
    if y.dim() == 3:
        y = y.unsqueeze(1)
    assert y.dim() == 4 and y.shape[1] == 1, f"unexpected label shape {tuple(y.shape)}"
    if mode == "max":
        return F.adaptive_max_pool2d(y, size)
    return F.adaptive_avg_pool2d(y, size)


def hgsl_loss(score_maps, label, weight=HGSL_LAMBDA_DEFAULT,
              stages=HGSL_STAGES_DEFAULT, mode="max", return_parts=False):
    """Hyperbolic Geometric Supervision Loss。

    `score_maps`: dict {stage_idx: (B,1,H_i,W_i)} —— **批量 B**（每对样品一张变化图）。
    `label`: (B,1,256,256) 或 (B,256,256)，`gray>=128` 已二值化。
    """
    if not score_maps:
        return None
    total = None
    parts = {}
    for i in sorted(score_maps.keys()):
        if i not in stages:
            continue
        s = score_maps[i]
        if s is None:
            continue
        y = downsample_gt_occupancy(label, s.shape[-1], mode=mode)
        # `BCEDiceLoss` 期望 (B,1,H,W) 的概率图与同形状目标
        term = BCEDiceLoss(s, y)
        parts[f"stage{i}"] = float(term.detach())
        total = term if total is None else total + term
    if total is None:
        return None
    n = len(parts)
    total = total / max(n, 1)
    out = float(weight) * total
    if return_parts:
        return out, parts
    return out
