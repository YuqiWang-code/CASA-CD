"""Run8：双曲流形空间的变化证据算子（Lorentz 模型，参数自由）。

## 为什么用 Lorentz 而不是 Poincaré 球

Lorentz（双叶双曲面）模型的距离只需要「半径 + 方向点积」，**不必显式构造 C+1 通道的
双曲张量**，对 C=96…768 的稠密特征图更省算力，数值也更稳（没有球边界 `atanh` 爆炸）。
两份调研（2026-10-13 §2.2 / 2026-10-14 §5.1）都给出了这两个模型并注明该优点。

## ⚠️ 必须避开的退化定理（调研 2026-10-13 §2.3 / 2026-10-14 §5.2）

若把单位化特征以**同一个固定半径** `r` 映射到双曲面，则

    cosh d_H = cosh²r − sinh²r·cosθ = 1 + sinh²r·(1 − cosθ)

`acosh` 严格递增 ⇒ `d_H` 与 `1−cosθ` **逐像素同序** ⇒ 走逐图 rank 之后权重逐位相同。
**「把 1−cos 换成双曲距离」因此不构成任何新增信息。**

**本模块的非退化机制（关键）**：半径**不是固定的**，而是由**同一时相的特征范数**决定，
并且两个分支用**两种不同的范数**：

| 分支 | 半径来源 | 语义 |
|---|---|---|
| `detail` | **局部离散度** `q = 1 − cos(x_i, mean_{3×3}(x_i))` → `r = r0 + a·q` | 局部/纹理/边缘结构强度 |
| `global` | **全局半径**：`x` 逐图归一化后相对全图均值的偏离 → `r = r0 + a·q_global` | 相对全图背景的整体语义角色 |

于是对**相同夹角 θ** 的两个位置，只要它们的范数不同，`cosh d_H` 就不同 ⇒ 排序改变。
这正是调研要求的「非退化性」：**在固定方向相似度下改变重要性排序**。

## 数学定义（实现逐项对应）

设时相 `t∈{A,B}`、位置 `i`，特征 `x_{t,i}∈R^C`：

1. `u_{t,i} = x_{t,i} / max(‖x_{t,i}‖, ε)`（单位方向）
2. `q^detail_{t,i} = clip( (1 − u_{t,i}·m_{t,i}) / 2, 0, 1 )`，
   其中 `m_{t,i} = Normalize( mean_{3×3}(u_{t,·}) )`（局部中心方向）
3. `q^global_{t,i} = clip( ‖x_{t,i} − mean_{HW}(x_t)‖ / (‖x_{t,i}‖ + ε), 0, 1 )`
   （对该时相自己的全局均值；**只用自己的时相，不含跨时相信息**）
4. `r^b_{t,i} = r0 + a · q^b_{t,i}`，`b ∈ {detail, global}`；`r ∈ [r0, r0+a]`
5. Lorentz 测地距离（`κ=1`，点 `h=(cosh r, sinh r·u)`）：
   `cosh d = cosh r_A cosh r_B − sinh r_A sinh r_B (u_A·u_B)`，`d = acosh(max(1, cosh d))`
6. 有界变化强度 `s = 1 − exp(−d) ∈ [0, 1)`

`r0`/`a` **在预注册里冻结，禁止在 test 上搜索**（调研 §6.2 的纪律）。

## 数值纪律

- 全部在 **FP32** 下算，且只读（调用方 `torch.no_grad()`）；
- `acosh` 自变量先 `clamp_min(1.0)`；`cosh/sinh` 的半径为有界小量（默认 ≤2.0），不会溢出；
- `ε=1e-8`；全零输入时 `q=0` ⇒ `r=r0` ⇒ 退化为固定半径（此时按上面的定理与余弦同序，
  这是**正确的退化行为**，不是 bug）。
"""
import torch
import torch.nn.functional as F

__all__ = [
    "R0_DEFAULT", "A_DEFAULT",
    "unit_direction", "local_center_direction", "detail_scatter",
    "global_radius_scatter", "branch_radii", "lorentz_change_score",
    "euclidean_change_score", "cosine_change_score", "fixed_radius_change_score",
    "rank_normalize_2d", "branch_rank_disagreement",
]

R0_DEFAULT = 0.5      # 半径下界（远离双曲数值边界）
A_DEFAULT = 0.5       # 半径跨度 ⇒ r ∈ [0.5, 1.0]


def unit_direction(x, eps=1e-8):
    """`u = x / max(‖x‖, ε)`；(B,C,H,W) → (B,C,H,W)，FP32。"""
    x = x.float()
    n = x.pow(2).sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    return x / n


def local_center_direction(u, ksize=3):
    """`m = Normalize(mean_{ksize×ksize}(u))`；(B,C,H,W) → (B,C,H,W)。"""
    pad = ksize // 2
    m = F.avg_pool2d(u, kernel_size=ksize, stride=1, padding=pad,
                     count_include_pad=False)
    n = m.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
    return m / n


def detail_scatter(u, ksize=3):
    """局部离散度 `q = clip((1 − u·m)/2, 0, 1)`；(B,C,H,W) → (B,1,H,W)。"""
    m = local_center_direction(u, ksize=ksize)
    cos = (u * m).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    return ((1.0 - cos) * 0.5).clamp(0.0, 1.0)


def global_radius_scatter(x, eps=1e-8):
    """全局语义半径代理 `q = clip(‖x − mean_HW(x)‖ / (‖x‖ + ε), 0, 1)`。

    只用**该时相自己**的全局均值 ⇒ 单时相统计量，A/B 严格对称，不含跨时相泄漏。
    """
    x = x.float()
    mu = x.mean(dim=(2, 3), keepdim=True)
    dev = (x - mu).pow(2).sum(dim=1, keepdim=True).clamp_min(0.0).sqrt()
    nrm = x.pow(2).sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    return (dev / (nrm + eps)).clamp(0.0, 1.0)


def branch_radii(xa, xb, branch="detail", r0=R0_DEFAULT, a=A_DEFAULT, ksize=3):
    """返回 `(r_A, r_B, u_A, u_B)`，全部 (B,C,H,W) / 半径 (B,1,H,W)。"""
    assert branch in ("detail", "global"), branch
    ua = unit_direction(xa)
    ub = unit_direction(xb)
    if branch == "detail":
        qa = detail_scatter(ua, ksize=ksize)
        qb = detail_scatter(ub, ksize=ksize)
    else:
        qa = global_radius_scatter(xa)
        qb = global_radius_scatter(xb)
    ra = (r0 + a * qa)
    rb = (r0 + a * qb)
    return ra, rb, ua, ub


def lorentz_change_score(xa, xb, branch="detail", r0=R0_DEFAULT, a=A_DEFAULT, ksize=3):
    """Lorentz 测地距离 → 有界变化强度 `s = 1 − exp(−d)`；(B,C,H,W) → (B,1,H,W) ∈ [0,1)。"""
    ra, rb, ua, ub = branch_radii(xa, xb, branch=branch, r0=r0, a=a, ksize=ksize)
    cos_ab = (ua * ub).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    z = (torch.cosh(ra) * torch.cosh(rb)
         - torch.sinh(ra) * torch.sinh(rb) * cos_ab)
    d = torch.acosh(z.clamp_min(1.0))
    return (1.0 - torch.exp(-d)).clamp(0.0, 1.0)


def euclidean_change_score(xa, xb, branch="detail", r0=R0_DEFAULT, a=A_DEFAULT, ksize=3):
    """**同式欧氏对照**：用完全相同的 `r_A, r_B, u_A, u_B`，只把测地距离换成切向欧氏距离。

    调研（2026-10-13 §6.3 / 2026-10-14 §11.2 G4）要求必须有这个对照，否则无法区分
    「局部径向信息有用」与「负曲率有用」。
    """
    ra, rb, ua, ub = branch_radii(xa, xb, branch=branch, r0=r0, a=a, ksize=ksize)
    cos_ab = (ua * ub).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    d2 = (ra * ra + rb * rb - 2.0 * ra * rb * cos_ab).clamp_min(0.0)
    d = d2.sqrt()
    return (1.0 - torch.exp(-d)).clamp(0.0, 1.0)


def cosine_change_score(xa, xb):
    """现状口径对照：`1 − cos(x_A, x_B)`。

    ⚠️ 与生产实现 `caacp_ss2d.change_score_cosine_2d` **逐位同式**：只 `clamp(min=0.0)`，
    **不做上界夹紧**。这一点是 §2.3 退化定理能成立的前提：

    - `1−cos` 的天然上界是 2（当 `cos` 可为 −1 时），若写成 `.clamp(0,1)`，任何
      `1−cos > 1`（即夹角大于 90°）的位置会被**压平到同一个值 1** ⇒ 产生大量并列
      ⇒ **rank 不再一一对应**，从而**伪造出"排序分歧"**。本文件首版正是踩了这个坑
      （C2 退化检验 FAIL，`frac_rank_diff=0.482`）；改成只夹下界后 rank 分歧 = 0。
    - 同理，双曲路径内部必须先 `clamp(-1,1)` 才能喂 `acosh`；`1−cos` 路径内部
      `cos` 天然在 [−1,1]（unit 向量点积的浮点越界 ≤1e-7，不影响 rank）。
    """
    ua = unit_direction(xa)
    ub = unit_direction(xb)
    cos_ab = (ua * ub).sum(dim=1, keepdim=True)
    return (1.0 - cos_ab).clamp(min=0.0)


def fixed_radius_change_score(xa, xb, r=R0_DEFAULT + 0.5):
    """**退化负对照**：固定半径的双曲距离。按 §2.3 定理，它应与 `cosine` 逐像素同序。

    这个函数存在的唯一目的就是**在真实数据上复现退化定理**（smoke T3）。
    """
    ua = unit_direction(xa)
    ub = unit_direction(xb)
    cos_ab = (ua * ub).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    z = (torch.cosh(torch.as_tensor(r, dtype=torch.float32, device=ua.device)) ** 2
         - torch.sinh(torch.as_tensor(r, dtype=torch.float32, device=ua.device)) ** 2 * cos_ab)
    d = torch.acosh(z.clamp_min(1.0))
    return (1.0 - torch.exp(-d)).clamp(0.0, 1.0)


def rank_normalize_2d(s):
    """逐图 rank 归一化到 [0,1]（与 `caacp_ss2d.rank_normalize_2d` **逐位同口径**）。

    注意：这是**退化的来源**——`rank(·)` 只保留序、丢弃一切幅度信息，因此任何保序变换
    在此之后都不可能改变权重。Run8 的 SAT-QKV 因此**不用 rank**，改为在 Q/K/V 里
    直接消费数值分数（见 `layers/caacp_hqkv.py`）。本函数只保留给对照/诊断。
    """
    B, C, H, W = s.shape
    flat = s.flatten(2)                                   # (B,1,N)
    ranks = flat.argsort(dim=2).argsort(dim=2).float()
    return (ranks / max(flat.shape[2] - 1, 1)).view(B, C, H, W)


@torch.no_grad()
def branch_rank_disagreement(s_a, s_b):
    """两个分数图的排序分歧（Kendall τ 的 1−τ 近似 + 逐位置 rank 不同比例）。

    返回 `dict(frac_rank_diff=..., kendall_tau=...)`。用于 smoke 的**非退化性**判据：
    `frac_rank_diff` 必须显著 > 0，否则新几何只是余弦换皮。
    """
    a = s_a.flatten(1)
    b = s_b.flatten(1)
    ra = a.argsort(dim=1).argsort(dim=1).float()
    rb = b.argsort(dim=1).argsort(dim=1).float()
    frac = (ra != rb).float().mean().item()
    # Kendall τ（用中心化后的 Pearson 相关在秩上计算，等价且向量化）
    ra = ra - ra.mean(dim=1, keepdim=True)
    rb = rb - rb.mean(dim=1, keepdim=True)
    num = (ra * rb).sum(dim=1)
    den = ra.pow(2).sum(dim=1).sqrt() * rb.pow(2).sum(dim=1).sqrt()
    tau = (num / den.clamp_min(1e-12)).mean().item()
    return {"frac_rank_diff": frac, "kendall_tau": tau}
