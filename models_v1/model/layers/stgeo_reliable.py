"""STGEO-REL（方向 A）：**可靠性感知 + 尺度选择性的几何修正**。

## 动机（来自 Exp1 三臂的实测诊断，不是猜测）

| 观测 | 含义 |
|---|---|
| v1（直接注入）SYSU Precision **−2.04pp**、未匹配 <256px **+49%** | 几何修正"无差别地"作用于小目标/背景/大目标 |
| v2（鲁棒池化 + 跨尺度门）碎片 −28%、Precision 恢复，但 large pooled R **−0.34pp** | 用"喂通道"的方式让网络自己学选择性，仍不够精细 |
| B（监督）SYSU Recall **+1.62pp** 但 Precision **−1.48pp** | 同上：定向能力有，区域选择性不足 |

⇒ 问题不是"几何证据无效"，而是**缺少"这条结构变化值不值得改表征"的判据**，
以及**修正会同时动到大目标的低频响应**。本模块就补这两件事。

## 三个组件（全部参数自由，除 1 个零初始化 γ）

**(1) 可靠性 r（参数自由，无阈值）**
```
r_support = box_mean_{5×5}(s) / (s + box_mean_{5×5}(s))      ∈ [0,1)   # 中心-邻域一致性
r_xscale  = min_σ s_σ / (max_σ s_σ + ε)                      ∈ [0,1]    # 跨尺度一致性
r = r_support · r_xscale
```
两者都是**非负量的比值**，因而无阈值、无超参。语义：真实小目标的结构变化在邻域内
有支撑、在多个尺度上同现；孤立纹理响应两点都不满足。
**注意（对先前分析的修正）**：这里**不**要求"双时相都有相同边缘"——新增目标本就
只在一侧有结构；r 只度量"该结构变化是否在空间与尺度上被支持"。

**(2) 局部零均值（detail-only）修正**
```
Δ = γ · ( r ⊙ z − AvgPool_{9×9}( r ⊙ z ) )
```
减去局部均值 ⇒ 修正的**低频（DC）分量为 0**，因此它只能锐化局部结构，
**不能抬高/压低大目标所依赖的整体激活水平**。这是对 v2「大目标召回受损」的直接对策。

**(3) 同式欧氏对照（`geom="euc"`，用于回答"负曲率是否必要"）**
在同一 (κ, θ) 特征上把双曲距离换成**切空间欧氏距离**：
```
w = artanh(κ)·e^{iφ}  （过原点的切空间坐标），d_euc = √2·|w_A − w_B|
```
√2 使两者在恒等附近同尺度 ⇒ 这是**严格的度量-only 对照**（同一输入、同一池化、同一门）。
若 `hyp ≈ euc`，则只能声称"局部结构统计有用"，**不得**声称负曲率必要。

## 部署

`inject` 形态的变体：推理期真实算子（保留在部署图里），**不得**声称零推理代价；
新增 FLOPs 只有几个 5×5/9×9 平均池化（在 64² 上），量级可忽略但非零。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.st_geodesic import (
    TR_FLOOR,
    _pool_evidence,
    poincare_shape_distance,
    shape_disk_coords,
    spd_geodesic_evidence,
    structure_tensor,
)

STGEO_REL_FORM = "stgeo_rel_v1"
STGEO_REL_SEED = 31
GEOMS = ("hyperbolic", "euclidean")


# ---------------------------------------------------------------------------
# 1. 同式欧氏对照
# ---------------------------------------------------------------------------
def tangent_coords(zx, zy, kappa, eps=1e-8):
    """盘坐标 → 过原点的切空间坐标 w = artanh(κ)·e^{iφ}。"""
    k = kappa.clamp(min=eps, max=1.0 - 1e-6)
    r = torch.atanh(k)                                  # 测地半径
    inv = 1.0 / k
    return zx * inv * r, zy * inv * r


def tangent_euclidean_distance(zx1, zy1, k1, zx2, zy2, k2):
    """√2·|w_A − w_B|：与双曲距离在恒等附近同尺度，只换度量。"""
    ax, ay = tangent_coords(zx1, zy1, k1)
    bx, by = tangent_coords(zx2, zy2, k2)
    d2 = ((ax - bx) ** 2 + (ay - by) ** 2).clamp_min(0.0)
    return math.sqrt(2.0) * torch.sqrt(d2)


def _distance(zx1, zy1, k1, zx2, zy2, k2, geom="hyperbolic"):
    if geom == "hyperbolic":
        return poincare_shape_distance(zx1, zy1, k1, zx2, zy2, k2)
    return tangent_euclidean_distance(zx1, zy1, k1, zx2, zy2, k2)


# ---------------------------------------------------------------------------
# 2. 可靠性（参数自由）
# ---------------------------------------------------------------------------
def reliability_maps(shape_scores, eps=1e-8):
    """shape_scores: list（每尺度一张 (B,1,H,W) 全分辨率 d_shape）。

    返回 (r, r_support, r_xscale)，均 (B,1,H,W) ∈ [0,1]。
    """
    shape_scores = [x.clamp_min(0.0) for x in shape_scores]
    s0 = shape_scores[0]
    m = F.avg_pool2d(s0, 5, 1, 2, count_include_pad=False)   # 中心-邻域均值
    r_support = m / (s0 + m + eps)                       # ∈ [0,1)
    st = torch.stack(shape_scores, dim=0)
    r_xscale = st.amin(dim=0) / (st.amax(dim=0) + eps)   # ∈ [0,1]
    r = (r_support * r_xscale).clamp(0.0, 1.0)
    return r, r_support.clamp(0.0, 1.0), r_xscale


def local_zero_mean(x, k=9):
    """减去 k×k 局部均值 ⇒ 修正的 DC 分量为 0（保护大目标的低频响应）。

    `count_include_pad=False`：边界窗口只用有效像素求均值，避免零填充带来的边界偏置。
    """
    return x - F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)


# ---------------------------------------------------------------------------
# 3. 模块
# ---------------------------------------------------------------------------
class STGeoReliable(nn.Module):
    """可靠性感知 + 尺度选择性的几何修正（推理期真实算子）。"""

    def __init__(self, in_ch=3, sigmas=(1.0, 2.0, 4.0), mid=32, out_dim=192,
                 geom="hyperbolic", detail_only=True, local_k=9,
                 tr_floor=TR_FLOOR, pool_mode="topk_mean", init_seed=STGEO_REL_SEED):
        super().__init__()
        assert geom in GEOMS, geom
        self.sigmas = tuple(float(s) for s in sigmas)
        self.geom = geom
        self.detail_only = bool(detail_only)
        self.local_k = int(local_k)
        self.tr_floor = float(tr_floor)
        self.pool_mode = pool_mode
        self.k_per_scale = 5                              # d_shape,d_str,κa,κb,ori
        self.in_ch_ev = self.k_per_scale * len(self.sigmas) + 1   # + 可靠性通道
        self.out_dim = int(out_dim)
        self.mid = int(mid)
        self.form_tag = STGEO_REL_FORM

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            self.proj = nn.Conv2d(self.in_ch_ev, self.mid, 1, bias=False)
            self.dw = nn.Conv2d(self.mid, self.mid, 3, 1, 1, groups=self.mid, bias=False)
            self.bn = nn.BatchNorm2d(self.mid)
            self.act = nn.SiLU()
            self.out = nn.Conv2d(self.mid, self.out_dim, 1, bias=False)
            self.apply(self._init_weights)
        self.gamma = nn.Parameter(torch.zeros(1))          # 零初始化门
        self._diag = {}

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def forward(self, pre, post, ref_hw):
        evs, shapes, diags = [], [], {}
        for s in self.sigmas:
            ev, parts = spd_geodesic_evidence(pre, post, sigma_d=s,
                                              tr_floor=self.tr_floor)
            # 用 geom 开关覆盖 d_shape 通道（同式欧氏对照）
            if self.geom != "hyperbolic":
                jj = structure_tensor(pre, sigma_d=s)
                jk = structure_tensor(post, sigma_d=s)
                za, zb = shape_disk_coords(*jj, tr_floor=self.tr_floor), \
                    shape_disk_coords(*jk, tr_floor=self.tr_floor)
                d_alt = _distance(za[0], za[1], za[2], zb[0], zb[1], zb[2],
                                  geom=self.geom) * (za[3] * zb[3])
                ev = torch.cat([d_alt, ev[:, 1:]], dim=1)
                parts = dict(parts, d_shape=d_alt)
            evs.append(_pool_evidence(ev, ref_hw[0], mode=self.pool_mode))
            shapes.append(parts["d_shape"])
            with torch.no_grad():
                diags[f"sigma{s:g}"] = {
                    "d_shape_mean": float(parts["d_shape"].mean()),
                    "d_shape_max": float(parts["d_shape"].amax()),
                    "kappa_mean": float(parts["kappa_a"].mean()),
                    "valid_frac": float(parts["valid"].mean()),
                }
        r, r_sup, r_xs = reliability_maps(shapes)
        r_pool = F.adaptive_avg_pool2d(r, ref_hw[0])
        h = torch.cat(evs + [r_pool], dim=1)
        z = self.act(self.bn(self.dw(self.proj(h))))
        z = self.out(z)
        w = r_pool * z                                     # 软选择（不是硬性排除）
        if self.detail_only:
            w = local_zero_mean(w, k=self.local_k)         # 局部零均值 ⇒ 不动的 DC
        out = self.gamma * w
        with torch.no_grad():
            diags["reliability"] = {
                "r_mean": float(r.mean()), "r_p90": float(r.flatten().quantile(0.9)),
                "r_support_mean": float(r_sup.mean()),
                "r_xscale_mean": float(r_xs.mean()),
            }
            diags["correction"] = {
                "out_abs_mean": float(out.abs().mean()),
                "out_mean": float(out.mean()),             # ≈0（局部零均值）
                "w_abs_mean": float(w.abs().mean()),
            }
        self._diag = diags
        return out

    # ------------------------------------------------------------------
    def diag(self):
        return dict(self._diag)

    def param_report(self):
        n = sum(p.numel() for p in self.parameters())
        return {"form": self.form_tag, "sigmas": list(self.sigmas), "geom": self.geom,
                "detail_only": self.detail_only, "local_k": self.local_k,
                "pool_mode": self.pool_mode, "in_ch_ev": self.in_ch_ev,
                "mid": self.mid, "out_dim": self.out_dim, "params": int(n),
                "gamma_init_zero": bool(float(self.gamma.detach()) == 0.0),
                "gamma": float(self.gamma.detach())}
