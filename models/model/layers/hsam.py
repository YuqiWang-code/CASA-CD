"""Run9 核心：HSAM —— Hyperbolic Similarity Attention Module（**零参数**）。

来源：Yang et al., *Hyperboloid-Embedded Siamese Network for Change Detection in
Remote Sensing Images*, IEEE JSTARS 2024（`docs/参考文献/流形空间spd黎曼几何/`）。
按论文 Eq.(1)–(8) 实现，作用在编码器**最深两层**（论文实验的 (3,4) H-Layer 最优）。

## 论文要做什么（一句话）

把双时相特征各自 exp 映射到 **Poincaré 球**上，用**双曲距离**当相似度，
构造 `HW×HW` 注意力矩阵，做双向聚合后**残差加回**原特征。全程**无可学习参数**。

    F1, F2  (C,H,W)
      → exp_0 映射到 Poincaré 球（c=1）            Eq.(3)(4)
      → HD 两两距离 → M_HS (HW×HW)，除以 √C_h       Eq.(5)(6)
      → M_Ha = logmap(Softmax(expmap(M_HS)))        Eq.(7)
      → F^H_j = reshape(F_j) × M_Ha ⊕ F_j           Eq.(8)

## 三处**必须**的偏离（都有实测依据，不是口味问题）

### ① 取数位置：用 `outnorm` **之前**的激活
`VMambaBackbone.forward_features` 的每个 stage 输出都过 `outnorm{i}=LayerNorm2d(dim)`。
LayerNorm 在通道维把每个空间位置标准化成零均值单位方差 ⇒ **`‖v‖² = C` 精确成立**
（实测 `‖v‖/√C = 1.0000 ± 5e-7`，`1-‖p‖²` 对所有 token 恒为 `0.4200`）。
于是 `‖p‖ = tanh(1) = 0.761592` **所有点同半径**，代入 Eq.(5) 得

    ‖p−q‖² = 2r₀²(1−cosθ),  (1−‖p‖²)(1−‖q‖²) = sech⁴(1) = 常数
    ⇒ HD = arcosh(1 + K·(1−cosθ))        ← **余弦距离的严格单调函数**

实测秩相关 `rank_corr(HD, 1−cos) = 1.000001 / 1.000000`（stage2 / stage3）⇒
**"双曲"在这组特征上不携带任何额外信息**，模块等价于余弦注意力。
修正：改取 `outnorm` 之前的激活 `o`（`encoder.forward_pre_norm()`），其范数随内容变化
（实测 cv：stage2 4.9e-2、stage3 1.1e-1），恢复论文"未归一化 CNN 激活"的前提。
网络主干仍用归一化后的特征；HSAM 在 `outnorm` 之前混合，之后照常归一化 ⇒
**CTRL 路径一个字节都不变**。

### ② 全局 RMS 标量缩放
残差流范数随深度增长（stage3 实测 `‖v‖/√C = 2.71`）。不处理时 `1−‖p‖²` 最小掉到 `8.5e-6`，
HD 被这个分母主导，秩相关掉到 `0.0047` —— 换成了"半径完全主导"的另一种退化。
修正：exp 映射前**只除以整张量的一个标量** `g = rms(v)`：保序、保留 token 间相对幅度、
零参数、跨 stage 稳定。实测秩相关变为 `0.8922 / 0.5021`（角度与半径各占一部分）。

### ③ Eq.(6)(7) 的 logit 尺度（`logit_norm="std"`）
论文原式 `tanh(HD/√C)` 在 LayerNorm 特征上注意力熵 = **1.0000**（`prob_max` 恰为 `1/N`），
`agg/feat = 1.9e-4` ⇒ 等于没加。修正：对 `HD/√C` 做**逐样本标准化**再 softmax。
这是尺度无关的等价温度选择（对 HD 的任何常数缩放都不变），不是可搜索超参。
实测熵 0.89–0.92、`prob_max` 0.34–0.53。`--hsam_logit_norm paper` 可复现原式。

## 硬门：非退化证书

`rank_corr(HD, 1−cos)`（秩相关）逐 epoch 记录，**> 0.999 ⇒ 判退化、中止该 run**。
秩相关是判"是否只是单调变换"的正确统计量（Pearson 会被单调非线性本身拉低）。
两个方向的失败都要挡：
  * `rank_corr ≈ 1`  ⇒ 半径没变化（退化回余弦注意力）—— ① 没生效
  * 半径压倒角度  ⇒ `1−‖p‖²` 触底 —— ② 没生效
  后者的表征是 `min(1−‖p‖²)` 贴近 clamp 下界，一并记录。

## 半径可切换（预注册为**未启用**的备选臂）

`radius_mode`：
  * `"feature_energy"`（默认）：`ρ = ‖v/g‖/√C`，即论文的"激活强度当半径"，加 ② 的缩放；
  * `"local_dispersion"`：`ρ = σ_local/σ̄`，半径只由**局部离散度**决定（3×3 窗内方差均值），
    与 token 能量**解耦**。更彻底、离论文更远，本轮**不跑**，留作后续对照。

## 复杂度

只用 `(B,N,·)` 规模的中间量 + 3 次 matmul，**不构造 (B,N,N,C)**
（`‖v−u‖² = ‖v‖²+‖u‖²−2vᵀu`）⇒ 内存 O(N²)、算力 O(N²C)。
诊断只在 `collect_diag=True` 时算（默认关，避免每步 device sync）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HSAM_FORM = "hsam_poincare_v2"
CURVATURE_C = 1.0              # 论文：c = 1（固定，不搜索）
EPS_R = 1e-8
EPS_BALL = 1e-6
EPS_DELTA = 1e-6
EPS_ATANH = 1e-6
R_MAX = 30.0
ATTN_STAGES_DEFAULT = (2, 3)   # 0-based：stage2=1/16(384)、stage3=1/32(768) = 论文 (3,4) H-Layer
LOGIT_NORM_DEFAULT = "std"
RADIUS_MODE_DEFAULT = "feature_energy"
RADIUS_MODES = ("feature_energy", "local_dispersion")
ENT_MAX_OK = 0.995             # 归一化熵超过它 ⇒ 均匀注意力 ⇒ fail-fast
RANK_CORR_MAX_OK = 0.999       # 秩相关超过它 ⇒ 退化回余弦注意力 ⇒ fail-fast
DISP_WINDOW = 3                # local_dispersion 的局部窗

__all__ = [
    "HSAM", "build_hsam", "expmap0", "logmap0", "poincare_pairwise_distance",
    "hsam_logits", "local_sigma", "rank_corr",
    "HSAM_FORM", "ATTN_STAGES_DEFAULT", "CURVATURE_C", "RADIUS_MODES",
    "RANK_CORR_MAX_OK", "ENT_MAX_OK",
]


# ---------------------------------------------------------------------------
def expmap0(v, dim_scale=1.0, eps=EPS_R, r_max=R_MAX):
    """原点处指数映射：Eq.(3) `exp_0(v)=tanh(‖v‖)·v/(‖v‖+eps)`。**保持逐 token 半径可变**。"""
    v = v / float(dim_scale)
    r = v.pow(2).sum(dim=-1, keepdim=True).clamp_min(0.0).sqrt()
    return torch.tanh(r.clamp(max=float(r_max))) * v / (r + eps)


def expmap_radius(v, rho, eps=EPS_R):
    """给定**逐 token 半径参数** `rho`（(B,N)），把单位方向映射到球内：

        p_i = tanh(rho_i) · v_i / ‖v_i‖

    这样半径由 `rho` 决定（`feature_energy` 时是特征能量，`local_dispersion` 时是局部离散度），
    方向仍来自 `v`。两个来源互不混淆。
    """
    n = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.tanh(rho.clamp(max=R_MAX)).unsqueeze(-1) * v / n


def logmap0(x, eps=EPS_ATANH):
    """原点处对数映射（c=1）：`atanh(x)`，带 clamp。"""
    return torch.atanh(x.clamp(-1.0 + eps, 1.0 - eps))


def poincare_pairwise_distance(p, q, eps_ball=EPS_BALL, eps_delta=EPS_DELTA):
    """Poincaré 球两两双曲距离 Eq.(5)：`arcosh(1 + 2‖v−u‖² / ((1−‖v‖²)(1−‖u‖²)))`。

    `p`:(B,N,C)，`q`:(B,M,C) → (B,N,M)。用 `‖v−u‖²=‖v‖²+‖u‖²−2vᵀu` 避免 (B,N,M,C)。
    """
    p2 = p.pow(2).sum(dim=-1).clamp_(0.0, 1.0 - eps_ball)
    q2 = q.pow(2).sum(dim=-1).clamp_(0.0, 1.0 - eps_ball)
    g = p @ q.transpose(1, 2)
    d2 = (p2[:, :, None] + q2[:, None, :] - 2.0 * g).clamp_min(0.0)
    denom = (1.0 - p2)[:, :, None].clamp_min(eps_ball) * \
        (1.0 - q2)[:, None, :].clamp_min(eps_ball)
    # Shift the acosh argument away from 1 in float32, then subtract the same
    # constant so identical points still have zero distance and finite gradients.
    delta = 2.0 * d2 / denom
    base = torch.acosh(delta.new_tensor(1.0 + eps_delta))
    return torch.acosh(1.0 + delta + eps_delta) - base * torch.exp(-delta / 1e-4)


def hsam_logits(hd_raw, dim, logit_norm=LOGIT_NORM_DEFAULT):
    """双曲距离矩阵 → softmax 前 logits（见模块头 ③）。"""
    hd_n = hd_raw / math.sqrt(dim)
    if logit_norm == "paper":
        return torch.tanh(hd_n)
    if logit_norm == "raw":
        return hd_raw
    if logit_norm == "std":
        mu = hd_n.mean(dim=(-1, -2), keepdim=True)
        sd = hd_n.std(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        return (hd_n - mu) / sd
    raise ValueError(f"[HSAM] 未知 logit_norm={logit_norm!r}")


def local_sigma(x, k=DISP_WINDOW):
    """逐 token 的**局部离散度**：`k×k` 窗内方差的通道均值再开方。`(B,C,H,W)→(B,H,W)`。

    无参数、确定性。`local_dispersion` 半径模式用它当半径参数：
    纹理/边缘区域（离散度大）靠近球面，平坦区域靠近球心。
    """
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    mu = F.avg_pool2d(x, k, 1)
    m2 = F.avg_pool2d(x * x, k, 1)
    var = (m2 - mu * mu).clamp_min(0.0)
    return var.mean(dim=1).clamp_min(1e-12).sqrt()


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def rank_corr(a, b, n_max=100_000, seed=0):
    """秩相关（= 是否只是单调变换的判据）。大矩阵先下采样。"""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = a.numel()
    if n > n_max:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:n_max].to(a.device)
        a, b = a[idx], b[idx]

    def rk(x):
        order = torch.argsort(x)
        out = torch.empty_like(order, dtype=torch.float32)
        out[order] = torch.arange(x.numel(), dtype=torch.float32, device=x.device)
        return out

    return _pearson(rk(a), rk(b))


def _entropy_norm(prob):
    n = prob.shape[-1]
    p = prob.clamp_min(1e-12)
    return -(p * p.log()).sum(-1).mean() / math.log(max(2, n))


def _rms(x):
    return x.pow(2).mean().clamp_min(0.0).sqrt()


class HSAM(nn.Module):
    """双时相特征的**双曲相似度注意力**（零参数）。对每个被选中的 stage 独立做一次。"""

    def __init__(self, stage_dims=(96, 192, 384, 768), stages=ATTN_STAGES_DEFAULT,
                 logit_norm=LOGIT_NORM_DEFAULT,
                 radius_mode=RADIUS_MODE_DEFAULT,
                 feature_source="pre_norm"):
        super().__init__()
        self.stage_dims = tuple(int(d) for d in stage_dims)
        self.stages = tuple(int(s) for s in stages)
        self.logit_norm = str(logit_norm)
        self.radius_mode = str(radius_mode)
        self.feature_source = str(feature_source)
        if self.logit_norm not in ("std", "paper", "raw"):
            raise ValueError(f"[HSAM] logit_norm={logit_norm!r} 只接受 std|paper|raw")
        if self.radius_mode not in RADIUS_MODES:
            raise ValueError(f"[HSAM] radius_mode={radius_mode!r} 只接受 {RADIUS_MODES}")
        if self.feature_source not in ("pre_norm", "post_norm"):
            raise ValueError(f"[HSAM] feature_source={feature_source!r} 只接受 "
                             f"pre_norm|post_norm（post_norm 已被实测证明退化，仅作负对照）")
        for s in self.stages:
            if not (0 <= s < len(self.stage_dims)):
                raise ValueError(f"[HSAM] stage {s} 越界（共 {len(self.stage_dims)} 层）")
        self.collect_diag = False
        self._diag = {}

    # ------------------------------------------------------------------
    def _radius(self, x, s):
        """逐 token 半径参数 `rho` (B,N)，按 `radius_mode` 计算（见模块头）。"""
        B, C, H, W = x.shape
        g = x.pow(2).mean().clamp_min(1e-12).sqrt()          # 全局 RMS（标量，跨样本/位置）
        if self.radius_mode == "feature_energy":
            # 论文的"激活强度当半径"，加 ② 的全局标量缩放
            return (x.flatten(2).transpose(1, 2) / g).pow(2).sum(-1) \
                .clamp_min(0.0).sqrt() / math.sqrt(C)
        # local_dispersion：半径只由局部离散度决定，与能量解耦
        sig = local_sigma(x / g).flatten(1)                   # (B,N)
        sm = sig.mean().clamp_min(1e-12)
        return sig / sm                                       # 均值 ≈ 1 ⇒ 平均半径 tanh(1)

    def _one_stage(self, f1, f2, s, want_diag):
        B, C, H, W = f1.shape
        N = H * W
        v1 = f1.flatten(2).transpose(1, 2)                    # (B,N,C) 方向来源
        v2 = f2.flatten(2).transpose(1, 2)

        p1 = expmap_radius(v1, self._radius(f1, s))
        p2 = expmap_radius(v2, self._radius(f2, s))

        hd_raw = poincare_pairwise_distance(p1, p2)           # Eq.(5)
        logits = hsam_logits(hd_raw, C, self.logit_norm)      # Eq.(6)(7) + ③
        prob = torch.softmax(logits, dim=-1)
        attn = logmap0(prob)

        a1 = f1.flatten(2)                                    # (B,C,N)
        a2 = f2.flatten(2)
        out1 = (a1 @ attn + a1).view(B, C, H, W)              # Eq.(8)
        out2 = (a2 @ attn.transpose(1, 2) + a2).view(B, C, H, W)

        if want_diag:
            with torch.no_grad():
                rad = p1.norm(dim=-1)
                om = (1.0 - rad.pow(2))
                ent_used = _entropy_norm(prob)
                ent_paper = _entropy_norm(torch.softmax(torch.tanh(hd_raw / math.sqrt(C)), -1))
                n1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                n2 = v2 / v2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cos = n1 @ n2.transpose(1, 2)
                # 非对角元上算证书（对角恒 0 会人为抬高相关）
                hd0 = hd_raw[:2]
                c0 = (1.0 - cos)[:2]
                Nn = hd0.shape[-1]
                m = ~torch.eye(Nn, dtype=torch.bool, device=hd0.device) \
                    .unsqueeze(0).expand(hd0.shape[0], Nn, Nn)
                rk = rank_corr(hd0[m], c0[m])
                at_eps = float((om <= EPS_BALL * 1.01).float().mean())
                self._diag[f"stage{s}"] = {
                    "dim": C, "tokens": N, "h": H,
                    "logit_norm": self.logit_norm, "radius_mode": self.radius_mode,
                    "feature_source": self.feature_source,
                    "hd_mean": float(hd_raw.mean()),
                    "hd_cv": float(hd_raw.std() / hd_raw.mean().clamp_min(1e-12)),
                    "logit_std": float(logits.std()),
                    "attn_prob_max": float(prob.max()),
                    "attn_entropy_norm": float(ent_used),
                    "attn_entropy_paper": float(ent_paper),
                    "rank_corr_hd_vs_cos": float(rk),
                    "atanh_clip_frac": float(((prob < EPS_ATANH) |
                                              (prob > 1.0 - EPS_ATANH)).float().mean()),
                    "radius_mean": float(rad.mean()),
                    "radius_std": float(rad.std()),
                    "radius_cv": float(rad.std() / rad.mean().clamp_min(1e-12)),
                    "one_minus_r2_min": float(om.min()),
                    "one_minus_r2_cv": float(om.std() / om.mean().clamp_min(1e-12)),
                    "at_eps_frac": at_eps,
                    "agg_over_feat": float(_rms(out1 - f1) / _rms(f1).clamp_min(1e-12)),
                    "out_over_in": float(_rms(out1) / _rms(f1).clamp_min(1e-12)),
                    "contrib_norm": float((out1 - f1).norm()),
                    "finite": bool(torch.isfinite(out1).all() and torch.isfinite(out2).all()),
                }
        return out1, out2

    # ------------------------------------------------------------------
    def forward(self, feats_a, feats_b):
        """`feats_a`/`feats_b`：4 个 stage 的特征。返回**两个新 list**。

        只在 `self.stages` 里的层做双曲注意力，其余原样透传。
        """
        assert len(feats_a) == len(feats_b) == len(self.stage_dims)
        want = bool(self.collect_diag)
        if want:
            self._diag = {"form": HSAM_FORM, "stages": list(self.stages), "params": 0}
        oa, ob = [], []
        for i, (a, b) in enumerate(zip(feats_a, feats_b)):
            if i in self.stages:
                a2, b2 = self._one_stage(a, b, i, want)
                oa.append(a2)
                ob.append(b2)
            else:
                oa.append(a)
                ob.append(b)
        return oa, ob

    def diag(self):
        return dict(self._diag)

    def param_report(self):
        return {"form": HSAM_FORM, "stages": list(self.stages),
                "curvature_c": CURVATURE_C, "params": 0,
                "manifold": "poincare_ball", "geometry": "hyperbolic",
                "learnable": False, "logit_norm": self.logit_norm,
                "radius_mode": self.radius_mode,
                "feature_source": self.feature_source,
                "feature_scale": "global RMS scalar (non-learnable)",
                "rank_corr_max_ok": RANK_CORR_MAX_OK}


def build_hsam(stage_dims=(96, 192, 384, 768), stages=ATTN_STAGES_DEFAULT, **kw):
    return HSAM(stage_dims=stage_dims, stages=stages, **kw)
