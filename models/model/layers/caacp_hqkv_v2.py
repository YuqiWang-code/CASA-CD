"""Run8b 核心创新：CAACP-HQKV v2 —— **抗自锁**版双曲双分支 SAT 式 QKV。

## 为什么要有 v2（v1 的致命缺陷，2026-10-17 实测）

Run8 的 v1 在四个库上**全部**出现：

    [HQKV-GAMMA] detail ≈ ±5e-41    global = 0.0    lite = 0.0
    [HQKV-DIAG-*] contrib_norm = 0.0（四个 stage）, branch_out_ratio_db = 0.0

即 **模块输出恒为零张量、从未进入网络**；H1_HYP 与 CTRL_R8 的 logits 事实上是同一个模型
（四库 F1 差 −0.0016~+0.0001 全是数值噪声）。

**自锁机理**：`γ` 零初始化 ⇒ epoch-0 时 `∂r/∂γ = ‖Δ‖`，而 `r = γ·Δ`；
`γ` 停在 ~1e-41 后 **`Δ` 仍非零但 `r ≡ 0`**，于是 `L` 对 `r` 的梯度**被 `γ` 乘成 0**
⇒ `∂L/∂γ = ⟨∂L/∂r, Δ⟩ = 0` ⇒ **γ 永远出不了 0**。

### 另外两个**结构性**缺陷（不是初始化问题，必须改代码）

1. **`gamma_lite` 恒 0 是结构性的**：stage0 的输出走 `lite_fuse`，但最终输出乘的是
   `gamma_detail`（v1 的 `out = self.gamma_detail * acc_d`）⇒ `gamma_lite` **从未进入前向**，
   永远收不到梯度。
2. **`gamma_global` 没有独立残差路径**：v1 里 detail/global 两个分支的注意力输出
   被 `o = a + b` **先求和**再统一乘 `gamma_detail` ⇒ `gamma_global` 也**从未进入前向**。

## v2 的三条修法（对应上面三个缺陷）

| # | 修法 | 目的 |
|---|---|---|
| 1 | **每分支输出做 RMS 归一化**（`Δ_b ← Δ_b / rms(Δ_b)`） | 让 `‖Δ_b‖ ≈ √N`，**与 γ 的尺度无关**；于是 `∂L/∂γ_b = ⟨∂L/∂r, Δ_b⟩` 从第一步起就**不是 0** |
| 2 | **三个门各自独立**：`r = γ_detail·Δ_detail + γ_global·Δ_global + γ_lite·Δ_lite` | 每个门都真正进入前向 ⇒ 都能收到梯度 |
| 3 | **`γ` 不再零初始化**（初值 `GAMMA_INIT=0.5`） | 彻底避开"零初始化 + 乘性门"的自锁陷阱 |

**代价（必须在论文里如实披露）**：`γ≠0` ⇒ **epoch-0 不再与 CTRL 逐位一致**。
这是 v1 的"零初始化纪律"与"机制必须可学"之间的**真实取舍**：
- 零初始化能保证 epoch-0 恒等，但对**乘性门**会造成必死自锁；
- v2 选择"机制可学"，把"预训练是否被正确加载"改为由
  `[PRETRAIN-LOAD] worst_diff == 0`（直接可观测、更强）来保证。

## 数值与规模控制

- RMS 归一化后每个分支的输出 RMS = 1，`γ_b = 0.5` ⇒ 三支合计 RMS ≈ 0.866（不走 √n 放大）。
  与主干特征（`LayerNorm2d` 后逐通道单位方差）相比是**同一量级的小扰动**，不会淹没主干。
- 归一化**不带可学习参数**（不引入 BN/LN 的 running stats 问题）；
  分号：`rms = sqrt(mean(x²)) + eps`，在 `(B,C,H,W)` 上取全元素均值。

## 诊断（供 fail-fast 与裁决）

- `contrib_norm`：**已乘门**的输出 L2 范数（v1 里它是 0，是本轮要盯死的量）；
- `gamma`：三支门的终值；
- `grad`：三支门在**第一步**的梯度绝对值（自锁的直接判据）；
- `deg_d_vs_g_frac_rank_diff`：两分支排序分歧（非退化门）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.hyperbolic import (
    A_DEFAULT,
    R0_DEFAULT,
    euclidean_change_score,
    lorentz_change_score,
)
from model.layers.vmamba_kernel import backend_report  # noqa: F401

HQKV_FORM_V2 = "hqkv_v2_antilock"
HQKV_SEED = 17
ATTN_STAGES_DEFAULT = (1, 2, 3)     # 1/8、1/16、1/32 走注意力；1/4 走廉价融合
GAMMA_INIT = 0.5                    # **非零**初值：避开"零初始化 + 乘性门"自锁
RMS_EPS = 1e-6

__all__ = ["CAACPHQKVv2", "BranchSATAttentionV2", "build_caacp_hqkv_v2",
           "HQKV_FORM_V2", "GAMMA_INIT"]


def rms_normalize(x, eps=RMS_EPS):
    """让张量的 RMS = 1（**无参数**、可微）。`(B,C,H,W) -> 同形状`。

    为什么需要它：v1 的 `Δ` 是若干个 1×1 卷积输出的**裸和**，量级不可控；
    乘性门 `γ` 的梯度 `∂L/∂γ = ⟨∂L/∂r, Δ⟩` 与 `‖Δ‖` 成正比。
    把 `Δ` 归一到单位 RMS 后，门从第一步起就有固定量级的梯度信号。
    """
    r = x.pow(2).mean().clamp_min(0.0).sqrt()
    return x / (r + eps)


class BranchSATAttentionV2(nn.Module):
    """一个分支的特化 QKV + 非对称注意力（Q: N，K/V: k·N），输出宽度 `out_dim`。

    与 v1 的唯一差别：**输出做 RMS 归一化**（见 `rms_normalize`）。
    参数省法（同 v1）：`V` 用 per-channel depthwise 1×1（`groups=rank`），
    单分支参数 = `3·dim·rank + rank·out_dim`（dim=768/rank=32/out=192 时 93,696）。
    """

    def __init__(self, dim, rank=32, window=2, global_branch=False,
                 out_dim=192, init_seed=HQKV_SEED):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.window = int(window)
        self.global_branch = bool(global_branch)
        self.out_dim = int(out_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.q = nn.Conv2d(self.dim, self.rank, 1, bias=False)
            self.k = nn.Conv2d(self.dim, self.rank, 1, bias=False)
            self.v_split = nn.Conv2d(self.dim, self.rank, 1, groups=self.rank, bias=False)
            self.proj = nn.Conv2d(self.rank, self.out_dim, 1, bias=False)
            self.apply(self._init_weights)
        self.scale = self.rank ** -0.5

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_q, x_kv, score):
        """`x_q`/`x_kv` (B,C,H,W)；`score` (B,1,H,W) 该分支的双曲变化分数（数值，非 rank）。"""
        if self.global_branch:
            kv_src = global_deviation_weights(x_kv, window=self.window)
        else:
            kv_src = change_weighted_pool(x_kv, score, window=self.window)
        k = self.k(kv_src)
        v = self.v_split(kv_src)
        q = self.q(x_q)

        B, r, H, W = q.shape
        qf = q.flatten(2).transpose(1, 2)                 # (B, N, r)
        kf = k.flatten(2)                                 # (B, r, n)
        vf = v.flatten(2).transpose(1, 2)                 # (B, n, r)
        attn = torch.softmax(qf @ kf * self.scale, dim=-1)     # (B, N, n)
        out = (attn @ vf).transpose(1, 2).reshape(B, r, H, W)
        return rms_normalize(self.proj(out))

    def param_report(self):
        return {"dim": self.dim, "rank": self.rank, "window": self.window,
                "out_dim": self.out_dim, "global_branch": self.global_branch,
                "params": sum(p.numel() for p in self.parameters())}


# ---------------------------------------------------------------------------
# 两个不引入参数的压缩算子（与 v1 同式，保留以便对照）
# ---------------------------------------------------------------------------
def change_weighted_pool(x, score, window=2, eps=1e-6):
    """规则 window×window 网格内按**数值**变化分数加权聚合（非 rank、非 TopK）。"""
    B, C, H, W = x.shape
    assert H % window == 0 and W % window == 0, f"{H}×{W} 不能被 window={window} 整除"
    ho, wo = H // window, W // window
    s = score.view(B, 1, ho, window, wo, window)
    w = 1.0 + s
    w = w / w.sum(dim=(3, 5), keepdim=True).clamp_min(eps)
    xr = x.view(B, C, ho, window, wo, window)
    return (xr * w).sum(dim=(3, 5))


def global_deviation_weights(x, window=4, eps=1e-6):
    """全局分支的压缩：按「相对全图均值的偏离」加权池化（规则网格）。"""
    B, C, H, W = x.shape
    mu = x.mean(dim=(2, 3), keepdim=True)
    dev = (x - mu).pow(2).sum(dim=1, keepdim=True).clamp_min(0.0).sqrt()
    w = 1.0 + dev
    return F.avg_pool2d(x * w, kernel_size=window, stride=window) / \
        F.avg_pool2d(w, kernel_size=window, stride=window).clamp_min(eps)


class CAACPHQKVv2(nn.Module):
    """跨 4 个 stage 的双曲双分支 SAT-QKV（**抗自锁**）。

    输出：`r = γ_detail·Δ_detail + γ_global·Δ_global + γ_lite·Δ_lite`，
    三支各自 `rms_normalize` 到单位 RMS，`γ` 初值 0.5、三条**独立**。
    """

    def __init__(self, stage_dims=(96, 192, 384, 768), rank=32,
                 detail_window=2, global_window=4,
                 attn_stages=ATTN_STAGES_DEFAULT,
                 r0=R0_DEFAULT, a=A_DEFAULT, local_ksize=3,
                 lite_dim=192, init_seed=HQKV_SEED,
                 score_stages=(0, 1, 2, 3), gamma_init=GAMMA_INIT):
        super().__init__()
        self.stage_dims = tuple(int(d) for d in stage_dims)
        self.rank = int(rank)
        self.detail_window = int(detail_window)
        self.global_window = int(global_window)
        self.attn_stages = tuple(int(i) for i in attn_stages)
        self.r0 = float(r0)
        self.a = float(a)
        self.local_ksize = int(local_ksize)
        self.lite_dim = int(lite_dim)
        self.score_stages = tuple(int(i) for i in score_stages)
        self._scores = {}
        self._diag = {}

        self.detail_attn = nn.ModuleList()
        self.global_attn = nn.ModuleList()
        self.lite_attn = nn.ModuleList()
        self.lite_fuse = nn.ModuleList()
        for i, d in enumerate(self.stage_dims):
            if i in self.attn_stages:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(init_seed + i)
                    self.detail_attn.append(BranchSATAttentionV2(
                        d, rank=rank, window=detail_window, global_branch=False,
                        out_dim=lite_dim, init_seed=init_seed + i))
                    self.global_attn.append(BranchSATAttentionV2(
                        d, rank=rank, window=global_window, global_branch=True,
                        out_dim=lite_dim, init_seed=init_seed + 8 + i))
                self.lite_attn.append(None)
                self.lite_fuse.append(None)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(init_seed + 16 + i)
                    # 修复 v1 缺陷①：stage0 **不走 detail 分支**，而是自己一条
                    # `lite` 分支（3 路廉价融合），由 `gamma_lite` 独立控制。
                    fuse = nn.Sequential(
                        nn.Conv2d(3 * d, self.lite_dim, 1, bias=False),
                        nn.BatchNorm2d(self.lite_dim),
                        nn.SiLU(),
                        nn.Conv2d(self.lite_dim, self.lite_dim, 1, bias=False),
                    )
                    fuse.apply(self._init_weights)
                self.lite_fuse.append(fuse)
                self.detail_attn.append(None)
                self.global_attn.append(None)
                self.lite_attn.append(None)

        # 三个**独立**门，**非零**初值（修复缺陷③；缺陷②由"各自独立残差"解决）
        self.gamma_detail = nn.Parameter(torch.full((1,), float(gamma_init)))
        self.gamma_global = nn.Parameter(torch.full((1,), float(gamma_init)))
        self.gamma_lite = nn.Parameter(torch.full((1,), float(gamma_init)))
        # 记录"第一步"的门梯度（诊断自锁用；由调用方在 step 1 之后读取）
        self._grad_snapshot = {}

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def branch_scores(self, xa, xb, stage, geometry="hyperbolic"):
        score_fn = lorentz_change_score if geometry == "hyperbolic" else euclidean_change_score
        s_d = score_fn(xa, xb, branch="detail", r0=self.r0, a=self.a, ksize=self.local_ksize)
        s_g = score_fn(xa, xb, branch="global", r0=self.r0, a=self.a, ksize=self.local_ksize)
        return s_d, s_g

    def score_maps(self):
        """供 HGSL 使用：上一次 forward 缓存的每 stage 双曲变化分数（B,1,H_i,W_i）。"""
        return dict(self._scores)

    def diag(self):
        return dict(self._diag)

    def snapshot_gamma_grad(self):
        """在 `loss.backward()` 之后调用：把三支门的梯度绝对值记进诊断。"""
        def g(p):
            return float(p.grad.detach().abs().sum()) if p.grad is not None else None
        self._grad_snapshot = {"detail": g(self.gamma_detail),
                               "global": g(self.gamma_global),
                               "lite": g(self.gamma_lite)}

    def param_report(self):
        per = {}
        total = 0
        for i, d in enumerate(self.stage_dims):
            ent = {"dim": d}
            for name, mod in (("detail", self.detail_attn[i]), ("global", self.global_attn[i]),
                              ("lite_fuse", self.lite_fuse[i])):
                if mod is not None:
                    n = sum(p.numel() for p in mod.parameters())
                    ent[name] = n
                    total += n
            per[f"stage{i}"] = ent
        total += 3
        return {"form": HQKV_FORM_V2, "rank": self.rank,
                "detail_window": self.detail_window, "global_window": self.global_window,
                "attn_stages": list(self.attn_stages), "r0": self.r0, "a": self.a,
                "gamma_init": float(self.gamma_detail.detach().reshape(-1)[0]),
                "gammas": {"detail": float(self.gamma_detail.detach().sum()),
                           "global": float(self.gamma_global.detach().sum()),
                           "lite": float(self.gamma_lite.detach().sum())},
                "gamma_grad": dict(self._grad_snapshot),
                "per_stage": per, "total_params": total,
                # v2 **不承诺**零初始化（那正是自锁根源）
                "gamma_zero_init": False}

    # ------------------------------------------------------------------
    def forward(self, feats_a, feats_b, geometry="hyperbolic", inject_indices=None):
        """返回 `(out, diag)`；`out` (B, lite_dim, 64, 64)。

        `r = γ_detail·Δ_detail + γ_global·Δ_global + γ_lite·Δ_lite`（三支独立、各自 RMS=1）。
        """
        if inject_indices is None:
            inject_indices = (0, 1, 2, 3)
        assert len(feats_a) == len(feats_b) == len(self.stage_dims)
        ref_hw = feats_a[0].shape[-2:]
        acc_detail, acc_global, acc_lite = None, None, None
        diag = {"stages": {}}
        self._scores = {}

        for i in range(len(self.stage_dims)):
            if i in self.score_stages:
                xa = feats_a[i].float()
                xb = feats_b[i].float()
                self._scores[i] = self.branch_scores(xa, xb, i, geometry=geometry)[0]
            else:
                xa = feats_a[i].detach().float()
                xb = feats_b[i].detach().float()
            with torch.no_grad():
                if i in self.score_stages:
                    s_d = self._scores[i].detach()
                    s_g = self.branch_scores(xa, xb, i, geometry=geometry)[1].detach()
                else:
                    s_d, s_g = self.branch_scores(xa, xb, i, geometry=geometry)
                    s_d, s_g = s_d.detach(), s_g.detach()
                diag["stages"][f"stage{i}"] = {
                    "score_detail_mean": float(s_d.mean()),
                    "score_global_mean": float(s_g.mean()),
                }
                from model.layers.hyperbolic import branch_rank_disagreement
                dg = branch_rank_disagreement(
                    s_d.reshape(s_d.shape[0], -1), s_g.reshape(s_g.shape[0], -1))
                diag["stages"][f"stage{i}"].update(
                    {f"deg_d_vs_g_{k}": v for k, v in dg.items()})

            if i not in inject_indices:
                continue
            if i in self.attn_stages:
                a_out = self.detail_attn[i](feats_a[i], feats_a[i], s_d)      # RMS=1
                b_out = self.global_attn[i](feats_b[i], feats_b[i], s_g)      # RMS=1
                with torch.no_grad():
                    diag["stages"][f"stage{i}"]["branch_out_ratio_db"] = float(
                        a_out.detach().norm() / b_out.detach().norm().clamp_min(1e-12))
                # ⚠️ 不同 stage 的分辨率不同（1/8、1/16、1/32）⇒ **必须先在各自尺度上
                # RMS 归一化，再上采样到 ref_hw 相加**。若直接 `acc + a_out` 会因 (32 vs 16)
                # 维度不匹配而报错（首版实测踩到）。
                a_out = rms_normalize(a_out)
                b_out = rms_normalize(b_out)
                if a_out.shape[-2:] != ref_hw:
                    a_out = F.interpolate(a_out, size=ref_hw, mode="bilinear",
                                          align_corners=False)
                    b_out = F.interpolate(b_out, size=ref_hw, mode="bilinear",
                                          align_corners=False)
                acc_detail = a_out if acc_detail is None else acc_detail + a_out
                acc_global = b_out if acc_global is None else acc_global + b_out
            else:
                # stage0（或任何不在 attn_stages 的 stage）走 lite 分支
                l_out = self.lite_fuse[i](
                    torch.cat([feats_a[i], feats_b[i], (xa - xb).abs()], dim=1))
                l_out = rms_normalize(l_out)
                if l_out.shape[-2:] != ref_hw:
                    l_out = F.interpolate(l_out, size=ref_hw, mode="bilinear",
                                          align_corners=False)
                acc_lite = l_out if acc_lite is None else acc_lite + l_out

        # 每支的 `acc` 已经是若干"单位 RMS 分量"之和（每分量由 rms_normalize 保证 RMS=1），
        # 于是 `‖acc‖` 的量级是 O(√(C·H·W)·√stage数)，乘 γ 后累加成 `out`。
        out = feats_a[0].new_zeros(feats_a[0].shape[0], self.lite_dim, *ref_hw)
        parts = {}
        for name, acc, gamma in (("detail", acc_detail, self.gamma_detail),
                                 ("global", acc_global, self.gamma_global),
                                 ("lite", acc_lite, self.gamma_lite)):
            if acc is None:
                parts[name] = 0.0
                continue
            if acc.shape[-2:] != ref_hw:
                acc = F.interpolate(acc, size=ref_hw, mode="bilinear", align_corners=False)
            acc = rms_normalize(acc)
            out = out + gamma * acc
            parts[name] = float(acc.detach().norm())
        #
        # ⚠️ **绝对不能在最后再对 `out` 做一次 rms_normalize**（首版试过，必须记下来）：
        # 那会把 `γ` 的效果**归一化掉**（γ 变化不影响输出），并且 `∂L/∂γ` 会经由那个归一化
        # 被截断 —— 等于把自锁换了层皮。方向性错误，已撤回。
        # 正确做法：**用固定的结构常数缩放**（不参与梯度），既压住注入量级、又不影响 γ 的可学性。
        #   分母 = √(活跃分支数 × 参与叠加的 stage 数)
        #   初值时 out 的 RMS ≈ (γ_detail+γ_global+γ_lite)/√(分支数×stage数)
        #   = 1.5/√(3×3) = 0.5 ⇒ 与主干（LayerNorm2d 后单位方差）相比是**可控的小扰动**。
        n_parts = sum(1 for v in parts.values() if isinstance(v, float) and v > 0.0)
        n_stage = max(1, len([i for i in inject_indices if i in self.stage_dims]))
        out = out * (1.0 / math.sqrt(max(1, n_parts) * n_stage))
        # 三支相对量级（用于判断"是否只有一支在起作用"）
        tot = sum(v for v in parts.values() if isinstance(v, float)) or 1.0
        for k, v in parts.items():
            diag[f"part_{k}_norm"] = v
            diag[f"part_{k}_share"] = (v / tot) if isinstance(v, float) else 0.0

        self._diag = diag
        self._diag["gammas"] = {"detail": float(self.gamma_detail.detach().sum()),
                                "global": float(self.gamma_global.detach().sum()),
                                "lite": float(self.gamma_lite.detach().sum())}
        # **v1 的 contrib_norm 恒为 0；v2 把它作为 fail-fast 的判据**
        self._diag["contrib_norm"] = float(out.detach().norm())
        return out, diag


def build_caacp_hqkv_v2(stage_dims=(96, 192, 384, 768), **kwargs):
    return CAACPHQKVv2(stage_dims=stage_dims, **kwargs)
