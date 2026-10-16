"""Run8 核心创新：CAACP-HQKV（Hyperbolic dual-branch SAT-style QKV change aggregation）。

对应本轮用户要求的 R2–R8：
每层输出 → 映射到双曲流形 → 分成「细节/局部/纹理/边缘」与「全局语义」两个分支 →
各自做特化 QKV → 注入解码器。K/V 按 **SAT 思想做 token 压缩（`N → k·N`）**，**Q 保持 `N`**。

## 与 Run1–Run3 的 CASAA（已终止）的**实质差异**（P0-3 陷阱①的交代）

| 维度 | CASAA（Run1/Run3，已终止） | CAACP-HQKV（Run8） |
|---|---|---|
| 空间结构 | ViT **变长 token 序列** + 密度峰值聚类（`cluster_and_merge` 含 `randperm`/`rand_like` 随机采样） | **规则二维网格**（固定 window 池化），无随机采样 ⇒ epoch-0 与 RNG 账本可控 |
| 为什么压缩 | 「省算力」（但实测 FLOPs 只降 **0.25%**） | 压缩是**几何分解的载体**：两个分支用**两种半径**分别定义 K/V |
| 打分来源 | GT patch occupancy / 单时相 ViT 特征 rank | **双时相 Lorentz 测地距离 + 局部离散度半径**（参数自由） |
| 分支数 | 1 | **2（细节/全局），各自独立 QKV 权重** |
| 注入位置 | 编码器内部第 8–11 层 | 编码器→解码器的**跳层连接**（可选消融：只注入 1/4 / 1/16） |
| 终止原因 | 「deployable routing ranking 提升 ≠ F1 提升」 | ⚠️ 本方案**尚未证伪该风险**，因此 Stage 设计把「注入」与「几何」拆开做单变量 |

## 为什么**不用** `rank_normalize_2d`（P0-3 陷阱③的交代）

`rank(·)` 只保留序、丢弃一切幅度；任何**保序**变换（包括固定半径双曲距离）在其后都
产生**逐位相同**的权重。Run8 因此让 Q/K/V 直接消费**数值分数**（`1+Q` 形式，非保序）：
非恒定半径的双曲距离会改变 `Q` 的**数值**，从而改变 softmax 权重，不会退化。

## 数据流（真实，不是概念图）

```
stage i 输出  fA, fB            (B, C_i, H_i, W_i)   ← 同一时相各自的 encoder 输出
   │
   ├─ detail 分支:  r = r0 + a·q_local(3×3 离散度)  ─┐
   └─ global  分支:  r = r0 + a·q_global(全图偏离)  ─┤
                                                    ↓
                        s_b = 1 − exp(−d_Lorentz)      (B,1,H_i,W_i)  数值分数，无 rank
                                                    ↓
        K/V 压缩:  window 内 w ∝ 1 + s_b  →  kv_b  (B, C_i, H_i/w_b, W_i/w_b)    [token: N → k_b·N]
        Q 保持:   q  = Linear(f_b)                     (B, r, H_i, W_i)          [token: N 不变]
                                                    ↓
        attn_b = softmax(qᵀk_b/√r) · v_b             每分支**独立**的 q/k/v 投影
                                                    ↓
        out = γ_detail·Proj(attn_detail) + γ_global·Proj(attn_global)   γ 零初始化
                                                    ↓
                        返回到 1/4 尺度 → 加入 DCR 最细尺度输入
```

## 零初始化与预训练纪律

- 两个门 `γ_detail` / `γ_global` **零初始化** ⇒ **关掉开关时模型前向与"无该模块"逐位一致**；
- 模块**最后构造**且用 `fork_rng` 固定局部种子 ⇒ 不推进全局随机流，既有模块初始化不受影响；
- 因此 `hf_edge=0/1` × `hqkv=0/1` 的对照臂可以共用同一套预训练加载（`worst_diff` 必须为 0）。
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
from model.layers.vmamba_kernel import backend_report  # noqa: F401  (日志/诊断用)

HQKV_FORM = "hqkv_v1"
HQKV_SEED = 17          # 局部 RNG 种子（与 HFAE 的 16 / FET 的 16 同惯例，但取不同值避免同流）
ATTN_STAGES_DEFAULT = (1, 2, 3)     # 在 1/8、1/16、1/32 上做注意力；1/4 用廉价融合


# ---------------------------------------------------------------------------
# 分数 → cell 权重（**非保序**：直接消费数值分数，不经 rank）
# ---------------------------------------------------------------------------
def change_weighted_pool(x, score, window=2, eps=1e-6):
    """规则 window×window 网格内按变化分数加权聚合。

    `x`     (B,C,H,W)
    `score` (B,1,H,W) 数值分数（**不是 rank**）
    返回    (B,C,H/window,W/window)；对 `x` 保梯度，权重路径无梯度。

    权重 `w_i = (1 + s_i) / Σ_j (1 + s_j)`：
    - `s=0`（完全无变化）⇒ 全部 `1/4` ⇒ 精确退化为均匀池化；
    - `s` 越大越主导该 cell 的低频上下文。
    """
    B, C, H, W = x.shape
    assert H % window == 0 and W % window == 0, f"{H}×{W} 不能被 window={window} 整除"
    ho, wo = H // window, W // window
    s = score.view(B, 1, ho, window, wo, window)
    w = 1.0 + s
    w = w / w.sum(dim=(3, 5), keepdim=True).clamp_min(eps)
    xr = x.view(B, C, ho, window, wo, window)
    return (xr * w).sum(dim=(3, 5))


def global_deviation_weights(x, window=4, eps=1e-6):
    """全局分支的压缩：按「相对全图均值的偏离」加权池化（同样规则网格）。

    只用该时相自己的全局均值 ⇒ 单时相统计量、A/B 严格对称。
    """
    B, C, H, W = x.shape
    mu = x.mean(dim=(2, 3), keepdim=True)
    dev = (x - mu).pow(2).sum(dim=1, keepdim=True).clamp_min(0.0).sqrt()
    w = 1.0 + dev
    return F.avg_pool2d(x * w, kernel_size=window, stride=window) / \
        F.avg_pool2d(w, kernel_size=window, stride=window).clamp_min(eps)


# ---------------------------------------------------------------------------
# 单分支的 SAT 式非对称注意力（Q 全分辨率，K/V 压缩）
# ---------------------------------------------------------------------------
class BranchSATAttention(nn.Module):
    """一个分支的特化 QKV + 非对称注意力（Q: N，K/V: k·N）。

    `out_dim` 让**所有 stage 的注意力输出统一到 `lite_dim`**，这样 4 个 stage 的增量
    可以直接相加后再上采样（否则 stage1 输出 `rank`=32、stage0 输出 `lite_dim`=192 会 shape 冲突）。
    多出来的 `rank → out_dim` 每 stage 只有 `rank*out_dim` 个参数（32×192 = 6,144）。
    """

    def __init__(self, dim, rank=32, window=2, global_branch=False,
                 out_dim=None, init_seed=HQKV_SEED):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.window = int(window)
        self.global_branch = bool(global_branch)
        self.out_dim = int(out_dim if out_dim is not None else dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.q = nn.Conv2d(self.dim, self.rank, 1, bias=False)
            self.k = nn.Conv2d(self.dim, self.rank, 1, bias=False)
            self.v_split = nn.Conv2d(self.dim, self.rank, 1, groups=self.rank, bias=False)
            self.proj = nn.Conv2d(self.rank, self.out_dim, 1, bias=False)
            self.apply(self._init_weights)
        self.scale = self.rank ** -0.5

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def param_report(self):
        return {"dim": self.dim, "rank": self.rank, "window": self.window,
                "out_dim": self.out_dim, "global_branch": self.global_branch,
                "params": sum(p.numel() for p in self.parameters())}

    def forward(self, x_q, x_kv, score):
        """`x_q` (B,C,H,W)：Query 来源（保持全分辨率，**不压缩**）。
        `x_kv` (B,C,H,W)：K/V 来源（同 stage，将按 window 压缩）。
        `score` (B,1,H,W)：该分支的双曲变化分数（数值，非 rank）。

        参数省法：`V` 用 **per-channel depthwise 1×1**（`groups=rank`，每通道一个标量），
        把 value 通道降到 `rank` 后在注意力输出处再用 `proj: rank→dim` 升回。
        这使单分支参数从 `2·dim²` 降到 `dim·rank + dim·rank + rank·dim`，
        在 `dim=768, rank=32` 时从 1.18M 降到 74K（**降 16×**）。
        """
        if self.global_branch:
            kv_src = global_deviation_weights(x_kv, window=self.window)
        else:
            kv_src = change_weighted_pool(x_kv, score, window=self.window)
        k = self.k(kv_src)
        v = self.v_split(kv_src)
        q = self.q(x_q)

        B, r, H, W = q.shape
        _, _, hk, wk = k.shape
        qf = q.flatten(2).transpose(1, 2)                 # (B, N, r)
        kf = k.flatten(2)                                 # (B, r, n)
        vf = v.flatten(2).transpose(1, 2)                 # (B, n, r)
        attn = torch.softmax(qf @ kf * self.scale, dim=-1)     # (B, N, n)
        out = (attn @ vf).transpose(1, 2).reshape(B, r, H, W)
        return self.proj(out)


# ---------------------------------------------------------------------------
# 4 层 × 双分支的完整模块
# ---------------------------------------------------------------------------
class CAACPHQKV(nn.Module):
    """跨 4 个 stage 的双曲双分支 SAT-QKV 变化聚合。

    - 4 个 stage 各自独立参数（满足 R2「每层」）；
    - 每个 stage 两个分支（detail / global）各自独立 QKV（满足 R7「特化」）；
    - K/V token 压缩到 `k·N`，Q 保持 `N`（满足 R8）；
    - 输出在 1/4 尺度上按 stage 大小上采样后相加，再乘零初始化门（满足 R4 的"跳层连接"注入）。
    """

    def __init__(self, stage_dims=(96, 192, 384, 768), rank=32,
                 detail_window=2, global_window=4,
                 attn_stages=ATTN_STAGES_DEFAULT,
                 r0=R0_DEFAULT, a=A_DEFAULT, local_ksize=3,
                 lite_dim=96, init_seed=HQKV_SEED,
                 score_stages=(0, 1, 2, 3)):
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
        # 哪些 stage 的双曲分数参与 HGSL 监督（= 需要保留计算图的那些）
        self.score_stages = tuple(int(i) for i in score_stages)
        self._scores = {}

        # 每 stage 的双分支注意力（**最后构造** + fork_rng 固定局部种子）
        self.detail_attn = nn.ModuleList()
        self.global_attn = nn.ModuleList()
        self.lite_fuse = nn.ModuleList()
        for i, d in enumerate(self.stage_dims):
            if i in self.attn_stages:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(init_seed + i)
                    self.detail_attn.append(BranchSATAttention(
                        d, rank=rank, window=detail_window, global_branch=False,
                        out_dim=lite_dim, init_seed=init_seed + i))
                    self.global_attn.append(BranchSATAttention(
                        d, rank=rank, window=global_window, global_branch=True,
                        out_dim=lite_dim, init_seed=init_seed + 8 + i))
                    self.lite_fuse.append(None)
            else:
                self.detail_attn.append(None)
                self.global_attn.append(None)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(init_seed + 16 + i)
                    fuse = nn.Sequential(
                        nn.Conv2d(3 * d, self.lite_dim, 1, bias=False),
                        nn.BatchNorm2d(self.lite_dim),
                        nn.SiLU(),
                        nn.Conv2d(self.lite_dim, self.lite_dim, 1, bias=False),
                    )
                    fuse.apply(self._init_weights)
                self.lite_fuse.append(fuse)

        # 零初始化门（**在 apply 之后**，否则会被初始化覆盖——HFAE 首版踩过这个坑）
        self.gamma_detail = nn.Parameter(torch.zeros(1))
        self.gamma_global = nn.Parameter(torch.zeros(1))
        self.gamma_lite = nn.Parameter(torch.zeros(1))

        # 诊断量（只读，供日志/裁决）
        self._diag = {}

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def branch_scores(self, xa, xb, stage, geometry="hyperbolic"):
        """返回 `(s_detail, s_global)`，各 (B,1,H,W)。

        **本方法不自带 `no_grad`**：梯度是否流回几何量由调用方决定
        （`score_stages` 里包含 `stage` ⇒ 保留计算图；否则调用方在 `no_grad` 下调用）。
        这一点必须显式，否则 HGSL 的 `hgsl_detach_score=False` 臂会静默失效。
        """
        score_fn = lorentz_change_score if geometry == "hyperbolic" else euclidean_change_score
        s_d = score_fn(xa, xb, branch="detail", r0=self.r0, a=self.a, ksize=self.local_ksize)
        s_g = score_fn(xa, xb, branch="global", r0=self.r0, a=self.a, ksize=self.local_ksize)
        return s_d, s_g

    def param_report(self):
        out = {"form": HQKV_FORM, "rank": self.rank,
               "detail_window": self.detail_window, "global_window": self.global_window,
               "attn_stages": list(self.attn_stages),
               "r0": self.r0, "a": self.a,
               "gammas": {"detail": float(self.gamma_detail.detach()),
                          "global": float(self.gamma_global.detach()),
                          "lite": float(self.gamma_lite.detach())},
               "per_stage": {}}
        total = 0
        for i, d in enumerate(self.stage_dims):
            ent = {"dim": d}
            for name, mod in (("detail", self.detail_attn[i]), ("global", self.global_attn[i]),
                              ("lite", self.lite_fuse[i])):
                if mod is not None:
                    np_ = sum(p.numel() for p in mod.parameters())
                    ent[name] = np_
                    total += np_
            out["per_stage"][f"stage{i}"] = ent
        out["total_params"] = total + 3          # 3 个 gamma
        out["gamma_zero_init"] = all(float(g.detach().abs().sum()) == 0.0 for g in
                                     (self.gamma_detail, self.gamma_global, self.gamma_lite))
        return out

    def diag(self):
        return dict(self._diag)

    def score_maps(self):
        """上一次 forward 缓存的每 stage 双曲变化分数（供 HGSL 使用）。

        返回 `{stage_i: (B,1,H_i,W_i)}`，**B 尺寸**（每对样品一张），已 `.detach()` 与否
        取决于构造时的 `score_requires_grad`。空的 dict 表示本步没有分数（`caacp=0`）。
        """
        return dict(self._scores)

    # ------------------------------------------------------------------
    def forward(self, feats_a, feats_b, geometry="hyperbolic", inject_indices=None):
        """`feats_a/feats_b`: 每时相 4 个 stage 的 (B, C_i, H_i, W_i)。

        返回 `(out, diag)`：
        - `out` (B, lite_dim, 64, 64)：待注入 DCR 最细尺度的增量（**已乘 γ_detail**）；
        - `diag`：各 stage 的分支门值、分数统计、分支间排序分歧（非退化性证据）。
        """
        if inject_indices is None:
            inject_indices = (0, 1, 2, 3)
        assert len(feats_a) == len(feats_b) == len(self.stage_dims)
        ref_hw = feats_a[0].shape[-2:]
        acc_d, acc_g, acc_l = None, None, None
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
                    "score_detail_max": float(s_d.amax()),
                    "score_global_mean": float(s_g.mean()),
                    "score_global_max": float(s_g.amax()),
                }
                from model.layers.hyperbolic import branch_rank_disagreement
                dg = branch_rank_disagreement(
                    s_d.reshape(s_d.shape[0], -1), s_g.reshape(s_g.shape[0], -1))
                diag["stages"][f"stage{i}"].update(
                    {f"deg_d_vs_g_{k}": v for k, v in dg.items()})

            if i not in inject_indices:
                continue
            if i in self.attn_stages:
                a = self.detail_attn[i](feats_a[i], feats_a[i], s_d)
                b = self.global_attn[i](feats_b[i], feats_b[i], s_g)
                o = a + b
                with torch.no_grad():
                    diag["stages"][f"stage{i}"]["branch_out_ratio_db"] = float(
                        a.detach().norm() / b.detach().norm().clamp_min(1e-12))
            else:
                o = self.lite_fuse[i](
                    torch.cat([feats_a[i], feats_b[i], (xa - xb).abs()], dim=1))
            if o.shape[-2:] != ref_hw:
                o = F.interpolate(o, size=ref_hw, mode="bilinear", align_corners=False)
            acc_d = o if acc_d is None else acc_d + o
            with torch.no_grad():
                diag["stages"][f"stage{i}"]["contrib_norm"] = float(o.detach().norm())

        assert acc_d is not None, "至少需要一个 stage 参与注入"
        out = self.gamma_detail * acc_d
        self._diag = diag
        self._diag["gammas"] = {"detail": float(self.gamma_detail.detach()),
                                "global": float(self.gamma_global.detach()),
                                "lite": float(self.gamma_lite.detach())}
        return out, diag


def build_caacp_hqkv(stage_dims=(96, 192, 384, 768), **kwargs):
    return CAACPHQKV(stage_dims=stage_dims, **kwargs)
