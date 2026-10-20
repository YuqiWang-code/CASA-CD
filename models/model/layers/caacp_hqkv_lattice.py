"""CAACP-HQKV-Lattice：**拆除双曲几何与半径分解之后**的跨层 QKV 注入。

## 这次拆除做了什么（2026-10-20）

拆除前的实现（`_SUPERSEDED_hqkv_v2_2026-10-20/`，字节与 `Run8b/SOURCE_SHA256.txt` 逐项相等）
是一条「流形 → 半径分解 → 双分支特化 QKV」的链：

    T1/T2 各 stage 特征 → Lorentz 双曲映射 → 双时相差异感知的变化分数 r = r0 + a·q
      → 按「局部离散度 vs 全局半径」拆成 detail / global 两支
      → 两支各自一套 SAT 式 QKV（Q 全 N，K/V 压到 k·N）
      → 三支 γ 门控 → 注入 DCR；分数另喂 HGSL

实测结论（`B1_HYP` vs `CTRL_R8B`，四库同协议同 seed）：
**ΔF1 = {+0.0001, −0.0009, −0.0012, −0.0014}，均值 −0.00085**，还多付
+0.304 M 参数 / +0.46 G FLOPs / 约 +9% 训练时间 ⇒ **不划算**，故拆除。

**本模块保留的只有**：
1. **跨层 QKV 注入**：Q 取 T1 的**全分辨率** token（N），K/V 取 T2 经压缩后的 token（`k·N`）；
2. **SAT 式非对称注意力**（Q 长 N、K/V 长 n）—— 与 STORM/ICML2026 的约束相容：
   **不做 TopK / 聚类 / 随机采样**，只用**固定格点**（fixed cell lattice）压缩，
   这正是 README 里给 CAACP 划的 novelty 边界；
3. **反自锁的工程手段**（与几何无关，保留）：每支输出 `rms_normalize`、**每支独立 γ 门**、
   `γ` 非零初值、`contrib_norm` 诊断口 + fail-fast 判据。

**本模块不再有的东西**：任何流形/双曲/欧氏几何项、变化分数、半径 `r0`/`a`、
`detail_window`/`global_window` 的双窗口、detail/global 双分支、HGSL 的分数输入口。

## 为什么压缩不用分数加权了

拆除前用「变化分数」给池化加权；分数来自几何，几何拆了就没有分数。
换回**固定格点均值池化**并不是退步：它是**无参数、确定、与内容无关**的压缩，
恰好符合「structured context compression / 固定 cell lattice」的边界要求，
也不会引入「分数本身没学好 ⇒ 压缩权重乱掉」这一层间接失败模式。
如果将来要重新引入内容相关性，必须在**独立预注册**下作为唯一变量另开一轮。

## 参数量（`rank=32`、`lite_dim=192`、`window=2`、stage1/2/3 走注意力）

| stage | dim | 单套 QKV 参数 | 说明 |
|---|---:|---:|---|
| 0 | 96 | `lite_fuse` 3·96→192→192 | 1/4 分辨率走廉价融合（无注意力） |
| 1 | 192 | ≈ 24.6 k | 1/8，N=1024 → n=256 |
| 2 | 384 | ≈ 43.0 k | 1/16，N=256 → n=64 |
| 3 | 768 | ≈ 79.9 k | 1/32，N=64 → n=16 |
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HQKV_FORM = "hqkv_v3_lattice"
HQKV_SEED = 21
ATTN_STAGES_DEFAULT = (1, 2, 3)   # 1/8、1/16、1/32 走注意力；1/4 走廉价融合
LATTICE_WINDOW_DEFAULT = 2
GAMMA_INIT = 0.5                  # **非零**初值：避开「零初始化 × 乘性门」自锁
RMS_EPS = 1e-6

__all__ = [
    "CAACPHQKVLattice", "LatticeSATAttention", "build_caacp_hqkv_lattice",
    "rms_normalize", "lattice_pool",
    "HQKV_FORM", "GAMMA_INIT", "LATTICE_WINDOW_DEFAULT",
]


def rms_normalize(x, eps=RMS_EPS):
    """把张量归一到单位 RMS（**无参数**、可微）。`(B,C,H,W) -> 同形状`。

    为什么必须留着：乘性门 `γ` 的梯度是 `∂L/∂γ = ⟨∂L/∂r, Δ⟩`，与 `‖Δ‖` 成正比。
    不归一化时 `Δ` 的量级取决于初始化/分辨率，门可能一开始就收不到有效梯度
    （Run8 v1 的教训）。归一到单位 RMS 后，门从第一步起就有固定量级的梯度信号。
    """
    r = x.pow(2).mean().clamp_min(0.0).sqrt()
    return x / (r + eps)


def lattice_pool(x, window=LATTICE_WINDOW_DEFAULT):
    """**固定格点**均值池化：`(B,C,H,W) -> (B,C,H/w,W/w)`。

    - 无参数、无可学习内容、不做 TopK/聚类/随机采样（预注册约束）；
    - 等价于把特征图切成 `window×window` 的 cell，一个 cell 出一个 token；
    - `H % window == 0` 且 `W % window == 0` 是硬前提，不满足直接报错（不做静默 padding）。
    """
    window = int(window)
    assert window >= 1, window
    B, C, H, W = x.shape
    if H % window or W % window:
        raise ValueError(
            f"[LATTICE] {H}×{W} 不能被 window={window} 整除；"
            f"格点压缩要求精确划分（拒绝静默 padding）")
    return F.avg_pool2d(x, kernel_size=window, stride=window)


class LatticeSATAttention(nn.Module):
    """一个 stage 的跨层 QKV：Q 取 T1 全分辨率，K/V 取 T2 的格点压缩 token。

    形状（以 stage1、输入 32×32、window=2、rank=32 为例）：

        x_q  (B,C,32,32) → q (B,32,32,32) → qf (B, N=1024, 32)
        x_kv (B,C,32,32) → lattice_pool → (B,C,16,16) → k/v → kf (B, 32, n=256)
        attn = softmax(qf @ kf · rank^-0.5)   → (B, 1024, 256)      # Q 保持 N，K/V 压到 n
        out  = attn @ vf → (B,1024,32) → (B,32,32,32) → proj → (B,out_dim,32,32) → RMS=1

    参数（`rank` 通道、`out_dim` 输出）：
    `q: C·rank`，`k: C·rank`，`v`: depthwise `C`（`groups=rank` 的 1×1，只学逐通道缩放），
    `proj: rank·out_dim`。stage3（C=768, rank=32, out=192）= 24,576+24,576+768+6,144 ≈ 56.1 k。
    """

    def __init__(self, dim, rank=32, window=LATTICE_WINDOW_DEFAULT,
                 out_dim=192, init_seed=HQKV_SEED, kv_from="b"):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.window = int(window)
        self.out_dim = int(out_dim)
        self.kv_from = str(kv_from)
        assert self.kv_from in ("a", "b"), self.kv_from
        assert self.dim % self.rank == 0, \
            f"dim={self.dim} 必须能被 rank={self.rank} 整除（depthwise V 的 groups）"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            self.q = nn.Conv2d(self.dim, self.rank, 1, bias=False)
            self.k = nn.Conv2d(self.dim, self.rank, 1, bias=False)
            self.v = nn.Conv2d(self.dim, self.rank, 1, groups=self.rank, bias=False)
            self.proj = nn.Conv2d(self.rank, self.out_dim, 1, bias=False)
            self.apply(self._init_weights)
        self.scale = self.rank ** -0.5

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_q, x_kv):
        """`x_q`/`x_kv` (B,C,H,W) → (B,out_dim,H,W)，输出 RMS = 1。"""
        kv_src = x_kv if self.kv_from == "b" else x_q
        pooled = lattice_pool(kv_src, window=self.window)
        k = self.k(pooled)                     # (B,r,h,w)
        v = self.v(pooled)                     # (B,r,h,w)
        q = self.q(x_q)                        # (B,r,H,W)

        B, r, H, W = q.shape
        qf = q.flatten(2).transpose(1, 2)                # (B, N, r)
        kf = k.flatten(2)                                # (B, r, n)
        vf = v.flatten(2).transpose(1, 2)                # (B, n, r)
        attn = torch.softmax(qf @ kf * self.scale, dim=-1)   # (B, N, n)
        out = (attn @ vf).transpose(1, 2).reshape(B, r, H, W)
        return rms_normalize(self.proj(out))

    def token_counts(self, hw):
        (h, w) = hw
        return {"q_tokens": int(h * w),
                "kv_tokens": int((h // self.window) * (w // self.window)),
                "window": self.window}

    def param_report(self):
        return {"dim": self.dim, "rank": self.rank, "window": self.window,
                "out_dim": self.out_dim, "kv_from": self.kv_from,
                "params": sum(p.numel() for p in self.parameters())}


class CAACPHQKVLattice(nn.Module):
    """跨 4 个 stage 的格点压缩 SAT-QKV + 每支独立 γ 门控（**无任何几何项**）。

    输出：

        out = s · Σ_i γ_i · rms(Δ_i),      s = 1/√(项数)   ← 固定结构常数，不参与梯度

    其中 `Δ_i` 是第 i 个 stage 的注意力输出（或 stage0 的 lite 融合输出），
    每个 `Δ_i` 先归一到单位 RMS 再乘门，`γ_i` 初值 0.5 且**互相独立**。

    ⚠️ **绝对不要在最后再对整个 `out` 做一次 `rms_normalize`**（Run8b 首版试过并撤回）：
    那会把 `γ` 的效果归一化掉，且 `∂L/∂γ` 会经由归一化被截断 —— 等于把自锁换层皮。
    """

    def __init__(self, stage_dims=(96, 192, 384, 768), rank=32,
                 window=LATTICE_WINDOW_DEFAULT,
                 attn_stages=ATTN_STAGES_DEFAULT, lite_dim=192,
                 init_seed=HQKV_SEED, gamma_init=GAMMA_INIT, kv_from="b"):
        super().__init__()
        self.stage_dims = tuple(int(d) for d in stage_dims)
        self.rank = int(rank)
        self.window = int(window)
        self.attn_stages = tuple(int(i) for i in attn_stages)
        self.lite_dim = int(lite_dim)
        self.kv_from = str(kv_from)
        self._diag = {}
        self._grad_snapshot = {}

        self.attn = nn.ModuleList()
        self.lite_fuse = nn.ModuleList()
        gamma_names = []
        for i, d in enumerate(self.stage_dims):
            if i in self.attn_stages:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(init_seed) + i)
                    self.attn.append(LatticeSATAttention(
                        d, rank=rank, window=window, out_dim=lite_dim,
                        init_seed=int(init_seed) + i, kv_from=kv_from))
                self.lite_fuse.append(None)
                gamma_names.append(f"stage{i}")
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(init_seed) + 16 + i)
                    fuse = nn.Sequential(
                        nn.Conv2d(3 * d, self.lite_dim, 1, bias=False),
                        nn.BatchNorm2d(self.lite_dim),
                        nn.SiLU(),
                        nn.Conv2d(self.lite_dim, self.lite_dim, 1, bias=False),
                    )
                    fuse.apply(self._init_fuse_weights)
                self.attn.append(None)
                self.lite_fuse.append(fuse)
                gamma_names.append(f"stage{i}_lite")

        # 每支**独立**门、**非零**初值 —— 这是 Run8b 反自锁的核心，与几何无关，保留
        self._gamma_names = tuple(gamma_names)
        self._gate_of = {i: (f"stage{i}" if i in self.attn_stages else f"stage{i}_lite")
                         for i in range(len(self.stage_dims))}
        for n in self._gamma_names:
            self.register_parameter(n, nn.Parameter(torch.full((1,), float(gamma_init))))

    @staticmethod
    def _init_fuse_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------ 诊断
    def gamma_dict(self):
        return {n: float(getattr(self, n).detach().sum()) for n in self._gamma_names}

    def snapshot_gamma_grad(self):
        """在 `loss.backward()` 之后调用：记录每支门梯度的绝对值和（自锁直接判据）。"""

        def g(p):
            return float(p.grad.detach().abs().sum()) if p.grad is not None else None

        self._grad_snapshot = {n: g(getattr(self, n)) for n in self._gamma_names}

    def score_maps(self):
        """**恒为空**。拆除几何后本模块不再产生任何变化分数。

        保留这个方法是**故意的**：调用方（`train.py` 的 HGSL 口、`casa_vmamba_net_v2`）
        如果还按老约定来取分数，会拿到 `{}` 而不是 `AttributeError`；
        而 `hgsl=1` 已经在网络构造时 **fail-closed 报错**，不会走到这里还不知道。
        """
        return {}

    def diag(self):
        return dict(self._diag)

    def param_report(self):
        per, total = {}, 0
        for i, d in enumerate(self.stage_dims):
            ent = {"dim": d}
            for name, mod in (("attn", self.attn[i]), ("lite_fuse", self.lite_fuse[i])):
                if mod is not None:
                    n = sum(p.numel() for p in mod.parameters())
                    ent[name] = n
                    total += n
            if self.attn[i] is not None:
                ent["window"] = self.attn[i].window
                ent["rank"] = self.attn[i].rank
            per[f"stage{i}"] = ent
        total += len(self._gamma_names)
        return {"form": HQKV_FORM, "rank": self.rank, "window": self.window,
                "attn_stages": list(self.attn_stages), "kv_from": self.kv_from,
                "gammas": self.gamma_dict(), "gamma_grad": dict(self._grad_snapshot),
                "gamma_init": float(getattr(self, self._gamma_names[0]).detach().sum()),
                "gamma_zero_init": False,
                "per_stage": per, "total_params": total,
                # 明确标注：本模块**不含任何几何/流形/半径**项
                "geometry": "none", "manifold": "none",
                "radius_decomposition": "none", "score_weighted_pooling": "none"}

    # ------------------------------------------------------------------
    def forward(self, feats_a, feats_b, inject_indices=None):
        """`feats_a`/`feats_b`：4 个 stage 的 (B,C_i,H_i,W_i)。返回 `(out, diag)`。

        `out`：`(B, lite_dim, H_0, W_0)`（H_0/W_0 = stage0 分辨率，即 1/4 输入）。
        """
        if inject_indices is None:
            inject_indices = tuple(range(len(self.stage_dims)))
        assert len(feats_a) == len(feats_b) == len(self.stage_dims)
        ref_hw = feats_a[0].shape[-2:]
        diag = {"stages": {}}
        accs = {}

        for i in range(len(self.stage_dims)):
            if i not in inject_indices:
                continue
            if i in self.attn_stages:
                out_i = self.attn[i](feats_a[i], feats_b[i])          # 已 RMS=1
                tc = self.attn[i].token_counts(feats_a[i].shape[-2:])
                diag["stages"][f"stage{i}"] = dict(tc, kind="lattice_sat_qkv")
            else:
                xa = feats_a[i].float()
                xb = feats_b[i].float()
                out_i = self.lite_fuse[i](torch.cat([xa, xb, (xa - xb).abs()], dim=1))
                out_i = rms_normalize(out_i)
                diag["stages"][f"stage{i}"] = {"kind": "lite_fuse",
                                               "q_tokens": int(feats_a[i].shape[-2:][0] *
                                                               feats_a[i].shape[-2:][1])}
            # 各 stage 分辨率不同 ⇒ 先在**各自尺度**上 RMS 归一化，再上采样到 ref_hw
            out_i = rms_normalize(out_i)
            if out_i.shape[-2:] != ref_hw:
                out_i = F.interpolate(out_i, size=ref_hw, mode="bilinear",
                                      align_corners=False)
            accs[i] = out_i

        out = feats_a[0].new_zeros(feats_a[0].shape[0], self.lite_dim, *ref_hw)
        parts = {}
        for i, acc in accs.items():
            gname = self._gate_of[i]
            gamma = getattr(self, gname)
            acc = rms_normalize(acc)
            out = out + gamma * acc
            parts[gname] = float(acc.detach().norm())
        # 固定结构缩放（不参与梯度）：n 项各 RMS=1、γ=0.5 ⇒ out 的 RMS ≈ 0.5·√n/√n = 0.5
        n_terms = max(1, len(parts))
        out = out * (1.0 / math.sqrt(n_terms))
        tot = sum(parts.values()) or 1.0
        for k, v in parts.items():
            diag[f"part_{k}_norm"] = v
            diag[f"part_{k}_share"] = v / tot

        self._diag = diag
        self._diag["gammas"] = self.gamma_dict()
        self._diag["contrib_norm"] = float(out.detach().norm())
        return out, diag


def build_caacp_hqkv_lattice(stage_dims=(96, 192, 384, 768), **kwargs):
    return CAACPHQKVLattice(stage_dims=stage_dims, **kwargs)
