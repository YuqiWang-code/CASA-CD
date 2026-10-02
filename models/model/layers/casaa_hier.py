"""HierCASAA：统一的变化感知非对称注意力模块（主线重构 · CASA-STR 创新点一）。

本地实施注意事项 §24：full / content / change 三种 context 构造共享同一套
qkv/proj/β 参数化，唯一变量 = K/V context 构造；不要复制多份实现。

参数化（与官方 SHViT SHSA 同构，便于论文表述）：
    qkv : Conv2d 1×1  C -> (qk_dim + qk_dim + C)   （单头，qk_dim=16）
    proj: Conv2d 1×1  C -> C
    pre_norm: GroupNorm(1, C)（SHSA 风格）
    beta : scalar（zero-init gate，F' = F + beta * out）

mode:
    full    : K = N（全注意力，N=256 @1/16）
    content : K = keep_ratio*N，全内容密度聚类（SAA-style 对照）
    change  : Kc = change_share*K 变化候选直保留 + Kb 稳定背景共享 assignment 聚合
              （change score 来自外部 1/8 双时相特征，参数自由）

复用 models/model/layers/casaa.py 的已验证机制：
    deterministic_density_assign / aggregate_with_shared_assignment /
    norm_preserve / rank_normalize_per_image / change_score_cosine。

P0（本地实施注意事项 §4）：只使用一个 zero-init 机制——qkv/proj 正常初始化 +
external beta=0；禁止同时把 proj BN gamma 置 0（会与 β 形成双零初始化死锁）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.casaa import (
    change_score_cosine,
    rank_normalize_per_image,
    deterministic_density_assign,
    aggregate_with_shared_assignment,
    norm_preserve,
)


def compute_f2_change_score(f2a, f2b):
    """1/8 特征的双时相变化 score（参数自由、T1/T2 对称、与 CASAA@1/16 对齐）。

    f2a/f2b: (B, C64, 32, 32) -> AvgPool2x2 -> (B, C, 16, 16) -> (B, 256, C)
    s = 1 - cos(f2a_i, f2b_i) -> rank-normalize per image。
    只用于 routing，不进入梯度图（调用方 no_grad）。
    """
    p1 = F.avg_pool2d(f2a, kernel_size=2, stride=2)      # (B, C, 16, 16)
    p2 = F.avg_pool2d(f2b, kernel_size=2, stride=2)
    p1 = p1.flatten(2).transpose(1, 2)
    p2 = p2.flatten(2).transpose(1, 2)
    s = 1.0 - F.cosine_similarity(p1, p2, dim=-1, eps=1e-8)   # (B, 256)
    return rank_normalize_per_image(s)


class HierCASAA(nn.Module):
    def __init__(self, dim=128, qk_dim=16, mode="change",
                 keep_ratio=0.25, change_share=0.5):
        super().__init__()
        assert mode in ("full", "content", "change")
        self.dim = dim
        self.qk_dim = qk_dim
        self.mode = mode
        self.keep_ratio = keep_ratio
        self.change_share = change_share
        self.scale = qk_dim ** -0.5

        self.pre_norm = nn.GroupNorm(1, dim)
        self.qkv = nn.Conv2d(dim, qk_dim + qk_dim + dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))

        self._routing = None

    # ------------------------------------------------------------------
    def _compute_contexts(self, x1, x2, f2a, f2b):
        """构造两个时相的 K/V context（(B, K, C) 各一）。routing 在 no_grad。"""
        B, C, H, W = x1.shape
        N = int(H * W)
        x1t = x1.flatten(2).transpose(1, 2)          # (B, N, C)
        x2t = x2.flatten(2).transpose(1, 2)

        K = max(int(round(self.keep_ratio * N)), 1) if self.mode != "full" else N
        Kc = min(int(round(self.change_share * K)), K) if self.mode == "change" else 0
        Kb = K - Kc

        with torch.no_grad():
            if self.mode == "full":
                c1, c2 = x1t, x2t
                assign = Ic = s = None
            else:
                if self.mode == "change":
                    assert f2a is not None and f2b is not None, "change mode needs 1/8 features"
                    s = compute_f2_change_score(f2a, f2b)         # (B, N)
                    Ic = torch.topk(s, Kc, dim=-1).indices        # (B, Kc)
                    mask = torch.ones(B, N, dtype=torch.long, device=x1.device)
                    mask.scatter_(1, Ic, 0)
                    mask = mask.bool()
                else:  # content
                    s = Ic = None
                    mask = torch.ones(B, N, dtype=torch.bool, device=x1.device)
                assign = None
                if Kb > 0:
                    x1_bg = x1t[mask]
                    x2_bg = x2t[mask]
                    z_bg = F.normalize((x1_bg + x2_bg) / 2.0, dim=-1).view(B, N - Kc, C)
                    assign = deterministic_density_assign(z_bg, Kb)

            # 聚合/收集：带梯度的原 feature
            if self.mode == "full":
                c1, c2 = x1t, x2t
            else:
                if Kc > 0:
                    c1_change = x1t.gather(1, Ic[..., None].expand(B, Kc, C))
                    c2_change = x2t.gather(1, Ic[..., None].expand(B, Kc, C))
                else:
                    c1_change = c2_change = None
                if Kb > 0:
                    x1_bg = x1t[mask].view(B, N - Kc, C)
                    x2_bg = x2t[mask].view(B, N - Kc, C)
                    agg1, agg2 = aggregate_with_shared_assignment(x1_bg, x2_bg, assign, Kb)
                    agg1 = norm_preserve(agg1, x1t)
                    agg2 = norm_preserve(agg2, x2t)
                else:
                    agg1 = agg2 = None
                parts1 = [p for p in (c1_change, agg1) if p is not None]
                parts2 = [p for p in (c2_change, agg2) if p is not None]
                c1 = torch.cat(parts1, dim=1) if parts1 else x1t[:, :0]
                c2 = torch.cat(parts2, dim=1) if parts2 else x2t[:, :0]

        self._routing = {"N": N, "K": K, "Kc": Kc, "Kb": Kb,
                         "s": s, "Ic": Ic, "assign": assign}
        return c1, c2, x1t, x2t

    # ------------------------------------------------------------------
    def forward(self, x1, x2, f2a=None, f2b=None):
        """paired CASAA：x1/x2 (B,C,H,W) -> y1/y2 (B,C,H,W)，N 不变。

        F' = F + beta * y（beta 由顶层网络施加；此处只返回 attention 输出）。
        """
        c1, c2, x1t, x2t = self._compute_contexts(x1, x2, f2a, f2b)
        B, C, H, W = x1.shape
        N = int(H * W)
        K = c1.shape[1]

        q1 = self.qkv(self.pre_norm(x1)).flatten(2)             # (B, 2q+C, N)
        q2 = self.qkv(self.pre_norm(x2)).flatten(2)
        q1, k1_full, v1_full = q1.split([self.qk_dim, self.qk_dim, C], dim=1)
        q2, k2_full, v2_full = q2.split([self.qk_dim, self.qk_dim, C], dim=1)

        # 上下文投影：权重同一 qkv（对 (B,C,1,K) 的 token 序列做 1×1 卷积）
        kv1 = F.conv2d(self.pre_norm(c1.transpose(1, 2).reshape(B, C, 1, K)),
                       self.qkv.weight, None).squeeze(2)         # (B, 2q+C, K)
        kv2 = F.conv2d(self.pre_norm(c2.transpose(1, 2).reshape(B, C, 1, K)),
                       self.qkv.weight, None).squeeze(2)
        _, k1c, v1c = kv1.split([self.qk_dim, self.qk_dim, C], dim=1)
        _, k2c, v2c = kv2.split([self.qk_dim, self.qk_dim, C], dim=1)

        if self.mode == "full":
            k1, v1, k2, v2 = k1_full, v1_full, k2_full, v2_full
        else:
            k1, v1, k2, v2 = k1c, v1c, k2c, v2c

        a1 = (q1.transpose(-2, -1) @ k1) * self.scale             # (B, N, K)
        a1 = a1.softmax(dim=-1)
        y1 = (v1 @ a1.transpose(-2, -1)).reshape(B, C, H, W)      # (B, C, N)
        a2 = (q2.transpose(-2, -1) @ k2) * self.scale
        a2 = a2.softmax(dim=-1)
        y2 = (v2 @ a2.transpose(-2, -1)).reshape(B, C, H, W)

        y1 = self.proj(y1)
        y2 = self.proj(y2)
        return y1, y2

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 128, 16, 16
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)
    f2a = torch.randn(B, 64, 32, 32)
    f2b = torch.randn(B, 64, 32, 32)
    for mode in ("full", "content", "change"):
        m = HierCASAA(dim=C, mode=mode)
        y1, y2 = m(x1, x2, f2a, f2b)
        r = m._routing
        assert y1.shape == (B, C, H, W), (mode, y1.shape)
        print(f"mode={mode}: N={r['N']} K={r['K']} Kc={r['Kc']} Kb={r['Kb']} out={tuple(y1.shape)}")
    # change routing 对称性
    m = HierCASAA(dim=C, mode="change")
    _, _ = m(x1, x2, f2a, f2b)
    r1 = m._routing
    _, _ = m(x2, x1, f2b, f2a)
    r2 = m._routing
    assert torch.equal(r1["Ic"], r2["Ic"]) and torch.equal(r1["assign"], r2["assign"])
    print("change routing swap-symmetric OK")
    print("HierCASAA mechanics smoke ALL OK")
