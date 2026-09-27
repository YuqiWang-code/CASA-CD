"""SAA (Selective Aggregation Attention) — 从 SAT (CVPR 2026) 官方代码提取的独立版。

来源：https://github.com/PhuTran1005/SAT  （basicsr/archs/sat_arch.py）
许可：随上游仓库（MIT，见 LICENSE 说明）。

CASA-CD 的用途：这是 CASAA（Change-Aware Asymmetric Token Modeling）的直接模板——
「完整 Query + 压缩 K/V」的非对称注意力：

    attn = softmax(Q_full @ K_comp^T / sqrt(d)) @ V_comp
    Q: (B, N, cr)   —— 全部 token，逐位置判别能力完整保留
    K/V: (B, K, ...) —— 由 cluster_and_merge 聚类合并后的压缩上下文，K = M*N << N
    复杂度 O(N^2) -> O(NK)

已做的简化（相对上游）：
- 去掉 timm / einops / basicsr 依赖，仅用 torch；
- 保留算法原样（聚类采样、密度峰值选心、加权合并、范数保持均未改动）；
- 附一个 __main__ 冒烟测试：构造 SAA 并对比全注意力 FLOPs。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cluster_and_merge(x, cluster_num, subsample_factor=4):
    """把 N 个 token 聚类合并为 cluster_num 个「背景/上下文」token（SAT 原文实现）。

    算法：密度峰值聚类（density-peak）在 S=min(N, max(2K, 4K)) 个子采样点上选 K 个中心，
    再按余弦相似度把全部 token 分配到最近中心，最后每个簇做加权平均。
    """
    B, N, C = x.shape
    device = x.device
    K = cluster_num

    x_proj = x
    x_norm = F.normalize(x_proj, dim=-1)  # (B, N, C)

    S = min(N, max(2 * K, subsample_factor * K))  # S >= 2K，上限 N

    samples_per_region = S // K
    sub_idx = []
    for i in range(K):
        start_idx = i * (N // K)
        end_idx = (i + 1) * (N // K) if i < K - 1 else N
        region_size = end_idx - start_idx
        n_samples = min(samples_per_region, region_size)
        if region_size > 0:
            region_perm = torch.randperm(region_size, device=device)[:n_samples]
            sub_idx.append(start_idx + region_perm)

    sub_idx = torch.cat(sub_idx)
    if len(sub_idx) < S:
        remaining = S - len(sub_idx)
        all_idx = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[sub_idx] = False
        additional = all_idx[mask][torch.randperm((~mask).sum(), device=device)[:remaining]]
        sub_idx = torch.cat([sub_idx, additional])

    x_norm_sub = x_norm[:, sub_idx]  # (B, S, C)

    # 子采样点之间的余弦相似度
    sim_sub = x_norm_sub @ x_norm_sub.transpose(1, 2)  # (B, S, S)
    torch.diagonal(sim_sub, dim1=1, dim2=2).fill_(-1)

    k = min(K, S - 1)
    sim_topk_sub, _ = torch.topk(sim_sub, k=k, dim=-1)
    density_sub = sim_topk_sub.mean(dim=-1)  # (B, S) 局部密度 ρ
    density_sub = density_sub + torch.rand_like(density_sub) * 1e-6

    mask_higher_density = (density_sub[:, None, :] > density_sub[:, :, None]).float()
    masked_sim_sub = sim_sub * mask_higher_density - 1e9 * (1.0 - mask_higher_density)
    max_sim_to_higher, _ = masked_sim_sub.max(dim=-1)

    delta_sub = 1.0 - max_sim_to_higher  # 距离 δ
    max_density_mask_sub = (mask_higher_density.sum(dim=-1) == 0)
    min_sim_global = sim_sub.min(dim=-1)[0]
    delta_sub[max_density_mask_sub] = (1.0 - min_sim_global)[max_density_mask_sub]
    delta_sub = torch.clamp(delta_sub, min=0.0)

    score_sub = density_sub * delta_sub  # γ = ρ × δ
    _, center_idx_in_sub = torch.topk(score_sub, k=K, dim=-1)
    center_idx = sub_idx[center_idx_in_sub]  # (B, K)

    centers_norm = torch.gather(
        x_norm, 1, center_idx[..., None].expand(B, K, x_norm.shape[-1]))

    sim_token_center = x_norm @ centers_norm.transpose(1, 2)  # (B, N, K)
    assign_idx = sim_token_center.argmax(dim=-1)  # (B, N)

    out = x.new_zeros(B, K, C)
    one_hot = F.one_hot(assign_idx, num_classes=K).type_as(x)  # (B, N, K)
    cluster_counts = one_hot.sum(dim=1, keepdim=True).clamp(min=1e-6)
    out = torch.einsum("bnc,bnk->bkc", x, one_hot) / cluster_counts.transpose(1, 2)

    return out


class SAA(nn.Module):
    """Selective Aggregation Attention（SAT 原文实现，仅去掉框架依赖）。

    完整 Query（通道压缩比 c_ratio），K/V 来自聚类合并后的 M*N 个压缩 token。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., c_ratio=0.5, M=0.03):
        super(SAA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.cr = int(dim * c_ratio)
        self.scale = qk_scale or (self.cr // num_heads) ** -0.5
        self.M = M  # 压缩后 token 占比 NF = M*N

        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H=None, W=None, prev_attn=None, image=None):
        B, N, C = x.shape
        T = x

        NF = int(self.M * N)

        # 聚类合并背景 token → 压缩 KV
        T_avg = cluster_and_merge(T, NF)
        # 范数保持
        norms = torch.norm(T, dim=-1)
        max_norm = norms.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        avg_norm = torch.norm(T_avg, dim=-1, keepdim=True)
        epsilon = 1e-6
        mask = avg_norm > epsilon
        scaled = (T_avg / (avg_norm + epsilon)) * max_norm
        T_avg = torch.where(mask, scaled, T_avg)

        KV_comp = T_avg
        K_size = KV_comp.shape[1]

        # 交叉注意力：完整 Q × 压缩 K/V
        q = self.q(x).reshape(B, N, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(KV_comp).reshape(B, K_size, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(KV_comp).reshape(B, K_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


if __name__ == "__main__":
    # 冒烟测试：随机 token 上验证形状与 FLOPs 对比
    torch.manual_seed(0)
    B, N, C, heads, M = 2, 256, 192, 6, 0.03
    x = torch.randn(B, N, C)

    saa = SAA(dim=C, num_heads=heads, c_ratio=0.5, M=M)
    out = saa(x)
    assert out.shape == (B, N, C), f"bad shape {out.shape}"
    print(f"SAA 输出形状 OK: {tuple(out.shape)}，压缩 token 数 NF={int(M * N)}（N={N}）")

    def flops_full_attn():
        q = k = v = N
        d = C
        return 2 * B * N * N * d  # qk^T + attn@v

    def flops_saa():
        K = int(M * N)
        d = C
        cr = C // 2
        return 2 * B * N * K * cr + B * N * K * d  # q@k^T (cr 维) + attn@v (C 维)

    print(f"全注意力 FLOPs ≈ {flops_full_attn() / 1e6:.2f} M")
    print(f"SAA FLOPs     ≈ {flops_saa() / 1e6:.2f} M（不含聚类开销）")
    print("SAA smoke test OK")
