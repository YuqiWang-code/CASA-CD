"""CASAA (Change-Aware Selective Aggregation Attention) — CASA-CD 创新主线一。

「完整 Query + 变化感知压缩 K/V」的非对称注意力：

    Y_t = Softmax(Q_t @ K_t^T / sqrt(d)) @ V_t,  t in {1, 2}
    Q_t: (B, N, C)     —— N 个 query 完整保留（逐位置判别能力不变）
    K_t/V_t: (B, K, C) —— K = keep_ratio * N 个压缩上下文 token，K << N
    注意力交互量：O(N^2) -> O(NK)（routing 之外）

与 SAT SAA（others/SAT/saa.py）的关系——复用其思想，不照搬参数化：
  - 复用：density-peak 选心、token-to-center 分配、簇内均值、Full Q / 压缩 K/V
    的注意力结构、norm preservation。
  - 改动：deterministic stratified sampling（eval routing 与 RNG 无关）；
    双时相共享路由；疑似变化 token 原位保留 + 稳定背景 token 强聚合；
    原 DeiT-Tiny fused qkv 权重切片（qkv/proj 参数名与形状不变，零新增参数）。

router 模式：
  - 'change'  (CASAA, A2)：Kc=change_share*K 个 change-score TopK token 原位保留，
    Kb=K-Kc 个稳定背景 token 按共享内容聚类；C_t = [X_t[Ic]; mean_cluster(X_t_bg)]。
    change score = 1 - cos(x1_i, x2_i)（参数自由、T1/T2 对称）。
  - 'content' (SAA-style 对照, A1)：Kc=0，全部 token 按共享内容聚类成 K 个原型。
  - 'oracle'  (A3, 仅诊断)：change score 直接用 GT patch occupancy
    （Encoder 从 label 计算后以 score_hint 传入）。用于把「机制问题」与
    「scorer 问题」拆开——DIAGNOSTIC-ONLY，不可部署，绝不作方法结果。

设计要点：
  - 切片 qkv 投影：对 full X 只算 Q，对压缩上下文只算 K/V（权重仍为原 fused
    qkv 的行切片，数学与原「整段 qkv 后切片」等价，不白算 full K/V 与 compressed Q）。
  - routing 索引（TopK / assignment）在 torch.no_grad() 下计算；聚合/收集仍使用
    带梯度的原 feature tensor，梯度可回传到背景 token。
  - _routing 记录上一次 forward_pair 的诊断统计（score / 索引 / 聚类大小 /
    bank 范数 / 注意力质量），供 analyse/casaa_router_diagnostic.py 做 Router Audit。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention


def change_score_cosine(x1, x2):
    """参数自由的双时相变化显著性：s_i = 1 - cos(x1_i, x2_i)。

    Args:
        x1, x2: (B, N, C)
    Returns:
        s: (B, N)，越大越可能是变化区域。
    """
    x1_n = F.normalize(x1, dim=-1)
    x2_n = F.normalize(x2, dim=-1)
    return 1.0 - (x1_n * x2_n).sum(dim=-1)


def deterministic_stratified_subindex(M, K, S, device):
    """从 M 个 token 里确定性分层取 S 个子采样点（每区均匀分布，与 RNG 无关）。

    只使用整数运算（对 fvcore tracer 友好）。
    """
    samples_per_region = S // K
    sub = []
    for i in range(K):
        start = i * (M // K)
        end = (i + 1) * (M // K) if i < K - 1 else M
        region_size = end - start
        if region_size > 0:
            n = min(samples_per_region, region_size)
            if n == 1:
                pos = torch.zeros(1, dtype=torch.long, device=device)
            else:
                j = torch.arange(n, dtype=torch.long, device=device)
                pos = j * (region_size - 1) // (n - 1)   # 整数除法，均匀分布
            sub.append(start + pos)
    sub_idx = torch.cat(sub) if sub else torch.zeros(0, dtype=torch.long, device=device)
    if len(sub_idx) < S:
        all_idx = torch.arange(M, device=device)
        mask = torch.ones(M, dtype=torch.bool, device=device)
        mask[sub_idx] = False
        sub_idx = torch.cat([sub_idx, all_idx[mask][: S - len(sub_idx)]])
    return sub_idx


def deterministic_density_assign(z, K):
    """在 M 个（已 L2 归一化的）token 上做确定性 density-peak 聚类。

    算法结构沿用 SAT 的 cluster_and_merge，但子采样是确定性的（分层均匀采样），
    且只在 background 集合上执行。

    Args:
        z: (B, M, C) 归一化 token（Run1 为共享背景描述 Norm((x1+x2)/2)）。
        K: 聚类数。
    Returns:
        assign_idx: (B, M) long，每个 token 所属 cluster id ∈ [0, K)。
    """
    B, M, C = z.shape
    # 显式转 int：JIT 追踪下 shape 是符号量，Python 侧整数运算（range/min//）需要真 int
    B, M, C = int(B), int(M), int(C)
    device = z.device
    assert M >= K, f"density assign needs M({M}) >= K({K})"

    S = min(M, max(2 * K, 4 * K))
    sub_idx = deterministic_stratified_subindex(M, K, S, device)

    z_sub = z[:, sub_idx]                      # (B, S, C)
    sim_sub = z_sub @ z_sub.transpose(1, 2)    # 余弦相似度 (B, S, S)
    torch.diagonal(sim_sub, dim1=1, dim2=2).fill_(-1)

    k = min(K, S - 1)
    sim_topk_sub, _ = torch.topk(sim_sub, k=k, dim=-1)
    density_sub = sim_topk_sub.mean(dim=-1)    # (B, S) 局部密度 rho
    # 确定性 tie-break（SAT 用 rand*1e-6 随机扰动；这里用与索引相关的固定扰动）
    density_sub = density_sub + torch.arange(S, device=device, dtype=z.dtype) * 1e-6

    mask_higher = (density_sub[:, None, :] > density_sub[:, :, None]).float()
    masked_sim = sim_sub * mask_higher - 1e9 * (1.0 - mask_higher)
    max_sim_to_higher, _ = masked_sim.max(dim=-1)

    delta_sub = 1.0 - max_sim_to_higher          # 距离 delta
    max_density_mask = (mask_higher.sum(dim=-1) == 0)
    min_sim_global = sim_sub.min(dim=-1)[0]
    delta_sub[max_density_mask] = (1.0 - min_sim_global)[max_density_mask]
    delta_sub = delta_sub.clamp(min=0.0)

    score_sub = density_sub * delta_sub          # gamma = rho * delta
    _, center_idx_in_sub = torch.topk(score_sub, k=K, dim=-1)
    center_idx = sub_idx[center_idx_in_sub]      # (B, K)

    centers = z.gather(1, center_idx[..., None].expand(B, K, C))
    sim_token_center = z @ centers.transpose(1, 2)   # (B, M, K)
    assign_idx = sim_token_center.argmax(dim=-1)     # (B, M)
    return assign_idx


def aggregate_with_shared_assignment(x1, x2, assign_idx, K):
    """按共享 assignment 对两个时相分别做簇内均值（特征不混时相）。

    Args:
        x1, x2: (B, M, C) 两个时相的 background token。
        assign_idx: (B, M) 共享 cluster id。
        K: 聚类数。
    Returns:
        agg1, agg2: (B, K, C)
    """
    B, M, C = x1.shape
    one_hot = F.one_hot(assign_idx, num_classes=K).type_as(x1)   # (B, M, K)
    counts = one_hot.sum(dim=1, keepdim=True).clamp(min=1e-6)    # (B, 1, K)
    agg1 = torch.einsum("bmc,bmk->bkc", x1, one_hot) / counts.transpose(1, 2)
    agg2 = torch.einsum("bmc,bmk->bkc", x2, one_hot) / counts.transpose(1, 2)
    return agg1, agg2


def norm_preserve(agg, ref):
    """把聚合原型范数拉回原 token 的最大范数（沿用 SAT 的 norm preservation）。

    聚合平均会压低原型范数，用原 token 最大范数对其重标定。
    """
    norms = torch.norm(ref, dim=-1)                                  # (B, M)
    max_norm = norms.max(dim=-1, keepdim=True)[0].unsqueeze(-1)      # (B, 1, 1)
    avg_norm = torch.norm(agg, dim=-1, keepdim=True)                 # (B, K, 1)
    eps = 1e-6
    scaled = (agg / (avg_norm + eps)) * max_norm
    mask = avg_norm > eps
    return torch.where(mask, scaled, agg)


class CASAAAttention(Attention):
    """变化感知非对称注意力：完整 Q、压缩 K/V；qkv/proj 与 DeiT 完全同名同形。

    继承 Attention 以保留 fused `qkv`（C -> 3C）与 `proj` 的参数名和形状，
    DeiT-Tiny 预训练 Q/K/V 权重可原位加载；单时相 `forward(x)` 仍是原
    self-attention（用于回归/对照）。

    前向只做切片投影：对 full X 只算 Q、对压缩上下文只算 K/V（权重是 fused qkv
    的行切片，数学与整段 qkv 后切片等价，不白算 full K/V 与 compressed Q）。

    Args:
        dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop: 与 Attention 一致。
        keep_ratio: K/N，K/V 压缩 token 总数比例（Run1/2: 0.25）。
        change_share: K 中「疑似变化 token 直保留」占比（router='change'/'oracle' 时）。
        router: 'change'（cosine score）/ 'content'（A1 对照）/ 'oracle'（GT 诊断）。
        norm_preserve: 是否对聚合原型做范数保持。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0,
                 keep_ratio=0.25, change_share=0.5, router='change',
                 norm_preserve=True):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                         proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        assert router in ('change', 'content', 'oracle'), f"unknown router {router}"
        self.dim = dim
        self.keep_ratio = keep_ratio
        self.change_share = change_share
        self.router = router
        self.norm_preserve = norm_preserve
        self._routing = None   # 上一次 forward_pair 的 routing 统计（供 smoke/诊断）

    # ------------------------------------------------------------------
    # 切片 qkv 投影（Run2 等价优化：只算需要的 Q / K / V）
    # ------------------------------------------------------------------
    def _proj_q(self, x):
        """只算 full X 的 Q：qkv.weight[0:C] / qkv.bias[0:C]。"""
        w = self.qkv.weight[: self.dim]
        b = self.qkv.bias[: self.dim] if self.qkv.bias is not None else None
        return F.linear(x, w, b)

    def _proj_kv(self, c):
        """只算压缩上下文的 K/V：qkv.weight[C:3C] / qkv.bias[C:3C]。

        Returns:
            kv: (B, K, 2*C)，[..., :C] 为 K，[..., C:] 为 V。
        """
        w = self.qkv.weight[self.dim:]
        b = self.qkv.bias[self.dim:] if self.qkv.bias is not None else None
        return F.linear(c, w, b)

    # ------------------------------------------------------------------
    # 变化 score 接口
    # ------------------------------------------------------------------
    def _change_score(self, x1, x2, score_hint):
        if self.router == 'oracle':
            # DIAGNOSTIC-ONLY：score 来自 GT patch occupancy（Encoder 传入）
            assert score_hint is not None, "oracle router requires the GT score_hint"
            return score_hint
        return change_score_cosine(x1, x2)

    # ------------------------------------------------------------------
    # 路由与上下文构造
    # ------------------------------------------------------------------
    def compute_contexts(self, x1, x2, score_hint=None):
        """计算两个时相的压缩上下文 (C1, C2) 与 routing 统计。

        routing（change score / TopK / assignment）在 no_grad 下计算；
        聚合与收集仍用带梯度的原 feature，梯度可回传。
        """
        B, N, C = x1.shape
        # 显式转 int：JIT 追踪下 shape 是符号量，Python 侧整数运算需要真 int
        B, N, C = int(B), int(N), int(C)
        K = max(int(round(self.keep_ratio * N)), 1)
        if self.router in ('change', 'oracle'):
            Kc = min(int(round(self.change_share * K)), K)
        else:
            Kc = 0
        Kb = K - Kc

        with torch.no_grad():
            s = self._change_score(x1, x2, score_hint)   # (B, N)
            if Kc > 0:
                Ic = torch.topk(s, Kc, dim=-1).indices   # (B, Kc) 疑似变化位置
                # int scatter（fvcore tracer 不支持 bool scalar 的 scatter_.value）
                mask = torch.ones(B, N, dtype=torch.long, device=x1.device)
                mask.scatter_(1, Ic, 0)
                mask = mask.bool()
            else:
                Ic = None
                mask = torch.ones(B, N, dtype=torch.bool, device=x1.device)

            assign = None
            if Kb > 0:
                # 共享背景描述：只在背景 token 上聚类（仅用于算 assignment）
                x1_bg_n = x1[mask]                       # (B*Nbg, C)
                x2_bg_n = x2[mask]
                z_bg = F.normalize((x1_bg_n + x2_bg_n) / 2.0, dim=-1).view(B, N - Kc, C)
                assign = deterministic_density_assign(z_bg, Kb)   # (B, Nbg)

        # 聚合/收集：使用带梯度的原 feature tensor
        if Ic is not None:
            c1_change = x1.gather(1, Ic[..., None].expand(B, Kc, C))
            c2_change = x2.gather(1, Ic[..., None].expand(B, Kc, C))
        if Kb > 0:
            x1_bg = x1[mask].view(B, N - Kc, C)
            x2_bg = x2[mask].view(B, N - Kc, C)
            agg1, agg2 = aggregate_with_shared_assignment(x1_bg, x2_bg, assign, Kb)
            if self.norm_preserve:
                agg1 = norm_preserve(agg1, x1)
                agg2 = norm_preserve(agg2, x2)
            if Ic is not None:
                C1 = torch.cat([c1_change, agg1], dim=1)   # (B, K, C)
                C2 = torch.cat([c2_change, agg2], dim=1)
            else:
                C1, C2 = agg1, agg2
        else:
            C1, C2 = c1_change, c2_change

        # 诊断统计（Router Audit 用；no_grad 下已是常量）
        cluster_size = None
        if assign is not None:
            cluster_size = F.one_hot(assign, num_classes=Kb).sum(dim=1).float()   # (B, Kb)
        c_norm_mean = torch.norm(C1[:, :Kc], dim=-1).mean().item() if Kc > 0 else float("nan")
        bg_norm_mean = torch.norm(C1[:, Kc:], dim=-1).mean().item() if Kb > 0 else float("nan")

        self._routing = {"N": N, "K": K, "Kc": Kc, "Kb": Kb,
                         "s": s, "Ic": Ic, "assign": assign,
                         "cluster_size": cluster_size,
                         "c_norm_mean": c_norm_mean, "bg_norm_mean": bg_norm_mean,
                         "attn_mass_change": None, "attn_mass_bg": None}
        return C1, C2

    def forward_pair(self, x1, x2, score_hint=None):
        """双时相 paired attention：Q 完整，K/V 来自压缩上下文，输出 N 不变。

        Args:
            x1, x2: (B, N, C)
            score_hint: (B, N) 外部变化 score（仅 router='oracle' 使用）。
        Returns:
            y1, y2: (B, N, C)
        """
        B, N, C = x1.shape
        C1, C2 = self.compute_contexts(x1, x2, score_hint=score_hint)
        K = C1.shape[1]
        H = self.num_heads
        hd = C // H
        Kc = self._routing["Kc"]

        # 切片投影：full X -> Q only；压缩上下文 -> K/V only
        q1 = self._proj_q(x1).reshape(B, N, H, hd).permute(0, 2, 1, 3) * self.scale
        q2 = self._proj_q(x2).reshape(B, N, H, hd).permute(0, 2, 1, 3) * self.scale
        kv1 = self._proj_kv(C1).reshape(B, K, 2, H, hd).permute(2, 0, 3, 1, 4)
        kv2 = self._proj_kv(C2).reshape(B, K, 2, H, hd).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]

        a1 = (q1 @ k1.transpose(-2, -1))
        a1_s = a1.softmax(dim=-1)
        a1 = self.attn_drop(a1_s)
        y1 = (a1 @ v1).transpose(1, 2).reshape(B, N, C)
        a2 = (q2 @ k2.transpose(-2, -1))
        a2_s = a2.softmax(dim=-1)
        a2 = self.attn_drop(a2_s)
        y2 = (a2 @ v2).transpose(1, 2).reshape(B, N, C)

        y1 = self.proj_drop(self.proj(y1))
        y2 = self.proj_drop(self.proj(y2))

        # 注意力质量统计：query 对 change bank / bg bank 的平均 attention mass
        m_change = (a1_s[..., :Kc].mean() + a2_s[..., :Kc].mean()) / 2.0 if Kc > 0 else None
        m_bg = (a1_s[..., Kc:].mean() + a2_s[..., Kc:].mean()) / 2.0 if K > Kc else None
        self._routing["attn_mass_change"] = float(m_change.item()) if m_change is not None else float("nan")
        self._routing["attn_mass_bg"] = float(m_bg.item()) if m_bg is not None else float("nan")
        return y1, y2


if __name__ == "__main__":
    # CPU 单 block 机制冒烟（无需 GPU / 预训练权重）：
    #   cd <repo_root> && python -m models.model.layers.casaa
    torch.manual_seed(0)
    B, N, C, H = 2, 256, 192, 6
    x1 = torch.randn(B, N, C, requires_grad=True)
    x2 = torch.randn(B, N, C, requires_grad=True)

    print("== change score ==")
    s12 = change_score_cosine(x1, x2)
    s21 = change_score_cosine(x2, x1)
    assert torch.allclose(s12, s21), "change score must be T1/T2 symmetric"
    print(f"  symmetric OK, shape {tuple(s12.shape)}")

    print("== qkv sliced projection equivalence ==")
    attn = CASAAAttention(dim=C, num_heads=H, qkv_bias=True)
    with torch.no_grad():
        # 参考实现：整段 qkv 后切片（旧逻辑）
        def ref_forward_pair(a, xa, xb):
            Ca, Cb = a.compute_contexts(xa, xb)
            Kref = Ca.shape[1]
            qkv1 = a.qkv(xa).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
            qkv2 = a.qkv(xb).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
            qkv1c = a.qkv(Ca).reshape(B, Kref, 3, H, C // H).permute(2, 0, 3, 1, 4)
            qkv2c = a.qkv(Cb).reshape(B, Kref, 3, H, C // H).permute(2, 0, 3, 1, 4)
            q1, k1, v1 = qkv1[0] * a.scale, qkv1[1], qkv1[2]
            q2, k2, v2 = qkv2[0] * a.scale, qkv2[1], qkv2[2]
            _, k1c, v1c = qkv1c
            _, k2c, v2c = qkv2c
            y1 = ((q1 @ k1c.transpose(-2, -1)).softmax(-1) @ v1c).transpose(1, 2).reshape(B, N, C)
            y2 = ((q2 @ k2c.transpose(-2, -1)).softmax(-1) @ v2c).transpose(1, 2).reshape(B, N, C)
            return a.proj_drop(a.proj(y1)), a.proj_drop(a.proj(y2))

        for r in ("change", "content"):
            aa = CASAAAttention(dim=C, num_heads=H, qkv_bias=True, router=r)
            aa.load_state_dict(attn.state_dict())
            y1, y2 = aa.forward_pair(x1, x2)
            yr1, yr2 = ref_forward_pair(aa, x1, x2)
            d = max((y1 - yr1).abs().max().item(), (y2 - yr2).abs().max().item())
            assert d < 1e-6, f"router={r} sliced vs full qkv mismatch: {d}"
            print(f"  router={r}: sliced == full-qkv, max_abs_err = {d:.2e}")

    print("== CASAA (router=change) ==")
    y1, y2 = attn.forward_pair(x1, x2)
    assert y1.shape == y2.shape == (B, N, C), (y1.shape, y2.shape)
    r = attn._routing
    print(f"  N={r['N']} K={r['K']} Kc={r['Kc']} Kb={r['Kb']}  output {tuple(y1.shape)}")
    assert (r["N"], r["K"], r["Kc"], r["Kb"]) == (256, 64, 32, 32)
    assert r["s"].shape == (B, N) and r["cluster_size"].shape == (B, 32)
    assert r["attn_mass_change"] is not None and r["attn_mass_bg"] is not None
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    # 确定性：同一输入两次 routing 一致
    _ = attn.forward_pair(x1, x2)
    r2 = attn._routing
    assert torch.equal(r["Ic"], r2["Ic"]) and torch.equal(r["assign"], r2["assign"])
    # T1/T2 交换对称：routing 一致
    _ = attn.forward_pair(x2, x1)
    r3 = attn._routing
    assert torch.equal(r["Ic"], r3["Ic"]) and torch.equal(r["assign"], r3["assign"])
    print("  deterministic + swap-symmetric routing OK")
    # 梯度回传（qkv / proj / 上游 token）
    loss = (y1 * torch.randn_like(y1)).sum() + (y2 * torch.randn_like(y2)).sum()
    loss.backward()
    assert attn.qkv.weight.grad is not None and attn.qkv.weight.grad.norm() > 0
    assert attn.proj.weight.grad is not None and attn.proj.weight.grad.norm() > 0
    assert torch.isfinite(attn.qkv.weight.grad).all() and torch.isfinite(attn.proj.weight.grad).all()
    assert x1.grad is not None and x1.grad.norm() > 0 and torch.isfinite(x1.grad).all()
    print("  gradients OK (qkv/proj/upstream tokens finite & nonzero)")

    print("== SAA-style control (router=content) ==")
    attn_c = CASAAAttention(dim=C, num_heads=H, qkv_bias=True, router='content')
    with torch.no_grad():
        _, _ = attn_c.forward_pair(x1, x2)
    rc = attn_c._routing
    print(f"  N={rc['N']} K={rc['K']} Kc={rc['Kc']} Kb={rc['Kb']}")
    assert (rc["K"], rc["Kc"], rc["Kb"]) == (64, 0, 64)
    assert rc["Ic"] is None

    print("== Oracle router (DIAGNOSTIC-ONLY) ==")
    attn_o = CASAAAttention(dim=C, num_heads=H, qkv_bias=True, router='oracle')
    hint = torch.rand(B, N)
    with torch.no_grad():
        _, _ = attn_o.forward_pair(x1, x2, score_hint=hint)
        ro1 = attn_o._routing
        _, _ = attn_o.forward_pair(x2, x1, score_hint=hint)
        ro2 = attn_o._routing
    assert (ro1["K"], ro1["Kc"], ro1["Kb"]) == (64, 32, 32)
    assert torch.equal(ro1["s"], hint), "oracle score must equal the GT hint"
    assert torch.equal(ro1["Ic"], torch.topk(hint, 32, dim=-1).indices)
    assert torch.equal(ro1["Ic"], ro2["Ic"]) and torch.equal(ro1["assign"], ro2["assign"])
    try:
        attn_o.forward_pair(x1, x2)   # 无 hint 必须报错
        raise AssertionError("oracle without hint should raise")
    except AssertionError:
        pass
    print("  oracle score == hint, TopK correct, swap-invariant, requires hint")

    print("== single-phase forward (pretrained-path regression) ==")
    y = attn(x1)
    assert y.shape == (B, N, C)
    print("  OK")
    print("CASAA mechanics smoke ALL OK")
