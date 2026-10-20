#!/usr/bin/env python
"""Run9 HSAM CPU 回归门：数学正确性 + **退化性** + 反静默 + fail-closed。

    cd models && python test_hsam_cpu.py

不训练、不需要数据集、不需要 GPU（CUDA 上的真实特征 smoke 走服务器脚本
`train_scripts/CASA-TViM/Run9/_r9_smoke.sh`）。

## 为什么这个测试的重点是"退化"

Run8 的教训：`contrib_norm` 恒为 0 却让训练白跑 4 个 run。
HSAM 是**零参数确定算子**，不会"学不出来"，但它有另一类静默失败：
论文 Eq.(6)(7) 的字面形式在 LayerNorm 特征上**注意力熵 = 1.0000**（完全均匀），
此时模块等价于均值池化、`agg/feat ≈ 1.9e-4`（对本项目等于零）。
所以本测试把"论文原式必然退化"写成一条**固定断言**（H4），
把"实际用的 std 模式必须非退化"写成另一条（H5）。任何一边变了都会当场报警。
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model.layers.hsam import (  # noqa: E402
    ATTN_STAGES_DEFAULT,
    CURVATURE_C,
    ENT_MAX_OK,
    HSAM,
    HSAM_FORM,
    expmap0,
    hsam_logits,
    logmap0,
    poincare_pairwise_distance,
)

DIMS = (96, 192, 384, 768)
HW = (64, 32, 16, 8)
_total, _fail, _skip = [], [], []


def check(name, cond, detail=""):
    _total.append(name)
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        _fail.append(name)
    return ok


def norm_feats(seed=0, C=None, H=None, n=2, requires_grad=False):
    """模拟 LayerNorm2d 的输出：逐通道标准化 ⇒ token 范数几乎恒定（HSAM 的最坏情形）。"""
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(n, C, H, H, generator=g)
    mu = t.mean(dim=(2, 3), keepdim=True)
    sd = t.std(dim=(2, 3), keepdim=True)
    t = (t - mu) / (sd + 1e-5)
    return t.requires_grad_(requires_grad)


def main():
    print("=== Run9 HSAM CPU 回归门 ===")

    # ---------------- A. 数学正确性 ----------------
    print("\n[A] Poincaré 球运算")
    v = torch.tensor([[0.3, -0.4, 0.0], [1.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    p = expmap0(v, dim_scale=1.0)
    check("A1 exp 映射后范数严格 < 1（落在球内）",
          bool((p.norm(dim=-1) < 1.0).all()),
          f"{[round(float(x), 6) for x in p.norm(dim=-1)]}")
    check("A2 零向量映射到原点", float(p[2].norm()) < 1e-12, f"{float(p[2].norm()):.2e}")

    # 闭式：exp(v) = tanh(‖v‖)·v/‖v‖
    r = v[1].norm()
    closed = math.tanh(float(r)) * v[1] / float(r)
    check("A3 exp 与闭式 tanh(‖v‖)·v/‖v‖ 一致",
          float((p[1] - closed).abs().max()) < 1e-6,
          f"maxdiff={float((p[1] - closed).abs().max()):.2e}")

    # logmap 是 expmap 的逆（1-D 情形）
    x = torch.tensor([0.0, 0.1, 0.5, 0.9])
    check("A4 logmap(expmap(x)) ≈ x", float((logmap0(expmap0(x.unsqueeze(-1), 1.0).squeeze(-1)) - x).abs().max()) < 1e-6,
          f"maxdiff={float((logmap0(expmap0(x.unsqueeze(-1), 1.0).squeeze(-1)) - x).abs().max()):.2e}")

    # 距离：自身为 0、对称、闭式一致
    q = expmap0(torch.tensor([[0.3, -0.4, 0.0], [0.1, 0.1, 0.1]]), 1.0)   # (2,3)
    qa = q[:1].unsqueeze(1)      # (1,1,3) 当作 B=1,N=1
    qb = q[1:].unsqueeze(1)
    d_self = poincare_pairwise_distance(qa, qa)
    check("A5 HD(x,x)=0", float(d_self.abs().max()) < 1e-6, f"{float(d_self.abs().max()):.2e}")
    d12 = poincare_pairwise_distance(qa, qb)
    d21 = poincare_pairwise_distance(qb, qa)
    check("A6 HD 对称", float((d12 - d21.transpose(1, 2)).abs().max()) < 1e-6)
    # 闭式暴力核对（逐对算，不用 matmul 技巧）
    a, b = q[0], q[1]
    num = 2.0 * float((a - b).pow(2).sum())
    den = (1.0 - float(a.pow(2).sum())) * (1.0 - float(b.pow(2).sum()))
    brute = math.acosh(1.0 + num / den)
    check("A7 HD 与逐对闭式一致", abs(float(d12[0, 0, 0]) - brute) < 1e-5,
          f"got={float(d12[0, 0, 0]):.6f} brute={brute:.6f}")

    # 极端输入不能 NaN：tanh 饱和的情形
    big = torch.full((1, 4, 3), 1e3)
    pb = expmap0(big, dim_scale=1.0)
    check("A8 极端大幅值输入不产生 NaN（r_max 兜底）",
          bool(torch.isfinite(pb).all()) and float(pb.norm(dim=-1).max()) < 1.0,
          f"‖p‖={float(pb.norm(dim=-1).max()):.6f}")

    # ---------------- B. 退化性（本测试的核心） ----------------
    print("\n[B] 退化性：论文原式 vs 实际用法")
    m_paper = HSAM(DIMS, stages=ATTN_STAGES_DEFAULT, logit_norm="paper")
    m_std = HSAM(DIMS, stages=ATTN_STAGES_DEFAULT, logit_norm="std")
    m_std.collect_diag = True
    m_paper.collect_diag = True
    fa = [norm_feats(1, d, HW[i], requires_grad=True) for i, d in enumerate(DIMS)]
    fb = [norm_feats(2, d, HW[i], requires_grad=True) for i, d in enumerate(DIMS)]
    _, _ = m_paper(fa, fb)
    _, _ = m_std(fa, fb)
    dp = m_paper.diag()
    ds = m_std.diag()

    e_paper = [dp[f"stage{s}"]["attn_entropy_norm"] for s in dp["stages"]]
    a_paper = [dp[f"stage{s}"]["agg_over_feat"] for s in dp["stages"]]
    check("B1 论文原式（tanh(HD/√C)）注意力**必然**均匀（熵 > 0.9999）",
          all(e > 0.9999 for e in e_paper), f"ent={['%.6f' % e for e in e_paper]}")
    check("B2 论文原式的贡献 ≈ 0（agg/feat < 1e-3）⇒ 对本项目等于没加",
          all(a < 1e-3 for a in a_paper), f"agg/feat={['%.2e' % a for a in a_paper]}")

    e_std = [ds[f"stage{s}"]["attn_entropy_norm"] for s in ds["stages"]]
    p_std = [ds[f"stage{s}"]["attn_prob_max"] for s in ds["stages"]]
    a_std = [ds[f"stage{s}"]["agg_over_feat"] for s in ds["stages"]]
    check("B3 std 模式**非退化**（熵 < ENT_MAX_OK）",
          all(e < ENT_MAX_OK for e in e_std), f"ent={['%.4f' % e for e in e_std]} (阈值 {ENT_MAX_OK})")
    check("B4 std 模式的 prob_max 明显高于 1/N（有真实结构）",
          all(p > 3.0 / ds[f"stage{s}"]["tokens"] for s, p in zip(ds["stages"], p_std)),
          f"prob_max={['%.4e' % p for p in p_std]}")
    check("B5 std 模式的贡献非零（agg/feat > 1e-2）",
          all(a > 1e-2 for a in a_std), f"agg/feat={['%.4e' % a for a in a_std]}")
    check("B6 两模式可区分（std 的熵严格低于 paper）",
          all(s < p for s, p in zip(e_std, e_paper)),
          f"std={['%.4f' % e for e in e_std]} paper={['%.4f' % e for e in e_paper]}")

    cv = [ds[f"stage{s}"]["radius_cv"] for s in ds["stages"]]
    check("B7 半径**非常数**（r_cv > 1e-3）", all(c > 1e-3 for c in cv),
          f"r_cv={['%.4e' % c for c in cv]}")

    # B8/B9 非退化证书：喂"精确归一化到 √C"的特征（= outnorm 的输出）必须被判定退化，
    # 喂未归一化特征（= outnorm 之前）必须不是退化的。这是模块存在意义的判据。
    def exact_norm_feats(seed=5, n=2):
        g = torch.Generator().manual_seed(seed)
        out = []
        for i, d in enumerate(DIMS):
            t = torch.randn(n, d, HW[i], HW[i], generator=g)
            t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12) * (d ** 0.5)
            out.append(t)
        return out

    m_cert = HSAM(DIMS, stages=ATTN_STAGES_DEFAULT, logit_norm="std",
                  radius_mode="feature_energy", feature_source="post_norm")
    m_cert.collect_diag = True
    na = exact_norm_feats(5)
    nb = exact_norm_feats(6)
    m_cert(na, nb)
    d_post = m_cert.diag()
    rc_post = [d_post[f"stage{s}"]["rank_corr_hd_vs_cos"] for s in d_post["stages"]]
    check("B8 post_norm 特征（‖v‖²=C 精确）⇒ rank_corr ≈ 1（**退化，作负对照**）",
          all(r > 0.999 for r in rc_post), f"rank_corr={['%.6f' % r for r in rc_post]}")

    # 未归一化：把范数乘上一个随机幅度
    g = torch.Generator().manual_seed(7)
    ua = [t * (0.5 + torch.rand(t.shape[0], 1, 1, 1, generator=g) * 1.5) for t in na]
    ub = [t * (0.5 + torch.rand(t.shape[0], 1, 1, 1, generator=g) * 1.5) for t in nb]
    m_cert(ua, ub)
    d_pre = m_cert.diag()
    rc_pre = [d_pre[f"stage{s}"]["rank_corr_hd_vs_cos"] for s in d_pre["stages"]]
    check("B9 未归一化特征（有能量变化）⇒ rank_corr < 0.999（**修法生效**）",
          all(r < 0.999 for r in rc_pre), f"rank_corr={['%.6f' % r for r in rc_pre]}")
    check("B10 两种取数的 rank_corr 可区分（pre 明显低于 post）",
          all(p < q for p, q in zip(rc_pre, rc_post)),
          f"pre={['%.4f' % r for r in rc_pre]} post={['%.4f' % r for r in rc_post]}")

    # B11 radius_mode 可切换且确实改变了半径
    m_ld = HSAM(DIMS, stages=ATTN_STAGES_DEFAULT, radius_mode="local_dispersion")
    m_ld.collect_diag = True
    m_ld(ua, ub)
    d_ld = m_ld.diag()
    r_ld = [d_ld[f"stage{s}"]["radius_cv"] for s in d_ld["stages"]]
    check("B11 local_dispersion 半径模式可用且半径非常数",
          all(r > 1e-3 for r in r_ld) and all(r > 0 for r in r_ld),
          f"r_cv={['%.4e' % r for r in r_ld]}")
    check("B12 local_dispersion 与 feature_energy 的半径分布不同",
          any(abs(a - b) > 1e-3 for a, b in zip(r_ld, [d_pre[f"stage{s}"]["radius_cv"]
                                                       for s in d_pre["stages"]])),
          f"ld={['%.4e' % r for r in r_ld]}")

    # ---------------- C. 形状 / 残差 / 梯度 ----------------
    print("\n[C] 形状、残差结构、梯度")
    oa, ob = m_std(fa, fb)
    check("C1 输出 stage 数与被选中 stage 数一致",
          len(oa) == len(oa) and len(oa) == len(DIMS), f"{len(oa)}")
    check("C2 被选中 stage 形状不变",
          all(tuple(oa[s].shape) == tuple(fa[s].shape) for s in ATTN_STAGES_DEFAULT),
          f"e.g. {tuple(oa[2].shape)}")
    check("C3 未选中 stage **原样透传**（同一对象）",
          all(oa[s] is fa[s] for s in range(len(DIMS)) if s not in ATTN_STAGES_DEFAULT))
    check("C4 finite 全 True",
          all(ds[f"stage{s}"]["finite"] for s in ds["stages"]))
    loss = sum(t.pow(2).mean() for t in oa) + sum(t.pow(2).mean() for t in ob)
    loss.backward()
    gsum = float(sum(t.grad.abs().sum() for t in fa if t.grad is not None))
    check("C5 反向能传到输入特征（梯度非零）", gsum > 0, f"sum|grad|={gsum:.4e}")
    check("C6 模块**零参数**", sum(p.numel() for p in m_std.parameters()) == 0,
          f"{sum(p.numel() for p in m_std.parameters())}")
    rep = m_std.param_report()
    check("C7 form / manifold / curvature 正确",
          rep["form"] == HSAM_FORM and rep["manifold"] == "poincare_ball"
          and rep["curvature_c"] == CURVATURE_C and rep["params"] == 0,
          f"{rep['form']} c={rep['curvature_c']}")

    # ---------------- D. 接入与 fail-closed ----------------
    print("\n[D] 网络接入与 fail-closed")
    from model.casa_vmamba_net_v2 import CASAVMambaSTRNetV2  # noqa: E402
    import io
    from contextlib import redirect_stdout

    def raises(**kw):
        try:
            with redirect_stdout(io.StringIO()):
                CASAVMambaSTRNetV2(vmamba_pretrained_path=None, **kw)
        except SystemExit:
            return True
        return False

    check("D1 hsam=1 且 hgsl=1 ⇒ SystemExit（HSAM 不产生变化分数）",
          raises(hqkv=False, geometry="lattice", hsam=True, hgsl=True))
    net0 = CASAVMambaSTRNetV2(vmamba_pretrained_path=None, hqkv=False,
                              geometry="lattice", hsam=False, hgsl=False)
    check("D2 hsam=0 时不构造 HSAM（CTRL 的结构不变）", net0.hsam_op is None)
    net1 = CASAVMambaSTRNetV2(vmamba_pretrained_path=None, hqkv=False,
                              geometry="lattice", hsam=True, hgsl=False)
    check("D3 hsam=1 时构造 HSAM 且参数量仍为 0",
          net1.hsam_op is not None
          and sum(p.numel() for p in net1.hsam_op.parameters()) == 0)
    pb = net1.param_breakdown()
    check("D4 param_breakdown 里 hsam=0（不改变部署参数）", pb["hsam"] == 0, f"{pb['hsam']}")
    check("D5 HSAM 没有改变主干参数（total 与 hsam=0 时相同）",
          net0.param_breakdown()["total"] == pb["total"],
          f"{net0.param_breakdown()['total']} vs {pb['total']}")

    print(f"\n=== 结果：{len(_total) - len(_fail)}/{len(_total)} PASS ===")
    if _fail:
        print("FAILED: " + ", ".join(_fail))
        return 1
    print("HSAM 回归门全绿：球内映射正确、距离闭式一致、std 非退化、paper 原式的退化被固定成断言、零参数、fail-closed。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
