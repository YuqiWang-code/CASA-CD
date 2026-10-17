"""ST-GEO 数学与实现验收（纯 CPU，无外部数据；FAIL STOP）。

对应 st_geodesic.py 的命题 1/2/3：
  T1  生产实现（测地极坐标闭式）== eigh 参考的 AIM 距离
  T1b **反例守卫**：把 z=tanh(u)e^{2iθ} 当等距盘坐标的错误公式必须被判为不一致
  T2  对比度不变性（随机 SPD）：d_shape == 0、d_scale == 2√2|log c|
  T2c 对比度不变性（强各向异性条纹图，最坏数值情形）⇒ d_shape == 0
  T3  偏置不变性（图像域）⇒ 两项均为 0
  T4  分解恒等式 d²_AIM = d²_shape + 2(log σ₁/σ₂)²（命题 3）
  T5  对称性   T6 平坦区无 NaN/valid=0   T7 解析特例 ⇒ √2·u
  T8  梯度链非零有限   T9 双时相同一变换 ⇒ 距离不变
  T10 γ 零初始化 ⇒ 模块输出恒为 0      T11 抗伪变化曲线（|ΔI| 涨、d_shape 平）
  T12 真实结构变化必须有响应
用法：python test_st_geodesic.py
"""
import math
import sys

import torch

from model.layers.st_geodesic import (
    STGeoEvidence, det_scale_distance, poincare_shape_distance, shape_disk_coords,
    spd_geodesic_evidence, tr_based_shape_distance,
)

FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILED.append(name)


def disk_distance_wrong(zx1, zy1, zx2, zy2):
    """**错误**做法（仅用于 T1b 反例）：把 z = tanh(u)e^{2iθ} 当作等距盘坐标。"""
    dx, dy = zx1 - zx2, zy1 - zy2
    num2 = dx * dx + dy * dy
    den = 1.0 - (zx1 * zx2 + zy1 * zy2)
    cross = zx1 * zy2 - zy1 * zx2
    den2 = den * den + cross * cross
    ratio = torch.sqrt((num2 / den2.clamp_min(1e-12)).clamp(0.0, 1.0))
    return math.sqrt(2.0) * torch.atanh(ratio.clamp(max=1.0 - 1e-7))


def aim_reference(j1, j2):
    """参考实现：‖log(J1^{-1/2} J2 J1^{-1/2})‖_F（eigh，慢但权威）。"""
    evals, evecs = torch.linalg.eigh(j1)
    inv_sqrt = (evecs @ torch.diag_embed(evals.clamp_min(1e-30).rsqrt())
                @ evecs.transpose(-1, -2))
    m = inv_sqrt @ j2 @ inv_sqrt
    ev = torch.linalg.eigvalsh(0.5 * (m + m.transpose(-1, -2))).clamp_min(1e-30)
    return ev.log().pow(2).sum(dim=-1).sqrt()


def spd_from_ab(a, b, c):
    m = torch.zeros(a.shape[0], 2, 2, dtype=a.dtype)
    m[:, 0, 0] = a
    m[:, 0, 1] = b
    m[:, 1, 0] = b
    m[:, 1, 1] = c
    return m


def rand_spd(n, gen, scale=1.0):
    x = torch.randn(n, 2, 2, generator=gen)
    m = x @ x.transpose(-1, -2) + 0.05 * torch.eye(2)
    return m * scale


def edge_image(n=1, h=48, w=48):
    """强各向异性合成图：单方向正弦条纹（κ→1，条件数最坏）。"""
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                          torch.arange(w, dtype=torch.float32), indexing="ij")
    img = torch.sin(2 * math.pi * x / 6.0)
    return img.view(1, 1, h, w).repeat(n, 3, 1, 1)


def main():
    gen = torch.Generator().manual_seed(7)

    # ---------------- T1 生产实现 vs eigh 参考 ----------------
    j1 = rand_spd(256, gen)
    j2 = rand_spd(256, gen)
    za = shape_disk_coords(j1[:, 0, 0:1], j1[:, 0, 1:2], j1[:, 1, 1:2])
    zb = shape_disk_coords(j2[:, 0, 0:1], j2[:, 0, 1:2], j2[:, 1, 1:2])
    d_prod = poincare_shape_distance(za[0], za[1], za[2], zb[0], zb[1], zb[2])

    def det1(j):
        det = (j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] ** 2).clamp_min(1e-30).sqrt()
        return spd_from_ab(j[:, 0, 0] / det, j[:, 0, 1] / det, j[:, 1, 1] / det)

    d_ref = aim_reference(det1(j1), det1(j2))
    err = (d_prod[:, 0] - d_ref).abs().max().item()
    check("T1 production closed form == AIM reference (eigh)",
          err < 2e-3, f"max|d|={err:.3e}")

    # ---------------- T1b 错误公式必须被判为不一致（反例守卫）----------------
    d_wrong = disk_distance_wrong(za[0], za[1], zb[0], zb[1])
    err_wrong = (d_wrong[:, 0] - d_ref).abs().max().item()
    check("T1b wrong disk formula is inconsistent (guard)", err_wrong > 0.1,
          f"max|d|={err_wrong:.3e} (should be large)")

    # ---------------- T2 对比度不变性（随机 SPD）----------------
    c = 1.7
    jb = j1 * (c * c)
    zb2 = shape_disk_coords(jb[:, 0, 0:1], jb[:, 0, 1:2], jb[:, 1, 1:2])
    d_shape_c = poincare_shape_distance(za[0], za[1], za[2], zb2[0], zb2[1], zb2[2])
    d_scale_c = det_scale_distance(j1[:, 0, 0:1], j1[:, 0, 1:2], j1[:, 1, 1:2],
                                   jb[:, 0, 0:1], jb[:, 0, 1:2], jb[:, 1, 1:2])
    check("T2a J -> c^2 J => d_shape == 0 (exact, FP32)",
          float(d_shape_c.max()) == 0.0, f"max={float(d_shape_c.max()):.3e}")
    want = 2.0 * math.sqrt(2.0) * abs(math.log(c))
    check("T2b J -> c^2 J => d_scale == 2sqrt2|log c|",
          abs(float(d_scale_c.mean()) - want) < 1e-5,
          f"got={float(d_scale_c.mean()):.6f} want={want:.6f}")

    # ---------------- T2c 最坏情形：强各向异性条纹图 + 对比度 ----------------
    e = edge_image(2)
    _, pe0 = spd_geodesic_evidence(e, e * 1.35, sigma_d=1.0)
    _, pe1 = spd_geodesic_evidence(e, e * 0.55, sigma_d=1.0)
    kappa_edge = float(pe0["kappa_a"].max())
    worst = max(float(pe0["d_shape"].max()), float(pe1["d_shape"].max()))
    check("T2c anisotropic (kappa->1) worst case: contrast => d_shape == 0",
          worst == 0.0, f"max kappa={kappa_edge:.4f} max d_shape={worst:.3e}")

    # ---------------- T3 偏置不变性 ----------------
    img = torch.randn(2, 3, 64, 64, generator=gen) * 0.5
    _, p1 = spd_geodesic_evidence(img, img + 0.35, sigma_d=1.0)
    check("T3 image + offset => d_shape == 0 and d_str == 0",
          float(p1["d_shape"].abs().max()) == 0.0 and float(p1["d_str"].abs().max()) == 0.0,
          f"shape={float(p1['d_shape'].abs().max()):.2e} "
          f"str={float(p1['d_str'].abs().max()):.2e}")

    # ---------------- T4 分解恒等式（命题 3）----------------
    jj1 = rand_spd(128, gen, scale=0.7)
    jj2 = rand_spd(128, gen, scale=1.9)
    a1 = shape_disk_coords(jj1[:, 0, 0:1], jj1[:, 0, 1:2], jj1[:, 1, 1:2])
    a2 = shape_disk_coords(jj2[:, 0, 0:1], jj2[:, 0, 1:2], jj2[:, 1, 1:2])
    ds = poincare_shape_distance(a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])
    dsc = det_scale_distance(jj1[:, 0, 0:1], jj1[:, 0, 1:2], jj1[:, 1, 1:2],
                             jj2[:, 0, 0:1], jj2[:, 0, 1:2], jj2[:, 1, 1:2])
    d_full = (ds * ds + dsc * dsc).sqrt()
    d_ref_full = aim_reference(jj1, jj2).view_as(d_full)
    e4 = (d_full - d_ref_full).abs().max().item()
    check("T4 d_full == d_shape (+) d_scale == d_AIM(J1,J2)", e4 < 3e-3,
          f"max|d|={e4:.3e}")

    # ---------------- T4b τ 参考实现与生产实现一致 ----------------
    a1d = det1(jj1)
    a2d = det1(jj2)
    d_tau = tr_based_shape_distance(a1d[:, 0, 0:1], a1d[:, 0, 1:2], a1d[:, 1, 1:2],
                                    a2d[:, 0, 0:1], a2d[:, 0, 1:2], a2d[:, 1, 1:2])
    # τ 写法在强各向异性下病态，这里只要求两者在 1e-2 内一致（并记录差异）
    e4b = (ds - d_tau).abs().max().item()
    check("T4b production == tau-based reference (<=1e-2)", e4b < 1e-2,
          f"max|d|={e4b:.3e}")

    # ---------------- T5 对称性 ----------------
    ds_sw = poincare_shape_distance(a2[0], a2[1], a2[2], a1[0], a1[1], a1[2])
    check("T5 symmetry", float((ds - ds_sw).abs().max()) < 1e-6)

    # ---------------- T6 平坦区 ----------------
    flat = torch.zeros(1, 3, 32, 32)
    ev_f, p_f = spd_geodesic_evidence(flat, flat, sigma_d=1.0)
    check("T6 flat input: finite, valid=0, d_shape=0",
          bool(torch.isfinite(ev_f).all()) and float(p_f["valid"].max()) == 0.0
          and float(p_f["d_shape"].abs().max()) == 0.0)

    # ---------------- T7 解析特例 ----------------
    u = 0.8
    jA = spd_from_ab(torch.tensor([math.exp(u)]), torch.tensor([0.0]),
                     torch.tensor([math.exp(-u)]))
    jB = spd_from_ab(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0]))
    aa = shape_disk_coords(jA[:, 0, 0:1], jA[:, 0, 1:2], jA[:, 1, 1:2])
    bb = shape_disk_coords(jB[:, 0, 0:1], jB[:, 0, 1:2], jB[:, 1, 1:2])
    d7 = poincare_shape_distance(aa[0], aa[1], aa[2], bb[0], bb[1], bb[2]).item()
    check("T7 diag(e^u,e^-u) vs I => sqrt2*u", abs(d7 - math.sqrt(2.0) * u) < 1e-4,
          f"got={d7:.6f} want={math.sqrt(2.0)*u:.6f}")

    # ---------------- T8 梯度 ----------------
    img_a = torch.randn(1, 3, 48, 48, generator=gen).requires_grad_(True)
    img_b = torch.randn(1, 3, 48, 48, generator=gen)
    ev, _ = spd_geodesic_evidence(img_a, img_b, sigma_d=1.0)
    ev.sum().backward()
    g = img_a.grad
    check("T8 finite non-zero gradient wrt input",
          g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0)

    # ---------------- T9 同一变换作用于双时相 ----------------
    x = torch.randn(1, 1, 40, 40, generator=gen).repeat(1, 3, 1, 1)
    y = torch.randn(1, 1, 40, 40, generator=gen).repeat(1, 3, 1, 1)
    _, p0 = spd_geodesic_evidence(x, y, sigma_d=1.0)
    _, p90 = spd_geodesic_evidence(x.transpose(2, 3), y.transpose(2, 3), sigma_d=1.0)
    d0, d90 = float(p0["d_shape"].mean()), float(p90["d_shape"].mean())
    check("T9 identical transform on both timesteps => distance preserved",
          abs(d0 - d90) / max(d0, 1e-9) < 0.05, f"d0={d0:.6f} d90={d90:.6f}")

    # ---------------- T10 零初始化门 ----------------
    m = STGeoEvidence(in_ch=3, out_dim=8, mid=4, form="inject")
    out, _ = m(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32), (8, 8))
    check("T10 gamma zero-init => module output identically 0",
          float(out.detach().abs().max()) == 0.0 and m.param_report()["gamma_init_zero"],
          f"params={m.param_report()['params']}")

    # ---------------- T11 抗伪变化曲线 ----------------
    base = torch.randn(1, 3, 64, 64, generator=gen) * 0.4 + 0.5
    rows = []
    for c in (1.0, 1.1, 1.25, 1.5, 2.0):
        _, pc = spd_geodesic_evidence(base, base * c, sigma_d=1.0)
        rows.append((c, float((base * c - base).abs().mean()),
                     float(pc["d_shape"].max()), float(pc["d_str"].mean())))
    print("       c | mean|dI| | max d_shape | mean d_str")
    for c, raw, dsm, dsc in rows:
        print(f"      {c:4.2f} | {raw:8.5f} | {dsm:11.3e} | {dsc:10.4f}")
    ok11 = all(r[2] == 0.0 for r in rows) and rows[-1][1] > 5 * rows[1][1]
    check("T11 radiometric change: |dI| grows, d_shape stays 0", ok11)

    # ---------------- T12 真实结构变化必须有响应 ----------------
    base2 = torch.randn(1, 3, 64, 64, generator=gen) * 0.4 + 0.5
    changed = base2.clone()
    changed[:, :, 20:28, 20:28] += 0.6
    _, pst = spd_geodesic_evidence(base2, changed, sigma_d=1.0)
    resp = float(pst["d_shape"][0, 0, 18:30, 18:30].max())
    check("T12 8x8 structural change => d_shape responds (>0.1)",
          resp > 0.1, f"max d_shape in region={resp:.4f}")

    # ---------------- T13 尺度通道对结构变化同样有响应且连续 ----------------
    ratio = float(pst["d_str"][0, 0, 18:30, 18:30].max())
    check("T13 scale channel responds to structural change (>0.1)", ratio > 0.1,
          f"max d_str in region={ratio:.4f}")

    # ==================== v2 验收 ====================
    from model.layers.st_geodesic import (  # noqa: E402
        STGEO_FORM_V2, _cell_topk_mean, _pool_evidence,
    )

    # T14 鲁棒池化：单点尖峰 vs 2×2 真实结构
    spike = torch.zeros(1, 1, 4, 4)
    spike[0, 0, 1, 1] = 1.0                       # 单点噪声
    block = torch.zeros(1, 1, 4, 4)
    block[0, 0, 1:3, 1:3] = 1.0                   # 2×2 真实小结构
    mx_s = float(torch.nn.functional.max_pool2d(spike, 4).max())
    tk_s = float(_cell_topk_mean(spike, 1, k=4).max())
    mx_b = float(torch.nn.functional.max_pool2d(block, 4).max())
    tk_b = float(_cell_topk_mean(block, 1, k=4).max())
    check("T14 鲁棒池化：单点尖峰被压到 1/4，而 2x2 结构无损",
          abs(tk_s - 0.25) < 1e-6 and abs(mx_s - 1.0) < 1e-6
          and abs(tk_b - 1.0) < 1e-6 and abs(mx_b - 1.0) < 1e-6,
          f"spike: max={mx_s:.3f} topk={tk_s:.3f} | block: max={mx_b:.3f} topk={tk_b:.3f}")

    # T15 跨尺度门：通道数与零初始化门
    m2 = STGeoEvidence(in_ch=3, out_dim=8, mid=4, form="inject",
                       pool_mode="topk_mean", xscale_gate=True)
    out2, _ = m2(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32), (8, 8))
    rep2 = m2.param_report()
    check("T15 v2：in_ch=16（3 尺度×5 + 跨尺度门）、form 标记、γ=0 ⇒ 输出恒 0",
          rep2["in_ch_ev"] == 16 and rep2["form"] == STGEO_FORM_V2
          and float(out2.detach().abs().max()) == 0.0,
          f"in_ch={rep2['in_ch_ev']} form={rep2['form']} params={rep2['params']}")

    # T16 v2 证据在纯辐射变化下仍恒为 0（含跨尺度门）
    evs, gates = [], []
    for s_ in (1.0, 2.0, 4.0):
        ev_, pr_ = spd_geodesic_evidence(base, base * 1.6, sigma_d=s_)
        pool_ = _pool_evidence(ev_, 16, mode="topk_mean")
        evs.append(pool_)
        gates.append(float(pool_[:, 0:1].abs().max()))
    gate = torch.stack([e[:, 0:1] for e in evs], dim=0).amin(dim=0)
    check("T16 v2（topk_mean + 跨尺度门）在纯对比度变化下 d_shape 与门恒为 0",
          all(g == 0.0 for g in gates) and float(gate.abs().max()) == 0.0,
          f"per-scale max={gates} gate max={float(gate.abs().max()):.1e}")

    print()
    if FAILED:
        print(f"RESULT: FAIL ({len(FAILED)}): {FAILED}")
        return 1
    print("RESULT: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
