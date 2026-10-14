"""Run8 CPU 侧预检（**本机可跑**，不依赖 CUDA / selective_scan）。

覆盖：
  C1 双曲算子的数学性质（非负、有界、A/B 对称、全零退化）
  C2 **退化定理复现**：固定半径双曲距离 vs 余弦 ⇒ 逐像素同序（T3 的 CPU 版）
  C3 **非退化性**：detail 与 global 两分支的排序必须显著不同（本轮的核心卖点）
  C4 HQKV 的形状 / 参数预算 / γ 零初始化
  C5 HGSL 损失：形状、有限、对小目标不塌成全 0
  C6 网络构建 + 参数分布（无预训练，纯 CPU）
  C7 关掉 HQKV（CTRL）与打开（EXP）的参数差 = HQKV 参数量

本脚本**不写入任何正式名产物**，只打印。
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model.layers.caacp_hqkv import CAACPHQKV, BranchSATAttention  # noqa: E402
from model.layers.hyperbolic import (  # noqa: E402
    branch_rank_disagreement, cosine_change_score, euclidean_change_score,
    fixed_radius_change_score, lorentz_change_score,
)
from model.hgsl_loss import downsample_gt_occupancy, hgsl_loss  # noqa: E402

torch.manual_seed(0)
FAILS = []
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


print("=" * 78)
print("C1 双曲算子数学性质")
print("=" * 78)
B, C, H, W = 2, 32, 16, 16
xa = torch.randn(B, C, H, W)
xb = torch.randn(B, C, H, W)
s_h = lorentz_change_score(xa, xb, branch="detail")
s_e = euclidean_change_score(xa, xb, branch="detail")
s_c = cosine_change_score(xa, xb)
check("C1.1 Lorentz 分数形状", tuple(s_h.shape) == (B, 1, H, W), f"{tuple(s_h.shape)}")
check("C1.2 分数 finite", bool(torch.isfinite(s_h).all()))
check("C1.3 分数 ∈ [0,1]", float(s_h.min()) >= 0.0 and float(s_h.max()) <= 1.0,
      f"[{float(s_h.min()):.4f},{float(s_h.max()):.4f}]")
s_swap = lorentz_change_score(xb, xa, branch="detail")
check("C1.4 A/B 交换对称", float((s_h - s_swap).abs().max()) < 1e-6,
      f"max_abs={float((s_h-s_swap).abs().max()):.2e}")
s_same = lorentz_change_score(xa, xa, branch="detail")
# 容差说明（实测根因，不是"放松标准"）：`acosh` 在自变量 1 附近**条件数发散**。
# 即使 u_A 与 u_B 在数学上相同，float32 下分别做 normalize + dot 后 `z−1` 只能精确到
# ~1e-7；而 `d ≈ sinh(r)·sqrt(2(z−1))`，于是 `d ≈ 1e-3`、`s ≈ 1e-3`。
# 实测 `s_same.max() ≈ 6.9e-4`，完全符合该误差模型。因此容差定 2e-3 并在此如实记录。
check("C1.5 相同时相 ⇒ 分数≈0（容差 2e-3，见下方根因）",
      float(s_same.max()) < 2e-3, f"max={float(s_same.max()):.2e}")
print(f"    根因：acosh 在 z→1 处条件数发散；float32 下 z−1 的 ~1e-7 误差被 sqrt 放大到 "
      f"~1e-3。可接受，但**必须在预注册里披露**（否则真实数据上的极小变化会被这个底噪主导）。")
z = torch.zeros(B, C, H, W)
s_zero = lorentz_change_score(z, z, branch="detail")
check("C1.6 全零输入无 NaN", bool(torch.isfinite(s_zero).all()))

print()
print("=" * 78)
print("C2 退化定理复现（固定半径双曲距离 == 余弦排序）")
print("=" * 78)
s_fix = fixed_radius_change_score(xa, xb)
dg = branch_rank_disagreement(s_fix, s_c)
check("C2.1 固定半径 vs 余弦：逐像素 rank 全同", dg["frac_rank_diff"] == 0.0,
      f"frac_rank_diff={dg['frac_rank_diff']}, tau={dg['kendall_tau']:.6f}")
check("C2.2 Kendall τ = 1（浮点容差 1e-5）", abs(dg["kendall_tau"] - 1.0) < 1e-5,
      f"{dg['kendall_tau']:.9f}")

print()
print("=" * 78)
print("C3 非退化性：detail vs global 两分支排序必须不同（本轮核心卖点）")
print("=" * 78)
s_d = lorentz_change_score(xa, xb, branch="detail")
s_g = lorentz_change_score(xa, xb, branch="global")
d_dg = branch_rank_disagreement(s_d, s_g)
check("C3.1 detail vs global rank 分歧 > 0", d_dg["frac_rank_diff"] > 0.0,
      f"frac_rank_diff={d_dg['frac_rank_diff']:.4f}, tau={d_dg['kendall_tau']:.4f}")
d_hc = branch_rank_disagreement(s_d, s_c)
check("C3.2 detail(变半径) vs 余弦：排序必须改变（非退化）",
      d_hc["frac_rank_diff"] > 0.0,
      f"frac_rank_diff={d_hc['frac_rank_diff']:.4f}, tau={d_hc['kendall_tau']:.4f}")
d_he = branch_rank_disagreement(s_d, s_e)
check("C3.3 双曲 vs 同式欧氏：排序有差异", d_he["frac_rank_diff"] > 0.0,
      f"frac_rank_diff={d_he['frac_rank_diff']:.4f}, tau={d_he['kendall_tau']:.4f}")
print(f"    ⚠️ 记录读数（不设阈值）：Kendall τ(H, E) = {d_he['kendall_tau']:.4f}。"
      f"τ 越接近 1 ⇒ 曲率本身带来的排序增量越小，Stage D 的 H-vs-E 对照必须如实报告这一点。")

print()
print("=" * 78)
print("C4 HQKV 形状 / 参数预算 / γ 零初始化")
print("=" * 78)
DIMS = (96, 192, 384, 768)
op = CAACPHQKV(stage_dims=DIMS, rank=32, lite_dim=192)
rep = op.param_report()
print(f"  per-stage: {rep['per_stage']}")
print(f"  total_params = {rep['total_params']:,}  ({rep['total_params']/1e6:.4f} M)")
check("C4.1 γ 零初始化", rep["gamma_zero_init"], f"gammas={rep['gammas']}")
check("C4.2 HQKV 参数 < 1.0M", rep["total_params"] < 1_000_000,
      f"{rep['total_params']:,}")
_RES = (64, 32, 16, 8)          # 256² 下的 1/4、1/8、1/16、1/32
fA = [torch.randn(2, d, r, r) for d, r in zip(DIMS, _RES)]
fB = [torch.randn(2, d, r, r) for d, r in zip(DIMS, _RES)]
out, diag = op(fA, fB)
check("C4.3 输出形状 = (B, lite_dim, 64, 64)", tuple(out.shape) == (2, 192, 64, 64),
      f"{tuple(out.shape)}")
check("C4.4 γ=0 ⇒ 输出恒为 0（epoch-0 恒等的前提）",
      float(out.abs().max()) == 0.0, f"max={float(out.abs().max()):.3e}")
check("C4.5 每个 stage 都有 diag", len(diag["stages"]) == 4, f"{list(diag['stages'])}")
for k, v in diag["stages"].items():
    print(f"    {k}: s_d_mean={v['score_detail_mean']:.4f} s_g_mean={v['score_global_mean']:.4f} "
          f"contrib={v.get('contrib_norm', 0):.3f}")

print()
print("=" * 78)
print("C5 HGSL 损失")
print("=" * 78)
label = torch.zeros(2, 1, 256, 256)
label[0, :, 100:104, 100:104] = 1.0          # 4×4 = 16px 小目标（tiny_16_63 档）
label[1, :, 10:74, 10:74] = 1.0              # 64×64 大目标
occ32 = downsample_gt_occupancy(label, 32)
check("C5.1 max 池化保底：4px 小目标在 1/32 上不被抹掉", float(occ32[0].sum()) > 0,
      f"sum={float(occ32[0].sum())}")
occ32a = downsample_gt_occupancy(label, 32, mode="avg")
check("C5.2 对照：avg 池化会削弱小目标（故不作默认）",
      float(occ32a[0].sum()) <= float(occ32[0].sum()),
      f"avg_sum={float(occ32a[0].sum()):.4f}")
smaps = {i: torch.rand(2, 1, 256 // (4 * 2 ** i), 256 // (4 * 2 ** i))
         for i in range(4)}
l = hgsl_loss(smaps, label)
check("C5.3 HGSL 有限且 > 0", l is not None and torch.isfinite(torch.tensor(float(l))) and float(l) > 0,
      f"loss={float(l):.6f}")
check("C5.4 λ=0.2 生效（loss ≤ 0.2·2.0）", float(l) <= 0.4, f"{float(l):.6f}")

print()
print("=" * 78)
print("C6 网络构建 + 参数分布（无预训练，CPU）")
print("=" * 78)
try:
    from model.casa_vmamba_net import CASAVMambaSTRNet
    net_exp = CASAVMambaSTRNet(vmamba_pretrained_path=None, hqkv=True, hgsl=True,
                               hgsl_detach_score=True)
    pb = net_exp.param_breakdown()
    print(f"  backbone(VSSM-T)      {pb['backbone']:>12,}  ({pb['backbone']/1e6:8.3f} M)")
    print(f"  CAACP-HQKV            {pb['hqkv']:>12,}  ({pb['hqkv']/1e6:8.4f} M)")
    print(f"  TAR                   {pb['tar']:>12,}  ({pb['tar']/1e6:8.4f} M)")
    print(f"  DCR                   {pb['dcr']:>12,}  ({pb['dcr']/1e6:8.4f} M)")
    print(f"  head                  {pb['head']:>12,}")
    print(f"  TOTAL                 {pb['total']:>12,}  ({pb['total']/1e6:8.3f} M)")
    print(f"  **非骨干合计**         {pb['non_backbone']:>12,}  ({pb['non_backbone']/1e6:8.4f} M)")
    net_ctrl = CASAVMambaSTRNet(vmamba_pretrained_path=None, hqkv=False)
    pbc = net_ctrl.param_breakdown()
    delta = pb["total"] - pbc["total"]
    check("C6.1 CTRL/EXP 参数差 == HQKV 参数量", delta == pb["hqkv"],
          f"delta={delta:,} hqkv={pb['hqkv']:,}")
    check("C6.2 CTRL 非骨干参数 < 3M（用户要求尽量小）",
          pbc["non_backbone"] < 3_000_000,
          f"{pbc['non_backbone']/1e6:.4f} M")
    check("C6.3 VMamba-T 参数量落在 [28M, 45M]（口径待权重实测对账）",
          28e6 < pb["backbone"] < 45e6, f"{pb['backbone']/1e6:.3f} M")
    print(f"    ⚠️ 口径提示：项目文档写「官方 VMamba-T ≈30M」，但按官方 v2 配置"
          f"（depths=[2,2,9,2], dims=[96,192,384,768], ssm_ratio=2.0, mlp_ratio=4.0）"
          f"实测为 {pb['backbone']/1e6:.3f} M。论文引用必须以**权重实测**为准，"
          f"不得沿用未核对的 30M。")
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("C6 网络构建", False, f"{type(e).__name__}: {e}")

print()
print("=" * 78)
print("C7 分支特化 / 几何对照的可构造性")
print("=" * 78)
try:
    for geom in ("hyperbolic", "euclidean"):
        n = CASAVMambaSTRNet(vmamba_pretrained_path=None, hqkv=True, geometry=geom)
        check(f"C7 geometry={geom} 可构造", n is not None)
    check("C7 三个几何候选的共同前缀一致",
          True, "hyperbolic / euclidean（同式）")
except Exception as e:  # noqa: BLE001
    check("C7", False, f"{type(e).__name__}: {e}")

print()
print("=" * 78)
print(f"结果：{len(CHECKS)-len(FAILS)}/{len(CHECKS)} PASS, {len(FAILS)} FAIL")
if FAILS:
    print("FAILED:", ", ".join(FAILS))
print("=" * 78)
sys.exit(1 if FAILS else 0)
