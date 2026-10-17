"""Run8b CPU 预检：证明**抗自锁**确实生效（对比 v1 的自锁）。

核心判据（本轮唯一要证的事）：
  C1  γ 初值非零，且三支门**都进入前向**（v1 的 global/lite 从未进入）
  C2  `contrib_norm` 在 epoch-0 **非零**（v1 恒为 0）
  C3  **三个门的梯度都非零**（v1 的 global/lite 梯度恒为 0；detail 的 γ 停在 1e-41）
  C4  一次 SGD 更新后 `contrib_norm` 仍非零，且 γ 发生变化（真的在学）
  C5  γ=0 复现 v1 自锁（负对照）：contrib_norm == 0 且三支梯度全 0
  C6  三支 share 之和 ≈ 1（说明三支都在出力，而不是只有一支）
  C7  形状 / 参数预算 / 预训练加载接口
  C8  Run8b 与 Run8 的差异**只在抗自锁**（同 stage 数、同 rank、同 window）

本脚本**不写任何正式名产物**，只打印。
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model.layers.caacp_hqkv_v2 import (  # noqa: E402
    CAACPHQKVv2, GAMMA_INIT, HQKV_FORM_V2, rms_normalize,
)

torch.manual_seed(0)
DIMS = (96, 192, 384, 768)
RES = (64, 32, 16, 8)
FAILS, CHECKS = [], []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def make_feats(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = [torch.randn(2, d, r, r, generator=g) for d, r in zip(DIMS, RES)]
    b = [torch.randn(2, d, r, r, generator=g) for d, r in zip(DIMS, RES)]
    return a, b


def build(gamma_init):
    return CAACPHQKVv2(stage_dims=DIMS, rank=32, lite_dim=192,
                       detail_window=2, global_window=4, attn_stages=(1, 2, 3),
                       gamma_init=gamma_init)


print("=" * 80)
print("C0 基本：RMS 归一化与模块形状")
print("=" * 80)
x = torch.randn(2, 64, 16, 16) * 37.0          # 故意放大 37 倍
y = rms_normalize(x)
check("C0.1 rms_normalize 使 RMS=1", abs(float(y.pow(2).mean().sqrt()) - 1.0) < 1e-4,
      f"rms={float(y.pow(2).mean().sqrt()):.6f}")
check("C0.2 与输入尺度无关",
      abs(float(rms_normalize(x * 1e-3).pow(2).mean().sqrt()) - 1.0) < 1e-3)
op = build(GAMMA_INIT)
rep = op.param_report()
fA, fB = make_feats()
out, diag = op(fA, fB)
check("C0.3 输出形状 (B,192,64,64)", tuple(out.shape) == (2, 192, 64, 64),
      f"{tuple(out.shape)}")
check("C0.4 form 标记正确", rep["form"] == HQKV_FORM_V2, rep["form"])
print(f"  total_params = {rep['total_params']:,}")

print()
print("=" * 80)
print("C1 γ 初值非零，且三支都进入前向（v1 的 global/lite 从未进入）")
print("=" * 80)
check("C1.1 γ 初值 = GAMMA_INIT = 0.5（非零）",
      abs(rep["gammas"]["detail"] - GAMMA_INIT) < 1e-6
      and abs(rep["gammas"]["global"] - GAMMA_INIT) < 1e-6
      and abs(rep["gammas"]["lite"] - GAMMA_INIT) < 1e-6, f"gammas={rep['gammas']}")
check("C1.2 gamma_zero_init == False（v2 明确不承诺零初始化）",
      rep["gamma_zero_init"] is False)
for k in ("detail", "global", "lite"):
    sh = diag.get(f"part_{k}_share")
    check(f"C1.3 {k} 分支 share > 0（真的出力）", sh is not None and sh > 0.0,
          f"share={sh}")

print()
print("=" * 80)
print("C2 contrib_norm 非零（v1 恒为 0 —— 这是本轮最关键的判据）")
print("=" * 80)
cn = diag.get("contrib_norm", 0.0)
check("C2.1 contrib_norm > 0", cn > 0.0, f"contrib_norm={cn:.6e}")
# ⚠️ `contrib_norm` 是**未归一化的 L2 范数**，量级必然 O(√(B·C·H·W))（本例 ≈1280），
# 因此判据不能用"0.1~100"。真正该断言的是 **RMS**（= norm/√N），它才与分辨率无关：
# 期望 RMS ≈ (γ_d+γ_g+γ_l)/√(n_branch·n_stage) = 1.5/3 = 0.5。
_n = out.numel()
_rms = cn / (_n ** 0.5)
check("C2.2 注入的 RMS 量级合理（0.1 ~ 2.0，期望 ≈0.5）", 0.1 < _rms < 2.0,
      f"rms={_rms:.4f} (norm={cn:.2f}, numel={_n})")
for k in ("detail", "global", "lite"):
    nv = diag.get(f"part_{k}_norm", 0.0)
    check(f"C2.3 {k} 分支输出范数 > 0", isinstance(nv, float) and nv > 0.0, f"{nv:.4f}")

print()
print("=" * 80)
print("C3 三个门的梯度都非零（v1 的 global/lite 恒 0）")
print("=" * 80)
op.zero_grad()
out, _ = op(*make_feats(seed=1))
loss = out.pow(2).mean()
loss.backward()
op.snapshot_gamma_grad()
gg = op.param_report()["gamma_grad"]
print(f"  gamma_grad = {gg}")
for k in ("detail", "global", "lite"):
    check(f"C3.{k} 梯度非零", gg.get(k) is not None and gg[k] > 0.0, f"{gg.get(k)}")

print()
print("=" * 80)
print("C4 一次 SGD 更新后 γ 真的变了，且 contrib_norm 仍非零")
print("=" * 80)
g0 = op.param_report()["gammas"]
opt = torch.optim.SGD(op.parameters(), lr=1e-2)
op.zero_grad()
out, _ = op(*make_feats(seed=2))
loss = out.pow(2).mean()
loss.backward()
opt.step()
out2, diag2 = op(*make_feats(seed=2))
g1 = op.param_report()["gammas"]
moved = {k: abs(g1[k] - g0[k]) for k in g0}
check("C4.1 三个 γ 都在一步更新后改变",
      all(v > 0 for v in moved.values()), f"delta={moved}")
check("C4.2 更新后 contrib_norm 仍非零",
      diag2.get("contrib_norm", 0.0) > 0.0, f"{diag2.get('contrib_norm'):.6e}")

print()
print("=" * 80)
print("C5 负对照：γ=0 必须**复现 v1 的自锁**（否则说明诊断量本身失效）")
print("=" * 80)
op0 = build(0.0)
out0, diag0 = op0(*make_feats(seed=3))
check("C5.1 γ=0 ⇒ contrib_norm == 0（v1 的实测症状）",
      diag0.get("contrib_norm", -1.0) == 0.0, f"{diag0.get('contrib_norm')}")
op0.zero_grad()
o0, _ = op0(*make_feats(seed=3))
o0.pow(2).mean().backward()
op0.snapshot_gamma_grad()
gg0 = op0.param_report()["gamma_grad"]
check("C5.2 γ=0 ⇒ 三个门的梯度全为 0（自锁的直接证据）",
      all((gg0.get(k) or 0.0) == 0.0 for k in ("detail", "global", "lite")), f"{gg0}")
print("    ⇒ 这条对照说明：v1 不是'梯度小'，而是**结构上不可能有梯度**。")

print()
print("=" * 80)
print("C6 三支 share 之和 ≈ 1")
print("=" * 80)
tot = sum(diag.get(f"part_{k}_share", 0.0) for k in ("detail", "global", "lite"))
check("C6.1 share 之和 ≈ 1", abs(tot - 1.0) < 1e-6, f"sum={tot:.6f}")

print()
print("=" * 80)
print("C7 形状 / 预算 / 与 Run8 v1 的同构性")
print("=" * 80)
check("C7.1 参数量与 v1 同量级（250k~400k）",
      250_000 < rep["total_params"] < 400_000, f"{rep['total_params']:,}")
check("C7.2 attn_stages=(1,2,3) 与 v1 默认一致", rep["attn_stages"] == [1, 2, 3])
check("C7.3 rank/window 与 v1 默认一致",
      rep["rank"] == 32 and rep["detail_window"] == 2 and rep["global_window"] == 4)
check("C7.4 detail/global 在 stage1-3 都有、stage0 只有 lite",
      all(("detail" in rep["per_stage"][f"stage{i}"]
           and "global" in rep["per_stage"][f"stage{i}"]) for i in (1, 2, 3))
      and "lite_fuse" in rep["per_stage"]["stage0"])

print()
print("=" * 80)
print(f"结果：{len(CHECKS)-len(FAILS)}/{len(CHECKS)} PASS, {len(FAILS)} FAIL")
if FAILS:
    print("FAILED:", ", ".join(FAILS))
print("=" * 80)
sys.exit(1 if FAILS else 0)
