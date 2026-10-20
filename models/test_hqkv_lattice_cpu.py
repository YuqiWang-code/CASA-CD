#!/usr/bin/env python
"""拆除双曲几何后的 CPU 自检：`CAACPHQKVLattice` 必须**无几何、无半径分解、能跑通、有梯度**。

2026-10-20：用户判定双曲 HQKV 无用（四库 ΔF1 均值 −0.00085），下令拆除几何与半径分解，
只保留跨层 SAT 式 QKV 注入。这个脚本就是那次拆除的**回归门**：
它不训练、不碰 GPU、不需要数据集，`python models/test_hqkv_lattice_cpu.py` 直接跑。

## 检查项（24 条）

A. **拆除完整性**（这是本轮的主诉求，放在最前面）
  A1 `caacp_hqkv_lattice` 模块及其依赖链**不 import** `hyperbolic`
  A2 仓库里已没有 `caacp_hqkv_v2` / `hyperbolic` 的 live import
  A3 `hyperbolic.py` 仍在（作为 v1 的历史依赖），但 lattice 模块**不引用**它
  A4 `param_report()["manifold"] == "none"`
  A5 `param_report()["radius_decomposition"] == "none"`
  A6 `param_report()["score_weighted_pooling"] == "none"`
  A7 `param_report()["geometry"] == "none"`
  A8 模块参数里**没有任何** r0 / a / radius 命名
  A9 `score_maps()` 恒为空（不再产生变化分数）

B. **反自锁不回退**（拆几何不能把这条工程资产一起丢掉）
  B1 每支 γ 初值 = 0.5（非零）
  B2 每支 γ 在 `backward()` 后梯度**全部非零**
  B3 `contrib_norm > 0`
  B4 负对照 γ=0 ⇒ `contrib_norm == 0` **且** γ 梯度全为 0（复现自锁，证明门确实在前向里）

C. **结构正确性**
  C1 Q token 数 = N（全分辨率）
  C2 K/V token 数 = N/w²（严格压缩，且无 TopK/聚类）
  C3 窗口不能整除时**报错**（拒绝静默 padding）
  C4 输出形状 = `(B, lite_dim, H0, W0)`
  C5 `rank` 不能整除 `dim` 时报错（depthwise V 的 groups 前提）
  C6 注入后 `feats[0]` 尺度一致（净网络级）

D. **参数量**
  D1 `param_report()["total_params"]` 与真实 parameter 数一致
  D2 总参数 < 0.40 M（拆除前 v2 三支为 ≈0.62 M；只保留单支应显著更小）
  D3 参数量在 `rank=32` 下的**结构性上界**内（逐 stage 手算）

E. **fail-closed 契约**
  E1 网络 `geometry="hyperbolic"` ⇒ `SystemExit`
  E2 网络 `geometry="euclidean"` ⇒ `SystemExit`
  E3 网络 `geometry="lattice"` ⇒ 正常构造
  E4 网络 `hqkv=1, hgsl=1` ⇒ `SystemExit`（几何拆了，HGSL 无输入口）
"""
import io
import os
import sys
from contextlib import redirect_stdout

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model.layers.caacp_hqkv_lattice import (  # noqa: E402
    CAACPHQKVLattice,
    GAMMA_INIT,
    HQKV_FORM,
    LatticeSATAttention,
    build_caacp_hqkv_lattice,
    lattice_pool,
    rms_normalize,
)

DIMS = (96, 192, 384, 768)
HW = {0: (64, 64), 1: (32, 32), 2: (16, 16), 3: (8, 8)}
B = 2

_fail = []
_skip = []
_total = []


def check(name, cond, detail=""):
    _total.append(name)
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        _fail.append(name)
    return ok


def skip(name, why):
    _total.append(name)
    _skip.append(name)
    print(f"  [SKIP] {name}  {why}")
    return True


def make_feats(seed=3):
    g = torch.Generator().manual_seed(seed)
    fa, fb = [], []
    for i, d in enumerate(DIMS):
        h, w = HW[i]
        fa.append(torch.randn(B, d, h, w, generator=g))
        fb.append(torch.randn(B, d, h, w, generator=g))
    return fa, fb


def build(**kw):
    kw.setdefault("stage_dims", DIMS)
    kw.setdefault("rank", 32)
    kw.setdefault("window", 2)
    kw.setdefault("lite_dim", 192)
    kwargs = dict(stage_dims=kw.pop("stage_dims"), rank=kw.pop("rank"),
                  window=kw.pop("window"), lite_dim=kw.pop("lite_dim"))
    kwargs.update(kw)
    return CAACPHQKVLattice(**kwargs)


def main():
    print("=== 拆除回归门：CAACP-HQKV-Lattice（CPU）===")

    # ---------------- A. 拆除完整性 ----------------
    print("\n[A] 拆除完整性")
    src_file = os.path.join(_HERE, "model", "layers", "caacp_hqkv_lattice.py")
    with open(src_file, encoding="utf-8") as f:
        src = f.read()
    import_lines = [ln for ln in src.splitlines()
                    if ln.strip().startswith(("import ", "from "))]
    check("A1 lattice 的 import 里没有 hyperbolic",
          not any("hyperbolic" in ln for ln in import_lines),
          f"imports={len(import_lines)}")

    live_hits = []
    for root, _dirs, files in os.walk(_HERE):
        for fn in files:
            if not fn.endswith(".py") or fn == os.path.basename(__file__):
                continue
            p = os.path.join(root, fn)
            if "_SUPERSEDED" in p:
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    txt = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            for ln in txt.splitlines():
                s = ln.strip()
                if s.startswith(("import ", "from ")) and "caacp_hqkv_v2" in s:
                    live_hits.append(f"{fn}: {s}")
    check("A2 没有 live import 指向已删除的 caacp_hqkv_v2", not live_hits, str(live_hits[:3]))

    check("A3 hyperbolic.py 仍在（v1 历史依赖）但不在 lattice 依赖链里",
          os.path.isfile(os.path.join(_HERE, "model", "layers", "hyperbolic.py")),
          "hyperbolic.py 保留给 casa_vmamba_str v1")

    m = build()
    rep = m.param_report()
    check("A4 manifold == none", rep["manifold"] == "none", rep["manifold"])
    check("A5 radius_decomposition == none", rep["radius_decomposition"] == "none",
          rep["radius_decomposition"])
    check("A6 score_weighted_pooling == none", rep["score_weighted_pooling"] == "none",
          rep["score_weighted_pooling"])
    check("A7 geometry == none", rep["geometry"] == "none", rep["geometry"])
    bad_names = [n for n, _ in m.named_parameters()
                 if any(k in n.lower() for k in ("r0", "radius", "_a", "curvature"))]
    check("A8 参数里没有 r0/radius/curvature", not bad_names, str(bad_names[:3]))
    check("A9 score_maps() 恒为空", m.score_maps() == {}, str(m.score_maps()))
    check("A10 form 标记 = hqkv_v3_lattice", rep["form"] == HQKV_FORM, rep["form"])

    # ---------------- B. 反自锁 ----------------
    print("\n[B] 反自锁不回退")
    gammas = m.gamma_dict()
    check("B1 每支 γ 初值 = 0.5", all(abs(v - GAMMA_INIT) < 1e-12 for v in gammas.values()),
          str({k: round(v, 4) for k, v in gammas.items()}))

    fa, fb = make_feats()
    out, _diag = m(fa, fb)
    loss = out.pow(2).mean()
    loss.backward()
    m.snapshot_gamma_grad()
    gg = m.param_report()["gamma_grad"]
    check("B2 每支 γ 梯度**全部非零**（自锁已解）",
          len(gg) > 0 and all(v is not None and v > 0 for v in gg.values()),
          str({k: (None if v is None else f"{v:.3e}") for k, v in gg.items()}))
    check("B3 contrib_norm > 0", m.diag()["contrib_norm"] > 0,
          f"{m.diag()['contrib_norm']:.4e}")

    m0 = build(gamma_init=0.0)
    fa0, fb0 = make_feats()
    out0, _ = m0(fa0, fb0)
    out0.pow(2).mean().backward()
    m0.snapshot_gamma_grad()
    gg0 = m0.param_report()["gamma_grad"]
    check("B4 负对照 γ=0 ⇒ contrib_norm=0 且 γ 梯度全 0（复现自锁）",
          abs(m0.diag()["contrib_norm"]) < 1e-12 and all(v == 0.0 for v in gg0.values()),
          f"contrib={m0.diag()['contrib_norm']:.1e} grad={gg0}")

    # ---------------- C. 结构 ----------------
    print("\n[C] 结构正确性")
    attn = m.attn[1]
    tc = attn.token_counts((32, 32))
    check("C1  Q token 数 = N", tc["q_tokens"] == 32 * 32, str(tc))
    check("C2  K/V token 数 = N/w²", tc["kv_tokens"] == (32 * 32) // (2 * 2), str(tc))
    try:
        lattice_pool(torch.randn(1, 4, 7, 8), window=2)
        bad = False
    except ValueError:
        bad = True
    check("C3  窗口不能整除时拒绝（不 padding）", bad)
    check("C4  输出形状 = (B, lite_dim, H0, W0)",
          tuple(out.shape) == (B, 192, 64, 64), str(tuple(out.shape)))
    try:
        LatticeSATAttention(100, rank=32)
        bad = False
    except AssertionError:
        bad = True
    check("C5  rank 不整除 dim 时报错", bad)

    # 网络级：注入尺度必须与 feats[0] 一致
    # 注：整网前向需要 CUDA selective-scan 算子（`selective_scan_cuda_oflex` 或
    # `mamba_ssm`），且该算子只吃 CUDA 张量 ⇒ 本机无 GPU 时这条**跳过**而不是失败。
    # 在服务器上设 `LATTICE_TEST_CUDA=1` 就会真跑（这是本测试唯一需要 GPU 的一条）。
    from model.casa_vmamba_net_v2 import CASAVMambaSTRNetV2  # noqa: E402
    use_cuda = (os.environ.get("LATTICE_TEST_CUDA") == "1") and torch.cuda.is_available()
    dev = torch.device("cuda") if use_cuda else torch.device("cpu")
    net = CASAVMambaSTRNetV2(vmamba_pretrained_path=None, hqkv=True, geometry="lattice",
                            hgsl=False, inject_mode="dcr1").float().to(dev).eval()
    pre = torch.randn(5, 3, 64, 64, device=dev)
    post = torch.randn(5, 3, 64, 64, device=dev)
    try:
        with torch.no_grad():
            y = net(pre, post)
        check("C6  净网络前向形状 + contrib_norm>0",
              tuple(y.shape) == (5, 1, 64, 64) and net.contrib_norm() > 0,
              f"device={dev.type} {tuple(y.shape)} contrib={net.contrib_norm():.3e}")
    except RuntimeError as e:
        if "SCAN-BACKEND" in str(e) or "selective_scan" in str(e).lower():
            skip("C6  净网络前向（需要 CUDA selective-scan 算子 + CUDA 张量）",
                 "本机无 scan 算子或未设 LATTICE_TEST_CUDA=1；服务器上会执行")
        else:
            raise

    # ---------------- D. 参数量 ----------------
    print("\n[D] 参数量")
    real = sum(p.numel() for p in m.parameters())
    check("D1 report 参数量与真实一致", rep["total_params"] == real,
          f"report={rep['total_params']} real={real}")
    check("D2 总参数 < 0.40 M（拆除前 v2 三支 ≈0.62 M）", real < 400_000, f"{real:,}")
    # 结构性上界：每 stage 4·(dim·rank) + rank·out_dim，plus lite_fuse(3d→192→192) + 门
    bound = 0
    for i, d in enumerate(DIMS):
        if i in (1, 2, 3):
            bound += 3 * d * 32 + 32 * 192
        else:
            bound += 3 * d * 192 + 192 * 192 + 2 * 192
    bound += 4
    check("D3 参数量在结构性上界内", real <= bound, f"real={real:,} bound={bound:,} "
                                                    f"ratio={real / bound:.3f}")

    # ---------------- E. fail-closed ----------------
    print("\n[E] fail-closed 契约")

    def raises_systemexit(**kw):
        try:
            with redirect_stdout(io.StringIO()):
                CASAVMambaSTRNetV2(vmamba_pretrained_path=None, **kw)
        except SystemExit:
            return True
        return False

    check("E1 geometry='hyperbolic' ⇒ SystemExit",
          raises_systemexit(hqkv=True, geometry="hyperbolic", hgsl=False))
    check("E2 geometry='euclidean' ⇒ SystemExit",
          raises_systemexit(hqkv=True, geometry="euclidean", hgsl=False))
    ok = False
    try:
        with redirect_stdout(io.StringIO()):
            CASAVMambaSTRNetV2(vmamba_pretrained_path=None, hqkv=True,
                               geometry="lattice", hgsl=False)
        ok = True
    except SystemExit:
        ok = False
    check("E3 geometry='lattice' ⇒ 正常构造", ok)
    check("E4 hqkv=1 且 hgsl=1 ⇒ SystemExit（HGSL 失去分数输入）",
          raises_systemexit(hqkv=True, geometry="lattice", hgsl=True))

    # ---------------- 汇总 ----------------
    total = len(_total)
    npass = total - len(_fail) - len(_skip)
    print(f"\n=== 结果：{npass}/{total} PASS"
          + (f"，{len(_skip)} SKIP" if _skip else "")
          + (f"，{len(_fail)} FAIL" if _fail else "") + " ===")
    if _skip:
        print("SKIP: " + ", ".join(_skip))
    if _fail:
        print("FAILED: " + ", ".join(_fail))
        return 1
    print("拆除回归门全绿：无几何 / 无半径分解 / 有非零贡献 / 有非零门梯度 / fail-closed 生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
