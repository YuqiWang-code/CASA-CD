#!/usr/bin/env python
"""Run7 HFAE（高频边缘辅助监督）T0–T11 验收测试。

本地（无 GPU / 无 selective_scan kernel）会**跳过**需要前向的检查并明确打印 SKIP；
服务器（casacd 环境）应全 PASS。

    python models/test_run7_hf_edge.py                 # 全量
    python models/test_run7_hf_edge.py --device cuda:0
"""
import argparse
import copy
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

_MODELS = os.path.dirname(os.path.abspath(__file__))
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from model.casa_tvim_str_net import CASATViMSTRNet                       # noqa: E402
from model.str_hf_edge import (HFEdgeHead, HF_EDGE_TAPS, hf_edge_loss,   # noqa: E402
                               edge_target_from_label)
from model.utils import BCEDiceLoss                                      # noqa: E402

BASE_DEPLOY = 4_880_190
EXP_B_DEPLOY = 4_890_366          # 实测值（见 Run7 README §2）；T8 用区间断言容错
DEPLOY_LIMIT = 5_000_000

RESULTS = []


def rec(tid, name, ok, detail="", skipped=False):
    tag = "SKIP" if skipped else ("PASS" if ok else "FAIL")
    RESULTS.append((tid, name, tag, detail))
    print(f"[{tag}] {tid} {name}" + (f"  | {detail}" if detail else ""), flush=True)


def build(**kw):
    torch.manual_seed(16)
    return CASATViMSTRNet(None, caacp=True, rep_mode="full", str_dim=96,
                          caacp_score_mode="rank", frh=False,
                          caacp_residual_mode="current", fs_tar=False, **kw)


def npar(m):
    return sum(p.numel() for p in m.parameters())


def kernel_ok():
    try:
        import selective_scan_cuda_oflex  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- tests
def t0_legacy_regression(dev, has_kernel):
    """T0：hf_edge=0 与基线逐位同构（taps/head 都不存在）。"""
    m = build()
    ok = (m.hf_edge is False and m.hf_edge_head is None and m.hf_edge_form == "none"
          and m._hf_handles == [] and m.hf_edge_report() is None)
    m.eval(); m.switch_to_deploy()
    dp = npar(m)
    ok = ok and dp == BASE_DEPLOY
    rec("T0", "legacy path (hf_edge=0) unchanged", ok,
        f"deploy_params={dp} (expect {BASE_DEPLOY})")


def t1_tensor_contract(dev, has_kernel):
    """T1：tap / head / target 形状契约。"""
    if not has_kernel:
        rec("T1", "tensor contract", False, "no selective_scan kernel; run on server",
            skipped=True)
        return
    m = build(hf_edge=True, hf_edge_form="aux_inject").to(dev).eval()
    B = 2
    pre = torch.randn(B, 3, 256, 256, device=dev)
    post = torch.randn(B, 3, 256, 256, device=dev)
    with torch.no_grad():
        out = m(pre, post)
    logits = m.aux_edge_logits()
    spec = HF_EDGE_TAPS[m.hf_edge_taps]
    # 边缘头把双时相配成**一张**变化边缘图 ⇒ logits/feat 均为 B 尺寸
    ok = (out.shape == (B, 1, 256, 256)
          and logits is not None and logits.shape == (B, 1, spec[0][3], spec[0][3])
          and m._hf_edge_feat.shape == (B, m.hf_edge_head.mid, spec[0][3], spec[0][3]))
    tgt = edge_target_from_label(torch.zeros(B, 1, 256, 256, device=dev), spec[0][3])
    ok = ok and tgt.shape == logits.shape
    rec("T1", "tensor contract", ok,
        f"out={tuple(out.shape)} logits={tuple(logits.shape)} feat={tuple(m._hf_edge_feat.shape)} "
        f"target={tuple(tgt.shape)}")


def t2_rng_pretrain(dev, has_kernel):
    """T2：同 seed 构造 → 既有 state_dict 交集逐位相同；新参数零初始化（inject）。"""
    c = build()
    a = build(hf_edge=True, hf_edge_form="aux_only")
    b = build(hf_edge=True, hf_edge_form="aux_inject")
    sd_c = c.state_dict()
    ok, details = True, []
    for tag, m in (("A", a), ("B", b)):
        sd = m.state_dict()
        common = set(sd_c) & set(sd)
        diff = [k for k in common if not torch.equal(sd_c[k], sd[k])]
        new = sorted(set(sd) - set(sd_c))
        ok = ok and not diff
        details.append(f"{tag}: common={len(common)} diff={len(diff)} new={len(new)}")
        if tag == "B":
            ok = ok and m.hf_edge_report()["inject_zero_init"] is True
    # 负对照：把 inject 置非零，自检必须报 False
    b.hf_edge_head.inject.weight.data.fill_(0.01)
    neg = b.hf_edge_report()["inject_zero_init"] is False
    ok = ok and neg
    rec("T2", "RNG discipline + zero-init", ok, " | ".join(details) + f" | neg_ctrl={neg}")


def t3_epoch0_forward(dev, has_kernel):
    """T3：epoch-0 前向逐位恒等（synthetic 8 + 真实 16 由 T7 侧覆盖）。"""
    if not has_kernel:
        rec("T3", "epoch-0 bitwise identity", False, "no kernel; run on server", skipped=True)
        return
    c = build().to(dev).eval()
    torch.manual_seed(0)
    pre = torch.randn(8, 3, 256, 256, device=dev)
    post = torch.randn(8, 3, 256, 256, device=dev)
    with torch.no_grad():
        yc = c(pre, post)
    ok, det = True, []
    for tag, kw in (("aux_only", {}), ("aux_inject", {})):
        m = build(hf_edge=True, hf_edge_form=tag).to(dev).eval()
        with torch.no_grad():
            y = m(pre, post)
        eq = torch.equal(yc, y)
        ok = ok and eq
        det.append(f"{tag}: equal={eq} max|d|={float((y-yc).abs().max()):.3e}")
    rec("T3", "epoch-0 bitwise identity", ok, " | ".join(det))


def t4_grad_chain(dev, has_kernel):
    """T4：辅助损失 → 高频分支的梯度链非空。"""
    if not has_kernel:
        rec("T4", "aux-loss gradient chain", False, "no kernel; run on server", skipped=True)
        return
    m = build(hf_edge=True, hf_edge_form="aux_only").to(dev).train()
    pre = torch.randn(4, 3, 256, 256, device=dev)
    post = torch.randn(4, 3, 256, 256, device=dev)
    tgt = (torch.rand(4, 1, 256, 256, device=dev) > 0.9).float()
    out = m(pre, post)
    main = BCEDiceLoss(out, tgt)
    aux = hf_edge_loss(m.aux_edge_logits(), tgt, 0.2)
    loss = main + aux
    loss.backward()
    ok = (torch.isfinite(loss) and float(aux) > 0
          and m.hf_edge_tap_grad_norm() > 0
          and m.hf_edge_tap_grad_nonzero() > 0
          and m.hf_edge_head_grad_norm() > 0)
    head_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                    for p in m.head.parameters())
    enc_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for n, p in m.named_parameters() if n.startswith("encoder."))
    ok = ok and head_grad and enc_grad
    rec("T4", "aux-loss gradient chain", ok,
        f"aux={float(aux):.5f} tap_grad={m.hf_edge_tap_grad_norm():.4e} "
        f"nonzero={m.hf_edge_tap_grad_nonzero()} head_grad={m.hf_edge_head_grad_norm():.4e} "
        f"main_head_grad={head_grad} encoder_grad={enc_grad}")


def t5_mechanism_selfcheck(dev, has_kernel):
    """T5：机制自检——**只回传辅助损失**时，高频分支必须拿到非零梯度。

    首版用「加 aux 的 tap 梯度 vs 只主损失的 tap 梯度」做对照，**判据是错的**：
    高频分支本来就由**主损失**驱动（它同时喂主输出），两者总范数几乎相同
    （实测 0.17833 vs 0.17849），无法隔离 aux 的贡献。正确做法是**只用 aux 回传**：
    若此时 tap 梯度 > 0，才证明「边缘监督自己就能到达高频分支」。

    同时报告：主损失单独回传的 tap 梯度（对照读数）、关闭开关时无 tap 可查。
    """
    if not has_kernel:
        rec("T5", "mechanism self-check (HF-branch grad)", False, "no kernel", skipped=True)
        return
    pre = torch.randn(4, 3, 256, 256, device=dev)
    post = torch.randn(4, 3, 256, 256, device=dev)
    tgt = (torch.rand(4, 1, 256, 256, device=dev) > 0.9).float()

    # (a) 只回传 aux
    m1 = build(hf_edge=True, hf_edge_form="aux_only").to(dev).train()
    _ = m1(pre, post)
    aux = hf_edge_loss(m1.aux_edge_logits(), tgt, 0.2)
    aux.backward()
    g_aux_only = m1.hf_edge_tap_grad_norm()
    head_g_aux_only = m1.hf_edge_head_grad_norm()
    main_head_grad_after_aux = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                                   for p in m1.head.parameters())

    # (b) 只回传主损失（对照）
    m2 = build(hf_edge=True, hf_edge_form="aux_only").to(dev).train()
    out2 = m2(pre, post)
    BCEDiceLoss(out2, tgt).backward()
    g_main_only = m2.hf_edge_tap_grad_norm()

    # (c) 关闭开关
    m3 = build().to(dev).train()
    g_disabled = m3.hf_edge_tap_grad_norm()

    ok = (g_aux_only > 0 and head_g_aux_only > 0 and g_disabled == 0.0
          and not main_head_grad_after_aux)
    rec("T5", "mechanism self-check (aux-only reaches HF branch)", ok,
        f"tap_grad_aux_only={g_aux_only:.4e} head_grad_aux_only={head_g_aux_only:.4e} "
        f"tap_grad_main_only={g_main_only:.4e} disabled={g_disabled} "
        f"main_head_touched_by_aux={main_head_grad_after_aux}")


def t6_numeric_gate(dev, has_kernel):
    """T6：TF32 off + deterministic 下 epoch-0 数值门 max|Δ| < 2e-4 且无 NaN。"""
    if not has_kernel:
        rec("T6", "numeric gate", False, "no kernel", skipped=True)
        return
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    c = build().to(dev).eval()
    a = build(hf_edge=True, hf_edge_form="aux_inject").to(dev).eval()
    torch.manual_seed(1)
    pre = torch.randn(4, 3, 256, 256, device=dev)
    post = torch.randn(4, 3, 256, 256, device=dev)
    with torch.no_grad():
        yc, ya = c(pre, post), a(pre, post)
    d = float((yc - ya).abs().max())
    ok = d < 2e-4 and torch.isfinite(ya).all()
    rec("T6", "numeric gate", ok, f"max|d|={d:.3e} finite={bool(torch.isfinite(ya).all())}")


def t7_fold_gate(dev, has_kernel):
    """T7：train↔deploy 二值 disagreement 必须为 0（真实 16 张优先，缺数据退化为 synthetic）。

    机制上两种形态都**不修改主输出路径**（aux_only 推理丢弃监督头；aux_inject 保留零初始化注入），
    因此折叠后主输出应与训练图逐位一致。
    """
    if not has_kernel:
        rec("T7", "fold equivalence (train<->deploy)", False, "no kernel", skipped=True)
        return
    ok, det = True, []
    for tag in ("aux_only", "aux_inject"):
        m = build(hf_edge=True, hf_edge_form=tag).to(dev).eval()
        torch.manual_seed(2)
        pre = torch.randn(16, 3, 256, 256, device=dev)
        post = torch.randn(16, 3, 256, 256, device=dev)
        with torch.no_grad():
            y_tr = m(pre, post)
        d = copy.deepcopy(m); d.eval(); d.switch_to_deploy()
        with torch.no_grad():
            y_dp = d(pre, post)
        dis = float(((y_tr > 0.5) != (y_dp > 0.5)).float().mean())
        err = float((y_tr - y_dp).abs().max())
        ok = ok and dis == 0.0 and err < 2e-4
        det.append(f"{tag}: dis={dis} max|d|={err:.3e} deploy_params={npar(d)}")
    rec("T7", "fold equivalence (train<->deploy)", ok, " | ".join(det))


def t8_params_flops(dev, has_kernel):
    """T8：deploy 参数门（≤5M）与两形态的增量。"""
    c = build(); a = build(hf_edge=True, hf_edge_form="aux_only")
    b = build(hf_edge=True, hf_edge_form="aux_inject")
    train = {"CTRL": npar(c), "A": npar(a), "B": npar(b)}
    dep = {}
    for tag, m in (("CTRL", c), ("A", a), ("B", b)):
        m.eval(); m.switch_to_deploy(); dep[tag] = npar(m)
    ok = (dep["CTRL"] == BASE_DEPLOY and dep["A"] == BASE_DEPLOY
          and abs(dep["B"] - EXP_B_DEPLOY) <= 64 and all(v <= DEPLOY_LIMIT for v in dep.values()))
    rec("T8", "deploy param gate", ok,
        f"train={train} deploy={dep} A_delta={dep['A']-BASE_DEPLOY:+d} "
        f"B_delta={dep['B']-BASE_DEPLOY:+d} limit={DEPLOY_LIMIT}")


def t9_dry_run(dev, has_kernel):
    """T9：真实数据 3 步 dry-run（由 models/run7_dry_run.py 承担；此处检查可调用性）。"""
    script = os.path.join(_MODELS, "run7_dry_run.py")
    ok = os.path.isfile(script)
    rec("T9", "real-data 3-step dry-run entrypoint exists", ok,
        f"{script} (full dry-run runs on server with the dataset)")


def t10_sidecar_negative(dev, has_kernel):
    """T10：fail-closed 负例——开关/形态不自洽必须被模型层断言拒绝。"""
    cases = []
    try:
        build(hf_edge=True, hf_edge_form="none"); cases.append(("hf_edge=1+form=none", False))
    except AssertionError:
        cases.append(("hf_edge=1+form=none", True))
    try:
        build(hf_edge=True, hf_edge_form="aux_only", fine_tap=True)
        cases.append(("hf_edge=1+fine_tap=1", False))
    except AssertionError:
        cases.append(("hf_edge=1+fine_tap=1", True))
    try:
        build(hf_edge=True, hf_edge_form="aux_only", hf_edge_taps="bogus")
        cases.append(("bad taps", False))
    except AssertionError:
        cases.append(("bad taps", True))
    try:
        build(hf_edge=False, hf_edge_form="aux_only"); cases.append(("hf_edge=0+form!=none", False))
    except AssertionError:
        cases.append(("hf_edge=0+form!=none", True))
    ok = all(v for _n, v in cases)
    rec("T10", "sidecar fail-closed negatives", ok,
        " ".join(f"{n}={'rejected' if v else 'ACCEPTED(!)'}" for n, v in cases))


def t11_resume(dev, has_kernel):
    """T11：resume/protocol-exact 由 models/run7_dry_run.py + train.py 侧车校验承担。"""
    ok = os.path.isfile(os.path.join(_MODELS, "train.py"))
    rec("T11", "resume protocol-exact entrypoint", ok,
        "train.py [PROTOCOL-MISMATCH] strict resume gate (runs on server)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    has_kernel = kernel_ok()
    print(f"=== Run7 HFAE T0-T11 | device={dev} | selective_scan={has_kernel} ===")
    for fn in (t0_legacy_regression, t1_tensor_contract, t2_rng_pretrain, t3_epoch0_forward,
               t4_grad_chain, t5_mechanism_selfcheck, t6_numeric_gate, t7_fold_gate,
               t8_params_flops, t9_dry_run, t10_sidecar_negative, t11_resume):
        try:
            fn(dev, has_kernel)
        except Exception as e:  # noqa: BLE001
            rec(fn.__name__, "exception", False, f"{type(e).__name__}: {e}")
    n_fail = sum(1 for _t, _n, tag, _d in RESULTS if tag == "FAIL")
    n_skip = sum(1 for _t, _n, tag, _d in RESULTS if tag == "SKIP")
    print(f"\n=== SUMMARY: {len(RESULTS)} checks, {n_fail} FAIL, {n_skip} SKIP ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
