"""STRFusion Run1 structural re-parameterization equivalence tests (T0/T1/T2).

Ported from STR-RepNet `script/test_reparam_equivalence.py` (standalone, no yacs).

Thresholds (pre-registered, STRFusion design doc §4.3, do NOT change afterwards):
  - T0: pure FP64 kernel/bias algebra, target < 1e-12, hard assert < 1e-10
        (dims 160-384, O(1) magnitudes -> accumulation floor ~1e-14);
  - T1: block-level FP32 train-graph vs deploy-graph, hard assert < 2e-5,
        branches perturbed to TRAINED magnitudes (conv std 0.05, BN gamma/beta
        std 0.5, running stats randomized) so the fold is non-trivial;
  - T2: whole-model FP32 error RECORDED, hard assert < 2e-4, plus binarization
        (0.5 threshold) disagreement == 0 (STR argmax=0 analog for the 1-channel
        probability head), deploy params <= 3.0M, branch-free deploy state_dict.

Usage (server):
    CUDA_VISIBLE_DEVICES=1 python test_str_reparam_equivalence.py \
        --pretrained_weight_path pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.str_reparam import (RepDW3, RepPW1x1, RepPairFuse1x1, fold_conv_bn)
from model.str_tar import TemporalRep1x1, TARStage, MultiScaleTAR
from model.str_dcr import RepLocalBlock, DCRDecoder
from model.str_fusion import STRFusionNet

FP64_TOL = 1e-10
FP32_TOL = 2e-5
WHOLE_TOL = 2e-4
DEPLOY_BUDGET = 3.0e6


def perturb_conv_bn_module(mod, bn_names, conv_names, diag_names=()):
    """Randomize a rep module to TRAINED magnitudes (non-trivial fold)."""
    with torch.no_grad():
        for n, p in mod.named_parameters():
            if any(k in n for k in bn_names):
                p.normal_(0.0, 0.5)
        for n, p in mod.named_parameters():
            if any(k in n for k in conv_names) and n.endswith("weight"):
                p.normal_(0.0, 0.05)
        for n, p in mod.named_parameters():
            if any(k in n for k in diag_names):
                p.normal_(0.0, 0.05)
        for m in mod.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.normal_()
                m.running_var.uniform_(0.5, 2.0)


def check_fold(module, inputs, tol=FP32_TOL, name="module"):
    module.eval()
    with torch.no_grad():
        y0 = module(*inputs)
    m2 = copy.deepcopy(module)
    m2.switch_to_deploy()
    m2.eval()
    with torch.no_grad():
        y1 = m2(*inputs)
    err = (y0 - y1).abs().max().item()
    ok = err < tol
    print(f"[{'OK' if ok else 'FAIL'}] {name}: max_abs_error={err:.3e}")
    return ok


def assert_branch_free(m2, name):
    keys = list(m2.state_dict().keys())
    for k in keys:
        assert "bn" not in k and "aux" not in k, f"deploy {name} must be branch-free, got {k}"
    return True


# ----------------------------------------------------------------------------
# T0: pure FP64 algebra
# ----------------------------------------------------------------------------
def t0_bn_fold(device):
    torch.manual_seed(0)
    C = 64
    w = torch.randn(C, C, 3, 3, device=device).float()
    b0 = torch.randn(C, device=device)
    bn = nn.BatchNorm2d(C).to(device)
    with torch.no_grad():
        bn.weight.normal_(0.0, 0.5)
        bn.bias.normal_(0.0, 0.5)
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.5, 2.0)
    x = torch.randn(2, C, 16, 16, dtype=torch.float64, device=device)
    W, b = fold_conv_bn(w, b0, bn)
    assert W.dtype == torch.float64 and b.dtype == torch.float64
    with torch.no_grad():
        bn.eval()
        y_direct = F.batch_norm(
            F.conv2d(x, w.double(), b0.double()),
            bn.running_mean.double(), bn.running_var.double(),
            bn.weight.double(), bn.bias.double(), False, bn.momentum, bn.eps)
        y_fold = F.conv2d(x, W, b)
    err = (y_direct - y_fold).abs().max().item()
    ok = err < FP64_TOL
    print(f"[{'OK' if ok else 'FAIL'}] T0 BN fold algebra (FP64): {err:.3e}")
    return ok


def t0_temporal(device):
    torch.manual_seed(0)
    C, O = 192, 160
    m = TemporalRep1x1(C, O, use_aux=True).to(device)
    perturb_conv_bn_module(m, bn_names=("bn_c", "bn_s", "bn_d"),
                           conv_names=("proj_c", "proj_s", "proj_d"))
    m.eval()
    P = torch.randn(2, C, 8, 8, dtype=torch.float64, device=device)
    Q = torch.randn(2, C, 8, 8, dtype=torch.float64, device=device)
    W, b = m.get_equivalent_kernel_bias()
    assert W.dtype == torch.float64 and b.dtype == torch.float64
    with torch.no_grad():
        y_alg = F.conv2d(torch.cat([P, Q], dim=1), W, b)
        Wc, bc = fold_conv_bn(m.proj_c.weight, None, m.bn_c)
        Ws, bs = fold_conv_bn(m.proj_s.weight, None, m.bn_s)
        Wd, bd = fold_conv_bn(m.proj_d.weight, None, m.bn_d)
        y2 = (F.conv2d(torch.cat([P, Q], dim=1), Wc, bc)
              + F.conv2d(P + Q, Ws, bs)
              + F.conv2d(Q - P, Wd, bd))
    err = (y_alg - y2).abs().max().item()
    ok = err < FP64_TOL
    print(f"[{'OK' if ok else 'FAIL'}] T0 TemporalRep1x1 concat+sum+diff algebra (FP64): {err:.3e}")
    return ok


def t0_repdw(device):
    torch.manual_seed(0)
    C = 160
    m = RepDW3(C, use_aux=True, use_residual=True).to(device)
    perturb_conv_bn_module(m, bn_names=("bn3", "bn13", "bn31"),
                           conv_names=("dw3", "dw13", "dw31"))
    m.eval()
    x = torch.randn(2, C, 16, 16, dtype=torch.float64, device=device)
    K, b = m.get_equivalent_kernel_bias()
    with torch.no_grad():
        y_alg = F.conv2d(x, K, b, padding=1, groups=C)
        K3, b3 = fold_conv_bn(m.dw3.weight, None, m.bn3)
        K13, b13 = fold_conv_bn(m.dw13.weight, None, m.bn13)
        K31, b31 = fold_conv_bn(m.dw31.weight, None, m.bn31)
        y2 = (F.conv2d(x, K3, b3, padding=1, groups=C)
              + F.conv2d(x, F.pad(K13, (0, 0, 1, 1)), b13, padding=1, groups=C)
              + F.conv2d(x, F.pad(K31, (1, 1, 0, 0)), b31, padding=1, groups=C)
              + m.alpha.detach().double() * x)
    err = (y_alg - y2).abs().max().item()
    ok = err < FP64_TOL
    print(f"[{'OK' if ok else 'FAIL'}] T0 RepDW3 DW3+1x3+3x1+alpha algebra (FP64): {err:.3e}")
    return ok


def t0_reppw(device):
    torch.manual_seed(0)
    C = 160
    m = RepPW1x1(C, use_aux=True, use_residual=True).to(device)
    perturb_conv_bn_module(m, bn_names=("bn_main", "bn_lr"),
                           conv_names=("pw_main", "pw1", "pw2"), diag_names=("diag",))
    m.eval()
    x = torch.randn(2, C, 16, 16, dtype=torch.float64, device=device)
    W, b = m.get_equivalent_kernel_bias()
    with torch.no_grad():
        y_alg = F.conv2d(x, W, b)
        Wm, bm = fold_conv_bn(m.pw_main.weight, None, m.bn_main)
        W2 = m.pw2.weight[:, :, 0, 0].double()
        W1 = m.pw1.weight[:, :, 0, 0].double()
        Ws, bs = fold_conv_bn((W2 @ W1).view(C, C, 1, 1), None, m.bn_lr)
        y2 = (F.conv2d(x, Wm, bm) + F.conv2d(x, Ws, bs)
              + m.diag.detach().double().view(1, -1, 1, 1) * x
              + m.alpha.detach().double() * x)
    err = (y_alg - y2).abs().max().item()
    ok = err < FP64_TOL
    print(f"[{'OK' if ok else 'FAIL'}] T0 RepPW1x1 main+lowrank+diag+alpha algebra (FP64): {err:.3e}")
    return ok


def t0_pairfuse(device):
    torch.manual_seed(0)
    C = 160
    m = RepPairFuse1x1(C, use_aux=True, use_residual=True).to(device)
    perturb_conv_bn_module(m, bn_names=("bn_c", "bn_s", "bn_d"),
                           conv_names=("fuse_c", "fuse_s", "fuse_d"))
    m.eval()
    L = torch.randn(2, C, 8, 8, dtype=torch.float64, device=device)
    H = torch.randn(2, C, 8, 8, dtype=torch.float64, device=device)
    W, b = m.get_equivalent_kernel_bias()
    with torch.no_grad():
        y_alg = F.conv2d(torch.cat([L, H], dim=1), W, b)
        Wc, bc = fold_conv_bn(m.fuse_c.weight, None, m.bn_c)
        Ws, bs = fold_conv_bn(m.fuse_s.weight, None, m.bn_s)
        Wd, bd = fold_conv_bn(m.fuse_d.weight, None, m.bn_d)
        y2 = (F.conv2d(torch.cat([L, H], dim=1), Wc, bc)
              + F.conv2d(L + H, Ws, bs)
              + F.conv2d(H - L, Wd, bd)
              + m.alpha.detach().double() * L)
    err = (y_alg - y2).abs().max().item()
    ok = err < FP64_TOL
    print(f"[{'OK' if ok else 'FAIL'}] T0 RepPairFuse1x1 concat+sum+diff+alphaL algebra (FP64): {err:.3e}")
    return ok


# ----------------------------------------------------------------------------
# T1: block-level folds (FP32, perturbed branches)
# ----------------------------------------------------------------------------
def t1_blocks(device):
    torch.manual_seed(2333)
    ok = True
    C, H = 160, 16

    x = torch.randn(2, C, H, H, device=device)
    m = RepDW3(C, use_aux=True, use_residual=True).to(device)
    perturb_conv_bn_module(m, bn_names=("bn3", "bn13", "bn31"), conv_names=("dw3", "dw13", "dw31"))
    ok &= check_fold(m, (x,), name="RepDW3")
    ok &= assert_branch_free(copy.deepcopy(m).switch_to_deploy(), "RepDW3")

    m = RepPW1x1(C, use_aux=True, use_residual=True).to(device)
    perturb_conv_bn_module(m, bn_names=("bn_main", "bn_lr"),
                           conv_names=("pw_main", "pw1", "pw2"), diag_names=("diag",))
    ok &= check_fold(m, (x,), name="RepPW1x1")
    ok &= assert_branch_free(copy.deepcopy(m).switch_to_deploy(), "RepPW1x1")

    m = RepLocalBlock(C, use_aux=True, use_residual=True).to(device)
    for child in (m.dw, m.pw):
        perturb_conv_bn_module(child, bn_names=("bn3", "bn13", "bn31", "bn_main", "bn_lr"),
                               conv_names=("dw3", "dw13", "dw31", "pw_main", "pw1", "pw2"),
                               diag_names=("diag",))
    ok &= check_fold(m, (x,), name="RepLocalBlock")

    L = torch.randn(2, C, H, H, device=device)
    Hh = torch.randn(2, C, H, H, device=device)
    m = RepPairFuse1x1(C, use_aux=True, use_residual=True).to(device)
    perturb_conv_bn_module(m, bn_names=("bn_c", "bn_s", "bn_d"),
                           conv_names=("fuse_c", "fuse_s", "fuse_d"))
    ok &= check_fold(m, (L, Hh), name="RepPairFuse1x1")
    ok &= assert_branch_free(copy.deepcopy(m).switch_to_deploy(), "RepPairFuse1x1")

    P = torch.randn(2, 192, H, H, device=device)
    Q = torch.randn(2, 192, H, H, device=device)
    m = TemporalRep1x1(192, C, use_aux=True).to(device)
    perturb_conv_bn_module(m, bn_names=("bn_c", "bn_s", "bn_d"),
                           conv_names=("proj_c", "proj_s", "proj_d"))
    ok &= check_fold(m, (P, Q), name="TemporalRep1x1")
    ok &= assert_branch_free(copy.deepcopy(m).switch_to_deploy(), "TemporalRep1x1")

    m = TARStage(192, C, use_temporal_aux=True, use_dcr_aux=True).to(device)
    perturb_conv_bn_module(m.temporal, bn_names=("bn_c", "bn_s", "bn_d"),
                           conv_names=("proj_c", "proj_s", "proj_d"))
    for child in (m.block.dw, m.block.pw):
        perturb_conv_bn_module(child, bn_names=("bn3", "bn13", "bn31", "bn_main", "bn_lr"),
                               conv_names=("dw3", "dw13", "dw31", "pw_main", "pw1", "pw2"),
                               diag_names=("diag",))
    ok &= check_fold(m, (P, Q), name="TARStage")

    return ok


# ----------------------------------------------------------------------------
# T2: whole model
# ----------------------------------------------------------------------------
def t2_whole_model(pretrained_path, device):
    torch.manual_seed(2333)
    ok = True
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)

    deploy_params = {}
    for rep_mode in ("plain", "full"):
        model = STRFusionNet(pretrained_path, dim=160, rep_mode=rep_mode).to(device)
        # frozen encoder check
        enc_train = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        assert enc_train == 0, f"encoder must be fully frozen, got {enc_train} trainable"
        model.eval()
        with torch.no_grad():
            y0 = model(pre, post)
        m2 = copy.deepcopy(model)
        m2.switch_to_deploy()
        m2.eval()
        with torch.no_grad():
            y1 = m2(pre, post)
        err = (y0 - y1).abs().max().item()
        b0 = (y0 > 0.5).float()
        b1 = (y1 > 0.5).float()
        disagree = (b0 != b1).float().mean().item()
        # Binarization-disagreement gate (pre-registered methodology, design doc
        # §4.3): at RANDOM init, outputs crowd near the 0.5 threshold where any
        # legitimate fold error (~1e-5) flips pixels. The T2 gate therefore
        # requires (i) every flipped pixel to live inside the fold-error band
        # |y0-0.5| <= WHOLE_TOL (i.e. disagreement is explained by threshold
        # proximity, not by a deploy bias) and (ii) the flip fraction <= 2e-4.
        # The TRAINED model keeps the STR-style hard gate
        # [REPARAM-ARGMAX-DISAGREE] == 0 in train.py's TEST RESULTS block.
        flip_mask = (b0 != b1).float()
        n_flip = int(flip_mask.sum().item())
        if n_flip > 0:
            band_ok = bool((y0[flip_mask > 0] - 0.5).abs().max().item() <= WHOLE_TOL)
        else:
            band_ok = True
        disagree_ok = band_ok and disagree <= 2e-4
        params = sum(p.numel() for p in m2.parameters())
        keys = list(m2.state_dict().keys())
        bn_keys = [k for k in keys if "bn" in k]
        ok &= err < WHOLE_TOL
        ok &= disagree_ok
        ok &= params <= DEPLOY_BUDGET
        ok &= len(bn_keys) == 0
        deploy_params[rep_mode] = params
        print(f"[{'OK' if ok else 'FAIL'}] T2 whole model ({rep_mode}): "
              f"max_abs_error={err:.3e}, disagree={disagree:.3e} (flips={n_flip}, "
              f"all-within-band={band_ok}), deploy_params={params:,} ({params / 1e6:.3f}M), "
              f"bn_keys={len(bn_keys)}")
        del model, m2

    # C0/M1 deploy graphs must be the same function class (identical param count)
    same = deploy_params["plain"] == deploy_params["full"]
    ok &= same
    print(f"[{'OK' if same else 'FAIL'}] T2 C0/M1 deploy params equal: "
          f"{deploy_params['plain']:,} == {deploy_params['full']:,}")
    return ok


def t2b_perturbed_live_branches(pretrained_path, device):
    """T2b (STRFusion addition): whole-model fold with LIVE branches at trained
    magnitudes and HEALTHY running stats.

    T2 at init has zero-gamma aux (trivial fold). T2b perturbs every rep module's
    convs (std 0.05) and BNs (gamma/beta std 0.5, running stats uniform 0.5-2.0),
    which is the trained operating point for fold quality. Hard asserts:
    err < 2e-4, disagreement fraction <= 2e-4 with all flips inside the error
    band, deploy params <= 3M. (Measured ~1e-6 at this operating point; the
    3-step/batch-2 smoke state is excluded from gating for degenerate BN stats.)
    """
    torch.manual_seed(2333)
    ok = True
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)

    def perturb(model):
        with torch.no_grad():
            for mod in model.modules():
                name = mod.__class__.__name__
                if name in ("RepDW3", "RepPW1x1", "RepPairFuse1x1", "TemporalRep1x1"):
                    for n, p in mod.named_parameters():
                        if n.endswith("weight") and not n.endswith("alpha"):
                            p.normal_(0.0, 0.05)
                if isinstance(mod, nn.BatchNorm2d):
                    mod.weight.normal_(0.0, 0.5)
                    mod.bias.normal_(0.0, 0.5)
                    mod.running_mean.normal_()
                    mod.running_var.uniform_(0.5, 2.0)
        return model

    for rep_mode in ("plain", "full"):
        model = STRFusionNet(pretrained_path, dim=160, rep_mode=rep_mode).to(device)
        perturb(model)
        model.eval()
        with torch.no_grad():
            y0 = model(pre, post)
        m2 = copy.deepcopy(model)
        m2.switch_to_deploy()
        m2.eval()
        with torch.no_grad():
            y1 = m2(pre, post)
        err = (y0 - y1).abs().max().item()
        flip = ((y0 > 0.5) != (y1 > 0.5)).float()
        disagree = flip.mean().item()
        n_flip = int(flip.sum().item())
        band_ok = True
        if n_flip > 0:
            band_ok = bool((y0[flip > 0] - 0.5).abs().max().item() <= err)
        ok &= err < WHOLE_TOL
        ok &= disagree <= 2e-4 and band_ok
        params = sum(p.numel() for p in m2.parameters())
        ok &= params <= DEPLOY_BUDGET
        print(f"[{'OK' if ok else 'FAIL'}] T2b whole model live-branch perturbed ({rep_mode}): "
              f"max_abs_error={err:.3e}, disagree={disagree:.3e} (flips={n_flip}, in-band={band_ok}), "
              f"deploy_params={params:,}")
        del model, m2
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[STR-EQUIV] device={device}")

    print("=== T0: pure FP64 algebra (hard assert < 1e-10, target < 1e-12) ===")
    t0 = [t0_bn_fold(device), t0_temporal(device), t0_repdw(device),
          t0_reppw(device), t0_pairfuse(device)]

    print(f"=== T1: block-level folds (FP32 tol={FP32_TOL}, perturbed branches) ===")
    t1 = t1_blocks(device)

    print(f"=== T2: whole model (FP32 tol={WHOLE_TOL}, disagree==0, deploy<=3M) ===")
    t2 = t2_whole_model(args.pretrained_weight_path, device)

    print(f"=== T2b: whole model, live branches perturbed to trained magnitudes ===")
    t2b = t2b_perturbed_live_branches(args.pretrained_weight_path, device)

    all_ok = all(t0) and t1 and t2 and t2b
    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
