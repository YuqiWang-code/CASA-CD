"""R4-FET1 验收测试 T0–T9（方案 §7）：逐项硬断言，不接受“基本一致”。

运行（服务器，GPU0 独占）：

    cd models
    python test_run4_fine_tap.py \
      --tinyvim-pretrained-weight-path ../pretrained_weight/tinyvim_s_1000e.pth \
      --device cuda:0 \
      --real-dataset-root /share_datasets/CD/SYSU-CD-256 \
      --real-test-list /share_datasets/CD/SYSU-CD-256/list/test.txt --limit 16 \
      --dry-run-json /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_DRY/E6_FET1/dry_run.json

设计纪律：
  * 每个断言独立计分；任一 FAIL ⇒ 进程退出码 1（FAIL STOP）。
  * 需要 CUDA selective-scan 内核的测试（全模型 forward）在无内核环境标 SKIP-NO-KERNEL，
    **在目标服务器上必须实测通过**，SKIP 不计入 PASS。
  * 不修改任何历史权重/日志；T9 的注入实验全部在临时目录进行。
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.casa_tvim_str_net import CASATViMSTRNet                       # noqa: E402
from model.str_fine_tap import FineEvidenceTap1x1, FET_FORM, FET_FUSE, FET_SOURCE  # noqa: E402
from train import (measure_params, measure_trainable_params, measure_flops,  # noqa: E402
                   source_code_identity, PROTOCOL_VERSION, SOURCE_IDENTITY_FILES)

EXPECT = {
    "ctrl_train": 5_084_017,
    "ctrl_deploy": 4_880_190,
    "fet_train": 5_097_938,
    "fet_deploy": 4_894_110,
    "fet_train_inc": 13_921,
    "fet_deploy_inc": 13_920,
    "fet_mac": 64 * 64 * 144 * 96,      # 56,623,104
    "budget": 5_000_000,
}

FOLD_FP64_TOL = 1e-10
FOLD_FP32_TOL = 2e-5
WHOLE_FP32_TOL = 2e-4


class Harness(object):
    """逐项验收记录器（FAIL STOP 语义）。"""

    def __init__(self, verbose=True):
        self.rows = []
        self.verbose = verbose

    def check(self, tid, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        self.rows.append((tid, name, status, detail))
        if self.verbose:
            print(f"[{status}] {tid} {name}" + (f" :: {detail}" if detail else ""), flush=True)
        return bool(ok)

    def skip(self, tid, name, detail):
        self.rows.append((tid, name, "SKIP", detail))
        print(f"[SKIP] {tid} {name} :: {detail}", flush=True)
        return True

    def summary(self):
        print("\n=== R4 T0-T9 SUMMARY ===")
        for tid, name, status, detail in self.rows:
            print(f"{status:4s} | {tid:3s} | {name}" + (f" | {detail}" if detail else ""))
        n_fail = sum(1 for r in self.rows if r[2] == "FAIL")
        n_skip = sum(1 for r in self.rows if r[2] == "SKIP")
        n_pass = sum(1 for r in self.rows if r[2] == "PASS")
        print(f"--- PASS={n_pass} FAIL={n_fail} SKIP={n_skip} ---")
        return n_fail


def _sha256(path, nbytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() if nbytes is None else h.hexdigest()[:nbytes]


def build_model(fine_tap, pretrained_path, device, seed=16):
    """严格复刻 train.py 的构造序列（encoder → head → tar → decoder → FET）。"""
    torch.manual_seed(seed)
    model = CASATViMSTRNet(
        pretrained_path, caacp=True, rep_mode="full", str_dim=96,
        caacp_score_mode="rank", frh=False, caacp_residual_mode="current",
        fs_tar=False, fine_tap=bool(fine_tap)).float()
    return model.to(device)


def _kernel_ok(fn):
    """执行 fn；若因缺少 selective-scan 内核失败则返回 ('NOKERNEL', msg)。"""
    try:
        return True, fn()
    except RuntimeError as e:
        if "selective scan kernel" in str(e) or "mamba" in str(e):
            return "NOKERNEL", str(e)
        raise


# --------------------------------------------------------------------------- T0
def t0_identity(h, args, device):
    path = args.tinyvim_pretrained_weight_path
    if path is None or not os.path.isfile(path):
        h.check("T0", "pretrained weight present", False, f"missing {path}")
        return None
    base = os.path.basename(path)
    h.check("T0", "pretrained file is TinyViM-S 1000e (never a Run1 best ckpt)",
            base == "tinyvim_s_1000e.pth", base)
    sha_full = _sha256(path)
    print(f"[T0] pretrained sha256={sha_full}", flush=True)
    if args.source_identity_json and os.path.isfile(args.source_identity_json):
        with open(args.source_identity_json, encoding="utf-8") as f:
            expected = json.load(f)
        expected = expected.get("source_code_sha256", expected)
        cur = source_code_identity()
        bad = [k for k in cur if expected.get(k) not in (None, cur[k])]
        h.check("T0", "source code identity matches Run4 manifest", not bad,
                f"files={len(cur)} mismatched={bad}" if bad else f"files={len(cur)} all match")
    else:
        h.skip("T0", "source code identity vs manifest", "--source-identity-json not given (record manually)")

    model = build_model(1, path, device)
    ls = model.encoder.load_stats() or {}
    ok_load = (ls.get("retained") == ls.get("pretrained_keys")) and float(ls.get("worst_diff", 1.0)) == 0.0
    h.check("T0", "ImageNet backbone exact-loaded (retained==pretrained, worst_diff==0)", ok_load,
            f"retained={ls.get('retained')}/{ls.get('pretrained_keys')} worst_diff={ls.get('worst_diff')}")
    h.check("T0", "no FET key came from the pretrained file",
            all("fine_evidence_tap" not in k for k in (ls.get("loaded_keys") or [])),
            f"new_modules={len(ls.get('missing_new', []))}")
    return model, sha_full


# --------------------------------------------------------------------------- T1
def t1_ctrl_params(h, args, device):
    model = build_model(0, args.tinyvim_pretrained_weight_path, device)
    tr = measure_params(model)
    ok_tr = (tr == EXPECT["ctrl_train"])
    h.check("T1", f"fine_tap=0 train params == {EXPECT['ctrl_train']:,}", ok_tr, f"got {tr:,}")
    h.check("T1", "fine_tap=0 registers no FET module/param",
            getattr(model, "fine_evidence_tap", None) is None
            and not any(k.startswith("fine_evidence_tap") for k in model.state_dict()))
    n_keys = len(model.state_dict())
    model.switch_to_deploy()
    dp = measure_params(model)
    h.check("T1", f"fine_tap=0 deploy params == {EXPECT['ctrl_deploy']:,}", dp == EXPECT["ctrl_deploy"], f"got {dp:,}")
    h.check("T1", "fine_tap=0 deploy graph carries no FET key",
            not any(k.startswith("fine_evidence_tap") for k in model.state_dict()),
            f"train_keys={n_keys} deploy_keys={len(model.state_dict())} (fold removes TAR/DCR branch keys by design)")
    return True


# --------------------------------------------------------------------------- T2
def t2_epoch0(h, args, device):
    m0 = build_model(0, args.tinyvim_pretrained_weight_path, device)
    r0 = torch.get_rng_state().clone()
    m1 = build_model(1, args.tinyvim_pretrained_weight_path, device)
    r1 = torch.get_rng_state().clone()
    h.check("T2", "global CPU RNG state identical after both constructions", torch.equal(r0, r1))

    s0, s1 = m0.state_dict(), m1.state_dict()
    extra = sorted(k for k in s1 if k not in s0)
    missing = sorted(k for k in s0 if k not in s1)
    h.check("T2", "new keys are exactly fine_evidence_tap.*", not missing
            and all(k.startswith("fine_evidence_tap.") for k in extra), f"new={extra} missing={missing}")
    bad = [k for k in s0 if not torch.equal(s0[k], s1[k])]
    h.check("T2", "all pre-existing tensors bitwise identical (torch.equal)", not bad,
            f"n_keys={len(s0)} mismatched={bad[:4]}")
    gamma = m1.fet_gamma()
    dw = float(m1.fine_evidence_tap.diff.weight.abs().sum().item())
    h.check("T2", "gamma == 0 and W_diff == 0", gamma == 0.0 and dw == 0.0, f"gamma={gamma} |W_diff|_1={dw}")

    if args.device.startswith("cuda"):
        x = torch.randn(2, 3, 256, 256, device=device)
        m0.eval(); m1.eval()
        with torch.no_grad():
            y0 = m0(x, x)
            y1 = m1(x, x)
        h.check("T2", "epoch-0 forward bitwise identical (torch.equal, max_abs==0)", torch.equal(y0, y1),
                f"max_abs={(y0 - y1).abs().max().item():.3e} disagree={((y0 > .5) != (y1 > .5)).float().mean().item():.3e}")
    else:
        h.skip("T2", "epoch-0 forward bitwise identical", "needs CUDA selective-scan kernel")
    return m1


# --------------------------------------------------------------------------- T3
def t3_shapes(h, args, device):
    """FET 局部（无需内核）+ 全模型（需内核）形状契约。"""
    tap = FineEvidenceTap1x1(48, 96)
    for B in (1, 2, 16):
        P = torch.randn(B, 48, 64, 64)
        Q = torch.randn(B, 48, 64, 64)
        with torch.no_grad():
            Z = torch.cat([P, Q, (Q - P).abs()], dim=1)
            T = tap(P, Q)
        h.check("T3", f"local FET shapes B={B}",
                list(P.shape) == [B, 48, 64, 64] and list(Z.shape) == [B, 144, 64, 64]
                and list(T.shape) == [B, 96, 64, 64],
                f"P={list(P.shape)} Z={list(Z.shape)} T={list(T.shape)}")

    if not args.device.startswith("cuda"):
        h.skip("T3", "full-model shape contract", "needs CUDA selective-scan kernel")
        return
    model = build_model(1, args.tinyvim_pretrained_weight_path, device)
    model.eval()
    seen = {}

    def pre_hook(mod, inp):
        seen["P"], seen["Q"] = inp[0].shape, inp[1].shape

    def out_hook(mod, inp, out):
        seen["T"] = out.shape

    def head_hook(mod, inp):
        seen["Rprime"] = inp[0].shape

    hs = [model.fine_evidence_tap.register_forward_pre_hook(pre_hook),
          model.fine_evidence_tap.register_forward_hook(out_hook),
          model.head.register_forward_pre_hook(head_hook)]
    with torch.no_grad():
        y = model(torch.randn(2, 3, 256, 256, device=device), torch.randn(2, 3, 256, 256, device=device))
    for hh in hs:
        hh.remove()
    ok = (list(seen.get("P", [])) == [2, 48, 64, 64] and list(seen.get("Q", [])) == [2, 48, 64, 64]
          and list(seen.get("T", [])) == [2, 96, 64, 64] and list(seen.get("Rprime", [])) == [2, 96, 64, 64]
          and list(y.shape) == [2, 1, 256, 256])
    h.check("T3", "full-model shapes P/Q[2,48,64,64] T/R'[2,96,64,64] out[2,1,256,256]", ok, str(seen))


# --------------------------------------------------------------------------- T4
def t4_fold(h, args, device):
    torch.manual_seed(0)
    tap = FineEvidenceTap1x1(48, 96)
    with torch.no_grad():
        tap.gamma.fill_(0.3)
        tap.diff.weight.normal_(0.0, 0.05)
        tap.pq.bias.add_(torch.randn(96) * 0.02)
    P = torch.randn(4, 48, 64, 64)
    Q = torch.randn(4, 48, 64, 64)
    D = (Q - P).abs()

    # (1) FP64 代数恒等式：两侧都在 FP64 上求值，检验 γ[W_pq,W_Δ] 拼核的正确性
    Wpq64 = tap.pq.weight.detach().double()
    bpq64 = tap.pq.bias.detach().double()
    Wd64 = tap.diff.weight.detach().double()
    g64 = tap.gamma.detach().double().reshape(())
    train_ref = g64 * (torch.nn.functional.conv2d(torch.cat([P, Q], dim=1).double(), Wpq64, bpq64)
                       + torch.nn.functional.conv2d(D.double(), Wd64, None))
    W, b = tap.get_equivalent_kernel_bias()
    h.check("T4", "fused kernel is 96x144x1x1 (FP64)", list(W.shape) == [96, 144, 1, 1] and W.dtype == torch.float64,
            f"shape={list(W.shape)} dtype={W.dtype}")
    fused_ref = torch.nn.functional.conv2d(torch.cat([P, Q, D], dim=1).double(), W, b.reshape(-1))
    err64 = (train_ref - fused_ref).abs().max().item()
    h.check("T4", f"FP64 algebraic fold max_abs < {FOLD_FP64_TOL:g}", err64 < FOLD_FP64_TOL, f"max_abs={err64:.3e}")

    # (2) FP32 子模块等价：训练图模块 vs 折叠后模块
    train_f32 = tap(P, Q)
    tap.switch_to_deploy()
    fused_out = tap(P, Q)
    err32 = (train_f32 - fused_out).abs().max().item()
    h.check("T4", f"FP32 submodule train<->deploy max_abs < {FOLD_FP32_TOL:g}", err32 < FOLD_FP32_TOL,
            f"max_abs={err32:.3e} disagree={((train_f32 > .5) != (fused_out > .5)).float().mean().item():.3e}")
    keys = set(tap.state_dict())
    h.check("T4", "pq/diff/gamma removed; single Conv(144->96,k1) remains",
            keys == {"fused.weight", "fused.bias"} and tap.fused.kernel_size == (1, 1)
            and tap.fused.in_channels == 144 and tap.fused.out_channels == 96, f"keys={sorted(keys)}")
    rep = tap.param_report()
    h.check("T4", f"deploy param increment == {EXPECT['fet_deploy_inc']:,}", rep["fused"] == EXPECT["fet_deploy_inc"],
            f"got {rep['fused']:,}")
    n_fold = tap.fold_count
    tap.switch_to_deploy()
    out2 = tap(P, Q)
    h.check("T4", "second switch_to_deploy is idempotent (no re-fold, same output)",
            tap.fold_count == n_fold and torch.equal(out2, fused_out), f"fold_count={tap.fold_count}")


# --------------------------------------------------------------------------- T5
def t5_train_deploy(h, args, device, model=None):
    if not args.device.startswith("cuda"):
        h.skip("T5", "whole-model train<->deploy equivalence", "needs CUDA selective-scan kernel")
        return
    prev = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.deterministic)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    try:
        m = build_model(1, args.tinyvim_pretrained_weight_path, device)
        with torch.no_grad():                        # 非平凡：γ、W_diff 开门
            m.fine_evidence_tap.gamma.fill_(0.3)
            m.fine_evidence_tap.diff.weight.normal_(0.0, 0.02)
        m.eval()
        torch.manual_seed(16)
        pre_fix = torch.randn(8, 3, 256, 256, device=device)
        post_fix = torch.randn(8, 3, 256, 256, device=device)
        real = _load_real_batch(args, device)
        with torch.no_grad():
            y_tr = m(pre_fix, post_fix)
            y_tr_real = m(real[0], real[1]) if real is not None else None
        m.switch_to_deploy()
        m.eval()
        with torch.no_grad():
            y_dp = m(pre_fix, post_fix)
            y_dp_real = m(real[0], real[1]) if real is not None else None
        e = (y_tr - y_dp).abs().max().item()
        d = ((y_tr > 0.5) != (y_dp > 0.5)).float().mean().item()
        h.check("T5", f"random batch max_abs < {WHOLE_FP32_TOL:g} and binary disagreement == 0",
                e < WHOLE_FP32_TOL and d == 0.0, f"max_abs={e:.3e} disagree={d:.3e}")
        if y_tr_real is not None:
            er = (y_tr_real - y_dp_real).abs().max().item()
            dr = ((y_tr_real > 0.5) != (y_dp_real > 0.5)).float().mean().item()
            margin = (y_tr_real - 0.5).abs().min().item()
            h.check("T5", "real test batch (b=16) max_abs < 2e-4 and binary disagreement == 0",
                    er < WHOLE_FP32_TOL and dr == 0.0,
                    f"max_abs={er:.3e} disagree={dr:.3e} min|p-0.5|={margin:.3e}")
        else:
            h.skip("T5", "real test batch equivalence", "no --real-dataset-root/--real-test-list")
        dp = measure_params(m)
        h.check("T5", f"deploy graph total params == {EXPECT['fet_deploy']:,}", dp == EXPECT["fet_deploy"], f"got {dp:,}")
    finally:
        (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.deterministic) = prev


def _load_real_batch(args, device):
    if not (args.real_dataset_root and args.real_test_list):
        return None
    if not (os.path.isdir(args.real_dataset_root) and os.path.isfile(args.real_test_list)):
        return None
    import dataset.dataset as myDataLoader
    import dataset.Transforms as myTransforms
    tf = myTransforms.Compose([
        myTransforms.Normalize(mean=[0.406, 0.456, 0.485, 0.406, 0.456, 0.485],
                               std=[0.225, 0.224, 0.229, 0.225, 0.224, 0.229]),
        myTransforms.Scale(args.inWidth if hasattr(args, "inWidth") else 256, 256),
        myTransforms.ToTensor(),
    ])
    ds = myDataLoader.Dataset(file_root=args.real_dataset_root, list_path=args.real_test_list, transform=tf)
    n = min(int(args.limit), len(ds))
    imgs = torch.stack([ds[i][0] for i in range(n)])
    return imgs[:, 0:3].to(device), imgs[:, 3:6].to(device)


# --------------------------------------------------------------------------- T6
def t6_params_flops(h, args, device):
    m = build_model(1, args.tinyvim_pretrained_weight_path, device)
    tr, trn = measure_params(m), measure_trainable_params(m)
    ctrl = build_model(0, args.tinyvim_pretrained_weight_path, device)
    ctrl_tr = measure_params(ctrl)
    h.check("T6", f"train-graph params == {EXPECT['fet_train']:,}", tr == EXPECT["fet_train"], f"got {tr:,}")
    h.check("T6", f"FET train increment == {EXPECT['fet_train_inc']:,}", tr - ctrl_tr == EXPECT["fet_train_inc"],
            f"got {tr - ctrl_tr:,}")
    h.check("T6", "FET new params are trainable (no frozen backbone trick)", trn == tr, f"trainable={trn:,}")
    m.switch_to_deploy()
    dp, dpn = measure_params(m), measure_trainable_params(m)
    h.check("T6", f"deploy params == {EXPECT['fet_deploy']:,} and <= 5,000,000",
            dp == EXPECT["fet_deploy"] and dp <= EXPECT["budget"], f"got {dp:,}")
    h.check("T6", f"FET deploy increment == {EXPECT['fet_deploy_inc']:,}", dp - EXPECT["ctrl_deploy"] == EXPECT["fet_deploy_inc"],
            f"got {dp - EXPECT['ctrl_deploy']:,}")
    h.check("T6", "train/deploy params printed separately (train > 5M is allowed, deploy is the hard gate)",
            tr > EXPECT["budget"] >= dp, f"train={tr:,} deploy={dp:,}")
    mac = 64 * 64 * 144 * 96
    h.check("T6", f"FET fused MAC == {EXPECT['fet_mac']:,} (fvcore convention 1 MAC = 1 FLOP)",
            mac == EXPECT["fet_mac"], f"manual={mac:,} (0.0566231 G)")
    if args.device.startswith("cuda"):
        ok, res = _kernel_ok(lambda: measure_flops(m, size=256))
        if ok is True:
            flops, unsup = res
            h.check("T6", "deploy fvcore FLOPs measured (record [DEPLOY-FLOPS] + unsupported_ops)",
                    flops > 0, f"{flops:.4f} G unsupported_ops={unsup}")
        else:
            h.skip("T6", "deploy fvcore FLOPs", res)
    else:
        h.skip("T6", "deploy fvcore FLOPs", "needs CUDA")


# --------------------------------------------------------------------------- T7
def t7_gradients(h, args, device):
    if not args.device.startswith("cuda"):
        h.skip("T7", "3-step gradient path", "needs CUDA selective-scan kernel")
        return
    from model.utils import BCEDiceLoss
    m = build_model(1, args.tinyvim_pretrained_weight_path, device)
    m.train()
    backbone = [p for n, p in m.named_parameters() if n.startswith("encoder.")]
    new = [p for n, p in m.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.Adam([{"params": backbone, "lr": 2e-5, "lr_scale": 0.1, "name": "backbone"},
                            {"params": new, "lr": 2e-4, "lr_scale": 1.0, "name": "new"}],
                           2e-4, (0.9, 0.99), eps=1e-8, weight_decay=1e-4)
    torch.manual_seed(16)
    pre = torch.randn(4, 3, 256, 256, device=device)
    post = torch.randn(4, 3, 256, 256, device=device)
    tgt = (torch.rand(4, 1, 256, 256, device=device) > 0.7).float()
    lrs, gammas = [], []
    finite = True
    for step in range(3):
        out = m(pre, post, tgt)
        loss = BCEDiceLoss(out, tgt)
        opt.zero_grad()
        loss.backward()
        g = m.fine_evidence_tap.gamma.grad
        gammas.append(None if g is None else float(g.item()))
        lrs.append([round(pg["lr"], 12) for pg in opt.param_groups])
        if not torch.isfinite(loss):
            finite = False
        dw = m.fine_evidence_tap.diff.weight.grad
        if step >= 1 and (dw is None or float(dw.abs().sum().item()) == 0.0):
            finite = False
        opt.step()
    h.check("T7", "loss finite for 3 steps, no NaN", finite)
    h.check("T7", "step-1 gamma.grad finite and non-zero", gammas[0] is not None
            and gammas[0] != 0.0 and abs(gammas[0]) < 1e3, f"gamma.grad[0]={gammas[0]}")
    h.check("T7", "after gamma opens, W_diff gradient non-zero at steps 2 and 3",
            all(x is not None and x != 0.0 for x in gammas[1:]),
            f"gamma.grad={['%.3e' % x if x is not None else None for x in gammas]}")
    grp_ok = all(abs(l[0] / l[1] - 0.1) < 1e-9 for l in lrs)
    h.check("T7", "per-group lr ratio backbone:new == 0.1 preserved by the scheduler", grp_ok, f"lrs={lrs[0]}")
    miss = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or float(p.grad.abs().sum().item()) == 0.0)
            and not n.startswith("fine_evidence_tap.diff")]
    h.check("T7", "encoder/TAR/DCR/head all still receive gradients", not miss, f"zero-grad params={miss[:5]}")


# --------------------------------------------------------------------------- T8
def t8_dry_run(h, args, device):
    if not args.dry_run_json:
        h.skip("T8", "real-data 3-step dry-run", "run models/run4_dry_run.py then pass --dry-run-json")
        return
    if not os.path.isfile(args.dry_run_json):
        h.check("T8", "dry_run.json present", False, args.dry_run_json)
        return
    with open(args.dry_run_json, encoding="utf-8") as f:
        d = json.load(f)
    h.check("T8", "actual_steps == 3 (dry-only, no best ckpt written)",
            int(d.get("actual_steps", -1)) == 3 and not d.get("wrote_best", True),
            f"actual_steps={d.get('actual_steps')} wrote_best={d.get('wrote_best')}")
    h.check("T8", "loss finite on every step", all(bool(x) for x in d.get("loss_finite", [])),
            f"losses={d.get('losses')}")
    h.check("T8", "A/B/label share the same geometry mapping and label is gray>=128 binarized",
            bool(d.get("geom_sync")) and bool(d.get("label_gray128")),
            f"geom_sync={d.get('geom_sync')} label_gray128={d.get('label_gray128')}")
    h.check("T8", "gamma received gradient on step 1 and W_diff on steps 2-3",
            bool(d.get("gamma_grad_step1")) and bool(d.get("diff_grad_step23")))
    h.check("T8", "batch32 resource probe recorded", bool(d.get("batch32_ok")), f"bs={d.get('batch_size')}")


# --------------------------------------------------------------------------- T9
def t9_eval_injection(h, args, device):
    """用真实 eval.py 做错误组合注入：必须非零退出，不允许 strict=False 静默成功。"""
    tmp = tempfile.mkdtemp(prefix="r4_t9_")
    try:
        ctrl = build_model(0, args.tinyvim_pretrained_weight_path, "cpu")
        fet = build_model(1, args.tinyvim_pretrained_weight_path, "cpu")
        fet_deploy = build_model(1, args.tinyvim_pretrained_weight_path, "cpu")
        fet_deploy.switch_to_deploy()
        arch_fet = {"arch": "casa_tvim_str", "backbone": "tinyvim_s_slim", "caacp": 1,
                    "caacp_score_mode": "rank", "caacp_residual_mode": "current",
                    "frh": 0, "fs_tar": 0, "fine_tap": 1,
                    "fine_tap_form": FET_FORM, "fine_tap_source": FET_SOURCE, "fine_tap_fuse": FET_FUSE,
                    "rep_mode": "full", "str_dim": 96}
        arch_ctrl = dict(arch_fet, fine_tap=0, fine_tap_form="none",
                         fine_tap_source="none", fine_tap_fuse="none")
        cases = []

        def make_case(name, arch, state_dict, fine_tap_flag, expect_marker=None):
            d = os.path.join(tmp, name)
            os.makedirs(d, exist_ok=True)
            torch.save(state_dict, os.path.join(d, "best_F1=0.5000.pth"))
            if arch is not None:
                with open(os.path.join(d, "arch.json"), "w", encoding="utf-8") as f:
                    json.dump(arch, f)
            cases.append((name, d, fine_tap_flag, expect_marker))

        make_case("cli_fine_tap0_on_fet_ckpt", arch_fet, fet.state_dict(), 0, "[ARCH-MISMATCH]")
        make_case("missing_sidecar_with_fine_tap1", None, fet.state_dict(), 1, "[ARCH-MISSING]")
        make_case("wrong_fine_tap_form", dict(arch_fet, fine_tap_form="pq_abs_3x3_v1"),
                  fet.state_dict(), 1, "[ARCH-MISMATCH]")
        make_case("fet_ckpt_but_arch_says_ctrl", arch_ctrl, fet.state_dict(), 1, "[ARCH-MISMATCH]")
        make_case("missing_fet_weights", arch_fet, ctrl.state_dict(), 1, None)
        make_case("deploy_state_misused_as_train_ckpt", arch_fet, fet_deploy.state_dict(), 1, None)

        # T9 只检验 sidecar/strict-load 门；用结构合法但不含任何权重的假预训练文件，
        # 避免 FileNotFoundError 掩盖真正的判定（真实 T0 已单独核验预训练身份）。
        dummy_pre = os.path.join(tmp, "dummy_tinyvim_s_1000e.pth")
        torch.save({"model_ema": {}}, dummy_pre)
        pre_arg = str(args.tinyvim_pretrained_weight_path) if args.tinyvim_pretrained_weight_path \
            else dummy_pre

        for name, d, flag, marker in cases:
            cmd = [sys.executable, os.path.join(_MODELS_ROOT, "eval.py"),
                   "--arch", "casa_tvim_str", "--ckpt_dir", d, "--fine_tap", str(flag),
                   "--dataset", "SYSU-CD-256",
                   "--caacp", "1", "--caacp_score_mode", "rank", "--caacp_residual_mode", "current",
                   "--rep_mode", "full", "--str_dim", "96",
                   "--dataset_root", os.path.join(tmp, "nonexistent_data"),
                   "--test_list", os.path.join(tmp, "nonexistent_list.txt"),
                   "--pretrained_weight_path", pre_arg,
                   "--onGPU", "False"]
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=_MODELS_ROOT)
            out = (p.stdout or "") + (p.stderr or "")
            if marker is None:
                # 必须死在 strict load（键缺失/多余/形状不符），不能是别的原因
                load_fail = any(s in out for s in ("Missing key(s)", "Unexpected key(s)",
                                                   "size mismatch", "Error(s) in loading state_dict"))
                ok = p.returncode != 0 and load_fail
                tag = "strict-load" if load_fail else "OTHER-REASON"
            else:
                ok = p.returncode != 0 and marker in out
                tag = "ok" if marker in out else "MISSING"
            h.check("T9", f"reject: {name}", ok,
                    f"rc={p.returncode} marker={tag} "
                    f"tail={out.strip().splitlines()[-1][:120] if out.strip() else ''}")

        # 正向对照：正确的组合不得被 arch 检查误杀
        d = os.path.join(tmp, "correct_combo")
        os.makedirs(d, exist_ok=True)
        torch.save(fet.state_dict(), os.path.join(d, "best_F1=0.5000.pth"))
        with open(os.path.join(d, "arch.json"), "w", encoding="utf-8") as f:
            json.dump(arch_fet, f)
        cmd = [sys.executable, os.path.join(_MODELS_ROOT, "eval.py"),
               "--arch", "casa_tvim_str", "--ckpt_dir", d, "--fine_tap", "1",
               "--dataset", "SYSU-CD-256",
               "--caacp", "1", "--caacp_score_mode", "rank", "--caacp_residual_mode", "current",
               "--rep_mode", "full", "--str_dim", "96",
               "--dataset_root", os.path.join(tmp, "nonexistent_data"),
               "--test_list", os.path.join(tmp, "nonexistent_list.txt"),
               "--pretrained_weight_path", pre_arg,
               "--onGPU", "False"]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=_MODELS_ROOT)
        out = (p.stdout or "") + (p.stderr or "")
        h.check("T9", "positive control: correct combo passes the arch gate (strict load succeeds)",
                "[ARCH-MISMATCH]" not in out and "[ARCH-MISSING]" not in out
                and "size mismatch" not in out and "Missing key(s)" not in out
                and "Unexpected key(s)" not in out,
                f"rc={p.returncode}")

        # 训练后 checkpoint（可选）：裸 state_dict 必须 strict 加载
        if args.checkpoint and os.path.isfile(args.checkpoint):
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck)
            m = build_model(1, args.tinyvim_pretrained_weight_path, "cpu")
            res = m.load_state_dict(sd, strict=True)
            h.check("T9", "post-train checkpoint strict-loads into the fine_tap=1 train graph",
                    not res.missing_keys and not res.unexpected_keys, f"{args.checkpoint}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinyvim-pretrained-weight-path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--real-dataset-root", type=str, default=None)
    ap.add_argument("--real-test-list", type=str, default=None)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--source-identity-json", type=str, default=None)
    ap.add_argument("--dry-run-json", type=str, default=None)
    ap.add_argument("--skip-fold-heavy", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] {device} unavailable, falling back to cpu (heavy tests will SKIP)")
        device = "cpu"
        args.device = "cpu"

    h = Harness()
    print(f"[R4] protocol={PROTOCOL_VERSION} device={device} "
          f"pretrained={args.tinyvim_pretrained_weight_path}", flush=True)
    print("[R4] code identity: " + json.dumps(source_code_identity()), flush=True)

    t0_identity(h, args, device)
    t1_ctrl_params(h, args, device)
    t2_epoch0(h, args, device)
    t3_shapes(h, args, device)
    t4_fold(h, args, device)
    t5_train_deploy(h, args, device)
    t6_params_flops(h, args, device)
    t7_gradients(h, args, device)
    t8_dry_run(h, args, device)
    t9_eval_injection(h, args, device)

    n_fail = h.summary()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
