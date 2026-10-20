#!/usr/bin/env python
"""Run8b 真实数据 dry-run（抗自锁版）：3 步 + 6 道门 + **抗自锁证据** + 折叠刀锋 + batch32 显存。

**只读语义**：绝不写 `best_F1=*.pth`；不触碰 Run1–Run8 任何产物。

    python models/run8b_dry_run.py --variant B1_HYP \
      --dataset-root /share_datasets/CD/SYSU-CD-256 \
      --train-list /share_datasets/CD/SYSU-CD-256/list/train.txt \
      --vmamba-pretrained-weight-path .../vssm_tiny_0230_ckpt_epoch_262.pth \
      --device cuda:0 --output-dir .../Run8b_DRY/SYSU-CD-256/B1_HYP

变体（与 `train_scripts/CASA-TViM/Run8b/_r8b_run_one_ds.sh::variant_args` 必须一致）：
  CTRL_R8B  : hqkv=0,  hgsl=0                       （同期对照）
  B1_HYP    : hqkv=1 hyperbolic, hgsl=0, dcr1        （抗自锁主实验）
  B2_EUC    : hqkv=1 euclidean,  hgsl=0, dcr1        （曲率必要性对照）
  B3_HYP_GS : hqkv=1 hyperbolic, hgsl=1 detach=1     （+HGSL）
"""
import argparse
import copy
import json
import os
import sys

import torch

_MODELS = os.path.dirname(os.path.abspath(__file__))
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

import dataset.dataset as myDataLoader                                    # noqa: E402
import dataset.Transforms as myTransforms                                  # noqa: E402
from model.casa_vmamba_net_v2 import CASAVMambaSTRNetV2                    # noqa: E402
from model.hgsl_loss import HGSL_STAGES_DEFAULT, hgsl_loss                 # noqa: E402
from model.utils import BCEDiceLoss, init_seed                              # noqa: E402

VARIANTS = {
    "CTRL_R8B":  dict(hqkv=0, geometry="hyperbolic", hgsl=0, hgsl_detach_score=1,
                      inject_mode="none"),
    "B1_HYP":    dict(hqkv=1, geometry="hyperbolic", hgsl=0, hgsl_detach_score=1,
                      inject_mode="dcr1"),
    "B2_EUC":    dict(hqkv=1, geometry="euclidean", hgsl=0, hgsl_detach_score=1,
                      inject_mode="dcr1"),
    "B3_HYP_GS": dict(hqkv=1, geometry="hyperbolic", hgsl=1, hgsl_detach_score=1,
                      inject_mode="dcr1"),
}
MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
RMS_MIN = 0.05          # 注入 RMS 的下限（v1 恒为 0）


def build_transform():
    return myTransforms.Compose([myTransforms.Normalize(mean=MEAN6, std=STD6),
                                 myTransforms.Scale(256, 256), myTransforms.ToTensor()])


def make_model(variant, args, device):
    v = VARIANTS[variant]
    torch.manual_seed(args.seed)
    m = CASAVMambaSTRNetV2(
        vmamba_pretrained_path=args.vmamba_pretrained_weight_path,
        hqkv=bool(v["hqkv"]), str_dim=args.str_dim,
        geometry=v["geometry"],
        hgsl=bool(v["hgsl"]), hgsl_detach_score=bool(v["hgsl_detach_score"]),
        inject_mode=v["inject_mode"],
        hqkv_rank=args.hqkv_rank, hqkv_lite_dim=args.str_dim,
        hqkv_detail_window=args.hqkv_detail_window,
        hqkv_global_window=args.hqkv_global_window,
        hqkv_attn_stages=tuple(int(s) for s in args.hqkv_attn_stages.split(",") if s != ""),
        rep_mode="full", fs_tar=False,
        hqkv_r0=args.hqkv_r0, hqkv_a=args.hqkv_a,
        gamma_init=args.hqkv_gamma_init,
    ).float().to(device)
    return m


def npar(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--vmamba-pretrained-weight-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    ap.add_argument("--str-dim", type=int, default=192)
    ap.add_argument("--hqkv-rank", type=int, default=32)
    ap.add_argument("--hqkv-detail-window", type=int, default=2)
    ap.add_argument("--hqkv-global-window", type=int, default=4)
    ap.add_argument("--hqkv-attn-stages", default="1,2,3")
    ap.add_argument("--hqkv-r0", type=float, default=0.5)
    ap.add_argument("--hqkv-a", type=float, default=0.5)
    ap.add_argument("--hqkv-gamma-init", type=float, default=0.5)
    ap.add_argument("--hgsl-lambda", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logf = open(os.path.join(args.output_dir, "dry_run_log.txt"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    v = VARIANTS[args.variant]
    log(f"[DRY] variant={args.variant} spec={json.dumps(v)} device={device} "
        f"steps={args.steps} batch={args.batch_size} gamma_init={args.hqkv_gamma_init}")
    log("[DRY] never writes best_F1=*.pth; never touches Run1-Run8 artifacts")

    init_seed(args.seed)
    model = make_model(args.variant, args, device)
    st = model.pretrain_load_stats() or {}
    pre_ok = (st.get("worst_diff") == 0.0 and st.get("retained", 0) > 0)
    log(f"[DRY] PRETRAIN-LOAD retained={st.get('retained')}/{st.get('pretrained_keys')} "
        f"worst_diff={st.get('worst_diff')} -> {'PASS' if pre_ok else 'FAIL'}")
    pb = model.param_breakdown()
    log(f"[DRY] param_breakdown={json.dumps(pb)}")

    backbone = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    new = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.Adam(
        [{"params": backbone, "lr": args.lr * args.backbone_lr_ratio,
          "lr_scale": args.backbone_lr_ratio, "name": "backbone"},
         {"params": new, "lr": args.lr, "lr_scale": 1.0, "name": "new"}],
        args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    ds = myDataLoader.Dataset(file_root=args.dataset_root, list_path=args.train_list,
                              transform=build_transform())
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True, drop_last=False)
    log(f"[DRY] train samples={len(ds)} batches/epoch={len(loader)}")
    it = iter(loader)

    img0, tgt0 = next(it)
    pre0 = img0[:, 0:3].to(device).float(); post0 = img0[:, 3:6].to(device).float()
    t0v = tgt0.to(device).float()

    # ---------------- 抗自锁证据（epoch-0，未训练）
    rms0 = None
    if v["hqkv"]:
        model.eval()
        with torch.no_grad():
            y_exp = model(pre0, post0)
        d0 = model.hqkv_diag()
        cn0 = float(d0.get("contrib_norm", 0.0))
        n_el = y_exp.numel()      # 仅用于估计量级，不参与判据
        rms0 = cn0 / max(1.0, (args.batch_size * args.str_dim * 64 * 64) ** 0.5)
        log(f"[DRY] epoch-0 contrib_norm={cn0:.6e} rms≈{rms0:.4f} "
            f"gammas={d0.get('gammas')} shares={ {k: v2 for k, v2 in d0.items() if k.endswith('_share')} }")
    else:
        rms0 = 0.0

    # ---------------- 3 步训练
    model.train()
    losses, hgsls, cn_trace, rows = [], [], [], []
    for step in range(args.steps):
        img, target = next(it)
        pre = img[:, 0:3].to(device).float(); post = img[:, 3:6].to(device).float()
        tgt = target.to(device).float()
        out = model(pre, post, tgt)
        loss = BCEDiceLoss(out, tgt)
        hg = None
        if v["hgsl"]:
            hg = hgsl_loss(model.hgsl_score_maps(), tgt, args.hgsl_lambda,
                           stages=HGSL_STAGES_DEFAULT)
            if hg is None:
                log("[DRY][P0-FAIL] hgsl=1 但 score_maps 为空"); logf.close(); return 2
            loss = loss + hg
        opt.zero_grad(); loss.backward()
        gg = {}
        if v["hqkv"]:
            model.snapshot_gamma_grad()
            gg = (model.hqkv_report() or {}).get("gamma_grad", {})
        opt.step()
        finite = bool(torch.isfinite(loss).item())
        cn = model.contrib_norm() if v["hqkv"] else 0.0
        losses.append(float(loss.item())); cn_trace.append(cn)
        if hg is not None:
            hgsls.append(float(hg.item()))
        rows.append({"step": step + 1, "loss": losses[-1], "finite": finite,
                     "hgsl": (float(hg.item()) if hg is not None else None),
                     "contrib_norm": cn, "gamma_grad": gg})
        log(f"[DRY] step={step+1} loss={losses[-1]:.6f} finite={finite} "
            f"hgsl={rows[-1]['hgsl']} contrib_norm={cn:.6e} gamma_grad={gg}")
    gammas_after = model.hqkv_report()["gammas"] if v["hqkv"] else {}

    # ---------------- 真实 batch 折叠门
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model.eval()
    with torch.no_grad():
        y_tr = model(pre0, post0)
    dep = copy.deepcopy(model); dep.eval(); dep.switch_to_deploy()
    with torch.no_grad():
        y_dp = dep(pre0, post0)
    flip = (y_tr > 0.5) != (y_dp > 0.5)
    n_flip = int(flip.sum())
    err = float((y_tr - y_dp).abs().max())
    fm = float((y_tr - 0.5).abs()[flip].max()) if n_flip else None
    knife = bool(n_flip > 0 and fm is not None and fm < err)
    dep_params = npar(dep)
    log(f"[DRY] fold: disagreement={float(flip.float().mean())} max|d|={err:.3e} "
        f"n_flip={n_flip} flipped_margin_max={fm} -> "
        f"{'NO_FLIP' if n_flip == 0 else ('KNIFE_EDGE_ARTIFACT' if knife else 'NEEDS_REVIEW')}")

    # ---------------- batch32 显存
    peak_mb = None
    if device.startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
            model.train()
            out = model(pre0, post0, t0v)
            loss = BCEDiceLoss(out, t0v)
            if v["hgsl"]:
                hg2 = hgsl_loss(model.hgsl_score_maps(), t0v, args.hgsl_lambda)
                if hg2 is not None:
                    loss = loss + hg2
            opt.zero_grad(); loss.backward(); opt.step()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            log(f"[DRY] batch{args.batch_size} fwd+bwd OK peak={peak_mb:.0f} MB")
            del out, loss; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            log("[DRY][FAIL] batch32 OOM"); peak_mb = -1.0

    gates = {
        "pretrain_bitwise_load": bool(pre_ok),
        "all_losses_finite": bool(all(r["finite"] for r in rows)),
        # **本轮核心**：γ 非零 ⇒ 模块真的进入网络（v1 的对应量恒为 0）
        "hqkv_contrib_nonzero": bool((not v["hqkv"]) or (cn_trace and min(cn_trace) > 0.0)),
        "hqkv_rms_reasonable": bool((not v["hqkv"]) or (rms0 is not None and rms0 > RMS_MIN)),
        "gamma_grads_nonzero": bool((not v["hqkv"]) or
                                    all((rows[-1]["gamma_grad"].get(k) or 0.0) > 0.0
                                        for k in ("detail", "global", "lite"))),
        "fold_gate": bool(n_flip == 0 or knife),
        "non_backbone_params_small": bool(pb["non_backbone"] <= 4_000_000),
    }
    summary = {
        "variant": args.variant, "spec": v, "device": device,
        "steps": args.steps, "batch_size": args.batch_size, "seed": args.seed,
        "hqkv_gamma_init": args.hqkv_gamma_init,
        "param_breakdown": pb, "train_params": npar(model), "deploy_params": dep_params,
        "losses": losses, "hgsl_losses": hgsls, "contrib_norm_trace": cn_trace,
        "epoch0_rms_estimate": rms0, "gammas_after": gammas_after,
        "steps_detail": rows,
        "fold": {"n_flip": n_flip, "max_abs_diff": err, "flipped_margin_max": fm,
                 "verdict": ("NO_FLIP" if n_flip == 0 else
                             ("KNIFE_EDGE_ARTIFACT" if knife else "NEEDS_REVIEW"))},
        "peak_allocated_MB_batch32": peak_mb,
        "pretrain_load": {k: (len(v2) if k == "loaded_keys" else v2)
                          for k, v2 in st.items()},
        "hqkv_report": (model.hqkv_report() if v["hqkv"] else None),
        "gates": gates,
    }
    with open(os.path.join(args.output_dir, "dry_run_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    ok = all(gates.values())
    log(f"[DRY] GATES={json.dumps(gates)} -> {'PASS' if ok else 'FAIL'}")
    logf.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
