#!/usr/bin/env python
"""Run7 HFAE 真实数据 dry-run（3 步 + 机制自检 + 折叠等价 + batch32 显存探针）。

**只读语义**：绝不写 `best_F1=*.pth`，不触碰 Run1–Run6 / Diag1 任何产物；只写自己的 log 与 summary。

    python models/run7_dry_run.py --variant E9_HFAE_A \
      --dataset-root /share_datasets/CD/SYSU-CD-256 \
      --train-list /share_datasets/CD/SYSU-CD-256/list/train.txt \
      --tinyvim-pretrained-weight-path .../tinyvim_s_1000e.pth \
      --device cuda:0 --output-dir .../Run7_DRY/SYSU-CD-256/E9_HFAE_A
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

_MODELS = os.path.dirname(os.path.abspath(__file__))
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

import dataset.dataset as myDataLoader                                    # noqa: E402
import dataset.Transforms as myTransforms                                  # noqa: E402
from model.casa_tvim_str_net import CASATViMSTRNet                         # noqa: E402
from model.str_hf_edge import HF_EDGE_TAPS, edge_target_from_label, hf_edge_loss  # noqa: E402
from model.utils import BCEDiceLoss, init_seed                             # noqa: E402

# variant -> (hf_edge, hf_edge_form)
VARIANTS = {
    "M1_R7CTRL": (0, "none"),
    "E9_HFAE_A": (1, "aux_only"),
    "E10_HFAE_B": (1, "aux_inject"),
}
MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


def build_transform():
    return myTransforms.Compose([myTransforms.Normalize(mean=MEAN6, std=STD6),
                                 myTransforms.Scale(256, 256), myTransforms.ToTensor()])


def make_model(variant, args, device):
    hf_edge, form = VARIANTS[variant]
    torch.manual_seed(args.seed)
    m = CASATViMSTRNet(
        args.tinyvim_pretrained_weight_path, caacp=True, rep_mode="full", str_dim=96,
        caacp_score_mode="rank", frh=False, caacp_residual_mode="current", fs_tar=False,
        fine_tap=False, hf_edge=bool(hf_edge), hf_edge_form=form,
        hf_edge_taps=args.hf_edge_taps, hf_edge_lambda=args.hf_edge_lambda).float().to(device)
    return m


def npar(m):
    return sum(p.numel() for p in m.parameters())


def check_label_gray128(args, n=4):
    """label 契约：Transforms.Normalize 必须按 gray>=128 二值化（CDD 为 JPEG 有中间灰度）。"""
    import cv2
    rows, ok = [], True
    with open(args.train_list, encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()][:n]
    for nm in names:
        p = os.path.join(args.dataset_root, "label", nm)
        raw = cv2.imread(p, 0)
        if raw is None:
            rows.append({"name": nm, "missing": True}); ok = False; continue
        inter = int(((raw > 0) & (raw < 255)).sum())
        rows.append({"name": nm, "raw_min": int(raw.min()), "raw_max": int(raw.max()),
                     "intermediate_px": inter})
    return ok, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--tinyvim-pretrained-weight-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    ap.add_argument("--hf-edge-taps", default="1_4_1_8")
    ap.add_argument("--hf-edge-lambda", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "dry_run_log.txt")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    hf_edge, form = VARIANTS[args.variant]
    log(f"[DRY] variant={args.variant} hf_edge={hf_edge} form={form} device={device} "
        f"steps={args.steps} batch={args.batch_size} seed={args.seed} "
        f"taps={args.hf_edge_taps} lambda={args.hf_edge_lambda}")
    log("[DRY] never writes best_F1=*.pth; never touches Run1-Run6 / Diag1 artifacts")

    init_seed(args.seed)
    model = make_model(args.variant, args, device)
    rep = model.hf_edge_report()
    log(f"[DRY] train_params={npar(model)} hf_edge_report={json.dumps(rep, default=str)}")

    # ---------------- epoch-0 与 CTRL 逐位比对（同一 real batch，未训练前）
    ck = None
    if hf_edge:
        ck = make_model("M1_R7CTRL", args, device).eval()

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

    # ---------------------------------------------------------------- epoch-0 门
    img0, tgt0 = next(it)
    pre0 = img0[:, 0:3].to(device).float(); post0 = img0[:, 3:6].to(device).float()
    t0v = tgt0.to(device).float()
    model.eval()
    with torch.no_grad():
        y_exp = model(pre0, post0)
        if ck is not None:
            y_ctrl = ck(pre0, post0)
            eq0 = torch.equal(y_ctrl, y_exp)
            d0 = float((y_ctrl - y_exp).abs().max())
        else:
            eq0, d0 = True, 0.0
    log(f"[DRY] epoch-0 real-batch torch.equal(CTRL,{args.variant})={eq0} max|d|={d0:.3e}")
    if ck is not None:
        del ck
    if hf_edge and not eq0:
        log("[DRY][P0-FAIL] epoch-0 与 CTRL 不是逐位相同 —— 零初始化/注入路径有问题")
        logf.close(); return 2

    # ---------------------------------------------------------------- 3 步训练
    model.train()
    losses, auxes, tap_grads, head_grads = [], [], [], []
    step_rows = []
    for step in range(args.steps):
        img, target = next(it)
        pre = img[:, 0:3].to(device).float(); post = img[:, 3:6].to(device).float()
        tgt = target.to(device).float()
        out = model(pre, post, tgt)
        loss = BCEDiceLoss(out, tgt)
        aux = None
        if hf_edge:
            aux = hf_edge_loss(model.aux_edge_logits(), tgt, args.hf_edge_lambda)
            loss = loss + aux
        opt.zero_grad(); loss.backward()
        tg = float(model.hf_edge_tap_grad_norm()) if hf_edge else 0.0
        hg = float(model.hf_edge_head_grad_norm()) if hf_edge else 0.0
        opt.step()
        finite = bool(torch.isfinite(loss).item())
        losses.append(float(loss.item()))
        if aux is not None:
            auxes.append(float(aux.item()))
        tap_grads.append(tg); head_grads.append(hg)
        step_rows.append({"step": step + 1, "loss": losses[-1], "finite": finite,
                          "aux": (float(aux.item()) if aux is not None else None),
                          "tap_grad_norm": tg, "head_grad_norm": hg,
                          "lr": [float(pg["lr"]) for pg in opt.param_groups]})
        log(f"[DRY] step={step+1} loss={losses[-1]:.6f} finite={finite} "
            f"aux={step_rows[-1]['aux']} tap_grad={tg:.4e} head_grad={hg:.4e} "
            f"lr={['%.3e' % x for x in step_rows[-1]['lr']]}")

    # ---------------------------------------------------------------- 真实 batch 折叠门
    # 与项目既有折叠协议一致（D0 审计 / eval.py）：TF32 off + cudnn deterministic。
    # 不这么做会把 reparam 舍入误差放大到 1e-3 量级，导致个别元素跨过 0.5（实测 4.8e-7）。
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
    dis = float(((y_tr > 0.5) != (y_dp > 0.5)).float().mean())
    err = float((y_tr - y_dp).abs().max())
    dep_params = npar(dep)
    log(f"[DRY] real-batch fold (TF32 off, deterministic): disagreement={dis} "
        f"max|d|={err:.3e} deploy_params={dep_params}")

    # ---------------------------------------------------------------- 梯度覆盖
    zero_grad = [n for n, p in model.named_parameters()
                 if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    log(f"[DRY] zero-grad params n={len(zero_grad)} first={zero_grad[:8]}")

    # ---------------------------------------------------------------- label 契约
    lab_ok, lab_rows = check_label_gray128(args)
    log(f"[DRY] label gray>=128 check: ok={lab_ok} rows={json.dumps(lab_rows)}")

    # ---------------------------------------------------------------- batch32 显存
    peak_mb = None
    if device.startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
            model.train()
            out = model(pre0, post0, t0v)
            loss = BCEDiceLoss(out, t0v)
            if hf_edge:
                loss = loss + hf_edge_loss(model.aux_edge_logits(), t0v, args.hf_edge_lambda)
            opt.zero_grad(); loss.backward(); opt.step()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            log(f"[DRY] batch{args.batch_size} forward+backward OK peak_allocated={peak_mb:.0f} MB")
            del out, loss
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            log("[DRY][FAIL] batch32 OOM -> 同卡 2 并发不可行（不许降 batch，改同卡串行）")
            peak_mb = -1.0

    summary = {
        "variant": args.variant, "hf_edge": hf_edge, "hf_edge_form": form,
        "device": device, "steps": args.steps, "batch_size": args.batch_size,
        "seed": args.seed, "hf_edge_taps": args.hf_edge_taps,
        "hf_edge_lambda": args.hf_edge_lambda,
        "train_params": npar(model), "deploy_params": dep_params,
        "losses": losses, "aux_losses": auxes, "tap_grad_norms": tap_grads,
        "head_grad_norms": head_grads, "steps_detail": step_rows,
        "epoch0_real_batch_equal_vs_ctrl": bool(eq0), "epoch0_max_abs_diff": d0,
        "fold_disagreement_real": dis, "fold_max_abs_diff_real": err,
        "zero_grad_params": zero_grad, "label_gray128_ok": lab_ok, "label_rows": lab_rows,
        "peak_allocated_MB_batch32": peak_mb,
        "hf_edge_report": rep,
        "gates": {
            "all_losses_finite": bool(all(s["finite"] for s in step_rows)),
            "epoch0_bitwise_identity": bool(eq0),
            "fold_disagreement_zero": bool(dis == 0.0),
            "tap_grad_nonzero_step1": bool((not hf_edge) or tap_grads[0] > 0),
            "deploy_within_5M": bool(dep_params <= 5_000_000),
        },
    }
    with open(os.path.join(args.output_dir, "dry_run_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    ok = all(summary["gates"].values())
    log(f"[DRY] GATES={json.dumps(summary['gates'])} -> {'PASS' if ok else 'FAIL'}")
    logf.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
