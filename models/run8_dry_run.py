#!/usr/bin/env python
"""Run8 真实数据 dry-run（3 步 + 机制自检 + epoch-0 恒等 + 折叠等价 + batch32 显存探针）。

**只读语义**：绝不写 `best_F1=*.pth`，不触碰 Run1–Run7 / Diag1 任何产物；只写自己的 log 与 summary。

    python models/run8_dry_run.py --variant CTRL_R8 \
      --dataset-root /share_datasets/CD/SYSU-CD-256 \
      --train-list /share_datasets/CD/SYSU-CD-256/list/train.txt \
      --vmamba-pretrained-weight-path .../vssm_tiny_0230_ckpt_epoch_262.pth \
      --device cuda:0 --output-dir .../Run8_DRY/SYSU-CD-256/CTRL_R8

变体口径（与 `train_scripts/CASA-TViM/Run8/run_all.sh` 的 `variant_args()` 必须一致）：
  CTRL_R8   : hqkv=0,  hgsl=0                            （同期对照）
  H1_HYP    : hqkv=1 hyperbolic, hgsl=0                   （几何+双分支+注入）
  H2_EUC    : hqkv=1 euclidean,  hgsl=0                   （同式欧氏对照：曲率是否必要）
  H3_HYP_GS : hqkv=1 hyperbolic, hgsl=1 detach_score=1    （+几何监督损失）
"""
import argparse
import copy
import json
import os
import sys
import time

import torch

_MODELS = os.path.dirname(os.path.abspath(__file__))
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

import dataset.dataset as myDataLoader                                    # noqa: E402
import dataset.Transforms as myTransforms                                  # noqa: E402
from model.casa_vmamba_net import CASAVMambaSTRNet                         # noqa: E402
from model.hgsl_loss import HGSL_STAGES_DEFAULT, hgsl_loss                 # noqa: E402
from model.utils import BCEDiceLoss, init_seed                              # noqa: E402

# variant -> dict(hqkv, geometry, hgsl, hgsl_detach_score, inject_mode)
VARIANTS = {
    "CTRL_R8":   dict(hqkv=0, geometry="hyperbolic", hgsl=0, hgsl_detach_score=1,
                      inject_mode="none"),
    "H1_HYP":    dict(hqkv=1, geometry="hyperbolic", hgsl=0, hgsl_detach_score=1,
                      inject_mode="dcr1"),
    "H2_EUC":    dict(hqkv=1, geometry="euclidean", hgsl=0, hgsl_detach_score=1,
                      inject_mode="dcr1"),
    "H3_HYP_GS": dict(hqkv=1, geometry="hyperbolic", hgsl=1, hgsl_detach_score=1,
                      inject_mode="dcr1"),
}
MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


def build_transform():
    return myTransforms.Compose([myTransforms.Normalize(mean=MEAN6, std=STD6),
                                 myTransforms.Scale(256, 256), myTransforms.ToTensor()])


def make_model(variant, args, device):
    v = VARIANTS[variant]
    torch.manual_seed(args.seed)
    m = CASAVMambaSTRNet(
        vmamba_pretrained_path=args.vmamba_pretrained_weight_path,
        hqkv=bool(v["hqkv"]), str_dim=args.str_dim,
        geometry=v["geometry"], branch=args.hqkv_branch,
        hgsl=bool(v["hgsl"]), hgsl_detach_score=bool(v["hgsl_detach_score"]),
        inject_mode=v["inject_mode"],
        hqkv_rank=args.hqkv_rank, hqkv_lite_dim=args.str_dim,
        hqkv_detail_window=args.hqkv_detail_window,
        hqkv_global_window=args.hqkv_global_window,
        hqkv_attn_stages=tuple(int(s) for s in args.hqkv_attn_stages.split(",") if s != ""),
        rep_mode="full", fs_tar=False,
        hqkv_r0=args.hqkv_r0, hqkv_a=args.hqkv_a,
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
    ap.add_argument("--hqkv-branch", default="both")
    ap.add_argument("--hqkv-r0", type=float, default=0.5)
    ap.add_argument("--hqkv-a", type=float, default=0.5)
    ap.add_argument("--hgsl-lambda", type=float, default=0.2)
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
    v = VARIANTS[args.variant]
    log(f"[DRY] variant={args.variant} spec={json.dumps(v)} device={device} "
        f"steps={args.steps} batch={args.batch_size} seed={args.seed} "
        f"str_dim={args.str_dim} rank={args.hqkv_rank}")
    log("[DRY] never writes best_F1=*.pth; never touches Run1-Run7 / Diag1 artifacts")

    init_seed(args.seed)
    model = make_model(args.variant, args, device)
    st = model.pretrain_load_stats() or {}
    log(f"[DRY] PRETRAIN-LOAD retained={st.get('retained')}/{st.get('pretrained_keys')} "
        f"worst_diff={st.get('worst_diff')} missing_new={len(st.get('missing_new', []))} "
        f"dropped={len(st.get('dropped_intentional', []))}")
    # ⚠️ 这条是 Run8 的 P0：VMamba 官方 222 个 key 必须**逐位**加载（worst_diff == 0），
    # 否则"换骨干"就变成"随机初始化一个 43M 的大网络"，与预训练无关。
    pre_ok = (st.get("worst_diff") == 0.0 and st.get("retained", 0) > 0)
    log(f"[DRY] P0 pretrain bitwise load: {'PASS' if pre_ok else 'FAIL'}")
    pb = model.param_breakdown()
    log(f"[DRY] params breakdown={json.dumps(pb)}")
    if model.hqkv_op is not None:
        log(f"[DRY] hqkv_report={json.dumps(model.hqkv_report(), default=str)}")

    # ---------------- epoch-0 与 CTRL 逐位比对（同一 real batch，未训练前）
    ck = None
    if v["hqkv"]:
        ck = make_model("CTRL_R8", args, device).eval()

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
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if v["hqkv"] and not eq0:
        log("[DRY][P0-FAIL] epoch-0 与 CTRL 不是逐位相同 —— "
            "γ 零初始化 / 注入路径 / fork_rng 顺序有问题")
        logf.close(); return 2

    # ---------------------------------------------------------------- 3 步训练
    model.train()
    losses, hgsls, step_rows = [], [], []
    for step in range(args.steps):
        img, target = next(it)
        pre = img[:, 0:3].to(device).float(); post = img[:, 3:6].to(device).float()
        tgt = target.to(device).float()
        out = model(pre, post, tgt)
        loss = BCEDiceLoss(out, tgt)
        hg = None
        if v["hgsl"]:
            smaps = model.hgsl_score_maps()
            hg = hgsl_loss(smaps, tgt, args.hgsl_lambda, stages=HGSL_STAGES_DEFAULT)
            if hg is None:
                log("[DRY][P0-FAIL] hgsl=1 但 score_maps 为空")
                logf.close(); return 2
            loss = loss + hg
        opt.zero_grad(); loss.backward()
        # 机制自检：γ 是否真的拿到梯度（否则"注入"这一路等于没接上）
        gdet = (float(model.hqkv_op.gamma_detail.grad.abs().sum())
                if (model.hqkv_op is not None and model.hqkv_op.gamma_detail.grad is not None)
                else 0.0)
        opt.step()
        finite = bool(torch.isfinite(loss).item())
        losses.append(float(loss.item()))
        if hg is not None:
            hgsls.append(float(hg.item()))
        step_rows.append({"step": step + 1, "loss": losses[-1], "finite": finite,
                          "hgsl": (float(hg.item()) if hg is not None else None),
                          "gamma_detail_grad": gdet,
                          "lr": [float(pg["lr"]) for pg in opt.param_groups]})
        log(f"[DRY] step={step+1} loss={losses[-1]:.6f} finite={finite} "
            f"hgsl={step_rows[-1]['hgsl']} gamma_detail_grad={gdet:.4e} "
            f"lr={['%.3e' % x for x in step_rows[-1]['lr']]}")
    if model.hqkv_op is not None:
        diag = model.hqkv_diag()
        log(f"[DRY] hqkv_diag={json.dumps(diag, default=str)}")

    # ---------------------------------------------------------------- 真实 batch 折叠门
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
    # ---- 刀锋像素定位（与 analyse/run7_fold_knife_edge.py 的判据**同一定义**）
    # `fold_disagreement_zero` 是**硬门**；但当它非零时不能直接判"实现错误"，必须先区分：
    #   KNIFE_EDGE_ARTIFACT ⇔ 存在翻转 且 **所有**翻转像素的 |p_train − 0.5| 都 < 本次折叠的最大绝对误差
    #   ⇒ 翻转由「该像素在阈值上没有 margin、二值化本就不适定」造成，而非折叠实现错误。
    # Run7 已用这个判据把 4 次折叠门异常全部判定为 KNIFE_EDGE_ARTIFACT（预注册 §7.6 有披露）。
    flip = (y_tr > 0.5) != (y_dp > 0.5)
    n_flip = int(flip.sum())
    margin_tr = (y_tr - 0.5).abs()
    fm = float(margin_tr[flip].max()) if n_flip else None
    knife_edge = bool(n_flip > 0 and fm is not None and fm < err)
    verdict = ("NO_FLIP" if n_flip == 0 else
               ("KNIFE_EDGE_ARTIFACT" if knife_edge else "NEEDS_REVIEW"))
    log(f"[DRY] real-batch fold (TF32 off, deterministic): disagreement={dis} "
        f"max|d|={err:.3e} deploy_params={dep_params}")
    log(f"[DRY] fold knife-edge: n_flip_px={n_flip} flipped_margin_max={fm} "
        f"max_abs_diff={err:.3e} verdict={verdict}")

    # ---------------------------------------------------------------- 梯度覆盖
    zero_grad = [n for n, p in model.named_parameters()
                 if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    log(f"[DRY] zero-grad params n={len(zero_grad)} first={zero_grad[:8]}")

    # ---------------------------------------------------------------- batch32 显存
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
            log(f"[DRY] batch{args.batch_size} forward+backward OK peak_allocated={peak_mb:.0f} MB")
            del out, loss
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            log("[DRY][FAIL] batch32 OOM -> 同卡 2 并发不可行（不许降 batch，改同卡串行）")
            peak_mb = -1.0

    summary = {
        "variant": args.variant, "spec": v, "device": device, "steps": args.steps,
        "batch_size": args.batch_size, "seed": args.seed,
        "str_dim": args.str_dim, "hqkv_rank": args.hqkv_rank,
        "param_breakdown": pb,
        "train_params": npar(model), "deploy_params": dep_params,
        "losses": losses, "hgsl_losses": hgsls, "steps_detail": step_rows,
        "epoch0_real_batch_equal_vs_ctrl": bool(eq0), "epoch0_max_abs_diff": d0,
        "fold_disagreement_real": dis, "fold_max_abs_diff_real": err,
        "fold_knife_edge": {"n_flip_px": n_flip, "flipped_margin_max": fm,
                            "max_abs_diff": err, "verdict": verdict},
        "zero_grad_params": zero_grad,
        "peak_allocated_MB_batch32": peak_mb,
        "pretrain_load": {k: (v2 if k != "loaded_keys" else len(v2))
                          for k, v2 in st.items()},
        "hqkv_report": (model.hqkv_report() if model.hqkv_op is not None else None),
        "hqkv_diag": (model.hqkv_diag() if model.hqkv_op is not None else None),
        "gates": {
            "pretrain_bitwise_load": bool(pre_ok),
            "all_losses_finite": bool(all(s["finite"] for s in step_rows)),
            "epoch0_bitwise_identity": bool(eq0),
            "fold_disagreement_zero": bool(dis == 0.0),
            "fold_knife_edge_artifact": bool(dis == 0.0 or knife_edge),
            "_fold_gate_note": ("`fold_disagreement_zero` 是硬门；`fold_knife_edge_artifact` 是"
                                "Run7 预注册 §7.6 认可的等价判据（翻转像素全部无阈值 margin）。"
                                "两者任一为真即算过门，但**必须在报告里写明走的是哪一条**。"),
            "gamma_gets_gradient": bool((not v["hqkv"]) or
                                        any(s["gamma_detail_grad"] > 0 for s in step_rows)),
            "non_backbone_params_small": bool(pb["non_backbone"] <= 4_000_000),
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
