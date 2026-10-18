"""独立的正式 TEST 区块入口（用于训练已跑满但收尾崩溃的 run；**不重训、不覆盖曲线**）。

用法（服务器，GPU1）：
  cd models_v1 && CUDA_VISIBLE_DEVICES=1 python eval_stgeo_tb.py \
      --dataset LEVIR-CD-256 --dataset_root /share_datasets/CD/LEVIR-CD-256 \
      --test_list .../list/test.txt --ckpt_dir <ckpt_dir> \
      --vmamba_pretrained_weight_path ... --stgeo sup [--stgeo_sup_small 16 ...]
输出的 `=== TEST RESULTS === … === END TEST RESULTS ===` 与 train_stgeo.py **逐字同格式**，
可直接 append 到该 run 的 train_log.txt，使 `[ACTUAL-OPT-STEPS]/[PROTOCOL-VERSION]` 落在最后完整区块内。
"""
import argparse
import hashlib
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model.casa_vmamba_stgeo_net import CASAVMambaSTGeoNet  # noqa: E402
from model.layers.st_geodesic import STGEO_FORM  # noqa: E402
from model.layers.stgeo_sup import STGEO_SUP_FORM  # noqa: E402
from model.metric_tool import ConfuseMatrixMeter  # noqa: E402
from model.utils import BCEDiceLoss  # noqa: E402

import dataset.Transforms as myTransforms  # noqa: E402
import dataset.dataset as myDataLoader  # noqa: E402

PROTOCOL_VERSION = "exp1_stgeo_exact80k_v1"
MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


def log(m):
    print(m, flush=True)


def build_loader(ds_root, list_path, bs, nw):
    tf = myTransforms.Compose([myTransforms.Normalize(mean=MEAN6, std=STD6),
                               myTransforms.Scale(256, 256), myTransforms.ToTensor()])
    ds = myDataLoader.Dataset(file_root=ds_root, list_path=list_path, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False,
                                       num_workers=nw, pin_memory=True)


@torch.no_grad()
def evaluate(loader, model):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    for img, target in loader:
        pre, post = img[:, 0:3].cuda(), img[:, 3:6].cuda()
        target = target.cuda().float()
        out = model(pre.float(), post.float(), target)
        losses.append(float(BCEDiceLoss(out, target)))
        meter.update_cm(pr=(out > 0.5).long().cpu().numpy(), gt=target.cpu().numpy())
    return sum(losses) / max(1, len(losses)), meter.get_scores()


def sha16(p):
    if not p or not os.path.isfile(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--test_list", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--vmamba_pretrained_weight_path", required=True)
    ap.add_argument("--stgeo", default="sup")
    ap.add_argument("--stgeo_sigmas", default="1.0,2.0,4.0")
    ap.add_argument("--stgeo_mid", type=int, default=32)
    ap.add_argument("--stgeo_dim", type=int, default=192)
    ap.add_argument("--stgeo_pool", default="max_shape")
    ap.add_argument("--stgeo_xscale", type=int, default=0)
    ap.add_argument("--stgeo_sup_small", type=int, default=16)
    ap.add_argument("--stgeo_sup_lambda", type=float, default=0.2)
    ap.add_argument("--rep_mode", default="full")
    ap.add_argument("--str_dim", type=int, default=192)
    ap.add_argument("--backbone_lr_ratio", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu_id)
    model = CASAVMambaSTGeoNet(
        vmamba_pretrained_path=args.vmamba_pretrained_weight_path,
        stgeo=args.stgeo, stgeo_sigmas=tuple(float(s) for s in args.stgeo_sigmas.split(",")),
        stgeo_mid=args.stgeo_mid, stgeo_dim=args.stgeo_dim,
        stgeo_pool=args.stgeo_pool, stgeo_xscale=bool(args.stgeo_xscale),
        stgeo_sup_small=args.stgeo_sup_small, stgeo_sup_lambda=args.stgeo_sup_lambda,
        rep_mode=args.rep_mode, str_dim=args.str_dim).float().cuda()
    rep = model.stgeo_report()
    pb = model.param_breakdown()
    st = model.pretrain_load_stats() or {}

    cands = [f for f in os.listdir(args.ckpt_dir) if f.startswith("best_F1=") and f.endswith(".pth")]
    assert len(cands) == 1, f"best ckpt 必须恰好 1 个，实际 {cands}"
    best_name = cands[0]
    best_f1 = float(best_name[len("best_F1="):-len(".pth")])
    ck = torch.load(os.path.join(args.ckpt_dir, best_name), map_location="cpu",
                    weights_only=False)
    model.load_state_dict(ck.get("state_dict", ck))
    lastp = os.path.join(args.ckpt_dir, "last.pth")
    cur_iter, best_epoch = 0, -1
    if os.path.isfile(lastp):
        ckl = torch.load(lastp, map_location="cpu", weights_only=False)
        cur_iter = int(ckl.get("actual_steps", 0))
        best_epoch = ckl.get("best_epoch", -1)
    log(f"[EVAL-ONLY] ckpt={best_name} best_f1={best_f1:.4f} actual_steps={cur_iter} "
        f"best_epoch={best_epoch}")

    # ---- 折叠等价性（TF32 off + deterministic）
    prev_tf32 = torch.backends.cudnn.allow_tf32
    prev_det = torch.backends.cudnn.deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(16)
    pre_f = torch.randn(8, 3, 256, 256).cuda()
    post_f = torch.randn(8, 3, 256, 256).cuda()
    ld = build_loader(args.dataset_root, args.test_list, 16, args.num_workers)
    img_r, _ = next(iter(ld))
    pre_r, post_r = img_r[:, 0:3].cuda(), img_r[:, 3:6].cuda()
    model.eval()
    with torch.no_grad():
        y_tr = model(pre_f, post_f)
        y_tr_r = model(pre_r.float(), post_r.float())
    model.switch_to_deploy()
    model.eval()
    with torch.no_grad():
        y_dp = model(pre_f, post_f)
        y_dp_r = model(pre_r.float(), post_r.float())
    fold_err = float((y_tr - y_dp).abs().max())
    dis = float(((y_tr > 0.5) != (y_dp > 0.5)).float().mean())
    fold_err_r = float((y_tr_r - y_dp_r).abs().max())
    dis_r = float(((y_tr_r > 0.5) != (y_dp_r > 0.5)).float().mean())
    flips = ((y_tr_r > 0.5) != (y_dp_r > 0.5))
    n_flip = int(flips.sum())
    margin = float((y_tr_r - 0.5).abs()[flips].max()) if n_flip else 0.0
    knife = bool(n_flip > 0 and margin < fold_err_r)
    deploy_params = sum(p.numel() for p in model.parameters())

    test_loader = build_loader(args.dataset_root, args.test_list, 16, args.num_workers)
    _, sv = evaluate(test_loader, model)

    log("=== TEST RESULTS ===")
    log("[MODEL] CASA-VMamba-STGeo (VMamba-Tiny + SPD(2)/H2 geodesic evidence + TAR/DCR)")
    log(f"[ARCH] casa_vmamba_stgeo")
    log(f"[STGEO] {args.stgeo} form={rep.get('form')} sigmas={args.stgeo_sigmas} "
        f"mid={args.stgeo_mid} dim={args.stgeo_dim} pool={args.stgeo_pool} "
        f"xscale={args.stgeo_xscale} sup_small={args.stgeo_sup_small} "
        f"sup_lambda={args.stgeo_sup_lambda}")
    if args.stgeo == "sup":
        log(f"[STGEO-SUP] lambda={args.stgeo_sup_lambda} small={args.stgeo_sup_small} "
            f"form={STGEO_SUP_FORM} dropped_at_deploy=1")
    else:
        log(f"[STGEO-GAMMA] 0.000000e+00 (eval-only)")
    log(f"[REP-MODE] {args.rep_mode}")
    log(f"[BACKBONE-LR-RATIO] {args.backbone_lr_ratio}")
    log("[DATA-CONTRACT] legacy_6ch_reverse_v1")
    log("[DEPLOY-NUMERICS] tf32=off deterministic=on (STR T2 protocol)")
    log(f"[PRETRAIN-SHA256] {sha16(args.vmamba_pretrained_weight_path)}")
    log(f"[PRETRAIN-LOAD] retained={st.get('retained')}/{st.get('pretrained_keys')} "
        f"worst_diff={st.get('worst_diff'):.3e}")
    log(f"[PARAM-BREAKDOWN] backbone={pb['backbone']} stgeo={pb['stgeo']} tar={pb['tar']} "
        f"dcr={pb['dcr']} head={pb['head']} total={pb['total']} "
        f"non_backbone={pb['non_backbone']}")
    log(f"[REPARAM-MAX-ABS-ERROR] {fold_err:.3e} (train vs deploy, fixed random batch)")
    log(f"[REPARAM-ARGMAX-DISAGREE] {dis:.3e} (0.5-binarization; hard gate: must be 0)")
    log(f"[REPARAM-REAL-MAX-ABS-ERROR] {fold_err_r:.3e} (real batch 16)")
    log(f"[REPARAM-REAL-ARGMAX-DISAGREE] {dis_r:.3e} (operational gate: must be 0)")
    log(f"[FOLD-KNIFE-EDGE] n_flip={n_flip} flipped_margin_max={margin:.3e} "
        f"max_abs_diff={fold_err_r:.3e} verdict="
        f"{'NO_FLIP' if n_flip == 0 else ('KNIFE_EDGE_ARTIFACT' if knife else 'NEEDS_REVIEW')}")
    log(f"[DEPLOY-PARAMS] total={deploy_params / 1e6:.6f} M")
    log(f"Recall={sv['recall']:.4f} | Precision={sv['precision']:.4f} | OA={sv['OA']:.4f} | "
        f"F1={sv['F1']:.4f} | IoU={sv['IoU']:.4f} | Kappa={sv['Kappa']:.4f}")
    log(f"[BEST-F1] {best_f1:.4f} (epoch {best_epoch})")
    log(f"[ACTUAL-OPT-STEPS] {int(cur_iter)}")
    log(f"[PROTOCOL-VERSION] {PROTOCOL_VERSION}")
    log("=== END TEST RESULTS ===")
    log(f"[EVAL-ONLY-META] {json.dumps({'best_ckpt': best_name, 'deploy_params': deploy_params})}")
    torch.backends.cudnn.allow_tf32 = prev_tf32
    torch.backends.cudnn.deterministic = prev_det


if __name__ == "__main__":
    main()
