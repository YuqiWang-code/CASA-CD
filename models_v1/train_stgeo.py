"""探索线 Exp1 训练入口（models_v1 专用，**不触碰 models/train.py**）。

设计原则：
  1. **协议与 Run8 逐项一致**（80,000 optimizer.step / batch 32 / 256² / seed 16 /
     Adam(2e-4, 0.9/0.99, wd=1e-4) / poly^0.9 + 200 iter warmup / test-as-val /
     prob > 0.5 / gray ≥ 128），因此已跑完的 `CTRL_R8` 可作为**同期同协议对照**；
  2. **日志格式与 Run8 相同**（`=== TEST RESULTS ===` … `=== END TEST RESULTS ===`，
     内含 `[ACTUAL-OPT-STEPS]`、`[PROTOCOL-VERSION]`），使 analyse/ 的现成解析器
     （`extract_metrics_to_excel.py` / `tvim_diag_common.parse_test_block`）可直接使用；
  3. **fail-closed**：run_manifest.json 逐字段校验；`--resume` 在启动时禁止
     （崩溃恢复只允许从自己的 last.pth 自动续训）。
用法（服务器）：
  cd models_v1 && CUDA_VISIBLE_DEVICES=1 python train_stgeo.py --dataset SYSU-CD-256 ...
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

_MODELS_V1 = os.path.dirname(os.path.abspath(__file__))
if _MODELS_V1 not in sys.path:
    sys.path.insert(0, _MODELS_V1)

from model.casa_vmamba_stgeo_net import CASAVMambaSTGeoNet  # noqa: E402
from model.layers.st_geodesic import STGEO_FORM  # noqa: E402
from model.metric_tool import ConfuseMatrixMeter  # noqa: E402
from model.utils import BCEDiceLoss, adjust_learning_rate, init_seed  # noqa: E402

import dataset.Transforms as myTransforms  # noqa: E402
import dataset.dataset as myDataLoader  # noqa: E402

PROTOCOL_VERSION = "exp1_stgeo_exact80k_v1"
MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
SOURCE_FILES = (
    "models_v1/train_stgeo.py",
    "models_v1/model/casa_vmamba_stgeo_net.py",
    "models_v1/model/layers/st_geodesic.py",
    "models_v1/model/vmamba_tiny.py",
    "models_v1/model/layers/vmamba_kernel.py",
    "models_v1/model/str_tar.py",
    "models_v1/model/str_dcr.py",
    "models_v1/model/str_reparam.py",
    "models_v1/model/utils.py",
)


def log(msg):
    print(msg, flush=True)


def fmt_params(n):
    return f"{n / 1e6:.6f}"


def measure_params(model):
    return sum(p.numel() for p in model.parameters())


def source_code_identity():
    repo_root = os.path.dirname(_MODELS_V1)
    out = {}
    for rel in SOURCE_FILES:
        p = os.path.join(repo_root, rel.replace("/", os.sep))
        if not os.path.isfile(p):
            out[rel] = "MISSING"
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[rel] = h.hexdigest()[:16]
    return out


def sha16(path):
    if not path or not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_loader(args, list_path, batch_size, shuffle, train):
    if train:
        tf = myTransforms.Compose([
            myTransforms.Normalize(mean=args.mean, std=args.std),
            myTransforms.Scale(args.inWidth, args.inHeight),
            myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
            myTransforms.RandomFlip(),
            myTransforms.RandomExchange(),
            myTransforms.ToTensor(),
        ])
    else:
        tf = myTransforms.Compose([
            myTransforms.Normalize(mean=args.mean, std=args.std),
            myTransforms.Scale(args.inWidth, args.inHeight),
            myTransforms.ToTensor(),
        ])
    ds = myDataLoader.Dataset(file_root=args.dataset_root, list_path=list_path,
                              transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=args.num_workers, pin_memory=True,
                                       drop_last=False)


@torch.no_grad()
def evaluate(args, loader, model):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    for img, target in loader:
        pre, post = img[:, 0:3], img[:, 3:6]
        if args.onGPU:
            pre, post, target = pre.cuda(), post.cuda(), target.cuda()
        target = target.float()
        out = model(pre.float(), post.float(), target)
        losses.append(float(BCEDiceLoss(out, target).item()))
        pred = (out > 0.5).long()
        meter.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())
    return sum(losses) / max(1, len(losses)), meter.get_scores()


def train_one_epoch(args, loader, model, optimizer, epoch, max_batches, cur_iter,
                    remaining):
    model.train()
    meter = ConfuseMatrixMeter(n_class=2)
    losses, executed, lr = [], 0, None
    for it, (img, target) in enumerate(loader):
        if remaining is not None and executed >= remaining:
            break
        pre, post = img[:, 0:3], img[:, 3:6]
        lr = adjust_learning_rate(args, optimizer, epoch, it + cur_iter, max_batches)
        if args.onGPU:
            pre, post, target = pre.cuda(), post.cuda(), target.cuda()
        target = target.float()
        out = model(pre.float(), post.float(), target)
        loss = BCEDiceLoss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        executed += 1
        losses.append(float(loss.item()))
        with torch.no_grad():
            pred = (out > 0.5).long()
            meter.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())
    return (sum(losses) / max(1, len(losses)), meter.get_scores(), lr, executed)


def main():
    ap = argparse.ArgumentParser("Exp1 ST-GEO trainer (models_v1)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--test_list", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--vmamba_pretrained_weight_path", required=True)
    ap.add_argument("--pretrained_weight_path", default=None,
                    help="历史占位（本架构不用 DeiT）")
    ap.add_argument("--inWidth", type=int, default=256)
    ap.add_argument("--inHeight", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=80000)
    ap.add_argument("--exact_max_steps", type=int, default=1, choices=[0, 1])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--test_batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_mode", default="poly")
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--arch", default="casa_vmamba_stgeo")
    ap.add_argument("--stgeo", default="inject")
    ap.add_argument("--stgeo_sigmas", default="1.0,2.0,4.0")
    ap.add_argument("--stgeo_mid", type=int, default=32)
    ap.add_argument("--stgeo_dim", type=int, default=192)
    ap.add_argument("--stgeo_pool", default="max_shape",
                    choices=["max_shape", "topk_mean"])
    ap.add_argument("--stgeo_xscale", type=int, default=0, choices=[0, 1])
    ap.add_argument("--rep_mode", default="full", choices=["plain", "full"])
    ap.add_argument("--fs_tar", type=int, default=0, choices=[0, 1])
    ap.add_argument("--str_dim", type=int, default=192)
    ap.add_argument("--backbone_lr_ratio", type=float, default=0.1)
    ap.add_argument("--mean", type=float, nargs=6, default=MEAN6)
    ap.add_argument("--std", type=float, nargs=6, default=STD6)
    ap.add_argument("--onGPU", default=True, type=lambda x: str(x).lower() == "true")
    args = ap.parse_args()
    args.max_epochs = 0

    assert args.resume is None, "from-scratch: --resume 禁止在启动时使用"
    assert args.seed == 16, f"from-scratch: seed 必须为 16，得到 {args.seed}"
    assert os.path.basename(args.vmamba_pretrained_weight_path) == \
        "vssm_tiny_0230_ckpt_epoch_262.pth", "预训练必须是官方 vssm_tiny_0230 权重"

    if args.onGPU:
        torch.cuda.set_device(args.gpu_id)
    init_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    model = CASAVMambaSTGeoNet(
        vmamba_pretrained_path=args.vmamba_pretrained_weight_path,
        str_dim=args.str_dim, stgeo=args.stgeo,
        stgeo_sigmas=tuple(float(s) for s in args.stgeo_sigmas.split(",") if s),
        stgeo_mid=args.stgeo_mid, stgeo_dim=args.stgeo_dim,
        stgeo_pool=args.stgeo_pool, stgeo_xscale=bool(args.stgeo_xscale),
        rep_mode=args.rep_mode, fs_tar=bool(args.fs_tar)).float()
    if args.onGPU:
        model = model.cuda()

    bb = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    new = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.Adam(
        [{"params": bb, "lr": args.lr * args.backbone_lr_ratio,
          "lr_scale": args.backbone_lr_ratio, "name": "backbone"},
         {"params": new, "lr": args.lr, "lr_scale": 1.0, "name": "new"}],
        args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    st = model.pretrain_load_stats() or {}
    pb = model.param_breakdown()
    rep = model.stgeo_report()

    # ---------------- 配置头
    log("=" * 72)
    log(f"[MODEL] CASA-VMamba-STGeo (VMamba-Tiny + SPD(2)/H2 structure-tensor geodesic)")
    log(f"[ARCH] {args.arch}  [STGEO] {args.stgeo}  [REP-MODE] {args.rep_mode}")
    log(f"[CONFIG] protocol={PROTOCOL_VERSION} exact_max_steps={args.exact_max_steps} "
        f"max_steps={args.max_steps} batch={args.batch_size} seed={args.seed} "
        f"lr={args.lr} backbone_lr_ratio={args.backbone_lr_ratio}")
    log(f"[STGEO] form={rep['form']} sigmas={rep['sigmas']} mid={rep['mid']} "
        f"out_dim={rep['out_dim']} pool={rep['pool_mode']} xscale={rep['xscale_gate']} "
        f"in_ch={rep['in_ch_ev']} params={rep['params']} "
        f"gamma_init_zero={rep['gamma_init_zero']}")
    log(f"[PRETRAIN-LOAD] retained={st.get('retained')}/{st.get('pretrained_keys')} "
        f"worst_diff={st.get('worst_diff')} missing_new={len(st.get('missing_new', []))} "
        f"dropped={len(st.get('dropped_intentional', []))}")
    log(f"[PRETRAIN-SHA256] {sha16(args.vmamba_pretrained_weight_path)}")
    log(f"[PARAM-BREAKDOWN] backbone={pb['backbone']:,} stgeo={pb['stgeo']:,} "
        f"tar={pb['tar']:,} dcr={pb['dcr']:,} head={pb['head']:,} "
        f"total={pb['total']:,} non_backbone={pb['non_backbone']:,}")
    log(f"[TOTAL-PARAMS] {fmt_params(measure_params(model))} M")
    log(f"[DATA-CONTRACT] legacy_6ch_reverse_v1  [THRESHOLD] prob>0.5  [LABEL] gray>=128")
    log("=" * 72)
    assert st.get("worst_diff") == 0.0, f"[VMAMBA-PRETRAIN] worst_diff 必须为 0，得到 {st.get('worst_diff')}"

    # ---------------- run_manifest（fail-closed）
    manifest = {
        "arch": args.arch, "protocol_version": PROTOCOL_VERSION,
        "backbone": "vmamba_tiny_v2",
        "backbone_weight": os.path.basename(args.vmamba_pretrained_weight_path),
        "backbone_weight_sha256": sha16(args.vmamba_pretrained_weight_path),
        "stgeo": args.stgeo, "stgeo_sigmas": args.stgeo_sigmas,
        "stgeo_mid": args.stgeo_mid, "stgeo_dim": args.stgeo_dim,
        "stgeo_pool": args.stgeo_pool, "stgeo_xscale": int(args.stgeo_xscale),
        "stgeo_form_tag": (STGEO_FORM if args.stgeo != "none" else None),
        "rep_mode": args.rep_mode, "fs_tar": int(args.fs_tar), "str_dim": args.str_dim,
        "backbone_lr_ratio": args.backbone_lr_ratio,
        "data_contract": "legacy_6ch_reverse_v1", "seed": args.seed,
        "max_steps": args.max_steps, "batch_size": args.batch_size,
        "dataset": args.dataset, "exact_max_steps": int(args.exact_max_steps),
        "source_code_sha256": source_code_identity(),
        "resume_source": None,
    }
    mpath = os.path.join(args.ckpt_dir, "run_manifest.json")
    if os.path.isfile(mpath):
        old = json.load(open(mpath, encoding="utf-8"))
        for k in [k for k in manifest if k not in ("source_code_sha256", "resume_source")]:
            if old.get(k) != manifest[k]:
                raise SystemExit(f"[MANIFEST-MISMATCH] {k}: old={old.get(k)} "
                                 f"new={manifest[k]}; aborting")
        log(f"[MANIFEST] validated existing run_manifest.json")
    else:
        json.dump(manifest, open(mpath, "w", encoding="utf-8"), indent=2)
        log(f"[MANIFEST] run_manifest.json written")

    def save_arch():
        arch = dict(manifest)
        arch.pop("resume_source", None)
        arch["gamma"] = rep["gamma"]
        json.dump(arch, open(os.path.join(args.ckpt_dir, "arch.json"), "w",
                             encoding="utf-8"), indent=2)

    # ---------------- resume（只允许自己的 last.pth）
    start_epoch, best_f1, best_epoch, cur_iter = 0, -1.0, -1, 0
    auto = os.path.join(args.ckpt_dir, "last.pth")
    if os.path.isfile(auto):
        ck = torch.load(auto, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        if ck.get("optimizer") is not None:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0)
        best_f1 = ck.get("best_f1", -1.0)
        best_epoch = ck.get("best_epoch", -1)
        cur_iter = int(ck.get("actual_steps", 0))
        if ck.get("protocol_version") not in (None, PROTOCOL_VERSION):
            raise SystemExit(f"[PROTOCOL-MISMATCH] ckpt={ck.get('protocol_version')}")
        rng = ck.get("rng_state")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        log(f"[RESUME] last.pth actual_steps={cur_iter} epoch={start_epoch} "
            f"best_f1={best_f1:.4f}")

    train_loader = build_loader(args, args.train_list, args.batch_size, True, True)
    test_loader = build_loader(args, args.test_list, args.test_batch_size, False, False)
    max_batches = len(train_loader)
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    log(f"[MAX-STEPS] {args.max_steps} [iters/epoch] {max_batches} [max_epochs] {args.max_epochs}")
    log("EPOCH | Loss | Recall | Precision | OA | F1 | IoU | Kappa | t(s)")
    save_arch()

    def save_last(epoch):
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
                    "actual_steps": cur_iter, "protocol_version": PROTOCOL_VERSION,
                    "arch": args.arch, "stgeo": args.stgeo,
                    "rng_state": {"torch": torch.get_rng_state(),
                                  "numpy": np.random.get_state(),
                                  "python": random.getstate(),
                                  "cuda": (torch.cuda.get_rng_state_all()
                                           if torch.cuda.is_available() else None)}},
                   os.path.join(args.ckpt_dir, "last.pth"))

    exact = bool(args.exact_max_steps)
    for epoch in range(start_epoch, args.max_epochs):
        t0 = time.time()
        remaining = (args.max_steps - cur_iter) if exact else None
        if remaining is not None and remaining <= 0:
            log(f"[EXACT-STEPS] budget reached ({cur_iter})")
            break
        loss_tr, score_tr, lr, executed = train_one_epoch(
            args, train_loader, model, optimizer, epoch, max_batches, cur_iter, remaining)
        cur_iter += executed if exact else max_batches
        if exact:
            log(f"[ACTUAL-OPT-STEPS] epoch={epoch} executed={executed} "
                f"cumulative={cur_iter}/{args.max_steps}")
        torch.cuda.empty_cache()
        if epoch == 0:
            save_last(epoch)
            if exact and cur_iter >= args.max_steps:
                break
            continue
        loss_val, sv = evaluate(args, test_loader, model)
        log(f"Epoch {epoch}/{args.max_epochs} | Loss={loss_val:.4f} | "
            f"Recall={sv['recall']:.4f} | Precision={sv['precision']:.4f} | "
            f"OA={sv['OA']:.4f} | F1={sv['F1']:.4f} | IoU={sv['IoU']:.4f} | "
            f"Kappa={sv['Kappa']:.4f} | t={time.time() - t0:.0f}s")
        if sv["F1"] > best_f1:
            prev = os.path.join(args.ckpt_dir, f"best_F1={best_f1:.4f}.pth")
            if best_f1 >= 0 and os.path.isfile(prev):
                os.remove(prev)
            best_f1, best_epoch = sv["F1"], epoch
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, f"best_F1={best_f1:.4f}.pth"))
        save_last(epoch)
        save_arch()
        if exact and cur_iter >= args.max_steps:
            log(f"[EXACT-STEPS] reached {cur_iter}/{args.max_steps}, stopping")
            break
    log(f"[BEST] F1={best_f1:.4f} at epoch {best_epoch}")

    # ---------------- 正式测试（deploy 折叠图）
    best_path = os.path.join(args.ckpt_dir, f"best_F1={best_f1:.4f}.pth")
    if not os.path.isfile(best_path):
        log("[TEST] best checkpoint not found, skipping final test")
        return
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False))
    model = model.cuda()
    model.eval()
    prev_tf32 = torch.backends.cudnn.allow_tf32
    prev_det = torch.backends.cudnn.deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(16)
    pre_fix = torch.randn(8, 3, args.inWidth, args.inHeight).cuda()
    post_fix = torch.randn(8, 3, args.inWidth, args.inHeight).cuda()
    real_loader = build_loader(args, args.test_list, 16, False, False)
    img_r, _ = next(iter(real_loader))
    pre_r, post_r = img_r[:, 0:3].cuda(), img_r[:, 3:6].cuda()
    with torch.no_grad():
        y_tr = model(pre_fix, post_fix)
        y_tr_r = model(pre_r.float(), post_r.float())
        gamma_final = float(model.stgeo_op.gamma.detach())
    model.switch_to_deploy()
    model.eval()
    with torch.no_grad():
        y_dp = model(pre_fix, post_fix)
        y_dp_r = model(pre_r.float(), post_r.float())
    fold_err = float((y_tr - y_dp).abs().max())
    dis = float(((y_tr > 0.5) != (y_dp > 0.5)).float().mean())
    fold_err_r = float((y_tr_r - y_dp_r).abs().max())
    dis_r = float(((y_tr_r > 0.5) != (y_dp_r > 0.5)).float().mean())
    flips = ((y_tr_r > 0.5) != (y_dp_r > 0.5))
    n_flip = int(flips.sum())
    margin = float((y_tr_r - 0.5).abs()[flips].max()) if n_flip else 0.0
    knife = bool(n_flip > 0 and margin < fold_err_r)
    deploy_params = measure_params(model)

    _, sv = evaluate(args, test_loader, model)
    log("=== TEST RESULTS ===")
    log("[MODEL] CASA-VMamba-STGeo (VMamba-Tiny + SPD(2)/H2 geodesic evidence + TAR/DCR)")
    log(f"[ARCH] {args.arch}")
    log(f"[STGEO] {args.stgeo} sigmas={args.stgeo_sigmas} mid={args.stgeo_mid} "
        f"dim={args.stgeo_dim} pool={args.stgeo_pool} xscale={int(args.stgeo_xscale)} "
        f"form={model.stgeo_report()['form']}")
    log(f"[STGEO-GAMMA] {gamma_final:.6e} (zero-init gate; value on best ckpt)")
    log(f"[STGEO-DIAG] {json.dumps(model.stgeo_diag(), default=str)}")
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
    log(f"[DEPLOY-PARAMS] total={fmt_params(deploy_params)} M")
    log(f"Recall={sv['recall']:.4f} | Precision={sv['precision']:.4f} | OA={sv['OA']:.4f} | "
        f"F1={sv['F1']:.4f} | IoU={sv['IoU']:.4f} | Kappa={sv['Kappa']:.4f}")
    log(f"[BEST-F1] {best_f1:.4f} (epoch {best_epoch})")
    log(f"[ACTUAL-OPT-STEPS] {int(cur_iter)}")
    log(f"[PROTOCOL-VERSION] {PROTOCOL_VERSION}")
    log("=== END TEST RESULTS ===")
    torch.backends.cudnn.allow_tf32 = prev_tf32
    torch.backends.cudnn.deterministic = prev_det


if __name__ == "__main__":
    main()
