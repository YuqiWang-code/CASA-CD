"""R4 真实数据 3-step dry-run（方案 §7 T7/T8）。

用途：在**不启动任何正式 80K** 的前提下，用真实 A/B/label 走原 `BCEDiceLoss` 与原 Adam
参数组跑 3 次优化，逐项核验：

  * T8：A/B/label 几何映射同步（在每个几何变换之后逐一断言 img/label 空间尺寸一致）；
        label 按 `gray>=128` 二值化（与原始 PNG 逐像素比对）；
        loss 有限；batch32 显存可容纳（R8）；
  * T7：第 1 步 `gamma.grad` 有限非零；γ 开门后第 2/3 步 `diff.weight.grad` 非零；
        encoder/TAR/DCR/head 均仍有梯度；backbone:new 参数组 LR 比例 = 0.1。

纪律：只写 `--output-dir`（独立目录，例 `.../CASA-TViM/Run4_DRY/<variant>`）；
不保存 best、不写 `train_log.txt` 之外的任何历史产物、不触碰 Run1–Run3 / Diag1 文件。

    python run4_dry_run.py --variant E6_FET1 --device cuda:0 --steps 3 \
      --dataset-root /share_datasets/CD/SYSU-CD-256 \
      --train-list /share_datasets/CD/SYSU-CD-256/list/train.txt \
      --output-dir /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_DRY/E6_FET1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.casa_tvim_str_net import CASATViMSTRNet          # noqa: E402
from model.utils import BCEDiceLoss, init_seed               # noqa: E402
import dataset.dataset as myDataLoader                       # noqa: E402
import dataset.Transforms as myTransforms                    # noqa: E402

MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

VARIANTS = {
    "M1_R4CTRL": dict(fine_tap=0),
    "E6_FET1": dict(fine_tap=1),
}


class _Probe(object):
    """插在几何变换之间的探针：img/label 空间尺寸必须始终一致。

    注意布局差异：几何变换阶段 img 是 numpy HWC、label 是 HW；`ToTensor` 之后
    img 是 CHW、label 是 [1,H,W]。比较的是**空间**尺寸，不是前两维。
    """

    def __init__(self, name, rec):
        self.name = name
        self.rec = rec

    @staticmethod
    def _spatial(t):
        if hasattr(t, "dim") and t.dim() == 3:      # torch CHW / [1,H,W]
            return int(t.shape[-2]), int(t.shape[-1])
        return int(t.shape[0]), int(t.shape[1])     # numpy HWC / HW

    def __call__(self, img, label):
        ih, iw = self._spatial(img)
        lh, lw = self._spatial(label)
        self.rec.append({"stage": self.name, "img": [ih, iw], "label": [lh, lw],
                         "sync": bool(ih == lh and iw == lw)})
        return [img, label]


def build_train_transform(probe_rec, size=256):
    return myTransforms.Compose([
        myTransforms.Normalize(mean=MEAN, std=STD),
        myTransforms.Scale(size, size),
        _Probe("after_scale", probe_rec),
        myTransforms.RandomCropResize(int(7. / 224. * size)),
        _Probe("after_crop", probe_rec),
        myTransforms.RandomFlip(),
        _Probe("after_flip", probe_rec),
        myTransforms.RandomExchange(),
        _Probe("after_exchange", probe_rec),
        myTransforms.ToTensor(),
        _Probe("after_tensor", probe_rec),
    ])


def build_eval_transform(size=256):
    return myTransforms.Compose([
        myTransforms.Normalize(mean=MEAN, std=STD),
        myTransforms.Scale(size, size),
        myTransforms.ToTensor(),
    ])


def geom_probe_pass(args, n=4):
    """几何同步探针：在主进程里用**训练增广链**逐样本走一遍，记录每个几何变换后的空间尺寸。

    不能依赖 DataLoader 的 worker 进程（probe_rec 不会被写回父进程），
    因此显式 num_workers=0 地逐样本取样。
    """
    rec = []
    tf = build_train_transform(rec)
    ds = myDataLoader.Dataset(file_root=args.dataset_root, list_path=args.train_list, transform=tf)
    for i in range(min(n, len(ds))):
        _ = ds[i]
    return rec


def check_label_gray128(args, n=4):
    """label 张量必须逐位等于 (raw_gray >= 128)；同时统计 raw 中的中间灰度。

    必须用**独立**的确定性 eval 链数据集（不能复用带随机增广的训练 Dataset，
    否则 crop/flip 会让比较失去意义）。
    """
    import cv2
    tf = build_eval_transform()
    ds = myDataLoader.Dataset(file_root=args.dataset_root, list_path=args.train_list, transform=tf)
    rows, ok = [], True
    n_inter = 0
    for i in range(min(n, len(ds))):
        img, lab = ds[i]
        raw = cv2.imread(ds.gts[i], 0)
        n_inter += int(((raw > 0) & (raw < 255)).sum())
        ref = torch.from_numpy((raw >= 128).astype("int64")).unsqueeze(0)
        same = (lab.shape == ref.shape) and torch.equal(lab, ref)
        ok = ok and same
        rows.append({"idx": i, "same_as_gray128": bool(same),
                     "unique": sorted(set(int(v) for v in lab.flatten().tolist()))[:5],
                     "raw_min": int(raw.min()), "raw_max": int(raw.max()),
                     "raw_intermediate_px": int(((raw > 0) & (raw < 255)).sum())})
        _ = img
    return ok, rows, n_inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, required=True, choices=sorted(VARIANTS))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--train-list", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tinyvim-pretrained-weight-path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    args = ap.parse_args()

    cfg = VARIANTS[args.variant]
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "dry_run_log.txt")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    log(f"[DRY] variant={args.variant} fine_tap={cfg['fine_tap']} device={device} steps={args.steps} "
        f"batch={args.batch_size} seed={args.seed} out={args.output_dir}")
    log("[DRY] this run never writes best_F1=*.pth and never touches Run1-Run3 / Diag1 artifacts")

    init_seed(args.seed)

    # ---------------------------------------------------------------- model
    torch.manual_seed(args.seed)
    model = CASATViMSTRNet(
        args.tinyvim_pretrained_weight_path, caacp=True, rep_mode="full", str_dim=96,
        caacp_score_mode="rank", frh=False, caacp_residual_mode="current",
        fs_tar=False, fine_tap=bool(cfg["fine_tap"])).float().to(device)
    if cfg["fine_tap"]:
        log(f"[DRY] FET gamma_init={model.fet_gamma():.1e} "
            f"params={model.fine_evidence_tap.param_report()['train_new']}")

    backbone = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    new = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.Adam(
        [{"params": backbone, "lr": args.lr * args.backbone_lr_ratio,
          "lr_scale": args.backbone_lr_ratio, "name": "backbone"},
         {"params": new, "lr": args.lr, "lr_scale": 1.0, "name": "new"}],
        args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    # ---------------------------------------------------------------- data
    tf = build_train_transform([])
    ds = myDataLoader.Dataset(file_root=args.dataset_root, list_path=args.train_list, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=2, pin_memory=True, drop_last=False)
    log(f"[DRY] train samples={len(ds)} batches/epoch={len(loader)}")

    it = iter(loader)
    losses, gamma_grads, diff_grads, lr_rows = [], [], [], []
    loss_finite = []
    shapes = None
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        img, target = next(it)
        pre, post = img[:, 0:3].to(device), img[:, 3:6].to(device)
        tgt = target.to(device).float()
        out = model(pre.float(), post.float(), tgt)
        loss = BCEDiceLoss(out, tgt)
        opt.zero_grad()
        loss.backward()
        if cfg["fine_tap"]:
            g = model.fine_evidence_tap.gamma.grad
            gamma_grads.append(None if g is None else float(g.item()))
            dw = model.fine_evidence_tap.diff.weight.grad
            diff_grads.append(None if dw is None else float(dw.abs().sum().item()))
        lr_rows.append([float(pg["lr"]) for pg in opt.param_groups])
        losses.append(float(loss.item()))
        loss_finite.append(bool(torch.isfinite(loss).item()))
        shapes = {"img": list(img.shape), "out": list(out.shape), "target": list(tgt.shape)}
        opt.step()
        log(f"[DRY] step={step + 1} loss={losses[-1]:.6f} finite={loss_finite[-1]} "
            f"gamma.grad={gamma_grads[-1] if gamma_grads else 'n/a'} "
            f"|Wdiff.grad|_1={diff_grads[-1] if diff_grads else 'n/a'} "
            f"lr={['%.3e' % x for x in lr_rows[-1]]}")

    # ------------------------------------------------------- gradient coverage
    zero_grad_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or float(p.grad.abs().sum().item()) == 0.0:
            if n.startswith("fine_evidence_tap.diff"):
                continue                     # γ=0 时 W_diff 梯度为 0 是预期（T7 只要求 γ 开门后非零）
            zero_grad_params.append(n)
    if cfg["fine_tap"]:
        zero_grad_params = [n for n in zero_grad_params if not n.startswith("fine_evidence_tap.diff")]
    log(f"[DRY] zero-grad params (excluding FET diff while gamma==0): {zero_grad_params[:8]} "
        f"(n={len(zero_grad_params)})")

    # ------------------------------------------------------- label contract
    lab_ok, lab_rows, n_inter = check_label_gray128(args)
    log(f"[DRY] label gray>=128 check: ok={lab_ok} rows={json.dumps(lab_rows)}")

    probe_rec = geom_probe_pass(args)
    geom_sync = bool(probe_rec) and all(r["sync"] for r in probe_rec)
    log(f"[DRY] geometry sync probes={len(probe_rec)} all_sync={geom_sync}")
    for r in probe_rec:
        if not r["sync"]:
            log(f"[DRY][GEOM-FAIL] {r}")

    # ------------------------------------------------------- batch32 memory
    batch32_ok = True
    peak_mb = None
    if device.startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
            big_img, big_tgt = next(iter(torch.utils.data.DataLoader(
                myDataLoader.Dataset(file_root=args.dataset_root, list_path=args.train_list,
                                     transform=build_eval_transform()),
                batch_size=32, shuffle=False, num_workers=2, pin_memory=True)))
            pre = big_img[:, 0:3].to(device).float()
            post = big_img[:, 3:6].to(device).float()
            tgt = big_tgt.to(device).float()
            model.train()
            out = model(pre, post, tgt)
            loss = BCEDiceLoss(out, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            log(f"[DRY] batch32 forward+backward OK peak_allocated={peak_mb:.0f} MB "
                f"(two processes/GPU must fit -> R8)")
            del out, loss, pre, post, tgt, big_img, big_tgt
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            batch32_ok = False
            log(f"[DRY][OOM] batch32 does not fit on this GPU: {e}")

    report = {
        "variant": args.variant,
        "fine_tap": int(cfg["fine_tap"]),
        "device": device,
        "actual_steps": int(args.steps),
        "wrote_best": False,
        "losses": losses,
        "loss_finite": loss_finite,
        "gamma_grad_step1": bool(gamma_grads and gamma_grads[0] not in (None, 0.0)),
        "gamma_grads": gamma_grads,
        "diff_grad_step23": bool(len(diff_grads) >= 3 and all(x not in (None, 0.0) for x in diff_grads[1:3])),
        "diff_grads": diff_grads,
        "lr_rows": lr_rows,
        "lr_ratio_ok": all(abs(r[0] / r[1] - args.backbone_lr_ratio) < 1e-9 for r in lr_rows),
        "zero_grad_params": zero_grad_params,
        "geom_sync": geom_sync,
        "geom_probes": probe_rec,
        "label_gray128": bool(lab_ok),
        "label_rows": lab_rows,
        "raw_intermediate_px": n_inter,
        "batch_size": args.batch_size,
        "batch32_ok": bool(batch32_ok),
        "peak_memory_mb": peak_mb,
        "shapes": shapes,
        "elapsed_s": round(time.time() - t0, 1),
        "dataset_root": args.dataset_root,
        "train_list": args.train_list,
    }
    with open(os.path.join(args.output_dir, "dry_run.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"[DRY] wrote {os.path.join(args.output_dir, 'dry_run.json')}")

    hard_fail = (not all(loss_finite) or not geom_sync or not lab_ok or not batch32_ok
                 or (cfg["fine_tap"] and not report["gamma_grad_step1"])
                 or (cfg["fine_tap"] and not report["diff_grad_step23"])
                 or not report["lr_ratio_ok"] or bool(zero_grad_params))
    log(f"[DRY] RESULT={'FAIL' if hard_fail else 'PASS'}")
    logf.close()
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
