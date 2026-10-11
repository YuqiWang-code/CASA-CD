#!/usr/bin/env python
"""Run7 折叠门「刀锋像素定位」诊断（**真实数据 batch**，与 train.py 收尾所用完全同源）。

背景：Run7 预注册 §4 机制自检 ③ 要求「真实 batch 折叠 disagreement = 0」。
实测 `M1_R7CTRL/CDD-CD-256` 给出 `[REPARAM-REAL-ARGMAX-DISAGREE] 9.537e-07`
（= 1/1,048,576 像素，即 16×256×256 里恰 1 个像素翻转）。本工具给出**定位证据**，
用于判定该翻转是「该像素在 0.5 上没有 margin（二值化本就不适定）」还是「折叠实现错误」。

判据与 Run4 `analyse/run4_fold_knife_edge.py` **同一定义**：
    KNIFE_EDGE_ARTIFACT ⇔ 存在翻转 且 所有翻转像素的 |p_train − 0.5| 都 < 本次折叠的最大绝对误差。

与 train.py 的同源性（逐行对齐 train.py:884-911）：
    transform = Normalize(mean,std) -> Scale(256,256) -> ToTensor()
    loader    = Dataset(file_root=<dataset_root>, list_path=<test.txt>), batch=16, shuffle=False
    pre/post  = img[:,0:3] / img[:,3:6]（float）
    y_train   = model(pre, post)  ->  switch_to_deploy()  ->  y_deploy = model(pre, post)
    fold_err_real   = max|y_train - y_deploy|
    disagree_real   = mean((y_train>0.5) != (y_deploy>0.5))

    python analyse/run7_fold_knife_edge.py --variant M1_R7CTRL --dataset CDD-CD-256 \
        --ckpt-root /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM \
        --out /tmp/r7_knife_CDD_CTRL.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.abspath(os.path.join(_HERE, "..", "models"))
for p in (_HERE, _MODELS):
    if p not in sys.path:
        sys.path.insert(0, p)

from model.casa_tvim_str_net import CASATViMSTRNet  # noqa: E402
import dataset.dataset as myDataLoader  # noqa: E402
import dataset.Transforms as myTransforms  # noqa: E402

# train.py:1185-1188 的官方 ChangeViT 归一化（BGR，ImageNet×2）
MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

RUN7_VARIANTS = {"M1_R7CTRL": (0, "none"), "E9_HFAE_A": (1, "aux_only"),
                 "E10_HFAE_B": (1, "aux_inject")}


def build_model(variant, ckpt, device, pretrain):
    hf_edge, form = RUN7_VARIANTS[variant]
    m = CASATViMSTRNet(pretrain, caacp=True, rep_mode="full", str_dim=96,
                       caacp_score_mode="rank", frh=False, caacp_residual_mode="current",
                       fs_tar=False, fine_tap=False,
                       hf_edge=bool(hf_edge), hf_edge_form=form).float().to(device)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    res = m.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="M1_R7CTRL", choices=sorted(RUN7_VARIANTS))
    ap.add_argument("--dataset", default="CDD-CD-256")
    ap.add_argument("--ckpt-root", default="/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM")
    ap.add_argument("--run", default="Run7")
    ap.add_argument("--ckpt", default=None, help="显式指定权重；默认取该 run/variant/dataset 的 best_F1=*.pth")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--log-fold-err-real", type=float, default=None,
                    help="训练日志里的 [REPARAM-REAL-MAX-ABS-ERROR]，用于同源性自检")
    ap.add_argument("--log-disagree-real", type=float, default=None,
                    help="训练日志里的 [REPARAM-REAL-ARGMAX-DISAGREE]，用于同源性自检")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not str(args.device).startswith("cuda"):
        raise SystemExit("[R7-KNIFE] TinyViM 的 selective_scan 只有 CUDA 实现；请传 --device cuda:0")

    # 与 train.py:882-885 相同的数值设定
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(16)

    ds_root = args.dataset_root or f"/share_datasets/CD/{args.dataset}"
    ck = args.ckpt or os.path.join(args.ckpt_root, args.run, args.variant, args.dataset)
    if os.path.isdir(ck):
        cands = sorted(__import__("glob").glob(os.path.join(ck, "best_F1=*.pth")))
        if not cands:
            raise SystemExit(f"[R7-KNIFE] no best_F1=*.pth in {ck}")
        ck = cands[-1]
    pretrain = "/home/yqwang/projects/CASA-CD/pretrained_weight/tinyvim_s_1000e.pth"

    model = build_model(args.variant, ck, args.device, pretrain)
    print(f"[R7-KNIFE] {args.variant}/{args.dataset} ckpt={os.path.basename(ck)}", flush=True)

    tf = myTransforms.Compose([myTransforms.Normalize(mean=MEAN, std=STD),
                               myTransforms.Scale(256, 256),
                               myTransforms.ToTensor()])
    dset = myDataLoader.Dataset(file_root=ds_root, list_path=os.path.join(ds_root, "list", "test.txt"),
                                transform=tf)
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=2, pin_memory=False)
    img, _ = next(iter(loader))
    pre = img[:, 0:3].float().to(args.device)
    post = img[:, 3:6].float().to(args.device)

    with torch.no_grad():
        y_tr = model(pre, post)
        model.switch_to_deploy()
        model.eval()
        y_dp = model(pre, post)

    a = y_tr.cpu().numpy().astype(np.float64)
    b = y_dp.cpu().numpy().astype(np.float64)
    flip = (a > 0.5) != (b > 0.5)
    d = np.abs(a - b)
    margin_tr = np.abs(a - 0.5)
    margin_dp = np.abs(b - 0.5)
    max_abs = float(d.max())
    disagree = float(flip.mean())
    idx = np.argwhere(flip)
    flipped_margins = margin_tr[flip] if flip.any() else np.array([])
    knife_edge = bool(flip.any() and flipped_margins.max() < max_abs)

    payload = {
        "variant": args.variant, "dataset": args.dataset, "ckpt": ck,
        "batch_size": args.batch_size, "n_pixels": int(a.size),
        "max_abs_diff": max_abs, "n_flipped": int(flip.sum()),
        "disagree": disagree,
        "flipped_per_image": [int(flip[i].sum()) for i in range(flip.shape[0])],
        "flipped_pixels": [
            {"idx": [int(x) for x in p],
             "p_train": float(a[tuple(p)]), "p_deploy": float(b[tuple(p)]),
             "margin_train_abs_p_minus_half": float(abs(a[tuple(p)] - 0.5)),
             "margin_deploy_abs_p_minus_half": float(abs(b[tuple(p)] - 0.5)),
             "abs_delta": float(abs(a[tuple(p)] - b[tuple(p)]))}
            for p in idx[:50]],
        "flipped_margin_max": float(flipped_margins.max()) if flipped_margins.size else None,
        "n_pixels_with_margin_below_max_abs_diff": int((margin_tr < max_abs).sum()),
        "margin_percentiles_abs_p_minus_half": {
            "min": float(np.percentile(margin_tr, 0)), "p0.01": float(np.percentile(margin_tr, 0.01)),
            "p0.1": float(np.percentile(margin_tr, 0.1)), "p1": float(np.percentile(margin_tr, 1)),
            "median": float(np.median(margin_tr))},
        "verdict": ("NO_FLIP" if not flip.any() else
                    ("KNIFE_EDGE_ARTIFACT" if knife_edge else "NEEDS_REVIEW")),
        "note": ("KNIFE_EDGE_ARTIFACT = 所有翻转像素的 |p_train−0.5| 都小于本次折叠的最大绝对误差 "
                 "⇒ 翻转由「该像素在阈值上没有 margin、二值化本就不适定」造成，而非折叠实现错误。"),
    }
    # 同源性自检：与训练日志里的值对拍
    if args.log_fold_err_real is not None:
        # 日志只印 4 位有效数字，故用相对容差对拍（不是 1e-9/1e-12 那种绝对容差）
        r_fold = abs(max_abs - args.log_fold_err_real) / max(args.log_fold_err_real, 1e-30)
        payload["reproduce_check"] = {
            "log_fold_err_real": args.log_fold_err_real, "got_fold_err_real": max_abs,
            "fold_err_real_rel_diff": r_fold,
            "fold_err_real_match": r_fold < 1e-3,
            "log_disagree_real": args.log_disagree_real, "got_disagree_real": disagree,
            "disagree_real_rel_diff": (None if args.log_disagree_real is None else
                                       abs(disagree - args.log_disagree_real)
                                       / max(args.log_disagree_real, 1e-30)),
            "disagree_real_match": (args.log_disagree_real is None
                                    or abs(disagree - args.log_disagree_real)
                                    / max(args.log_disagree_real, 1e-30) < 1e-3),
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
