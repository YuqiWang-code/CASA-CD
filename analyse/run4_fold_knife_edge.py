#!/usr/bin/env python
"""Run4 §7-T5 文档要求的「刀锋像素定位」诊断：找出随机 batch 上 train/deploy 二值翻转的像素，
记录两侧概率、|p−0.5| margin、|Δp|，并统计全图 |p−0.5| 分布，判断翻转是否落在折叠数值误差之内。

    python analyse/run4_fold_knife_edge.py --ckpt <best_F1=*.pth> --variant M1_R4CTRL --device cuda:0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.casa_tvim_str_net import CASATViMSTRNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="M1_R4CTRL", choices=["M1_R4CTRL", "E6_FET1"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fine_tap = args.variant == "E6_FET1"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True

    model = CASATViMSTRNet(
        os.path.join(os.environ.get("CASA_PROJECT", "/home/yqwang/projects/CASA-CD"),
                     "pretrained_weight", "tinyvim_s_1000e.pth"),
        caacp=True, rep_mode="full", str_dim=96, caacp_score_mode="rank", frh=False,
        caacp_residual_mode="current", fs_tar=False, fine_tap=fine_tap).float().to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    res = model.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    torch.manual_seed(args.seed)
    pre = torch.randn(8, 3, 256, 256).to(args.device)
    post = torch.randn(8, 3, 256, 256).to(args.device)
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
    idx = np.argwhere(flip)
    # 每个翻转像素的 |p−0.5| 是否都小于该次折叠的最大绝对误差
    flipped_margins = margin_tr[flip] if flip.any() else np.array([])
    max_abs = float(d.max())
    knife_edge = bool(flip.any() and flipped_margins.max() < max_abs)

    payload = {
        "ckpt": args.ckpt, "variant": args.variant, "seed": args.seed,
        "n_pixels": int(a.size),
        "max_abs_diff": max_abs,
        "n_flipped": int(flip.sum()),
        "flipped_fraction": float(flip.mean()),
        "flipped_per_image": [int(flip[i].sum()) for i in range(flip.shape[0])],
        "flipped_pixels": [
            {"idx": [int(x) for x in p],
             "p_train": float(a[tuple(p)]), "p_deploy": float(b[tuple(p)]),
             "margin_train_abs_p_minus_half": float(abs(a[tuple(p)] - 0.5)),
             "abs_delta": float(abs(a[tuple(p)] - b[tuple(p)]))}
            for p in idx[:50]
        ],
        "flipped_margin_max": float(flipped_margins.max()) if flipped_margins.size else None,
        "margin_percentiles_abs_p_minus_half": {
            "min": float(np.percentile(margin_tr, 0)),
            "p0.01": float(np.percentile(margin_tr, 0.01)),
            "p0.1": float(np.percentile(margin_tr, 0.1)),
            "p1": float(np.percentile(margin_tr, 1)),
            "median": float(np.median(margin_tr)),
        },
        "n_pixels_with_margin_below_max_abs_diff": int((margin_tr < max_abs).sum()),
        "verdict": ("NO_FLIP" if not flip.any() else
                    ("KNIFE_EDGE_ARTIFACT" if knife_edge else "NEEDS_REVIEW")),
        "note": ("KNIFE_EDGE_ARTIFACT = 每个翻转像素的 |p−0.5| 都小于该次折叠的最大绝对误差 ⇒ 翻转由"
                 "「该输入在阈值上没有 margin」造成，而非折叠实现错误；§9-R6 要求定位数值与 FP64 融合，"
                 "本文件即该定位证据。NO_FLIP = 该 batch 无翻转。"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
