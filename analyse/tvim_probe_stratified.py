"""D3c：按 GT 面积分层的 probe 可读性评价（使用已保存的 PROBE_ONLY 权重，无需重新拟合）。

动机（Diag1 §10 新主方向）：D3 报告的 probe AP 是**像素加权 pooled AP**，被中/大目标主导，
因此"整体可读性 0.90"与 D1 的"small pooled pixel Recall 0.1974"并不矛盾。本脚本把 probe
按 GT 面积分层评估，回答：**probe 能否读出"小变化"**，还是只能读出"变化整体"。

口径（写进 probe_stratified.json:protocol）：
  * 正样本 = 指定 GT 面积组的像素；**负样本 = 全部背景像素（GT=0）**，即其它 GT 对象的像素
    从负样本中排除（避免把"正确检出其它对象"算作误报）。
  * AP 由 2000-bin 分数直方图累计（与 D3 同口径）；同时输出 pooled（全部 GT 像素 vs 背景）作为对拍。
  * per-object margin = 对象内 probe 概率均值 − 外侧 4px 环带（排除所有 GT 像素）均值；
    按组汇总 mean/median + image-level bootstrap CI。
  * 组定义：small(1–255px)、medium(256–1023)、large(≥1024)，4 邻接连通域，原生 256² GT。

用法：
    python analyse/tvim_probe_stratified.py --probe-dir "$DIAG/D3_concat/M1_FULL" \
        --dataset SYSU-CD-256 --run Run1 --variant M1_FULL --device cuda:0 \
        --out-dir "$DIAG/D3_stratified/M1_FULL"
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tvim_diag_common import (  # noqa: E402
    DATA_ROOT, VARIANT_CFG, build_model, ensure_dir, make_loader, pick_best_ckpt,
    read_list_names, set_eval_numerics, write_gate, write_json, bootstrap_ci,
)
from tvim_object_metrics import GROUP_ALIAS, bin_index, label_components  # noqa: E402
from tvim_stage_recoverability import NBINS, _ap_from_hist  # noqa: E402
from tvim_linear_probe import ENCODER_KEYS, _encoder_probe_input, _resolve_module  # noqa: E402

GROUPS = ("small", "medium", "large")


def _group_map(gt_bool):
    """返回 (group_id_map, {group: pixel_masks})；group_id_map: 0=背景,1..3=small/medium/large。"""
    lab, n = label_components(gt_bool, 4)
    gid = np.zeros(gt_bool.shape, dtype=np.int8)
    if n:
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        for j in range(1, n + 1):
            g = bin_index(int(counts[j]))
            gi = 1 if g in GROUP_ALIAS["small"] else (2 if g in GROUP_ALIAS["medium"] else 3)
            gid[lab == j] = gi
    return gid, lab, n


def run(args):
    import torch
    import torch.nn.functional as F
    from scipy import ndimage as ndi

    out_dir = ensure_dir(args.out_dir)
    device = args.device
    set_eval_numerics()
    ckpt_path, ckpt_meta = pick_best_ckpt(args.run, args.variant, args.dataset)
    model, _ = build_model(args.variant, device=device, ckpt_path=ckpt_path)

    probe_files = sorted(glob.glob(os.path.join(args.probe_dir, "probe_*_PROBE_ONLY.pth")))
    if not probe_files:
        raise SystemExit(f"no PROBE_ONLY weights in {args.probe_dir}")
    probes = {}
    for pf in probe_files:
        ck = torch.load(pf, map_location="cpu", weights_only=False)
        key = os.path.basename(pf)[len("probe_"):-len("_PROBE_ONLY.pth")]
        module = torch.nn.Conv2d(ck["C_in"], 1, 1).to(device)
        module.load_state_dict(ck["probe"])
        module.eval()
        probes[key] = {"module": module, "C_in": ck["C_in"],
                       "encoder_input": (ck.get("protocol") or {}).get("encoder_input", "absdiff")}
    keys = sorted(probes)
    print(f"[D3c] probes: {keys}", flush=True)

    # 逐层一次 test pass（每层都注册 hook，一次前向同时取全部节点）
    handles, cap = [], {}
    for k in keys:
        handles.append(_resolve_module(model, k).register_forward_hook(
            lambda m, i, o, kk=k: cap.__setitem__(kk, o.detach())))

    loader, list_path = make_loader(args.dataset, "test", batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    names = read_list_names(list_path)

    hist = {k: {"bg": np.zeros(NBINS)} for k in keys}
    for k in keys:
        for g in GROUPS:
            hist[k][g] = np.zeros(NBINS)
    margins = {k: {g: [] for g in GROUPS} for k in keys}
    margin_img = {k: {g: {} for g in GROUPS} for k in keys}
    n_seen = 0
    st = {k: {"tp": 0, "fp": 0, "fn": 0} for k in keys}

    try:
        with torch.no_grad():
            for img, label in loader:
                cap.clear()
                pre = img[:, 0:3].to(device).float()
                post = img[:, 3:6].to(device).float()
                _ = model(pre, post)
                B = pre.shape[0]
                gts = (label.numpy()[:, 0] > 0.5)
                for i in range(B):
                    nm = names[n_seen] if n_seen < len(names) else f"idx{n_seen}"
                    gt = gts[i]
                    if not gt.any():
                        n_seen += 1
                        continue
                    gid, lab, nlab = _group_map(gt)
                    bg = ~gt
                    for k in keys:
                        t = cap[k]
                        if k in ENCODER_KEYS:
                            feat = _encoder_probe_input(t[i].float(), t[B + i].float(),
                                                        probes[k]["encoder_input"])[None]
                        else:
                            feat = t[i].float()[None]
                        p = torch.sigmoid(probes[k]["module"](feat))[0, 0]
                        if p.shape != (256, 256):
                            p = F.interpolate(p[None, None], size=(256, 256), mode="bilinear",
                                              align_corners=False)[0, 0]
                        pr = p.cpu().numpy().astype(np.float64)
                        # 背景直方图（所有层共享同一背景像素集合）
                        hb, _ = np.histogram(pr[bg], bins=NBINS, range=(0.0, 1.0))
                        hist[k]["bg"] += hb
                        for gi, g in ((1, "small"), (2, "medium"), (3, "large")):
                            m = (gid == gi)
                            if not m.any():
                                continue
                            hp, _ = np.histogram(pr[m], bins=NBINS, range=(0.0, 1.0))
                            hist[k][g] += hp
                        # 预测二值（0.5）用于计数（仅 pooled 参考）
                        pb = pr > 0.5
                        st[k]["tp"] += int(np.count_nonzero(pb & gt))
                        st[k]["fp"] += int(np.count_nonzero(pb & ~gt))
                        st[k]["fn"] += int(np.count_nonzero(~pb & gt))
                        # per-object margin（按组）
                        if nlab:
                            objs = ndi.find_objects(lab)
                            counts = np.bincount(lab.ravel(), minlength=nlab + 1)
                            for j in range(1, nlab + 1):
                                g = bin_index(int(counts[j]))
                                grp = ("small" if g in GROUP_ALIAS["small"] else
                                       ("medium" if g in GROUP_ALIAS["medium"] else "large"))
                                mobj = (lab == j)
                                ring = ndi.binary_dilation(
                                    mobj, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool),
                                    iterations=4) & ~mobj & bg
                                if not ring.any():
                                    continue
                                mv = float(pr[mobj].mean() - pr[ring].mean())
                                margins[k][grp].append(mv)
                                margin_img[k][grp].setdefault(nm, []).append(mv)
                    n_seen += 1
                if args.limit and n_seen >= args.limit:
                    break
    finally:
        for h in handles:
            h.remove()

    results = {"protocol": {
        "positives_per_group": "指定 GT 面积组的像素",
        "negatives": "全部背景像素（GT=0）；其它 GT 对象的像素不计入负样本",
        "AP": f"{NBINS}-bin 分数直方图累计（与 D3 同口径）",
        "margin": "对象内 probe 概率均值 − 外侧 4px 环带（排除 GT 像素）均值",
        "groups": "small(1–255) / medium(256–1023) / large(≥1024)，4 邻接，原生 256² GT",
        "probe_dir": args.probe_dir, "dataset": args.dataset, "variant": args.variant,
        "n_images_scored": None,
    }, "per_layer": {}}
    for k in keys:
        bg_hist = hist[k]["bg"]
        pooled_pos = sum(hist[k][g] for g in GROUPS)
        entry = {"C_in": probes[k]["C_in"],
                 "encoder_input": probes[k]["encoder_input"],
                 # AP 的分母必须是"考虑集合的全部像素"（= 该组正样本 + 背景），
                 # 只传背景会因 cumsum(pos) > cumsum(bg) 产生 >1 的伪 AP（已修正）
                 "pooled_AP_all_gt_vs_bg": _ap_from_hist(pooled_pos, pooled_pos + bg_hist),
                 "pixel_counts": {kk: int(sum(hist[k][g].sum() for g in GROUPS)) if kk == "gt" else None
                                  for kk in ("gt",)},
                 "pred_counts": st[k],
                 "groups": {}}
        for g in GROUPS:
            pos = hist[k][g]
            n_pos = int(pos.sum())
            ap = _ap_from_hist(pos, pos + bg_hist) if n_pos > 0 else None
            mv = margins[k][g]
            per_img = [float(np.mean(v)) for v in margin_img[k][g].values()] if margin_img[k][g] else []
            entry["groups"][g] = {
                "n_positive_pixels": n_pos,
                "AP_vs_background": ap,
                "positive_prior_vs_bg": (n_pos / (n_pos + bg_hist.sum())) if bg_hist.sum() else None,
                "n_objects_with_margin": len(mv),
                "margin_mean": float(np.mean(mv)) if mv else None,
                "margin_median": float(np.median(mv)) if mv else None,
                "margin_image_level_bootstrap_CI": bootstrap_ci(per_img, n_boot=args.bootstrap, seed=16) if per_img else None,
            }
        results["per_layer"][k] = entry
    results["protocol"]["n_images_scored"] = n_seen
    write_json(os.path.join(out_dir, "probe_stratified.json"), results)
    write_json(os.path.join(out_dir, "probe_protocol_stratified.json"), results["protocol"])

    checks = {"n_layers": len(keys), "n_images_scored": n_seen,
              "full_test_set": (not args.limit),
              "pooled_AP": {k: results["per_layer"][k]["pooled_AP_all_gt_vs_bg"] for k in keys},
              "small_AP": {k: results["per_layer"][k]["groups"]["small"]["AP_vs_background"] for k in keys},
              "large_AP": {k: results["per_layer"][k]["groups"]["large"]["AP_vs_background"] for k in keys}}
    write_gate(out_dir, "D3c-STRATIFIED", "PASS" if (not args.limit) else "WARN", checks)
    print(f"[D3c] images={n_seen} gate={'PASS' if not args.limit else 'WARN'}", flush=True)
    for k in keys:
        e = results["per_layer"][k]
        g = e["groups"]
        print(f"  {k:26s} pooled={e['pooled_AP_all_gt_vs_bg']:.4f} "
              f"small={g['small']['AP_vs_background']} medium={g['medium']['AP_vs_background']} "
              f"large={g['large']['AP_vs_background']} "
              f"| margin small={None if g['small']['margin_mean'] is None else round(g['small']['margin_mean'],5)} "
              f"large={None if g['large']['margin_mean'] is None else round(g['large']['margin_mean'],5)}", flush=True)
    print(f"[D3c] wrote {out_dir}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--run", default="Run1")
    ap.add_argument("--variant", default="M1_FULL", choices=sorted(VARIANT_CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
