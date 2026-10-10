"""R4 §10.2 互补性 preflight（run4_fet_preflight_v1）——零正式训练的前测门。

**唯一问题**：Diag1 的 `L01b_norm0` small margin=0.0779 是**全体 GT small 对象**的均值，
不能推出 **M1 已漏检对象**仍有浅层辨识度。本脚本用固定 M1 checkpoint 与固定
`D3_concat/M1_FULL/probe_L01b_norm0_PROBE_ONLY.pth`（均只读、不重训、不改权重），
在 SYSU 4000 张 test 上一次性前向，计算：

  * 每张图 M1 `p>0.5` → GT small 对象（1–255px，4 连通）的 `Hit@25`；
    `Hit@25 < 25%` 记 **missed-small**；
  * 每张图 M1 `p>0.5` → 与 GT 做一次最大权重匹配（IoU≥0.10），仍未匹配且面积 1–255px
    的预测连通域记 **FP-like**（Diag1 为 2430 个）；
  * 对二者都用**冻结 shallow probe** 概率场计算
    `margin_probe = mean(对象内) − mean(外侧 4px 环带且剔除所有 GT)`（与 D3c 同口径）；
  * `rescue_rate = P(margin_probe ≥ 0.03 | missed-small)`，`fp_like_rate = 同 | FP-like`，
    `gap = rescue_rate − fp_like_rate`，`bootstrap=1000` **按图像抽样** 的 gap 95%CI。

**PASS 当且仅当**：`n_missed_small>=100`、`n_fp_like>=100`、`rescue_rate>=0.20`、
`gap>=0.10`、`gap CI 下界>0`。任一不满足 ⇒ `INCONCLUSIVE/FAIL → NO_80K`。
GPU 预算上界 **10 min**；超时记 `PRECHECK-INCOMPLETE`，不扩成无限诊断。
输出目录独立，绝不覆盖 Diag1 / Run1–3 产物。

    python analyse/tvim_run4_tap_preflight.py --device cuda:0 \
      --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
      --probe-dir "$DIAG/D3_concat/M1_FULL" --out-dir "$OUT"
"""
import argparse
import datetime
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvim_diag_common import (  # noqa: E402
    DATA_ROOT, build_model, ensure_dir, make_loader, pick_best_ckpt, read_list_names,
    set_eval_numerics, sha256_file, write_json,
)
from tvim_linear_probe import ENCODER_KEYS, _encoder_probe_input, _resolve_module  # noqa: E402
from tvim_object_metrics import (  # noqa: E402
    GROUP_ALIAS, bin_index, label_components, match_objects, object_iou_matrix,
)

PREFLIGHT_VERSION = "run4_fet_preflight_v1"
MARGIN_THRESHOLD = 0.03          # 预注册诊断阈值（不是分割阈值，不参与 E6 推理）
BOOTSTRAP = 1000
GATE = {
    "n_missed_small_min": 100,
    "n_fp_like_min": 100,
    "rescue_rate_min": 0.20,
    "gap_min": 0.10,
    "gap_ci_low_min": 0.0,
}
BUDGET_S = 600.0                  # §10.2：单次 10 min GPU 预算上界
SMALL_ALIAS = set(GROUP_ALIAS["small"])
M1_SYSU_SHA = "45d688a01af22fe22521f4dfb4579e80827ddcaa886e07bafc778fc288731457"


def _load_probe(probe_dir, key, device):
    """只加载指定 shallow probe；不重新拟合、不访问其它 checkpoint。"""
    path = os.path.join(probe_dir, f"probe_{key}_PROBE_ONLY.pth")
    if not os.path.isfile(path):
        raise SystemExit(f"[PREFLIGHT] missing probe: {path}")
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    module = torch.nn.Conv2d(ck["C_in"], 1, 1).to(device)
    module.load_state_dict(ck["probe"])
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module, ck, path


def _ring(mobj, gt, iterations=4):
    from scipy import ndimage as ndi
    st = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    return ndi.binary_dilation(mobj, structure=st, iterations=iterations) & ~mobj & ~gt


def _margin(prob, mask, gt):
    ring = _ring(mask, gt)
    if not ring.any() or not mask.any():
        return None
    return float(prob[mask].mean() - prob[ring].mean())


def _bootstrap_gap(recs, image_ids, n_boot=BOOTSTRAP, seed=16):
    """按图像重采样（像素/对象不作为独立样本）：返回 (gap, ci_low, ci_high)。"""
    rng = np.random.default_rng(seed)
    img_ids = np.asarray(sorted(image_ids))
    n_miss = np.array([r["n_missed"] for r in recs], dtype=np.float64)
    n_fp = np.array([r["n_fp"] for r in recs], dtype=np.float64)
    rescue = np.array([r["rescue"] for r in recs], dtype=np.float64)
    fpab = np.array([r["fp_above"] for r in recs], dtype=np.float64)
    obs = (rescue.sum() / max(1.0, n_miss.sum())) - (fpab.sum() / max(1.0, n_fp.sum()))
    n_img = len(img_ids)
    gaps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_img, size=n_img)
        nm, nf = n_miss[idx].sum(), n_fp[idx].sum()
        rr = rescue[idx].sum() / nm if nm > 0 else np.nan
        fr = fpab[idx].sum() / nf if nf > 0 else np.nan
        gaps[b] = rr - fr
    gaps = gaps[np.isfinite(gaps)]
    lo, hi = (float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))) if gaps.size else (None, None)
    return float(obs), lo, hi


def run(args):
    import torch
    import torch.nn.functional as F
    from scipy import ndimage as ndi

    t_start = time.time()
    out_dir = ensure_dir(args.out_dir)
    device = args.device
    set_eval_numerics()

    # ------------------------------------------------------------------ 只读资产
    ckpt_path, ckpt_meta = pick_best_ckpt(args.run, args.variant, args.dataset)
    m1_sha = sha256_file(ckpt_path) if ckpt_path and os.path.isfile(ckpt_path) else None
    probe, probe_ck, probe_path = _load_probe(args.probe_dir, args.probe_key, device)
    probe_sha = sha256_file(probe_path)
    list_path = os.path.join(DATA_ROOT, args.dataset, "list", "test.txt")
    list_sha = sha256_file(list_path)
    names = read_list_names(list_path)
    print(f"[PREFLIGHT] {PREFLIGHT_VERSION} dataset={args.dataset} run={args.run} "
          f"variant={args.variant} n_test_list={len(names)}", flush=True)
    print(f"[PREFLIGHT] M1 ckpt={ckpt_path}\n            sha256={m1_sha}", flush=True)
    print(f"[PREFLIGHT] probe={probe_path}\n            sha256={probe_sha} "
          f"encoder_input={probe_ck.get('protocol', {}).get('encoder_input')}", flush=True)

    model, minfo = build_model(args.variant, device=device, ckpt_path=ckpt_path, strict=True)
    assert not minfo["missing_keys"] and not minfo["unexpected_keys"], minfo

    # ------------------------------------------------------------------ 单次 test pass
    handles, cap = [], {}
    handles.append(_resolve_module(model, args.probe_key).register_forward_hook(
        lambda m, i, o: cap.__setitem__("f", o.detach())))
    loader, _ = make_loader(args.dataset, "test", batch_size=args.batch_size,
                            num_workers=args.num_workers)
    enc_input = probe_ck.get("protocol", {}).get("encoder_input", "absdiff")

    per_object = []
    per_image = []
    n_seen = 0
    small_px = {"n_obj": 0, "n_px": 0, "tp_px": 0, "hit25": 0}
    try:
        with torch.no_grad():
            for img, label in loader:
                cap.clear()
                pre = img[:, 0:3].to(device).float()
                post = img[:, 3:6].to(device).float()
                out = model(pre, post)
                B = pre.shape[0]
                gts = (label.numpy()[:, 0] > 0.5)
                probs = out[:, 0].float().cpu().numpy()
                for i in range(B):
                    nm = names[n_seen] if n_seen < len(names) else f"idx{n_seen}"
                    gt = gts[i]
                    pm1 = probs[i]
                    if args.probe_key in ENCODER_KEYS:
                        feat = _encoder_probe_input(cap["f"][i].float(), cap["f"][B + i].float(),
                                                    enc_input)[None]
                    else:
                        feat = cap["f"][i].float()[None]
                    pp = torch.sigmoid(probe(feat))[0, 0]
                    if pp.shape != (256, 256):
                        pp = F.interpolate(pp[None, None], size=(256, 256), mode="bilinear",
                                           align_corners=False)[0, 0]
                    pprobe = pp.cpu().numpy().astype(np.float64)

                    rec = {"image": nm, "n_missed": 0, "rescue": 0, "n_fp": 0, "fp_above": 0}
                    # ---- GT small objects
                    if gt.any():
                        lab, n_gt = label_components(gt, 4)
                        counts = np.bincount(lab.ravel(), minlength=n_gt + 1) if n_gt else np.zeros(1)
                        pb1 = pm1 > 0.5
                        tp_counts = (np.bincount(lab[pb1 & gt].ravel(), minlength=n_gt + 1)
                                     if (pb1 & gt).any() else np.zeros(n_gt + 1, dtype=np.int64))
                        for j in range(1, n_gt + 1):
                            area = int(counts[j])
                            if bin_index(area) not in SMALL_ALIAS:
                                continue
                            tp = int(tp_counts[j])
                            r = tp / area
                            hit25 = r >= 0.25
                            small_px["n_obj"] += 1
                            small_px["n_px"] += area
                            small_px["tp_px"] += tp
                            small_px["hit25"] += int(hit25)
                            if hit25:
                                continue
                            mobj = (lab == j)
                            mv = _margin(pprobe, mobj, gt)
                            if mv is None:
                                continue
                            rec["n_missed"] += 1
                            rec["rescue"] += int(mv >= MARGIN_THRESHOLD)
                            per_object.append({"image": nm, "kind": "missed_small", "area": area,
                                               "pixel_recall": float(r), "margin_probe": mv,
                                               "above_threshold": int(mv >= MARGIN_THRESHOLD)})
                    # ---- FP-like: M1 unmatched predicted CC (1–255 px)
                    if (pm1 > 0.5).any():
                        cache = object_iou_matrix(pm1 > 0.5, gt, 4)
                        mt = match_objects(pm1 > 0.5, gt, 4, cache=cache)
                        pr_lab = cache["pr_lab"]
                        matched = set(mt["matched_pred_idx"])
                        for j in range(cache["n_pred"]):
                            if j in matched:
                                continue
                            area = int(cache["pred_areas"][j])
                            if bin_index(area) not in SMALL_ALIAS:
                                continue
                            mobj = (pr_lab == j + 1)
                            mv = _margin(pprobe, mobj, gt)
                            if mv is None:
                                continue
                            rec["n_fp"] += 1
                            rec["fp_above"] += int(mv >= MARGIN_THRESHOLD)
                            per_object.append({"image": nm, "kind": "fp_like", "area": area,
                                               "pixel_recall": None, "margin_probe": mv,
                                               "above_threshold": int(mv >= MARGIN_THRESHOLD)})
                    per_image.append(rec)
                    n_seen += 1
                if args.limit and n_seen >= args.limit:
                    break
                if n_seen % 400 == 0:
                    el = time.time() - t_start
                    print(f"[PREFLIGHT] {n_seen} images  elapsed={el:.0f}s", flush=True)
                    if el > BUDGET_S:
                        print(f"[PREFLIGHT][TIMEOUT] budget {BUDGET_S:.0f}s exceeded at {n_seen} images",
                              flush=True)
                        break
    finally:
        for h in handles:
            h.remove()

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------ 汇总与判定
    n_missed = int(sum(r["n_missed"] for r in per_image))
    n_fp = int(sum(r["n_fp"] for r in per_image))
    rescue_n = int(sum(r["rescue"] for r in per_image))
    fp_above_n = int(sum(r["fp_above"] for r in per_image))
    rescue_rate = (rescue_n / n_missed) if n_missed else None
    fp_like_rate = (fp_above_n / n_fp) if n_fp else None
    gap, ci_lo, ci_hi = _bootstrap_gap(per_image, range(len(per_image)))

    small_recall = (small_px["tp_px"] / small_px["n_px"]) if small_px["n_px"] else None
    small_hit25 = (small_px["hit25"] / small_px["n_obj"]) if small_px["n_obj"] else None

    checks = {
        "n_missed_small>=100": {"value": n_missed, "threshold": GATE["n_missed_small_min"],
                                "pass": n_missed >= GATE["n_missed_small_min"]},
        "n_fp_like>=100": {"value": n_fp, "threshold": GATE["n_fp_like_min"],
                           "pass": n_fp >= GATE["n_fp_like_min"]},
        "rescue_rate>=0.20": {"value": rescue_rate, "threshold": GATE["rescue_rate_min"],
                              "pass": rescue_rate is not None and rescue_rate >= GATE["rescue_rate_min"]},
        "gap>=0.10": {"value": gap, "threshold": GATE["gap_min"],
                      "pass": gap >= GATE["gap_min"]},
        "gap_ci_low>0": {"value": ci_lo, "threshold": GATE["gap_ci_low_min"],
                         "pass": ci_lo is not None and ci_lo > GATE["gap_ci_low_min"]},
        "gpu_budget<=600s": {"value": round(elapsed, 1), "threshold": BUDGET_S,
                             "pass": elapsed <= BUDGET_S},
        "all_4000_test_images": {"value": n_seen, "threshold": len(names),
                                 "pass": (not args.limit) and n_seen == len(names)},
        "m1_ckpt_sha_expected": {"value": m1_sha, "threshold": M1_SYSU_SHA,
                                 "pass": m1_sha == M1_SYSU_SHA},
    }
    all_pass = all(c["pass"] for c in checks.values())
    if elapsed > BUDGET_S:
        status = "PRECHECK-INCOMPLETE"
    else:
        status = "PASS" if all_pass else "FAIL"

    payload = {
        "stage": "run4_preflight",
        "status": status,
        "preflight_version": PREFLIGHT_VERSION,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "run": args.run,
        "variant": args.variant,
        "probe_key": args.probe_key,
        "checks": checks,
        "n_images": n_seen,
        "n_list_images": len(names),
        "m1_ckpt": ckpt_path,
        "m1_ckpt_sha256": m1_sha,
        "probe_ckpt": probe_path,
        "probe_ckpt_sha256": probe_sha,
        "probe_encoder_input": enc_input,
        "test_list": list_path,
        "test_list_sha256": list_sha,
        "margin_threshold": MARGIN_THRESHOLD,
        "bootstrap": BOOTSTRAP,
        "bootstrap_unit": "image",
        "n_missed_small": n_missed,
        "n_fp_like": n_fp,
        "rescue_rate": rescue_rate,
        "fp_like_rate": fp_like_rate,
        "gap": gap,
        "gap_ci95": [ci_lo, ci_hi],
        "elapsed_s": round(elapsed, 1),
        "gpu_budget_s": BUDGET_S,
        "small_pooled_recall_1_255": small_recall,
        "small_hit25_1_255": small_hit25,
        "n_small_objects": small_px["n_obj"],
        "decision": ("80K ALLOWED (gate PASS)" if status == "PASS" else
                     "NO 80K (gate not PASS) - return to Diag1 D1/D3 FP profiling, no 1/8 or 3x3 search"),
    }
    write_json(os.path.join(out_dir, "gate.json"), payload)
    write_json(os.path.join(out_dir, "preflight.json"), payload)

    import csv
    with open(os.path.join(out_dir, "per_object.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "kind", "area", "pixel_recall",
                                          "margin_probe", "above_threshold"])
        w.writeheader()
        for r in per_object:
            w.writerow(r)

    print(f"[PREFLIGHT] images={n_seen} elapsed={elapsed:.0f}s", flush=True)
    print(f"[PREFLIGHT] n_missed_small={n_missed} rescue_rate={rescue_rate} "
          f"n_fp_like={n_fp} fp_like_rate={fp_like_rate}", flush=True)
    print(f"[PREFLIGHT] gap={gap} ci95=[{ci_lo}, {ci_hi}] (bootstrap={BOOTSTRAP}, image-level)", flush=True)
    print(f"[PREFLIGHT] small pooled recall(1-255)={small_recall} hit25={small_hit25} "
          f"(Diag1 M1: 0.1974 / 0.1896)", flush=True)
    print(f"[PREFLIGHT] STATUS={status} -> {os.path.join(out_dir, 'gate.json')}", flush=True)
    return 0 if status == "PASS" else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dataset", type=str, default="SYSU-CD-256")
    ap.add_argument("--run", type=str, default="Run1")
    ap.add_argument("--variant", type=str, default="M1_FULL")
    ap.add_argument("--probe-dir", type=str, required=True)
    ap.add_argument("--probe-key", type=str, default="L01b_norm0")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug only; non-zero forces gate FAIL")
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
