"""D1：修正后的小目标错误画像（zero-training，全 test 集，逐 GT 对象明细）。

输出（out_dir）：
  object_level.csv   每个 GT 连通域一行（几何 / recall / hit / 匹配 / 模型概率统计 / 图像上下文）
  image_level.csv    每张图一行（GT/预测像素、TP/FP/FN、对象计数、分组面积）
  summary.json       面积分组 / 形状分层 / 密度分层 / 边界带 / FP 画像 / 门控
  paired_delta.csv   与 --compare-variant 的 same-object Δrecall（若指定）
  paired_summary.json paired 统计 + image-level bootstrap CI
  examples/          --examples 时按**预注册规则**抽取的定性样例（仅服务器本地，不导出）

纪律：只读；不 train()；不写日志/权重；GT 仅用于离线指标与分桶。
"""
import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tvim_diag_common import (  # noqa: E402
    DATA_ROOT, VARIANT_CFG, build_model, ensure_dir, make_loader, pick_best_ckpt,
    read_list_names, set_eval_numerics, sha256_text_lines, write_gate, write_json,
    paired_bootstrap_ci, bootstrap_ci,
)
from tvim_object_metrics import (  # noqa: E402
    AREA_GROUPS, GROUP_ALIAS, METRICS_PROTOCOL, ObjectMetricAccumulator,
)

CSV_FIELDS = [
    "sample_name", "gt_component_id", "area", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "bbox_width", "bbox_height", "bbox_aspect_ratio", "perimeter_approx", "occupancy",
    "touches_border", "pixel_recall", "hit1", "hit25", "hit50",
    "matched_iou_010", "matched_iou_050", "iou_max_any_pred",
    "model_p_mean_in_gt", "model_p_max_in_gt", "model_p_p90_in_gt",
    "image_gt_area_ratio", "image_fp_pixels", "image_fn_pixels",
    "connectivity", "bbox_min_width", "width_stratum", "elongated", "density_stratum",
]

IMAGE_FIELDS = [
    "sample_name", "gt_positive_pixels", "pred_positive_pixels", "tp", "fp", "fn", "tn",
    "F1", "IoU", "recall", "precision", "n_pred_objects", "n_matched_loose",
    "n_matched_strict",
] + [f"gt_pixels_{g}" for g in AREA_GROUPS] + [f"n_gt_objects_{g}" for g in AREA_GROUPS]


# ---------------------------------------------------------------------------- helpers
def _agg_rows(rows, keyfn, label):
    """按 keyfn 分组聚合对象级统计（pooled pixel recall + macro recall + hit 率）。"""
    buckets = {}
    for r in rows:
        k = keyfn(r)
        if k is None:
            continue
        b = buckets.setdefault(k, {"n_objects": 0, "n_pixels": 0, "tp_pixels": 0,
                                   "macro_sum": 0.0, "hit1": 0, "hit25": 0, "hit50": 0,
                                   "n_images": set()})
        b["n_objects"] += 1
        b["n_pixels"] += int(r["area"])
        b["tp_pixels"] += int(round(r["pixel_recall"] * r["area"]))
        b["macro_sum"] += float(r["pixel_recall"])
        b["hit1"] += int(r["hit1"]); b["hit25"] += int(r["hit25"]); b["hit50"] += int(r["hit50"])
        b["n_images"].add(r["sample_name"])
    out = {}
    for k, b in buckets.items():
        n = b["n_objects"]
        out[str(k)] = {
            "n_objects": n, "n_pixels": b["n_pixels"], "tp_pixels": b["tp_pixels"],
            "pixel_recall_micro": b["tp_pixels"] / b["n_pixels"] if b["n_pixels"] else None,
            "object_macro_pixel_recall": b["macro_sum"] / n if n else None,
            "hit1": b["hit1"] / n if n else None,
            "hit25": b["hit25"] / n if n else None,
            "hit50": b["hit50"] / n if n else None,
            "n_images_with_group": len(b["n_images"]),
            "min_group_objects_for_claim": 100,
            "evidence_adequate": n >= 100,
        }
    return {"stratum": label, "buckets": out}


def _write_csv(path, rows, fields):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_one_variant(args, variant, run, out_dir, write_rows=True):
    """跑一个变体的全 test 集，返回 (summary, rows, per_image_accumulator)。"""
    import torch

    device = args.device
    set_eval_numerics()
    ckpt_path, ckpt_meta = pick_best_ckpt(run, variant, args.dataset)
    model, build_info = build_model(variant, device=device, ckpt_path=ckpt_path)
    loader, list_path = make_loader(args.dataset, "test", batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    names = read_list_names(list_path)

    acc = ObjectMetricAccumulator(connectivity=4, bands=(2, 4))
    idx = 0
    with torch.no_grad():
        for img, label in loader:
            pre = img[:, 0:3].to(device).float()
            post = img[:, 3:6].to(device).float()
            gt = (label.to(device) > 0.5)
            prob = model(pre, post)
            prob_np = prob.cpu().numpy()[:, 0]
            pred_np = prob_np > 0.5
            gt_np = gt.cpu().numpy()[:, 0].astype(bool)
            for i in range(pred_np.shape[0]):
                nm = names[idx] if idx < len(names) else f"idx{idx}"
                acc.add(pred_np[i], gt_np[i], name=nm, want_rows=write_rows, prob=prob_np[i])
                idx += 1
            if args.limit and idx >= args.limit:
                break

    summary = acc.summary(label=f"{run}/{variant}/{args.dataset}")
    summary["limited"] = bool(args.limit)
    summary["dry_run"] = bool(args.limit)
    summary["not_for_conclusion"] = bool(args.limit)
    summary["n_images_processed"] = idx
    summary["checkpoint"] = {"path": ckpt_path, "sha256": ckpt_meta["sha256"]}
    summary["build_info"] = build_info
    summary["test_list_sha256"] = sha256_text_lines(list_path)

    rows = list(acc.object_rows)
    if rows:
        summary["strata"] = {
            "width": _agg_rows(rows, lambda r: r.get("width_stratum"), "bbox_min_width"),
            "area_group": _agg_rows(rows, lambda r: _area_group_of(r["area"]), "area_group"),
            "elongated": _agg_rows(rows, lambda r: "elongated" if r["elongated"] else "compact",
                                   "aspect_ratio>=3"),
            "border": _agg_rows(rows, lambda r: "touches_border" if r["touches_border"] else "interior",
                                "touches_image_border"),
            "density": _agg_rows(rows, lambda r: r.get("density_stratum"), "image_density"),
            "occupancy": _agg_rows(rows, lambda r: ("<0.3" if r["occupancy"] < 0.3 else
                                                    ("<0.7" if r["occupancy"] < 0.7 else ">=0.7")),
                                   "bbox_occupancy"),
            "prob_mean": _agg_rows(rows, lambda r: _prob_bucket(r.get("model_p_mean_in_gt")),
                                   "model_p_mean_in_gt bucket"),
        }
        # 与旧 pseudo 指标并列（P0 修正的对照）
        summary["legacy_pseudo_vs_new"] = {
            "old_component_pr_pseudo": "small 0.1796 / medium 0.4620 / large 0.8273（FP 恒 0 的伪指标）",
            "new_small_recall_micro": summary["legacy_groups"]["small"]["pixel_recall_micro"],
            "new_small_object_macro_recall": summary["legacy_groups"]["small"]["object_macro_pixel_recall"],
            "new_small_hit25": summary["legacy_groups"]["small"]["hit25"],
            "new_ObjRecall_loose": summary["object"]["ObjRecall_loose"],
            "new_ObjPrecision_loose": summary["object"]["ObjPrecision_loose"],
        }
    ensure_dir(out_dir)
    if write_rows:
        _write_csv(os.path.join(out_dir, "object_level.csv"), rows, CSV_FIELDS)
        flat = []
        for img in acc.per_image:
            d = {k: v for k, v in img.items() if not isinstance(v, dict)}
            for g in AREA_GROUPS:
                d[f"gt_pixels_{g}"] = img["gt_area_by_group"][g]
                d[f"n_gt_objects_{g}"] = img["n_gt_objects_by_group"][g]
            flat.append(d)
        _write_csv(os.path.join(out_dir, "image_level.csv"), flat, IMAGE_FIELDS)
    summary["n_images_in_list"] = len(names)
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary, rows, acc


def _area_group_of(area):
    from tvim_object_metrics import bin_index
    return bin_index(int(area))


def _prob_bucket(p):
    if p is None:
        return None
    if p < 0.1:
        return "<0.1"
    if p < 0.3:
        return "0.1-0.3"
    if p < 0.5:
        return "0.3-0.5"
    return ">=0.5"


# ---------------------------------------------------------------------------- paired
def run_paired(args, base_rows, base_variant, cmp_rows, cmp_variant, out_dir):
    """same-object paired Δrecall（后 - 前）+ image-level bootstrap CI。"""
    key = lambda r: (r["sample_name"], int(r["gt_component_id"]))
    bmap = {key(r): r for r in base_rows}
    cmap = {key(r): r for r in cmp_rows}
    common = sorted(set(bmap) & set(cmap))
    per_obj = []
    for k in common:
        rb, rc = bmap[k], cmap[k]
        per_obj.append({
            "sample_name": k[0], "gt_component_id": k[1], "area": rb["area"],
            "recall_base": rb["pixel_recall"], "recall_cmp": rc["pixel_recall"],
            "delta_recall": rc["pixel_recall"] - rb["pixel_recall"],
            "hit25_base": rb["hit25"], "hit25_cmp": rc["hit25"],
            "width_stratum": rb.get("width_stratum"), "area_group": _area_group_of(rb["area"]),
            "density_stratum": rb.get("density_stratum"),
            "p_mean_base": rb.get("model_p_mean_in_gt"), "p_mean_cmp": rc.get("model_p_mean_in_gt"),
        })
    if not per_obj:
        return None
    # 图像级聚合后配对 bootstrap（抽样单位 = 图像）
    img_keys = sorted({p["sample_name"] for p in per_obj})
    per_img_delta = []
    for nm in img_keys:
        vals = [p["delta_recall"] for p in per_obj if p["sample_name"] == nm]
        per_img_delta.append(float(np.mean(vals)))
    ci = bootstrap_ci(per_img_delta, n_boot=args.bootstrap, seed=16)
    by_group = {}
    for g in AREA_GROUPS:
        vals = [p["delta_recall"] for p in per_obj if p["area_group"] == g]
        if vals:
            by_group[g] = {"n_objects": len(vals), "mean_delta_recall": float(np.mean(vals)),
                           "positive_frac": float(np.mean(np.array(vals) > 0)),
                           "negative_frac": float(np.mean(np.array(vals) < 0))}
    by_width = {}
    for w in ("<4", "4-7", ">=8"):
        vals = [p["delta_recall"] for p in per_obj if p["width_stratum"] == w]
        if vals:
            by_width[w] = {"n_objects": len(vals), "mean_delta_recall": float(np.mean(vals))}
    summary = {
        "base_variant": base_variant, "compare_variant": cmp_variant,
        "n_common_objects": len(per_obj),
        "n_images": len(img_keys),
        "mean_delta_recall_object_level": float(np.mean([p["delta_recall"] for p in per_obj])),
        "image_level_bootstrap_CI": ci,
        "by_area_group": by_group,
        "by_width_stratum": by_width,
        "note": "paired 视角以 sample_name+GT component_id 稳定匹配；CI 抽样单位=图像；仅表征有限 test 集采样不确定性",
    }
    _write_csv(os.path.join(out_dir, f"paired_delta_{base_variant}_vs_{cmp_variant}.csv"),
               per_obj,
               ["sample_name", "gt_component_id", "area", "area_group", "width_stratum",
                "density_stratum", "recall_base", "recall_cmp", "delta_recall",
                "hit25_base", "hit25_cmp", "p_mean_base", "p_mean_cmp"])
    write_json(os.path.join(out_dir, f"paired_summary_{base_variant}_vs_{cmp_variant}.json"), summary)
    return summary


# ---------------------------------------------------------------------------- examples
def make_examples(args, variant, run, out_dir, base_rows):
    """按预注册规则抽取定性样例（仅用于说明，不进入指标；图片仅留在服务器）。"""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    device = args.device
    model, _ = build_model(variant, device=device,
                           ckpt_path=pick_best_ckpt(run, variant, args.dataset)[0])
    loader, list_path = make_loader(args.dataset, "test", batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    names = read_list_names(list_path)
    by_img = {}
    for r in base_rows:
        by_img.setdefault(r["sample_name"], []).append(r)
    small_fn = {}
    small_fn_rate = {}
    fp_px = {}
    for nm, rs in by_img.items():
        sm = [r for r in rs if r["area"] < 256]
        small_fn[nm] = sum(1 for r in sm if not r["hit25"])
        small_fn_rate[nm] = (small_fn[nm] / len(sm)) if sm else 0.0
        fp_px[nm] = rs[0]["image_fp_pixels"]
    sel = []
    sel += sorted(small_fn, key=lambda n: -small_fn[n])[:6]
    sel += [n for n in sorted(small_fn_rate, key=lambda n: -small_fn_rate[n])
            if len([r for r in by_img[n] if r["area"] < 256]) >= 3][:6]
    sel += sorted(fp_px, key=lambda n: -fp_px[n])[:4]
    rng = np.random.default_rng(16)
    all_names = sorted(by_img)
    sel += list(rng.choice(all_names, size=min(8, len(all_names)), replace=False))
    sel = list(dict.fromkeys(sel))[:32]

    ex_dir = ensure_dir(os.path.join(out_dir, "examples"))
    root = os.path.join(DATA_ROOT, args.dataset)
    idx_of = {n: i for i, n in enumerate(names)}
    picked = set(sel)
    with torch.no_grad():
        idx = 0
        for img, label in loader:
            hit_names = []
            for i in range(img.shape[0]):
                nm = names[idx + i]
                if nm in picked:
                    hit_names.append((i, nm))
            if hit_names:
                pre = img[:, 0:3].to(device).float()
                post = img[:, 3:6].to(device).float()
                prob = model(pre, post).cpu().numpy()[:, 0]
                for i, nm in hit_names:
                    gt = (label[i, 0].numpy() > 0.5)
                    pr = prob[i] > 0.5
                    a = cv2.imread(os.path.join(root, "A", nm))
                    b = cv2.imread(os.path.join(root, "B", nm))
                    if a is None or b is None:
                        continue
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
                    titles = ["A (T1)", "B (T2)", "GT", "M1 pred", "error overlay"]
                    axes[0].imshow(a[:, :, ::-1]); axes[1].imshow(b[:, :, ::-1])
                    axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
                    axes[3].imshow(pr, cmap="gray", vmin=0, vmax=1)
                    ov = np.zeros((*gt.shape, 3), dtype=np.uint8)
                    ov[pr & gt] = (0, 200, 0)          # TP 绿
                    ov[(~pr) & gt] = (220, 0, 0)       # FN 红
                    ov[pr & (~gt)] = (230, 210, 0)     # FP 黄
                    axes[4].imshow(ov)
                    for ax, t in zip(axes, titles):
                        ax.set_title(t, fontsize=9); ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(os.path.join(ex_dir, f"{os.path.splitext(nm)[0]}.png"), dpi=110)
                    plt.close(fig)
            idx += img.shape[0]
    rules = {
        "rule": "预注册固定规则（§5.3）",
        "top6_small_FN_count": sel[:6],
        "top6_small_FN_rate_ge3small": sel[6:12],
        "top4_FP_pixels": sel[12:16],
        "random_seed16_8": sel[16:24],
        "total_selected": len(sel),
        "legend": {"TP": "green", "FN": "red", "FP": "yellow"},
        "note": "样例仅用于说明，不进入任何指标计算；图片保留在服务器，不导出",
    }
    write_json(os.path.join(ex_dir, "selection.json"), rules)
    return rules


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--run", default="Run1")
    ap.add_argument("--variant", default="M1_FULL", choices=sorted(VARIANT_CFG))
    ap.add_argument("--compare-run", default="Run1")
    ap.add_argument("--compare-variant", default="A2_STR", choices=sorted(VARIANT_CFG) + [""])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke：仅前 N 张（标 dry_run）")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    write_json(os.path.join(out_dir, "metrics_protocol.json"), METRICS_PROTOCOL)

    print(f"[D1] base variant={args.variant} run={args.run} dataset={args.dataset}", flush=True)
    base_summary, base_rows, base_acc = run_one_variant(args, args.variant, args.run, out_dir)

    paired = None
    cmp_summary = None
    if args.compare_variant and not args.no_compare:
        cmp_dir = ensure_dir(os.path.join(out_dir, f"compare_{args.compare_variant}"))
        print(f"[D1] compare variant={args.compare_variant}", flush=True)
        cmp_summary, cmp_rows, _ = run_one_variant(args, args.compare_variant, args.compare_run,
                                                   cmp_dir)
        paired = run_paired(args, base_rows, args.variant, cmp_rows, args.compare_variant, out_dir)

    examples = None
    if args.examples:
        print("[D1] generating examples (pre-registered selection rule)", flush=True)
        examples = make_examples(args, args.variant, args.run, out_dir, base_rows)

    # ---- Gate D1-PHENOMENON
    lg = base_summary["groups"]
    n_small = lg["small_64_255"]["n_objects"] + lg["tiny_1_15"]["n_objects"] + lg["tiny_16_63"]["n_objects"]
    n_large = lg["large_1024_inf"]["n_objects"]
    small_rec = base_summary["legacy_groups"]["small"]["pixel_recall_micro"]
    large_rec = base_summary["legacy_groups"]["large"]["pixel_recall_micro"]
    small_hit25 = base_summary["legacy_groups"]["small"]["hit25"]
    large_hit25 = base_summary["legacy_groups"]["large"]["hit25"]
    gap_pp = None
    if small_rec is not None and large_rec is not None:
        gap_pp = (large_rec - small_rec) * 100
    checks = {
        "n_images_processed": base_summary["n_images_processed"],
        "n_images_in_list": base_summary.get("n_images_in_list"),
        "full_test_set": (not args.limit) and (base_summary["n_images_processed"] == base_summary.get("n_images_in_list")),
        "small_objects_ge_100": n_small >= 100,
        "large_objects_ge_100": n_large >= 100,
        "small_pixel_recall_micro": small_rec,
        "large_pixel_recall_micro": large_rec,
        "small_hit25": small_hit25,
        "large_hit25": large_hit25,
        "size_gap_pp_pooled_recall": gap_pp,
        "size_gap_threshold_10pp_met": (gap_pp is not None and gap_pp >= 10.0),
        "old_pseudo_small_F1_replaced": True,
        "paired_available": paired is not None,
        "examples_available": examples is not None,
    }
    status = "PASS" if (checks["small_objects_ge_100"] and checks["large_objects_ge_100"]) else "WARN"
    if args.limit:
        status = "WARN"
    write_gate(out_dir, "D1-PHENOMENON", status, checks,
               extra={"note": "small 组对象数为 1<=area<256；旧 0.1796 为伪指标，见 legacy_pseudo_vs_new"})
    print(f"[D1] gate={status} small_n={n_small} large_n={n_large} "
          f"small_recall={small_rec} large_recall={large_rec} gap_pp={gap_pp}", flush=True)
    print(f"[D1] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
