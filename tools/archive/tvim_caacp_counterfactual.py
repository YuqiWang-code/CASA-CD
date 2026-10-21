"""D4：CAACP β 运行时受控反事实（ON/OFF，零训练；不可冒充从头重训消融）。

  Y_on  : 训练完成的 M1 原模型（原 β）。
  Y_off : deepcopy(M1)，仅 `encoder.caacp_op.beta.zero_()`，其余参数**逐位相同**。

比较同一批 A/B test 顺序上的：
  * 全图像素 Recall/Precision/F1/IoU（与官方口径一致）
  * GT small / tiny 组的 pooled pixel recall、object-macro recall、Hit@25
  * per-object paired Δrecall（same sample_name + GT component_id）+ image-level bootstrap CI

解释边界（写入报告）：`Y_on − Y_off` 只衡量**当前这组训练权重对 CAACP 修正项的局部依赖**；
M1 的其余参数是在 β 可学习条件下训练的，因此它**不是** A2（从头训练无 CAACP）的受控对照，
也不能证明"Stage3 是最早损失源"。
"""
import os
import sys
import csv
import copy
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tvim_diag_common import (  # noqa: E402
    VARIANT_CFG, build_model, ensure_dir, make_loader, pick_best_ckpt, read_list_names,
    set_eval_numerics, write_gate, write_json, bootstrap_ci, counts_to_scores,
)
from tvim_object_metrics import (  # noqa: E402
    AREA_GROUPS, ObjectMetricAccumulator, METRICS_PROTOCOL,
)


def _bn_checksum(model):
    import hashlib
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        if "running_" in k or k.endswith(".num_batches_tracked"):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _param_diff_keys(m1, m2):
    import torch
    d1, d2 = m1.state_dict(), m2.state_dict()
    keys = sorted(set(d1) | set(d2))
    diff = []
    for k in keys:
        if k not in d1 or k not in d2:
            diff.append(k)
            continue
        a, b = d1[k], d2[k]
        if a.shape != b.shape:
            diff.append(k)
            continue
        if not torch.equal(a.float(), b.float()):
            diff.append(k)
    return diff


def _beta_alias_keys(model):
    """同一 β 参数在 state_dict 中的全部别名键（caacp_block.op / network.4.8.op / caacp_op 指向同一模块）。"""
    import torch
    beta = model.encoder.caacp_op.beta
    ptr = beta.data_ptr()
    aliases = set()
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor) and v.data_ptr() == ptr and v.shape == beta.shape:
            aliases.add(k)
    return aliases


def _run_pass(model, args, device, names):
    """跑一遍全 test 集，返回 (summary, per_image, acc)。acc.object_rows 为逐 GT 对象明细。"""
    import torch
    loader, _ = make_loader(args.dataset, "test", batch_size=args.batch_size,
                            num_workers=args.num_workers)
    acc = ObjectMetricAccumulator(connectivity=4, bands=(2, 4))
    idx = 0
    with torch.no_grad():
        for img, label in loader:
            pre = img[:, 0:3].to(device).float()
            post = img[:, 3:6].to(device).float()
            gt = (label.to(device) > 0.5)
            prob = model(pre, post).cpu().numpy()[:, 0]
            pred = prob > 0.5
            gtn = gt.cpu().numpy()[:, 0].astype(bool)
            for i in range(pred.shape[0]):
                nm = names[idx] if idx < len(names) else f"idx{idx}"
                acc.add(pred[i], gtn[i], name=nm, want_rows=True, prob=prob[i])
                idx += 1
            if args.limit and idx >= args.limit:
                break
    return acc.summary(), acc.per_image, acc


def run_d4(args):
    import torch

    out_dir = ensure_dir(args.out_dir)
    device = args.device
    set_eval_numerics()
    ckpt_path, ckpt_meta = pick_best_ckpt(args.run, args.variant, args.dataset)
    model_on, build_info = build_model(args.variant, device=device, ckpt_path=ckpt_path)
    if model_on.encoder.caacp_op is None:
        write_gate(out_dir, "D4-COUNTERFACT", "FAIL",
                   {"caacp_present": False}, extra={"note": "variant has no CAACP op"})
        raise SystemExit("D4 requires a CAACP variant")
    beta_on = float(model_on.encoder.caacp_op.beta.detach().cpu().item())

    model_off = copy.deepcopy(model_on)
    with torch.no_grad():
        model_off.encoder.caacp_op.beta.zero_()
    beta_off = float(model_off.encoder.caacp_op.beta.detach().cpu().item())

    # ---- 受影响键核对（只应有 β；注意 β 在 state_dict 中存在别名键，指向同一 tensor）
    beta_alias = _beta_alias_keys(model_on)
    diff_keys = _param_diff_keys(model_on, model_off)
    only_beta_changed = bool(diff_keys) and set(diff_keys) == beta_alias
    bn_on_before, bn_off = _bn_checksum(model_on), _bn_checksum(model_off)

    loader, list_path = make_loader(args.dataset, "test", batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    names = read_list_names(list_path)

    # ---- ON 复现性：与独立 forward 逐位一致
    img_r, _ = next(iter(loader))
    pre_r = img_r[:, 0:3].to(device).float()
    post_r = img_r[:, 3:6].to(device).float()
    with torch.no_grad():
        y_a = model_on(pre_r, post_r)
        y_b = model_on(pre_r, post_r)
    on_reproducible = float((y_a - y_b).abs().max().item())

    print(f"[D4] ON pass (beta={beta_on:.6g})", flush=True)
    sum_on, imgs_on, acc_on = _run_pass(model_on, args, device, names)
    print("[D4] OFF pass (beta=0)", flush=True)
    sum_off, imgs_off, acc_off = _run_pass(model_off, args, device, names)

    bn_on_after = _bn_checksum(model_on)
    bn_unchanged = (bn_on_before == bn_on_after)

    # ---- per-object paired Δ（ON - OFF）
    kf = lambda r: (r["sample_name"], int(r["gt_component_id"]))
    on_map = {kf(r): r for r in acc_on.object_rows}
    off_map = {kf(r): r for r in acc_off.object_rows}
    common = sorted(set(on_map) & set(off_map))
    per_obj = []
    for k in common:
        a, b = on_map[k], off_map[k]
        per_obj.append({
            "sample_name": k[0], "gt_component_id": k[1], "area": a["area"],
            "recall_on": a["pixel_recall"], "recall_off": b["pixel_recall"],
            "delta_recall": a["pixel_recall"] - b["pixel_recall"],
            "hit25_on": a["hit25"], "hit25_off": b["hit25"],
            "p_mean_on": a.get("model_p_mean_in_gt"), "p_mean_off": b.get("model_p_mean_in_gt"),
            "width_stratum": a.get("width_stratum"), "area_group": _area_group(a["area"]),
        })
    obj_paired = None
    if per_obj:
        img_keys = sorted({p["sample_name"] for p in per_obj})
        per_img_delta = [float(np.mean([p["delta_recall"] for p in per_obj
                                       if p["sample_name"] == nm])) for nm in img_keys]
        obj_paired = {
            "n_common_objects": len(per_obj), "n_images": len(img_keys),
            "mean_delta_recall_object_level": float(np.mean([p["delta_recall"] for p in per_obj])),
            "image_level_bootstrap_CI": bootstrap_ci(per_img_delta, n_boot=args.bootstrap, seed=16),
            "by_area_group": {
                g: {"n_objects": int(sum(1 for p in per_obj if p["area_group"] == g)),
                    "mean_delta_recall": (float(np.mean([p["delta_recall"] for p in per_obj
                                                         if p["area_group"] == g]))
                                          if any(p["area_group"] == g for p in per_obj) else None)}
                for g in ("small", "medium", "large")},
            "small_objects_mean_delta": (
                float(np.mean([p["delta_recall"] for p in per_obj
                               if p["area"] < 256])) if any(p["area"] < 256 for p in per_obj) else None),
        }
        with open(os.path.join(out_dir, "paired_counterfactual.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_obj[0].keys()))
            w.writeheader()
            w.writerows(per_obj)

    pixel_on = sum_on["pixel"]; pixel_off = sum_off["pixel"]
    groups = {}
    for g in AREA_GROUPS:
        a, b = sum_on["groups"][g], sum_off["groups"][g]
        groups[g] = {
            "on_microR": a["pixel_recall_micro"], "off_microR": b["pixel_recall_micro"],
            "delta_microR": (None if (a["pixel_recall_micro"] is None or b["pixel_recall_micro"] is None)
                             else a["pixel_recall_micro"] - b["pixel_recall_micro"]),
            "on_macroR": a["object_macro_pixel_recall"], "off_macroR": b["object_macro_pixel_recall"],
            "on_hit25": a["hit25"], "off_hit25": b["hit25"],
            "delta_hit25": (None if (a["hit25"] is None or b["hit25"] is None) else a["hit25"] - b["hit25"]),
            "n_objects": a["n_objects"],
        }
    legacy = {}
    for lg in ("small", "medium", "large"):
        a, b = sum_on["legacy_groups"][lg], sum_off["legacy_groups"][lg]
        legacy[lg] = {
            "on_microR": a["pixel_recall_micro"], "off_microR": b["pixel_recall_micro"],
            "delta_microR": (None if (a["pixel_recall_micro"] is None or b["pixel_recall_micro"] is None)
                             else a["pixel_recall_micro"] - b["pixel_recall_micro"]),
            "on_hit25": a["hit25"], "off_hit25": b["hit25"],
        }

    # ---- image-level paired bootstrap（每图 F1 差）
    def per_image_f1(per_image):
        return np.array([im["F1"] for im in per_image], dtype=np.float64)

    f1_on = per_image_f1(imgs_on)
    f1_off = per_image_f1(imgs_off)
    n = min(len(f1_on), len(f1_off))
    ci_f1 = bootstrap_ci(f1_on[:n] - f1_off[:n], n_boot=args.bootstrap, seed=16)

    summary = {
        "variant": args.variant, "run": args.run, "dataset": args.dataset,
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_meta["sha256"]},
        "beta_on": beta_on, "beta_off": beta_off,
        "n_images_on": sum_on["n_images"], "n_images_off": sum_off["n_images"],
        "pixel": {
            "on": {k: pixel_on[k] for k in ("recall", "precision", "OA", "F1", "IoU", "Kappa")},
            "off": {k: pixel_off[k] for k in ("recall", "precision", "OA", "F1", "IoU", "Kappa")},
            "delta": {k: pixel_on[k] - pixel_off[k]
                      for k in ("recall", "precision", "OA", "F1", "IoU", "Kappa")},
            "counts_on": {k: pixel_on[k] for k in ("tp", "fp", "fn", "tn")},
            "counts_off": {k: pixel_off[k] for k in ("tp", "fp", "fn", "tn")},
        },
        "groups": groups, "legacy_groups": legacy,
        "object_level": {"on": sum_on["object"], "off": sum_off["object"]},
        "object_paired_delta": obj_paired,
        "image_level_F1_paired_bootstrap": ci_f1,
        "integrity": {
            "diff_state_dict_keys": diff_keys,
            "beta_alias_keys": sorted(beta_alias),
            "only_beta_changed": only_beta_changed,
            "bn_checksum_on_unchanged_by_eval": bn_unchanged,
            "on_forward_reproducible_max_abs": on_reproducible,
        },
        "interpretation_boundary": (
            "Y_on − Y_off 仅衡量当前训练权重对 CAACP 修正项的**局部依赖**；"
            "M1 其余参数是在 β 可学习条件下训练的，因此这不是 A2（从头训练无 CAACP）的受控对照，"
            "也不能据此断言 Stage3 是最早损失源。禁止表述为'关闭 CAACP 可提升模型'。"),
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    write_json(os.path.join(out_dir, "metrics_protocol.json"), METRICS_PROTOCOL)

    checks = {
        "only_beta_changed": summary["integrity"]["only_beta_changed"],
        "beta_off_is_zero": beta_off == 0.0,
        "bn_stats_unchanged": bn_unchanged,
        "on_forward_reproducible": on_reproducible == 0.0,
        "no_training": True,
        "both_passes_full": sum_on["n_images"] == sum_off["n_images"] == len(names) if not args.limit else False,
        "delta_F1": summary["pixel"]["delta"]["F1"],
        "delta_small_microR": legacy["small"]["delta_microR"],
        "delta_small_hit25": None if (legacy["small"]["on_hit25"] is None) else
                             (legacy["small"]["on_hit25"] - legacy["small"]["off_hit25"]),
    }
    ok = (checks["only_beta_changed"] and checks["beta_off_is_zero"] and checks["bn_stats_unchanged"]
          and checks["on_forward_reproducible"] and checks["both_passes_full"])
    write_gate(out_dir, "D4-COUNTERFACT", "PASS" if ok else "FAIL", checks)
    print(f"[D4] gate={'PASS' if ok else 'FAIL'} beta_on={beta_on:.6g} "
          f"dF1={checks['delta_F1']:+.6f} dsmall_microR="
          f"{None if checks['delta_small_microR'] is None else round(checks['delta_small_microR'], 6)}", flush=True)
    print(f"[D4] wrote {out_dir}", flush=True)
    return summary


def _area_group(area):
    from tvim_object_metrics import bin_index, GROUP_ALIAS
    g = bin_index(int(area))
    for lg, members in GROUP_ALIAS.items():
        if g in members:
            return lg
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--run", default="Run1")
    ap.add_argument("--variant", default="M1_FULL", choices=sorted(VARIANT_CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    run_d4(args)


if __name__ == "__main__":
    main()
