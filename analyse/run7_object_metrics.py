"""Run7 收口：对象级只读复算（与 Run5/Diag1 冻结口径逐字一致）。

对每个 (variant, dataset) 的 **best checkpoint** 做一次只读 test 推理，用
`tvim_object_metrics.ObjectMetricAccumulator` 一次算出：

  * GT 面积分层 pooled / macro Recall、Hit@1/25/50（legacy 分组 small/medium/large）；
  * 未匹配预测连通域数（`unmatched_pred_loose`）及**按面积组**直方图
    ⇒ 取 small 三组之和 = 与 Diag1/Run5 冻结值 "2430 unmatched <256px" 同口径的量；
  * ObjPrecision / ObjRecall / ObjF1（IoU≥0.10）与 n_gt / n_pred / n_matched；
  * 像素级六指标（与 `metric_tool` 同口径）。

**只读**：不训练、不改任何历史产物；`best_F1=*.pth` 只读加载 + `switch_to_deploy()`（推理图）。

    python analyse/run7_object_metrics.py --run Run7 --variant M1_R7CTRL --dataset SYSU-CD-256 \
      --out /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/_OBJECT_METRICS

也支持 `--compare` 读取同目录下已生成的若干 JSON 并打印对照表。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvim_diag_common import (  # noqa: E402
    build_model, env_info, make_loader, pick_best_ckpt, set_eval_numerics,
    sha256_file, write_json,
)
from tvim_object_metrics import (  # noqa: E402
    AREA_GROUPS, GROUP_ALIAS, ObjectMetricAccumulator,
)

# 与 Run6/Diag1 冻结参考（只用于对照报告，不参与任何判据计算）
FROZEN_RUN1_SYSU = {
    "small_pooled_recall": 0.1974,
    "small_hit25": 0.1896,
    "small_hit1": None,
    "n_gt_small_objects": 364,
    "unmatched_small_pred": 2430,
    "obj_precision_loose": 0.5320,
    "obj_recall_loose": 0.7674,
}
SMALL_GROUPS = list(GROUP_ALIAS["small"])

# Run7 三个变体 → (hf_edge, hf_edge_form)。**必须显式构造模型**：
# `tvim_diag_common.build_model()` 依赖 `VARIANT_CFG`，其中没有 Run7 变体；
# 即便补进去也不会传 `hf_edge*`，E9/E10 的 state_dict（含 hf_edge_head.*）会 strict 失败。
# 该缺陷在收口前用真实 ckpt 实测时发现（否则会在最终裁决那一步才炸）。
RUN7_VARIANTS = {"M1_R7CTRL": (0, "none"), "E9_HFAE_A": (1, "aux_only"),
                 "E10_HFAE_B": (1, "aux_inject")}


def build_run7_model(variant, device, ckpt_path, pretrain, strict=True):
    """按 Run7 变体显式构造 CASATViMSTRNet 并严格加载 train 图 state_dict。"""
    import torch
    from model.casa_tvim_str_net import CASATViMSTRNet
    if variant not in RUN7_VARIANTS:
        raise KeyError(f"unknown Run7 variant {variant}; expect {sorted(RUN7_VARIANTS)}")
    hf_edge, form = RUN7_VARIANTS[variant]
    m = CASATViMSTRNet(pretrain, caacp=True, rep_mode="full", str_dim=96,
                       caacp_score_mode="rank", frh=False,
                       caacp_residual_mode="current", fs_tar=False, fine_tap=False,
                       hf_edge=bool(hf_edge), hf_edge_form=form).float().to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    info = {"variant": variant, "hf_edge": hf_edge, "hf_edge_form": form,
            "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    if strict and (len(missing) or len(unexpected)):
        raise RuntimeError(f"[R7-OBJ] state_dict mismatch: missing={list(missing)[:5]} "
                           f"unexpected={list(unexpected)[:5]}")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, info


def _small_unmatched(d):
    h = d.get("unmatched_pred_cc_count_by_area") or {}
    return int(sum(h.get(g, 0) for g in SMALL_GROUPS))


def run_one(args):
    import torch
    # TinyViM 的 selective_scan kernel 只有 CUDA 实现；用 CPU 跑会在 forward 深处抛
    # 「Expected u.is_cuda() to be true」这种难以定位的错。这里提前给出明确指引。
    if not str(args.device).startswith("cuda"):
        raise SystemExit("[R7-OBJ] 本工具需要 CUDA（TinyViM 的 selective_scan kernel 仅有 CUDA 实现）；"
                         "请传 --device cuda:0")
    set_eval_numerics()
    device = args.device
    ck, meta = pick_best_ckpt(args.run, args.variant, args.dataset, ckpt_root=args.ckpt_root)
    sha = sha256_file(ck)
    print(f"[R7-OBJ] {args.variant}/{args.dataset} ckpt={os.path.basename(ck)} sha={sha[:16]}",
          flush=True)
    variant_cfg = args.variant
    if variant_cfg in RUN7_VARIANTS:
        from tvim_diag_common import PRETRAIN_TINYVIM
        model, info = build_run7_model(variant_cfg, device, ck, PRETRAIN_TINYVIM,
                                       strict=True)
        print(f"[R7-OBJ] built Run7 variant: hf_edge={info['hf_edge']} "
              f"form={info['hf_edge_form']} (missing={len(info['missing_keys'])} "
              f"unexpected={len(info['unexpected_keys'])})", flush=True)
    else:
        model, info = build_model(variant_cfg, device=device, ckpt_path=ck, strict=True)
        assert not info["missing_keys"] and not info["unexpected_keys"], \
            f"state_dict mismatch: {info}"
    model.eval()
    model.switch_to_deploy()          # 正式结果口径 = deploy 图（与 eval.py 一致）

    loader, list_path = make_loader(args.dataset, "test", batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    acc = ObjectMetricAccumulator(connectivity=4, bands=(2, 4))
    n_seen, nonfinite = 0, 0
    with torch.no_grad():
        for img, label in loader:
            pre = img[:, 0:3].to(device).float()
            post = img[:, 3:6].to(device).float()
            out = model(pre, post)
            if not torch.isfinite(out).all():
                nonfinite += int((~torch.isfinite(out)).sum().item())
            prob = out[:, 0].float().cpu().numpy()
            gts = label.numpy()[:, 0] > 0.5
            for i in range(prob.shape[0]):
                acc.add(prob[i] > 0.5, gts[i], name=None)
                n_seen += 1
    s = acc.summary(label=f"{args.variant}/{args.dataset}")

    lg = s["legacy_groups"]
    obj = s["object"]
    small_unmatched = _small_unmatched(obj)
    payload = {
        "run": args.run, "variant": args.variant, "dataset": args.dataset,
        "ckpt_path": ck, "ckpt_sha256": sha, "ckpt_candidates": meta.get("candidates"),
        "n_images_seen": n_seen, "nonfinite_values": nonfinite,
        "connectivity": s["connectivity"],
        "pixel": s["pixel"],
        "groups": s["groups"],
        "legacy_groups": lg,
        "object": {
            "n_gt": obj["n_gt"], "n_pred": obj["n_pred"],
            "n_matched_loose": obj["n_matched_loose"],
            "unmatched_pred_loose": obj["unmatched_pred_loose"],
            "unmatched_small_pred_area_1_255": small_unmatched,
            "ObjPrecision_loose": obj["ObjPrecision_loose"],
            "ObjRecall_loose": obj["ObjRecall_loose"],
            "ObjF1_loose": obj["ObjF1_loose"],
            "unmatched_pred_cc_count_by_area": obj["unmatched_pred_cc_count_by_area"],
            "pred_cc_count_by_area": obj["pred_cc_count_by_area"],
            "gt_cc_count_by_area": obj["gt_cc_count_by_area"],
        },
        "headline": {
            "small_pooled_recall": lg["small"]["pixel_recall_micro"],
            "small_object_macro_recall": lg["small"]["object_macro_pixel_recall"],
            "small_hit1": lg["small"]["hit1"],
            "small_hit25": lg["small"]["hit25"],
            "small_hit50": lg["small"]["hit50"],
            "n_gt_small_objects": lg["small"]["n_objects"],
            "medium_pooled_recall": lg["medium"]["pixel_recall_micro"],
            "large_pooled_recall": lg["large"]["pixel_recall_micro"],
            "unmatched_small_pred": small_unmatched,
            "ObjPrecision_loose": obj["ObjPrecision_loose"],
            "ObjRecall_loose": obj["ObjRecall_loose"],
            "F1": s["pixel"]["F1"],
        },
        "frozen_reference_run1_sysu": (FROZEN_RUN1_SYSU
                                       if args.dataset == "SYSU-CD-256" else None),
        "env": env_info(),
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"object_metrics_{args.variant}_{args.dataset}.json")
    write_json(out_path, payload)
    h = payload["headline"]
    print(f"[R7-OBJ] small pooledR={h['small_pooled_recall']:.6f} Hit@25={h['small_hit25']:.6f} "
          f"n_small={h['n_gt_small_objects']} unmatched_small={h['unmatched_small_pred']} "
          f"ObjP={h['ObjPrecision_loose']:.4f} ObjR={h['ObjRecall_loose']:.4f} F1={h['F1']:.4f}",
          flush=True)
    print(f"[R7-OBJ] wrote {out_path}", flush=True)
    return 0


def compare(args):
    rows = []
    for p in sorted(glob.glob(os.path.join(args.out, "object_metrics_*.json"))):
        rows.append(json.load(open(p, encoding="utf-8")))
    if not rows:
        print(f"[R7-OBJ] no object_metrics_*.json in {args.out}")
        return 1
    datasets = sorted({r["dataset"] for r in rows})
    variants = sorted({r["variant"] for r in rows})
    keys = ["small_pooled_recall", "small_hit25", "n_gt_small_objects", "unmatched_small_pred",
            "ObjPrecision_loose", "ObjRecall_loose", "medium_pooled_recall",
            "large_pooled_recall", "F1"]
    for ds in datasets:
        print(f"\n=== {ds} ===")
        print(f"{'metric':28s}" + "".join(f"{v:>18s}" for v in variants)
              + ("   E9-CTRL   E10-CTRL" if len(variants) > 1 else ""))
        base = {v: next((r["headline"] for r in rows
                         if r["dataset"] == ds and r["variant"] == v), None) for v in variants}
        for k in keys:
            line = f"{k:28s}"
            for v in variants:
                val = (base[v] or {}).get(k)
                line += f"{(f'{val:.6f}' if isinstance(val, float) else str(val)):>18s}"
            if len(variants) > 1:
                d = []
                for v in ("E9_HFAE_A", "E10_HFAE_B"):
                    if v in base and base[v] and base.get("M1_R7CTRL"):
                        a, b = base[v].get(k), base["M1_R7CTRL"].get(k)
                        d.append(f"{a-b:+.4f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "n/a")
                    else:
                        d.append("n/a")
                line += "   " + "   ".join(f"{x:>8s}" for x in d)
            print(line)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="Run7")
    ap.add_argument("--variant", default=None,
                    help="Run7 变体；留空则配合 --all-variants 逐个算")
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--variants", default="M1_R7CTRL,E9_HFAE_A,E10_HFAE_B")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--datasets", default="CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256")
    ap.add_argument("--ckpt-root",
                    default="/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        return compare(args)
    rc = 0
    variants = ([v.strip() for v in args.variants.split(",")] if args.all_variants
                else [args.variant])
    datasets = [d.strip() for d in args.datasets.split(",")] if args.all_variants else [args.dataset]
    for v in variants:
        for d in datasets:
            args.variant, args.dataset = v, d
            try:
                rc |= run_one(args)
            except FileNotFoundError as e:
                print(f"[R7-OBJ][SKIP] {v}/{d}: {e}")
            except Exception as e:  # noqa: BLE001
                print(f"[R7-OBJ][FAIL] {v}/{d}: {type(e).__name__}: {e}")
                rc |= 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
