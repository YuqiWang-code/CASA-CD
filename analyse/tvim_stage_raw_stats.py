"""D2b：从 stage_raw.csv 计算 per-image margin 的**配对** bootstrap CI（图像为抽样单位）。

补上 Diag1 的 [M] 缺口：D2 只给了 per-image AP 的 CI 与 margin 的 per-image CI，未给
"相邻节点 margin 差的配对 CI"。本脚本按 `sample_name` 对齐同一批图像，计算
`margin_to − margin_from`（节点级 small-object margin 的逐图值）的 image-level bootstrap CI，
并给出同分辨率相邻边界的判据结论（§10.2：相对下降 ≥10% 且 CI 同号）。

用法：
    python analyse/tvim_stage_raw_stats.py --stage-raw "$DIAG/D2/M1_FULL/stage_raw.csv" \
        --out "$DIAG/D2/M1_FULL/margin_paired_ci.json"
"""
import os
import sys
import csv
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tvim_diag_common import bootstrap_ci, paired_bootstrap_ci, write_json  # noqa: E402
from tvim_diag_report import NODE_ORDER, NODE_RES  # noqa: E402


def load(path):
    """返回 {node: {name: (score_in_gt, score_bg)}} 与逐图 AP/margin 列。"""
    per_node = {}
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        nodes = sorted({c[:-len("_score_in_gt")] for c in cols if c.endswith("_score_in_gt")})
        rows = list(rdr)
    for n in nodes:
        per_node[n] = {}
        for r in rows:
            v_in = r.get(f"{n}_score_in_gt")
            v_bg = r.get(f"{n}_score_bg")
            if v_in in (None, "", "None") or v_bg in (None, "", "None"):
                continue
            per_node[n][r["sample_name"]] = (float(v_in), float(v_bg))
    return per_node, nodes, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    per_node, nodes, n_rows = load(args.stage_raw)
    ordered = [n for n in NODE_ORDER if n in per_node]
    out = {"n_images": n_rows, "nodes": ordered, "method": {
        "paired_unit": "image (sample_name 对齐)",
        "bootstrap": args.bootstrap, "seed": 16,
        "statistic": "margin = mean(score in GT) − mean(score in background)，逐图；差值 = to − from",
    }, "per_node": {}, "adjacent": []}

    for n in ordered:
        d = per_node[n]
        in_gt = np.array([v[0] for v in d.values()])
        bg = np.array([v[1] for v in d.values()])
        out["per_node"][n] = {
            "resource": NODE_RES.get(n),
            "n_images": len(d),
            "score_in_gt_mean": float(in_gt.mean()),
            "score_bg_mean": float(bg.mean()),
            "margin_image_mean": float((in_gt - bg).mean()),
            "margin_image_bootstrap_CI": bootstrap_ci(in_gt - bg, n_boot=args.bootstrap, seed=16),
        }
    for a, b in zip(ordered[:-1], ordered[1:]):
        da, db = per_node[a], per_node[b]
        common = sorted(set(da) & set(db))
        if not common:
            continue
        va = np.array([da[k][0] - da[k][1] for k in common])
        vb = np.array([db[k][0] - db[k][1] for k in common])
        ci = paired_bootstrap_ci(vb, va, n_boot=args.bootstrap, seed=16)
        rel = (ci["mean"] / va.mean()) if va.mean() not in (0.0,) else None
        out["adjacent"].append({
            "from": a, "to": b,
            "res_from": NODE_RES.get(a), "res_to": NODE_RES.get(b),
            "same_resolution": NODE_RES.get(a) == NODE_RES.get(b),
            "n_paired_images": len(common),
            "margin_from_mean": float(va.mean()), "margin_to_mean": float(vb.mean()),
            "delta_margin_mean": ci["mean"], "delta_margin_CI": [ci["lo"], ci["hi"]],
            "relative_delta": rel,
            "CI_excludes_zero": bool(ci["lo"] > 0 or ci["hi"] < 0),
            "direction": int(np.sign(ci["mean"])) if ci["mean"] != 0 else 0,
            "criterion_relative_drop_10pct_same_sign": bool(
                rel is not None and rel <= -0.10 and ci["hi"] < 0),
        })
    write_json(args.out, out)
    print(f"wrote {args.out}")
    print(f"{'from -> to':46s} {'same_res':>8s} {'margin_from':>11s} {'margin_to':>10s} "
          f"{'delta':>10s} {'CI':>22s} {'rel':>8s} {'判据':>5s}")
    for r in out["adjacent"]:
        rel = r["relative_delta"]
        print(f"{r['from'] + ' -> ' + r['to']:46s} {str(r['same_resolution']):>8s} "
              f"{r['margin_from_mean']:11.5f} {r['margin_to_mean']:10.5f} "
              f"{r['delta_margin_mean']:10.5f} "
              f"[{r['delta_margin_CI'][0]:.5f},{r['delta_margin_CI'][1]:.5f}]".rjust(22) +
              f" {('—' if rel is None else f'{rel:8.2%}')} "
              f"{'是' if r['criterion_relative_drop_10pct_same_sign'] else '否':>5s}")


if __name__ == "__main__":
    main()
