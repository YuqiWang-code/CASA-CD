#!/usr/bin/env bash
# 对象级小目标指标：CTRL_R8 vs STG_INJ（LEVIR / SYSU），同一指标实现、同一 GPU
set -u
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
CA_ENV=/home/yqwang/miniforge3/envs/casacd
MAMBA_SITE=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate casacd 2>/dev/null || true; fi
export PATH="$CA_ENV/bin:${PATH}"
export PYTHONPATH="${MAMBA_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA_ENV/lib/python3.10/site-packages/torch/lib:$CA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU="${1:-1}"
export CUDA_VISIBLE_DEVICES=$GPU

REPO=/home/yqwang/projects/CASA-CD
DATA=/share_datasets/CD
OUT=/tmp/exp1/obj
mkdir -p "$OUT"
R8CK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8/CTRL_R8
ECK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1/STG_INJ

run() {  # run <arch> <pkgroot> <ckpt> <dataset>
  local arch="$1" pkg="$2" ck="$3" ds="$4"
  local tag="${arch}_${ds}"
  echo "########## $tag ##########"
  ( cd "$REPO/models_v1" && python eval_object_metrics.py --arch "$arch" --pkg-root "$pkg" \
      --ckpt "$ck" --dataset "$ds" --dataset-root "$DATA/$ds" \
      --test-list "$DATA/$ds/list/test.txt" --analyse-dir "$REPO/analyse" \
      --batch-size 16 --num-workers 4 --out "$OUT/$tag.json" ) \
      2>&1 | grep -vE "xFormers|warnings.warn|^\s*$" | tail -12
  echo
}

for ds in LEVIR-CD-256 SYSU-CD-256; do
  ck8=$(ls -1 "$R8CK/$ds/"best_F1=*.pth 2>/dev/null | head -1)
  cke=$(ls -1 "$ECK/$ds/"best_F1=*.pth 2>/dev/null | head -1)
  [ -n "$ck8" ] && run run8 "$REPO/models" "$ck8" "$ds" || echo "[WARN] 缺 CTRL_R8/$ds ckpt"
  [ -n "$cke" ] && run stgeo "$REPO/models_v1" "$cke" "$ds" || echo "[WARN] 缺 STG_INJ/$ds ckpt"
done

echo "########## 汇总对比 ##########"
python3 - "$OUT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
for ds in ["LEVIR-CD-256", "SYSU-CD-256"]:
    rows = {}
    for tag in ("run8", "stgeo"):
        p = os.path.join(root, f"{tag}_{ds}.json")
        if os.path.isfile(p):
            rows[tag] = json.load(open(p, encoding="utf-8"))
    if len(rows) < 2:
        print(f"{ds}: 缺数据 {list(rows)}")
        continue
    c, e = rows["run8"], rows["stgeo"]
    print(f"\n=== {ds} ===")
    print(f"{'metric':34s} {'CTRL_R8':>10s} {'STG_INJ':>10s} {'delta':>9s}")
    def line(name, a, b, fmt=".4f", scale=1.0):
        d = (b - a) * scale
        print(f"{name:34s} {a:10.4f} {b:10.4f} {d:+9.4f}")
    line("pixel F1", c["pixel"]["F1"], e["pixel"]["F1"])
    line("pixel Recall", c["pixel"]["recall"], e["pixel"]["recall"])
    line("pixel Precision", c["pixel"]["precision"], e["pixel"]["precision"])
    for g in ("small", "medium", "large"):
        line(f"{g} pooled Recall", c["legacy_groups"][g]["pixel_recall_micro"],
             e["legacy_groups"][g]["pixel_recall_micro"])
    for g in ("small", "medium", "large"):
        line(f"{g} Hit@25", c["legacy_groups"][g]["hit25"],
             e["legacy_groups"][g]["hit25"])
    line("ObjPrecision(loose)", c["object"]["ObjPrecision_loose"],
         e["object"]["ObjPrecision_loose"])
    line("ObjRecall(loose)", c["object"]["ObjRecall_loose"],
         e["object"]["ObjRecall_loose"])
    cu = sum(c["object"]["unmatched_pred_cc_count_by_area"][k] for k in ("tiny_1_15","tiny_16_63","small_64_255"))
    eu = sum(e["object"]["unmatched_pred_cc_count_by_area"][k] for k in ("tiny_1_15","tiny_16_63","small_64_255"))
    if cu is not None and eu is not None:
        print(f"{'unmatched <256px pred (count)':34s} {cu:10d} {eu:10d} {eu-cu:+9d}")
PY
