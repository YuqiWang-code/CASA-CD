#!/usr/bin/env bash
# v2 正式 TEST 区块 + 对象级指标（v2 单独跑；CTRL/v1 的 JSON 若已存在则复用）
set -u
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
CA_ENV=/home/yqwang/miniforge3/envs/casacd
MAMBA_SITE=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate casacd 2>/dev/null || true; fi
export PATH="$CA_ENV/bin:${PATH}"
export PYTHONPATH="${MAMBA_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA_ENV/lib/python3.10/site-packages/torch/lib:$CA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU="${1:-1}"; export CUDA_VISIBLE_DEVICES=$GPU
REPO=/home/yqwang/projects/CASA-CD
DATA=/share_datasets/CD
OUT=/tmp/exp1/obj2; mkdir -p "$OUT"
V2=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1/STG_V2
LOG=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/STG_V2

echo "################ v2 正式 TEST 区块 ################"
for d in LEVIR-CD-256 SYSU-CD-256; do
  echo "===== $d ====="
  awk '/=== TEST RESULTS ===/{f=1} f{print} /=== END TEST RESULTS ===/{f=0}' "$LOG/$d/train_log.txt" \
    | grep -E "^(Recall=|\[BEST-F1\]|\[STGEO|\[REPARAM|\[FOLD-KNIFE|\[DEPLOY-PARAMS\]|\[ACTUAL-OPT-STEPS\]|\[PROTOCOL-VERSION\])"
  echo
done

echo "################ 对象级指标：v2 ################"
for ds in LEVIR-CD-256 SYSU-CD-256; do
  ck=$(ls -1 "$V2/$ds/"best_F1=*.pth 2>/dev/null | head -1)
  [ -n "$ck" ] || { echo "[WARN] 缺 v2 $ds ckpt"; continue; }
  echo "--- STG_V2 / $ds ($(basename "$ck")) ---"
  ( cd "$REPO/models_v1" && python eval_object_metrics.py --arch stgeo --pkg-root "$REPO/models_v1" \
      --ckpt "$ck" --dataset "$ds" --dataset-root "$DATA/$ds" \
      --stgeo-pool topk_mean --stgeo-xscale 1 \
      --test-list "$DATA/$ds/list/test.txt" --analyse-dir "$REPO/analyse" \
      --batch-size 16 --num-workers 4 --out "$OUT/v2_$ds.json" ) \
      2>&1 | grep -vE "xFormers|warnings.warn" | tail -8
done

echo
echo "################ 三臂汇总（CTRL / v1 / v2）################"
python3 - "$OUT" "/tmp/exp1/obj" <<'PY'
import json, os, sys
new, old = sys.argv[1], sys.argv[2]
def load(tag, ds):
    for p in (os.path.join(new, f"{tag}_{ds}.json"), os.path.join(old, f"{tag}_{ds}.json")):
        if os.path.isfile(p):
            return json.load(open(p, encoding="utf-8"))
    return None
SMALL = ("tiny_1_15", "tiny_16_63", "small_64_255")
for ds in ["LEVIR-CD-256", "SYSU-CD-256"]:
    c = load("run8", ds) or load("ctr", ds)
    v1 = load("stgeo", ds)
    v2 = load("v2", ds)
    if not (c and v2):
        print(f"{ds}: 数据不全 c={bool(c)} v1={bool(v1)} v2={bool(v2)}"); continue
    def um(x):
        return sum(x["object"]["unmatched_pred_cc_count_by_area"][k] for k in SMALL)
    print(f"\n===== {ds} =====")
    print(f"{'metric':30s} {'CTRL':>9s} {'v1':>9s} {'v2':>9s}   {'v2-CTRL':>8s} {'v2-v1':>8s}")
    def row(name, f):
        a = f(c); b = f(v1) if v1 else float('nan'); d = f(v2)
        print(f"{name:30s} {a:9.4f} {b:9.4f} {d:9.4f}   {d-a:+8.4f} {d-b:+8.4f}")
    row("pixel F1", lambda x: x["pixel"]["F1"])
    row("pixel Recall", lambda x: x["pixel"]["recall"])
    row("pixel Precision", lambda x: x["pixel"]["precision"])
    row("small pooled R", lambda x: x["legacy_groups"]["small"]["pixel_recall_micro"])
    row("small Hit@25", lambda x: x["legacy_groups"]["small"]["hit25"])
    row("medium pooled R", lambda x: x["legacy_groups"]["medium"]["pixel_recall_micro"])
    row("large pooled R", lambda x: x["legacy_groups"]["large"]["pixel_recall_micro"])
    row("ObjP (loose)", lambda x: x["object"]["ObjPrecision_loose"])
    row("ObjR (loose)", lambda x: x["object"]["ObjRecall_loose"])
    for nm, f in (("unmatched <256px (count)", um),):
        a, b, d = f(c), (f(v1) if v1 else -1), f(v2)
        print(f"{nm:30s} {a:9d} {b:9d} {d:9d}   {d-a:+8d} {d-b:+8d}")
PY
