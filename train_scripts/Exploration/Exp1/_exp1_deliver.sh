#!/usr/bin/env bash
# 交付批次（远端部分）：四臂正式指标表 + B 臂对象级复算
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
OUT=/tmp/exp1/obj3; mkdir -p "$OUT"

echo "########## 1) 四臂正式指标（最后一个完整 TEST 区块）##########"
python3 - "$REPO" <<'PY'
import re, sys
base = "/home/yqwang/outputs/CASA-CD/CASA-TViM"
arms = [("CTRL_R8", f"{base}/Run8/CTRL_R8"),
        ("STG_INJ(v1)", f"{base}/Exploration/Exp1/STG_INJ"),
        ("STG_V2", f"{base}/Exploration/Exp1/STG_V2"),
        ("STG_SUP(B)", f"{base}/Exploration/Exp1/STG_SUP")]
def last(p):
    try: txt = open(p, encoding="utf-8", errors="replace").read()
    except FileNotFoundError: return None
    b = re.findall(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", txt, re.S)
    if not b: return None
    m = re.search(r"Recall=([\d.]+) \| Precision=([\d.]+) \| OA=([\d.]+) \| F1=([\d.]+) \| IoU=([\d.]+)", b[-1])
    return dict(zip(["R","P","OA","F1","IoU"], map(float, m.groups()))) if m else None
print(f"{'dataset':14s} {'arm':12s} {'F1':>7s} {'R':>7s} {'P':>7s} {'IoU':>7s}")
for ds in ["LEVIR-CD-256", "SYSU-CD-256"]:
    for name, path in arms:
        r = last(f"{path}/{ds}/train_log.txt")
        if r: print(f"{ds:14s} {name:12s} {r['F1']:7.4f} {r['R']:7.4f} {r['P']:7.4f} {r['IoU']:7.4f}")
PY

echo
echo "########## 2) B 臂对象级复算 ##########"
CK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1/STG_SUP
for ds in LEVIR-CD-256 SYSU-CD-256; do
  ck=$(ls -1 "$CK/$ds/"best_F1=*.pth 2>/dev/null | head -1)
  [ -n "$ck" ] || { echo "[WARN] 缺 $ds ckpt"; continue; }
  echo "--- STG_SUP / $ds ($(basename "$ck")) ---"
  ( cd "$REPO/models_v1" && python eval_object_metrics.py --arch stgeo --stgeo sup \
      --pkg-root "$REPO/models_v1" --ckpt "$ck" --dataset "$ds" \
      --dataset-root "$DATA/$ds" --test-list "$DATA/$ds/list/test.txt" \
      --analyse-dir "$REPO/analyse" --batch-size 16 --num-workers 4 \
      --out "$OUT/sup_$ds.json" ) 2>&1 | grep -vE "xFormers|warnings.warn" | tail -8
done

echo
echo "########## 3) 四臂对象级汇总 ##########"
python3 - "$OUT" /tmp/exp1/obj2 /tmp/exp1/obj <<'PY'
import json, os, sys
dirs = sys.argv[1:]
SMALL = ("tiny_1_15", "tiny_16_63", "small_64_255")
def load(tag, ds):
    for d in dirs:
        p = os.path.join(d, f"{tag}_{ds}.json")
        if os.path.isfile(p):
            return json.load(open(p, encoding="utf-8"))
    return None
def um(x):
    return sum(x["object"]["unmatched_pred_cc_count_by_area"][k] for k in SMALL)
for ds in ["LEVIR-CD-256", "SYSU-CD-256"]:
    rows = [("CTRL", load("run8", ds)), ("v1", load("stgeo", ds)),
            ("v2", load("v2", ds)), ("B", load("sup", ds))]
    if any(r[1] is None for r in rows):
        print(f"{ds}: 缺 {[n for n, r in rows if r is None]}"); continue
    print(f"\n===== {ds} =====")
    print(f"{'metric':28s} " + " ".join(f"{n:>9s}" for n, _ in rows))
    def line(name, f):
        print(f"{name:28s} " + " ".join(f"{f(r):9.4f}" for _, r in rows))
    line("pixel F1", lambda x: x["pixel"]["F1"])
    line("pixel Recall", lambda x: x["pixel"]["recall"])
    line("pixel Precision", lambda x: x["pixel"]["precision"])
    line("small pooled R", lambda x: x["legacy_groups"]["small"]["pixel_recall_micro"])
    line("small Hit@25", lambda x: x["legacy_groups"]["small"]["hit25"])
    line("large pooled R", lambda x: x["legacy_groups"]["large"]["pixel_recall_micro"])
    line("ObjP (loose)", lambda x: x["object"]["ObjPrecision_loose"])
    line("ObjR (loose)", lambda x: x["object"]["ObjRecall_loose"])
    print(f"{'unmatched<256px':28s} " + " ".join(f"{um(r):9d}" for _, r in rows))
PY
