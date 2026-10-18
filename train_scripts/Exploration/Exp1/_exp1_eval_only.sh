#!/usr/bin/env bash
# 为「训练已跑满但收尾崩溃」的 run 补出正式 TEST 区块（append 到同一 train_log.txt）
#   bash _exp1_eval_only.sh <VARIANT> <DATASET> <GPU>
set -u
VARIANT="${1:?usage: _exp1_eval_only.sh <VARIANT> <DS> <GPU>}"
DS="${2:?}"
GPU="${3:-1}"
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
CA_ENV=/home/yqwang/miniforge3/envs/casacd
MAMBA_SITE=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate casacd 2>/dev/null || true; fi
export PATH="$CA_ENV/bin:${PATH}"
export PYTHONPATH="${MAMBA_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA_ENV/lib/python3.10/site-packages/torch/lib:$CA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU

REPO=/home/yqwang/projects/CASA-CD
CODE=$REPO/models_v1
DATA=/share_datasets/CD
CKROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1
LOGROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1
CK="$CKROOT/$VARIANT/$DS"
LOG="$LOGROOT/$VARIANT/$DS/train_log.txt"

case "$VARIANT" in
  STG_SUP) EXTRA="--stgeo sup --stgeo_sup_small 16 --stgeo_sup_lambda 0.2" ;;
  STG_V2)  EXTRA="--stgeo inject --stgeo_pool topk_mean --stgeo_xscale 1" ;;
  STG_INJ) EXTRA="--stgeo inject --stgeo_pool max_shape --stgeo_xscale 0" ;;
  *) echo "unknown variant" >&2; exit 2 ;;
esac

[ -s "$LOG" ] || { echo "[ABORT] 缺训练日志 $LOG" >&2; exit 3; }
if grep -aq '=== END TEST RESULTS ===' "$LOG"; then
  echo "[SKIP] 已有完整 TEST 区块：$LOG"; exit 0
fi
echo "[EVAL-ONLY] ${VARIANT}/${DS} -> $LOG"
( cd "$CODE" && python eval_stgeo_tb.py \
    --dataset "$DS" --dataset_root "$DATA/$DS" --test_list "$DATA/$DS/list/test.txt" \
    --ckpt_dir "$CK" \
    --vmamba_pretrained_weight_path "$REPO/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth" \
    $EXTRA --num_workers 4 --gpu_id 0 ) 2>&1 | tee -a "$LOG" | grep -E "^\[(EVAL-ONLY|DEPLOY-PARAMS|REPARAM|FOLD|BEST-F1|ACTUAL|PROTOCOL|STGEO)" 
echo "[EVAL-ONLY] done; 校验最后一个 TEST 区块："
python3 - "$LOG" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
b = re.findall(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", txt, re.S)
ok = bool(b) and ("[ACTUAL-OPT-STEPS] 80000" in b[-1]) and ("[PROTOCOL-VERSION]" in b[-1])
print(f"[GATE] blocks={len(b)} last_ok={ok}")
raise SystemExit(0 if ok else 1)
PY
