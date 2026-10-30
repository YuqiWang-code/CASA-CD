#!/usr/bin/env bash
set -euo pipefail
VARIANT=${1:?variant required}
DATASET=${2:?dataset required}
GPU=${3:?physical GPU required}
case "$GPU" in 1) ;; *) exit 2 ;; esac
REPO=/home/yqwang/projects/CASA-CD
case "$VARIANT" in
  H_CLASS) RETRIEVAL=0.1 ;;
  H_NONE) RETRIEVAL=0.0 ;;
  *) exit 2 ;;
esac
LOG=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8/$VARIANT/$DATASET
CKPT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run8/$VARIANT/$DATASET
mkdir -p "$LOG" "$CKPT"
EXTRA=()
if grep -q '=== END TEST RESULTS ===' "$LOG/train_log.txt" 2>/dev/null; then
  exit 0
elif [ -e "$CKPT/last.pth" ]; then
  EXTRA=(--resume)
elif [ -s "$LOG/train_log.txt" ]; then
  echo "log exists without a resumable checkpoint for $VARIANT/$DATASET" >&2
  exit 3
fi
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH="/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO/models"
exec python train_CASA.py \
  --dataset-root "/share_datasets/CD/$DATASET" \
  --output-dir "$LOG" --checkpoint-dir "$CKPT" \
  --pretrained "$REPO/pretrained_weight/mobilenet_v2-b0353104.pth" \
  --module hyperbolic --guide-weight 0.1 --retrieval-weight "$RETRIEVAL" --refinement-beta 0 --seed 2333 --context-seed 2333 --encoder-lr-mult 1.0 --batch-size 32 --max-steps 40000 --num-workers 1 "${EXTRA[@]}" \
  >> "$LOG/train_log.txt" 2>&1
