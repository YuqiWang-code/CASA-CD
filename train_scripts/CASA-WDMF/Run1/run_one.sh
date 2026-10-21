#!/usr/bin/env bash
set -euo pipefail
VARIANT=${1:?variant required}
DATASET=${2:?dataset required}
REPO=/home/yqwang/projects/CASA-CD
case "$VARIANT" in
  M0_CTRL) MODULE=none ;;
  M1_HYPER_QKV) MODULE=hyperbolic ;;
  M2_CA_QKV) MODULE=qkv ;;
  *) exit 2 ;;
esac
LOG=/home/yqwang/outputs/CASA-CD/CASA-WDMF/Run1/$VARIANT/$DATASET
CKPT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-WDMF/Run1/$VARIANT/$DATASET
mkdir -p "$LOG" "$CKPT"
if [ -s "$LOG/train_log.txt" ] || [ -e "$CKPT/last.pth" ]; then
  echo "existing artifacts for $VARIANT/$DATASET" >&2
  exit 3
fi
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH="/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$REPO/models"
exec python train_wdmf.py \
  --dataset-root "/share_datasets/CD/$DATASET" \
  --output-dir "$LOG" --checkpoint-dir "$CKPT" \
  --pretrained "$REPO/pretrained_weight/mobilenet_v2-b0353104.pth" \
  --module "$MODULE" --batch-size 32 --max-steps 40000 --num-workers 1 \
  > "$LOG/train_log.txt" 2>&1
