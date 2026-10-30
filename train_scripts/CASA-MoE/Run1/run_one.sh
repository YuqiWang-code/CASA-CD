#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/environment.sh"
VARIANT=${1:?variant required}
DATASET=${2:?dataset required}
GPU=${3:?physical GPU required}
[[ "$GPU" == 1 ]] || exit 2
MODULE=$(python -c "import sys; sys.path.insert(0, '$HERE'); from config import VARIANTS; print(VARIANTS['$VARIANT'])")
OUT=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run1
RUNTIME=$(python -c "import json; print(json.load(open('$OUT/runtime.json'))['directory'])")
LOG="$OUT/$VARIANT/$DATASET"
CKPT="/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MoE/Run1/$VARIANT/$DATASET"
mkdir -p "$LOG" "$CKPT"
if grep -q '=== END TEST RESULTS ===' "$LOG/train_log.txt" 2>/dev/null; then exit 0; fi
if [[ -s "$LOG/train_log.txt" || -e "$CKPT/last.pth" ]]; then
  echo "Existing unfinished run: $VARIANT/$DATASET; explicit recovery is required" >&2
  exit 3
fi
cd "$RUNTIME/models_v1"
exec python train_CASA.py --dataset-root "/share_datasets/CD/$DATASET" \
  --output-dir "$LOG" --checkpoint-dir "$CKPT" \
  --pretrained /home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth \
  --module "$MODULE" --encoder-lr-mult 1.0 --batch-size 32 --max-steps 40000 --num-workers 1 \
  >> "$LOG/train_log.txt" 2>&1
