#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8
if pgrep -f '[/]CASA-Pyramid/Run8/pipeline.py' >/dev/null; then
  echo 'CASA-Pyramid Run8 is already running' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=1
mkdir -p "$OUT"
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
python "$REPO/train_scripts/CASA-Pyramid/Run8/configure_schedule.py"
setsid nohup python "$REPO/train_scripts/CASA-Pyramid/Run8/pipeline.py" \
  >> "$OUT/pipeline.log" 2>&1 < /dev/null &
PID=$!
sleep 5
kill -0 "$PID"
echo "pipeline pid=$PID"
tail -8 "$OUT/pipeline.log"
