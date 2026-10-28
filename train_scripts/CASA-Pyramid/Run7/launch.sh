#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run7
if pgrep -f '[/]CASA-Pyramid/Run7/pipeline.py' >/dev/null; then
  echo 'CASA-Pyramid Run7 is already running' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
python "$REPO/train_scripts/CASA-Pyramid/Run7/configure_schedule.py"
setsid nohup python "$REPO/train_scripts/CASA-Pyramid/Run7/pipeline.py" \
  >> "$OUT/pipeline.log" 2>&1 < /dev/null &
PID=$!
sleep 5
kill -0 "$PID"
echo "pipeline pid=$PID"
tail -8 "$OUT/pipeline.log"
