#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA/Run3
if pgrep -f '[/]CASA/Run3/pipeline.py' >/dev/null; then
  echo 'CASA Run3 is already running' >&2
  exit 3
fi
mkdir -p "$OUT"
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
setsid nohup python "$REPO/train_scripts/CASA/Run3/pipeline.py" \
  >> "$OUT/pipeline.log" 2>&1 < /dev/null &
PID=$!
sleep 5
kill -0 "$PID"
echo "pipeline pid=$PID"
tail -8 "$OUT/pipeline.log"
