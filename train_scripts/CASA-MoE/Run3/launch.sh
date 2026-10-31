#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/environment.sh"
OUT=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run3
if pgrep -f '[/]CASA-MoE/Run3/pipeline.py' >/dev/null; then
  echo 'CASA-MoE Run3 supervisor is already running' >&2
  exit 3
fi
RUNTIME=$(python -c "import json; print(json.load(open('$OUT/runtime.json'))['directory'])")
SCRIPTS="$RUNTIME/train_scripts/CASA-MoE/Run3"
python "$SCRIPTS/configure_schedule.py"
setsid nohup python "$SCRIPTS/pipeline.py" >> "$OUT/pipeline.log" 2>&1 < /dev/null &
PID=$!
sleep 5
kill -0 "$PID"
echo "CASA-MoE Run3 pipeline pid=$PID gpu=0"
tail -8 "$OUT/pipeline.log"
