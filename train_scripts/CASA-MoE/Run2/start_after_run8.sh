#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/environment.sh"
OUT=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run2
RUNTIME=$(python -c "import json; print(json.load(open('$OUT/runtime.json'))['directory'])")
nohup setsid python "$RUNTIME/train_scripts/CASA-MoE/Run2/handoff.py" >> "$OUT/handoff.log" 2>&1 < /dev/null &
PID=$!
sleep 2
kill -0 "$PID"
cat "$OUT/handoff_status.json"
