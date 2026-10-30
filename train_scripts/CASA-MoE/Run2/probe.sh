#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/environment.sh"
OUT=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run2
if pgrep -f '[/]CASA-MoE/Run2/health_probe.py|[/]CASA-MoE/Run2/pipeline.py' >/dev/null; then
  echo 'CASA-MoE probes or formal queue are already running' >&2
  exit 3
fi
RUNTIME=$(python -c "import json; print(json.load(open('$OUT/runtime.json'))['directory'])")
for DATASET in LEVIR-CD-256 SYSU-CD-256; do
  for MODULE in none static moe; do
  NAME=$(echo "${DATASET%%-*}" | tr '[:upper:]' '[:lower:]')_${MODULE}
  DIR="$OUT/probes/$NAME"
  mkdir -p "$DIR"
  python "$RUNTIME/train_scripts/CASA-MoE/Run2/health_probe.py" \
    --module "$MODULE" --dataset "$DATASET" --output-dir "$DIR" > "$DIR/probe.log" 2>&1 &
  PID=$!
  echo "$PID" > "$DIR/probe.pid"
  trap 'kill "$PID" 2>/dev/null || true' EXIT
  echo "probe=$NAME pid=$PID physical_gpu=1"
  wait "$PID"
  trap - EXIT
  done
done
echo '[ALL-PROBES-PASS]'
