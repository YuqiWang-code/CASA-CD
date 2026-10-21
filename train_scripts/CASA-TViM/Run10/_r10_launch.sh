#!/usr/bin/env bash
set -euo pipefail
RD=/home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run10
STATE=/tmp/run10
mkdir -p "$STATE"
if pgrep -f '[/]Run10/_r10_pipeline.sh|[/]Run10/_r10_run_one.sh' >/dev/null; then
  echo 'Run10 already running; refusing duplicate launch' >&2
  exit 3
fi
if pgrep -f '[p]ython train.py' >/dev/null; then
  echo 'another train.py is running; refusing launch' >&2
  exit 3
fi
setsid nohup bash "$RD/_r10_pipeline.sh" > "$STATE/launch.stdout" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$STATE/pipeline.pid"
sleep 8
kill -0 "$PID" 2>/dev/null || { tail -40 "$STATE/launch.stdout"; exit 1; }
echo "Run10 launched pid=$PID"
tail -8 "$STATE/pipeline.log" 2>/dev/null || true
