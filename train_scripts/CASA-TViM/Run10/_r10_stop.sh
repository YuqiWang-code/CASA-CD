#!/usr/bin/env bash
# Stop only Run10's process group; preserves outputs and checkpoints.
set -euo pipefail
STATE=/tmp/run10
PID=$(cat "$STATE/pipeline.pid" 2>/dev/null || true)
[ -n "$PID" ] || { echo 'no Run10 pipeline pid'; exit 0; }
CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || true)
case "$CMD" in *'_r10_pipeline.sh'*) ;; *) echo "pid $PID is not Run10 pipeline; refusing stop" >&2; exit 3 ;; esac
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
[ -n "$PGID" ] || exit 3
kill -TERM -- "-$PGID"
echo "TERM sent to Run10 process group $PGID"
