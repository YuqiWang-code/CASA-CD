#!/usr/bin/env bash
set -euo pipefail
OUT=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run2
for KIND in handoff pipeline; do
  if [[ -f "$OUT/$KIND.pid" ]]; then
    PID=$(cat "$OUT/$KIND.pid")
    if [[ -r "/proc/$PID/cmdline" ]]; then
      COMMAND=$(tr '\0' ' ' < "/proc/$PID/cmdline")
      if [[ "$COMMAND" == *"/CASA-MoE/Run2/$KIND.py"* ]]; then kill -TERM "$PID"; fi
    fi
  fi
done
