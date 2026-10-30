#!/usr/bin/env bash
set -euo pipefail
PID=$(cat /home/yqwang/outputs/CASA-CD/CASA-MoE/Run1/pipeline.pid)
if [[ -r "/proc/$PID/cmdline" ]]; then
  COMMAND=$(tr '\0' ' ' < "/proc/$PID/cmdline")
  if [[ "$COMMAND" == *'/CASA-MoE/Run1/pipeline.py'* ]]; then kill -TERM "$PID"; fi
fi
