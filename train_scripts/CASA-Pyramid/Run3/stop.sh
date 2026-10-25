#!/usr/bin/env bash
set -euo pipefail
PID=$(cat /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run3/pipeline.pid)
if [[ ! -r "/proc/$PID/cmdline" ]]; then
  echo 'CASA-Pyramid Run3 supervisor is not running'
  exit 0
fi
COMMAND=$(tr '\0' ' ' < "/proc/$PID/cmdline")
if [[ "$COMMAND" != *'/CASA-Pyramid/Run3/pipeline.py'* ]]; then
  echo 'CASA-Pyramid Run3 supervisor is not running'
  exit 0
fi
kill -TERM "$PID"
