#!/usr/bin/env bash
set -euo pipefail
PID=$(cat /home/yqwang/outputs/CASA-CD/CASA-WDMF/Run1/pipeline.pid)
CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline")
case "$CMD" in *'/CASA-WDMF/Run1/pipeline.py'*) kill -TERM "$PID" ;; *) echo 'pipeline PID no longer active'; exit 3 ;; esac
