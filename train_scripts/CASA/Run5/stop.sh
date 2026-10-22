#!/usr/bin/env bash
set -euo pipefail
PID=$(cat /home/yqwang/outputs/CASA-CD/CASA/Run5/pipeline.pid)
kill -TERM "$PID"
