#!/usr/bin/env bash
set -euo pipefail
PID=$(cat /home/yqwang/outputs/CASA-CD/CASA/Run6/pipeline.pid)
kill -TERM "$PID"
