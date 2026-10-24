#!/usr/bin/env bash
set -euo pipefail
PID=$(cat /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1/pipeline.pid)
kill -TERM "$PID"
