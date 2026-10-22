#!/usr/bin/env bash
set -euo pipefail
PID=$(cat /home/yqwang/outputs/CASA-CD/CASA-WDMF/Run2/pipeline.pid)
kill -TERM "$PID"
