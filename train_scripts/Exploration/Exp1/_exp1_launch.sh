#!/usr/bin/env bash
# 在 GPU1 上以脱离会话方式拉起 Exp1 主实验臂（四库串行，严格 80K）
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN=${REPO}/train_scripts/Exploration/Exp1/run_all.sh
LOGD=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1
mkdir -p "$LOGD" /tmp/exp1
if [ -f /tmp/exp1/run_all.pid ] && kill -0 "$(cat /tmp/exp1/run_all.pid)" 2>/dev/null; then
  echo "[ABORT] run_all 已在运行 pid=$(cat /tmp/exp1/run_all.pid)" >&2; exit 3
fi
setsid nohup bash "$RUN" STG_INJ 1 > "$LOGD/run_all.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > /tmp/exp1/run_all.pid
sleep 20
echo "[LAUNCHED] pid=$PID"
tail -5 "$LOGD/run_all.log" 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
