#!/usr/bin/env bash
# 通用启动器：以脱离会话方式拉起任意臂
#   bash _exp1_launch_arm.sh <VARIANT> <GPU> [DATASET...]
set -u
VARIANT="${1:?usage: _exp1_launch_arm.sh <VARIANT> <GPU> [DATASET...]}"
GPU="${2:-1}"
shift 2 || true
REPO=/home/yqwang/projects/CASA-CD
RUN=${REPO}/train_scripts/Exploration/Exp1/run_all.sh
LOGD=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1
mkdir -p "$LOGD" /tmp/exp1
PIDF=/tmp/exp1/run_all_${VARIANT}.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "[ABORT] ${VARIANT} 已在运行 pid=$(cat "$PIDF")" >&2; exit 3
fi
if pgrep -f "[t]rain_stgeo\.py" >/dev/null 2>&1; then
  echo "[ABORT] 已有 train_stgeo 进程在跑" >&2; exit 3
fi
setsid nohup bash "$RUN" "$VARIANT" "$GPU" "$@" \
  > "$LOGD/run_all_${VARIANT}.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDF"
sleep 25
echo "[LAUNCHED] ${VARIANT} pid=$PID"
tail -4 "$LOGD/run_all_${VARIANT}.log" 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
