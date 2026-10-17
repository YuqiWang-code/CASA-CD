#!/usr/bin/env bash
# 以脱离会话方式武装 v2 等卡督导（幂等：已在跑就不重复启动）
set -u
REPO=/home/yqwang/projects/CASA-CD
STATE=/tmp/exp1; mkdir -p "$STATE"
PIDF="$STATE/wait_v2.pid"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "[ABORT] 等卡督导已在运行 pid=$(cat "$PIDF")"; exit 3
fi
setsid nohup bash "$REPO/train_scripts/Exploration/Exp1/_exp1_wait_and_launch.sh" \
  1 STG_V2 "LEVIR-CD-256 SYSU-CD-256" > "$STATE/wait_v2.stdout" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDF"
sleep 8
echo "[ARMED] pid=$PID"
tail -5 "$STATE/wait_STG_V2.log" 2>/dev/null || tail -5 "$STATE/wait_v2.stdout"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
