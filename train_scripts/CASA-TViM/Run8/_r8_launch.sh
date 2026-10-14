#!/usr/bin/env bash
# Run8 波次启动器：把 run_all.sh 以**脱离会话**的方式拉起（ssh 通道关掉后仍继续）。
#   bash _r8_launch.sh --wave 1
#   bash _r8_launch.sh --wave all
#
# 纪律（Run7 血泪）：
#   - 必须重定向 stdin，否则 ssh 通道不关、调用超时；
#   - 写 pidfile 供 kill -0 判死活（**不要看日志判死活**，波内长时间静默是正常的）。
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
mkdir -p "$STATE"

WAVE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --wave) WAVE="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[ -n "$WAVE" ] || { echo "usage: $0 --wave {1|2|3|4|all}" >&2; exit 2; }

PIDFILE="$STATE/wave${WAVE}.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[LAUNCH] wave${WAVE} 已在运行 pid=$(cat "$PIDFILE")，拒绝重复启动"
  exit 3
fi

LOG="$STATE/wave${WAVE}.stdout"
cd "$REPO" || exit 1
setsid nohup bash "$RUN_DIR/run_all.sh" --wave "$WAVE" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"
sleep 3
echo "[LAUNCH] wave${WAVE} pid=$PID log=$LOG"
kill -0 "$PID" 2>/dev/null && echo "[LAUNCH] ALIVE" || echo "[LAUNCH] DIED-IMMEDIATELY (看 $LOG)"
