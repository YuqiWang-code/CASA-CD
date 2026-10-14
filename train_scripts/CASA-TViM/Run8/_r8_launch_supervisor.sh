#!/usr/bin/env bash
# 把 Run8 波次督导以**脱离会话**的方式拉起（ssh 通道关掉后仍继续）。
# 用法：bash _r8_launch_supervisor.sh
#
# 为什么不直接在 ssh 里内联：PowerShell 会吃掉内联命令里的 \$() 与引号
# （本项目第 1 号坑，Run7 期间反复发生）⇒ 一律写 .sh 文件再执行。
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
PIDFILE="$STATE/supervisor.pid"
mkdir -p "$STATE"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[SUP-LAUNCH] 已在运行 pid=$(cat "$PIDFILE")，拒绝重复启动"
  exit 3
fi

cd "$REPO" || exit 1
setsid nohup bash "$RUN_DIR/_r8_supervisor.sh" > "$STATE/sup.stdout" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"
sleep 6
echo "[SUP-LAUNCH] pid=$PID  log=$STATE/sup.stdout"
if kill -0 "$PID" 2>/dev/null; then
  echo "[SUP-LAUNCH] ALIVE"
else
  echo "[SUP-LAUNCH] DIED-IMMEDIATELY"
  tail -20 "$STATE/sup.stdout" 2>/dev/null
  exit 1
fi
echo "--- supervisor.log 尾部 ---"
tail -8 "$STATE/supervisor.log" 2>/dev/null || echo "  (尚无)"
