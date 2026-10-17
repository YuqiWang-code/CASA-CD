#!/usr/bin/env bash
# 等卡督导存活检查 + 当前 GPU 状态
set -u
P=$(cat /tmp/exp1/wait_v2.pid 2>/dev/null || echo "")
echo "pidfile=/tmp/exp1/wait_v2.pid content=${P:-none}"
if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then echo "supervisor=ALIVE"; else echo "supervisor=DEAD"; fi
pgrep -af "wait_and_launch" | head -3
echo "--- wait log tail ---"; tail -4 /tmp/exp1/wait_STG_V2.log 2>/dev/null
echo "--- gpu ---"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- gpu1 compute apps (owner) ---"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
  printf 'pid=%s user=%s\n' "$p" "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
done
echo "--- our exp1 processes ---"; pgrep -af "[t]rain_stgeo\.py" | cut -c1-80 || echo "  (none)"
