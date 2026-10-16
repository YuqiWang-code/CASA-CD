#!/usr/bin/env bash
# Exp1 只读状态快照
set -u
EXP=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1
echo "=== time ==="; date '+%F %T %z'
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== launcher ==="
if [ -f /tmp/exp1/run_all.pid ] && kill -0 "$(cat /tmp/exp1/run_all.pid)" 2>/dev/null; then
  echo "run_all ALIVE pid=$(cat /tmp/exp1/run_all.pid)"
else
  echo "run_all not alive"
fi
pgrep -af '[t]rain_stgeo.py' | head -2
echo "=== progress ==="
for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
  L="$EXP/STG_INJ/$d/train_log.txt"
  if [ -f "$L" ]; then
    step=$(grep -a '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -1 | sed 's/.*cumulative=//')
    last=$(grep -a '^Epoch ' "$L" | tail -1)
    blocks=$(grep -ac '=== END TEST RESULTS ===' "$L")
    printf '%-14s steps=%-14s blocks=%s\n   %s\n' "$d" "${step:-NA}" "$blocks" "${last:-（尚无 epoch 行）}"
  else
    printf '%-14s (no log)\n' "$d"
  fi
done
echo "=== tail ==="
tail -3 "$EXP/run_all.log" 2>/dev/null
