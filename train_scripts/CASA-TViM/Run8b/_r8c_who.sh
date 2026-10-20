#!/usr/bin/env bash
set -u
echo "=== $(date '+%F %T') ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "--- 进程（pid / 用户 / 显存 / cmdline 前 120 字符）---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r pid mem; do
  pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
  [ -d "/proc/$pid" ] || { echo "  pid=$pid mem=$mem (已退出)"; continue; }
  user=$(stat -c%U "/proc/$pid" 2>/dev/null || echo "?")
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120)
  echo "  pid=$pid user=$user mem=$mem :: $cmd"
done
echo "--- 我们是否还有 train.py ---"
for p in $(pgrep -f 'python train\.py' 2>/dev/null || true); do
  echo "  pid=$p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-110)"
done
echo "  (以上为空即我们没有在跑的训练)"
