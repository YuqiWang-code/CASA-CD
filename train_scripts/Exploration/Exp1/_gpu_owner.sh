#!/usr/bin/env bash
# 排卡探查：谁在 GPU 上、我们自己的进程在不在
set -u
echo "=== UUID 映射 ==="
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== compute apps + 归属 ==="
while IFS=, read -r uuid pid name mem; do
  uuid=$(echo "$uuid" | tr -d ' ')
  pid=$(echo "$pid" | tr -d ' ')
  user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-110)
  echo "gpu_uuid=$uuid pid=$pid user=${user:-?} mem=${mem} cmd=${cmd:-?}"
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)
echo "=== 我们自己的进程 ==="
echo "--- Exp1 (train_stgeo) ---"; pgrep -af "[t]rain_stgeo\.py" | cut -c1-100 | head -3 || echo "  (无)"
echo "--- Run8 (train.py) ---"; pgrep -af "[t]rain\.py --dataset" | sed 's/ --dataset_root.*//' | head -3 || echo "  (无)"
echo "=== 我（当前用户）==="; whoami; date '+%F %T %z'
