#!/usr/bin/env bash
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
echo "=== $(date '+%F %T') ==="
L="$ROOT/B1_HYP/LEVIR-CD-256/train_log.txt"
if [ -f "$L" ]; then
  echo "log size: $(stat -c%s "$L") bytes   mtime: $(stat -c%y "$L")"
  grep -E '^\[ACTUAL-OPT-STEPS\]' "$L" | tail -3
  echo "--- epoch 行尾 3 ---"; grep -E 'epoch=' "$L" | tail -3 | cut -c1-170
  echo "--- GRAD ---"; grep -h 'HQKV-GRAD' "$L" | head -1 | cut -c1-180
  echo "--- FAILFAST ---"; grep -hE '^\[FAILFAST' "$L" | head -3 | cut -c1-180
else
  echo "log 尚未生成"
fi
echo "--- GPU0 进程 ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | sed 's/^/  /'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
