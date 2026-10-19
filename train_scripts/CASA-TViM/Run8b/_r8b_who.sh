#!/usr/bin/env bash
# 查 GPU0 上的 foreign 进程是谁、占多少、在干什么（只读，绝不触碰）。
set -u
echo "=== $(date '+%F %T') ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "--- 全部 compute 进程 ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
echo "--- index/uuid ---"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo
echo "--- 逐个进程：user / 归属 / cmdline ---"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
  [ -z "$p" ] && continue
  u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
  et=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-160)
  cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null)
  echo "  pid=$p user=${u:-?} et=$et"
  echo "    cmd=$cmd"
  echo "    cwd=$cwd"
done
echo
echo "--- 我们的项目进程（对照）---"
pgrep -af 'CASA-CD' 2>/dev/null | grep -v 'bash -c' | cut -c1-120 | head -8
