#!/usr/bin/env bash
# 状态探针（只读）。
set -u
STATE=/tmp/run8b
echo "=== $(date '+%F %T') ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "--- GPU 进程 ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | sed 's/^/  /'
echo "--- 督导进程 ---"
for p in $(pgrep -f '_r8b' 2>/dev/null || true); do
  echo "  pid=$p  $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)"
done
echo "--- pipeline.log 尾 12 行 ---"
tail -12 "$STATE/pipeline.log" 2>/dev/null | sed 's/^/  /'
echo "--- 完成情况 ---"
if [ -f /home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run8b/_r8b_state.sh ]; then
  bash /home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run8b/_r8b_state.sh 2>&1 | sed 's/^/  /'
fi
