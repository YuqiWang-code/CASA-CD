#!/usr/bin/env bash
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run9
echo "=== $(date '+%F %T') ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "--- 计算进程 ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r pid mem; do
  pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
  [ -d "/proc/$pid" ] || continue
  echo "  pid=$pid mem=$mem :: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-150)"
done
echo "--- 我们的 train.py 全量（按启动时间）---"
ps -eo pid,lstart,cmd | grep -E 'python train\.py' | grep -v grep | cut -c1-160 | sed 's/^/  /'
echo "--- run_one_ds / supervisor ---"
ps -eo pid,lstart,cmd | grep -E '_r9_(run_one_ds|pipeline)' | grep -v grep | cut -c1-140 | sed 's/^/  /'
echo "--- 各库 born/mtime/步数 ---"
for V in S0_CTRL S1_HSAM; do
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    F="$ROOT/$V/$DS/train_log.txt"
    [ -f "$F" ] || continue
    printf '  %-9s %-14s born=%s mtime=%s steps=%s tests=%s\n' "$V" "$DS" \
      "$(stat -c%w "$F" 2>/dev/null | cut -d. -f1)" "$(stat -c%y "$F" 2>/dev/null | cut -d. -f1)" \
      "$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//')" \
      "$(grep -c '=== END TEST RESULTS ===' "$F" || true)"
  done
done
echo "--- pipeline.log 尾 12 ---"
tail -12 /tmp/run9/pipeline.log | sed 's/^/  /'
echo "--- stage.log 尾 8 ---"
tail -8 /tmp/run9/stage.log 2>/dev/null | sed 's/^/  /'
