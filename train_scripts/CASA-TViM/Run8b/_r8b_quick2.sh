#!/usr/bin/env bash
set -u
echo "=== $(date '+%F %T') ==="
tail -6 /tmp/run8b/pipeline.log 2>/dev/null | sed 's/^/  /'
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "--- 我们自己的 train.py ---"
for p in $(pgrep -f '^python train\.py' 2>/dev/null || true); do
  echo "  pid=$p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-160)"
done
R=/home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run8b
echo "--- arch.json (CTRL_R8B/CDD) ---"
find /home/yqwang/projects/CASA-CD -name arch.json -path '*Run8b*' 2>/dev/null | head -8 | while read -r f; do
  echo "  $f"; sed 's/^/    /' "$f" | head -30
done
