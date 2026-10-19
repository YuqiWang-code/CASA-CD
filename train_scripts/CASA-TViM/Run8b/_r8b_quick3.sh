#!/usr/bin/env bash
set -u
echo "=== $(date '+%F %T') ==="
echo "--- pipe1.stdout 尾 40 ---"
tail -40 /tmp/run8b/pipe1.stdout 2>/dev/null | sed 's/^/  /'
echo "--- pipeline.log 尾 25 ---"
tail -25 /tmp/run8b/pipeline.log 2>/dev/null | sed 's/^/  /'
echo "--- 进程（含 r8b 关键字）---"
ps -eo pid,ppid,etime,cmd | grep -E 'r8b|run_one_ds|train\.py' | grep -v grep | cut -c1-190 | sed 's/^/  /'
echo "--- 输出目录 ---"
ls -dt /home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run8b/*/ 2>/dev/null | head -20 | sed 's/^/  /'
find /home/yqwang/projects/CASA-CD -maxdepth 6 -type d -name '*B1_HYP*' 2>/dev/null | head -10 | sed 's/^/  /'
