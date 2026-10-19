#!/usr/bin/env bash
# Run8b 状态快照（只读）。
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"
VARIANTS="CTRL_R8B B1_HYP B2_EUC B3_HYP_GS"

echo "=== $(date '+%F %T') ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "--- 我们的 train.py（dataset + variant）---"
for p in $(pgrep -f '^python train\.py' 2>/dev/null || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  ds=$(echo "$cmd" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  ck=$(echo "$cmd" | sed -n 's/.*--ckpt_dir \([^ ]*\).*/\1/p')
  v=$(echo "$ck" | sed -n 's#.*/Run8b/\([^/]*\)/.*#\1#p')
  echo "  pid=$p $v / $ds"
done
echo "--- 督导 ---"
for p in $(pgrep -f '_r8b_pipeline_supervisor' 2>/dev/null || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  pid=$p $(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')"
done
echo
echo "--- 进度矩阵（steps 或 DONE）---"
for V in $VARIANTS; do
  printf '  %-10s' "$V"
  for DS in $DATASETS; do
    F="$ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf ' %s=--' "${DS%%-*}"; continue; fi
    TB=$(grep -c '=== END TEST RESULTS ===' "$F")
    if [ "$TB" -gt 0 ]; then printf ' %s=DONE' "${DS%%-*}"; else
      CUR=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//' | sed 's#/.*##')
      printf ' %s=%s' "${DS%%-*}" "${CUR:-0}"
    fi
  done
  echo
done
echo
echo "--- 督导日志尾 6 行 ---"
tail -6 /tmp/run8b/pipeline.log 2>/dev/null | sed 's/^/  /'
