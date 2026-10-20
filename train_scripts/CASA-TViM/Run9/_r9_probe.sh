#!/usr/bin/env bash
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run9
echo "=== $(date '+%F %T') ==="
echo "--- 当前 stage 日志尾 10 ---"
L=$(ls -t "$ROOT"/*/*/train_log.txt 2>/dev/null | head -1)
echo "  (log: $L)"
tail -10 "$L" 2>/dev/null | sed 's/^/  /'
echo "--- 关键行 ---"
grep -hE '^\[(PRETRAIN-LOAD|PARAM-BREAKDOWN|CONFIG|FAILFAST|HQKV\]|HSAM|PROTOCOL)' "$L" 2>/dev/null \
  | head -12 | sed 's/^/  /'
echo "--- 进度 ---"
for V in S0_CTRL S1_HSAM; do
  printf '  %-9s' "$V"
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    F="$ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf ' %s=--' "${DS%%-*}"; continue; fi
    if grep -q '=== END TEST RESULTS ===' "$F"; then printf ' %s=DONE' "${DS%%-*}"
    else printf ' %s=%s' "${DS%%-*}" "$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//;s#/.*##')"; fi
  done
  echo
done
echo "--- 督导尾 6 ---"
tail -6 /tmp/run9/pipeline.log 2>/dev/null | sed 's/^/  /'
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
