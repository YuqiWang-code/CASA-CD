#!/usr/bin/env bash
# Run8b 时间线：每个库的开始/结束时间、每 epoch 耗时、以及到下一波的剩余时间。
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"

echo "=== $(date '+%F %T') ==="
echo "--- 督导决策日志（尾 14 行）---"
tail -14 /tmp/run8b/pipeline.log | sed 's/^/  /'

echo
echo "--- CTRL_R8B 每个库：起止与速率 ---"
for d in $DATASETS; do
  F="$ROOT/CTRL_R8B/$d/train_log.txt"
  if [ ! -f "$F" ]; then echo "  $d: (无日志)"; continue; fi
  first_ts=$(stat -c '%y' "$F" | cut -c1-19)
  last_ts=$(stat -c '%y' "$F" | cut -c1-19)
  tb=$(grep -c '=== END TEST RESULTS ===' "$F")
  steps=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//' | sed 's#/.*##')
  et=$(grep -E '^Epoch ' "$F" | tail -1 | sed 's/.*t=//' | sed 's/s$//')
  n_ep=$(grep -cE '^Epoch ' "$F")
  echo "  $d: created=$first_ts lastwrite=$last_ts epochs_logged=$n_ep last_epoch_t=${et}s test_blocks=$tb steps=${steps:-0}"
done

echo
echo "--- 单库实测耗时（用 WHU 的 epoch_t 推）---"
F="$ROOT/CTRL_R8B/WHU-CD-256/train_log.txt"
if [ -f "$F" ]; then
  et=$(grep -E '^Epoch ' "$F" | tail -1 | sed 's/.*t=//' | sed 's/s$//')
  sps=$(awk -v i=186 -v t="${et:-60}" 'BEGIN{printf "%.4f", i/t}')
  echo "  WHU iters/epoch=186, epoch_t=${et}s -> ${sps} step/s"
  echo "  单库 80K = $(awk -v s="$sps" 'BEGIN{printf "%.2f", 80000/s/3600}') h"
  echo "  单波 4 库串行 = $(awk -v s="$sps" 'BEGIN{printf "%.2f", 4*80000/s/3600}') h"
  echo "  四波 = $(awk -v s="$sps" 'BEGIN{printf "%.2f", 16*80000/s/3600}') h = $(awk -v s="$sps" 'BEGIN{printf "%.1f", 16*80000/s/3600/24}') 天"
fi
