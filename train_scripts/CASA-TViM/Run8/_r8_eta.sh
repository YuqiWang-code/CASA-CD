#!/usr/bin/env bash
# 实测步速与 ETA（只读）：给出「每 step 秒数」与四个库的剩余时间估算。
set -u
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
V="${1:-CTRL_R8}"
ITERS_CDD=313; ITERS_LEVIR=223; ITERS_SYSU=375; ITERS_WHU=186
TOTAL=80000

echo "=== $(date '+%F %T')  变体=$V ==="
row() {
  local ds="$1" iters="$2"
  local F="$LOG_ROOT/$V/$ds/train_log.txt"
  if [ ! -f "$F" ]; then printf '  %-14s (未启动)\n' "$ds"; return; fi
  local cur tsum tcnt ta tb last_epoch
  cur=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//' | sed 's#/.*##')
  # 用最后两个 epoch 行测 epoch 耗时（比 [ACTUAL-OPT-STEPS] 可靠，后者每 epoch 只打一次）
  ta=$(grep -E '^Epoch ' "$F" | tail -2 | head -1 | sed 's/.*t=//' | sed 's/s$//')
  tb=$(grep -E '^Epoch ' "$F" | tail -1 | sed 's/.*t=//' | sed 's/s$//')
  if [ -n "${tb:-}" ]; then
    local sps; sps=$(awk -v i="$iters" -v t="$tb" 'BEGIN{printf "%.4f", i/t}')
    local eta; eta=$(awk -v tot="$TOTAL" -v c="${cur:-0}" -v s="$sps" 'BEGIN{printf "%.2f", (tot-c)/s/3600}')
    printf '  %-14s steps=%-7s epoch_t=%ss  iters=%s  step_s=%.4f  remain=%.2fh\n' \
      "$ds" "${cur:-0}" "$tb" "$iters" "$(awk -v s="$sps" 'BEGIN{printf "%.4f",1/s}')" "$eta"
  else
    printf '  %-14s steps=%-7s (还没有 epoch 行)\n' "$ds" "${cur:-0}"
  fi
}
row CDD-CD-256   $ITERS_CDD
row LEVIR-CD-256 $ITERS_LEVIR
row SYSU-CD-256  $ITERS_SYSU
row WHU-CD-256   $ITERS_WHU
