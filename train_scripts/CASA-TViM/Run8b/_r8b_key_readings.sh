#!/usr/bin/env bash
# Run8b 关键读数（只读）：fail-fast / 抗自锁证据 / 进度。
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
V="${1:-CTRL_R8B}"
DS="${2:-CDD-CD-256}"
F="$ROOT/$V/$DS/train_log.txt"

echo "=== $(date '+%F %T')  $V / $DS ==="
if [ ! -f "$F" ]; then echo "  (无日志)"; exit 0; fi
echo "--- 抗自锁：step-1 门梯度 ---"
grep -E '^\[HQKV-GRAD\]' "$F" | tail -2 | sed 's/^/  /' || echo "  (尚无)"
echo "--- fail-fast（step 1000）---"
grep -E '^\[FAILFAST' "$F" | tail -3 | sed 's/^/  /' || echo "  (尚未到 1000 步)"
echo "--- 进度 ---"
grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -2 | sed 's/^/  /'
grep -E '^Epoch ' "$F" | tail -2 | sed 's/^/  /'
echo "--- 每 epoch 的 HQKV-DIAG（核心：contrib_norm / gammas）---"
grep -E '^\[HQKV-DIAG\]' "$F" | tail -3 | sed 's/^/  /' || echo "  (尚未到第一个 epoch 末的 val)"
echo "--- 配置行 ---"
grep -E '^\[(CONFIG|FAILFAST|HQKV|PARAM-BREAKDOWN|PRETRAIN-LOAD)\]' "$F" | head -8 | sed 's/^/  /'
