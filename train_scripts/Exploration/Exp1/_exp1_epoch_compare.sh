#!/usr/bin/env bash
# 同期同 epoch 的 F1 对比（LEVIR），用于中途读趋势（不作结论）
set -u
B=/home/yqwang/outputs/CASA-CD/CASA-TViM
echo "epoch | CTRL_R8 | STG_INJ(v1) | STG_V2"
for e in 120 130 138 140 141; do
  a=$(grep -a "^Epoch $e/" "$B/Run8/CTRL_R8/LEVIR-CD-256/train_log.txt" | tail -1 | sed 's/.*F1=//;s/ |.*//')
  b=$(grep -a "^Epoch $e/" "$B/Exploration/Exp1/STG_INJ/LEVIR-CD-256/train_log.txt" | tail -1 | sed 's/.*F1=//;s/ |.*//')
  c=$(grep -a "^Epoch $e/" "$B/Exploration/Exp1/STG_V2/LEVIR-CD-256/train_log.txt" | tail -1 | sed 's/.*F1=//;s/ |.*//')
  printf '%5s | %-7s | %-11s | %-7s\n' "$e" "${a:-NA}" "${b:-NA}" "${c:-NA}"
done
echo
echo "--- 各臂在 LEVIR 上到 epoch 141 时已见的最大 F1 ---"
for v in "Run8/CTRL_R8" "Exploration/Exp1/STG_INJ" "Exploration/Exp1/STG_V2"; do
  m=$(grep -a '^Epoch ' "$B/$v/LEVIR-CD-256/train_log.txt" | head -141 | sed 's/.*F1=//;s/ |.*//' | sort -g | tail -1)
  printf '  %-26s %s\n' "$v" "${m:-NA}"
done
