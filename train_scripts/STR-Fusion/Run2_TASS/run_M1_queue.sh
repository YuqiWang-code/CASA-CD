#!/usr/bin/env bash
set -uo pipefail
# Run11 M1_TASS 串行队列（GPU0）：SYSU -> LEVIR -> WHU -> CDD（每个自带 retry）
cd /home/yqwang/projects/CASA-CD/train_scripts/STR-Fusion/Run2_TASS
for ds in SYSU-CD-256 LEVIR-CD-256 WHU-CD-256 CDD-CD-256; do
  echo "[QUEUE-M1] start $ds"
  bash M1_TASS/train_${ds}.sh
  echo "[QUEUE-M1] done $ds"
done
echo "[QUEUE-M1] ALL DONE"