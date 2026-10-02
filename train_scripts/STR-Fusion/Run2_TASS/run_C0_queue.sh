#!/usr/bin/env bash
set -uo pipefail
# Run11 C0_TOKEN 串行队列（GPU1）：SYSU -> LEVIR -> WHU -> CDD（每个自带 retry）
cd /home/yqwang/projects/CASA-CD/train_scripts/STR-Fusion/Run2_TASS
for ds in SYSU-CD-256 LEVIR-CD-256 WHU-CD-256 CDD-CD-256; do
  echo "[QUEUE-C0] start $ds"
  bash C0_TOKEN/train_${ds}.sh
  echo "[QUEUE-C0] done $ds"
done
echo "[QUEUE-C0] ALL DONE"