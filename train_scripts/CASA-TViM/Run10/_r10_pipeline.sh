#!/usr/bin/env bash
# Exactly four fresh runs, serially on GPU0. Any failure stops the queue.
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
RD=$REPO/train_scripts/CASA-TViM/Run10
STATE=/tmp/run10
mkdir -p "$STATE"
exec > >(tee -a "$STATE/pipeline.log") 2>&1
echo "[R10] start $(date '+%F %T')"
for DS in SYSU-CD-256 LEVIR-CD-256; do
  for VARIANT in S0_CTRL S1_HSAM_LD; do
    while true; do
      FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
      [ "${FREE:-0}" -ge 30000 ] && break
      echo "[R10] waiting GPU0: free=${FREE:-?} MiB, need=30000 MiB"
      sleep 20
    done
    echo "[R10] start $VARIANT/$DS $(date '+%F %T')"
    bash "$RD/_r10_run_one.sh" "$VARIANT" "$DS" 0
    echo "[R10] done $VARIANT/$DS $(date '+%F %T')"
  done
done
echo "[R10] all four complete $(date '+%F %T')"
