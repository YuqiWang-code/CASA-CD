#!/usr/bin/env bash
set -uo pipefail

# ChangeViT-T baseline / Run1 — wave 2 scheduler
# Waits for CDD + LEVIR to finish, then launches SYSU + WHU in parallel (GPU0).

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=/home/yqwang/outputs/CASA-CD/baseline/Run1

while true; do
    if grep -q "\[DONE\]" "${LOG_DIR}/CDD-CD-256/train_log.txt" 2>/dev/null \
       && grep -q "\[DONE\]" "${LOG_DIR}/LEVIR-CD-256/train_log.txt" 2>/dev/null; then
        echo "[WAVE2-START] $(date)" >> "${LOG_DIR}/queue.log"
        break
    fi
    sleep 300
done

bash "${SCRIPT_DIR}/train_SYSU-CD-256.sh" &
P1=$!
bash "${SCRIPT_DIR}/train_WHU-CD-256.sh" &
P2=$!
wait $P1 $P2

echo "[WAVE2-DONE] $(date)" >> "${LOG_DIR}/queue.log"
