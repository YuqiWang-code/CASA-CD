#!/usr/bin/env bash
set -uo pipefail

# Wait for WHU (GPU1) to finish, then resume SYSU on GPU1.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=/home/yqwang/outputs/CASA-CD/baseline/Run1

echo "[SYSU-WAIT] $(date) waiting for WHU to finish" >> "${LOG_DIR}/queue.log"
while true; do
    if grep -q "=== END TEST RESULTS ===" "${LOG_DIR}/WHU-CD-256/train_log.txt" 2>/dev/null; then
        echo "[SYSU-START-ON-GPU1] $(date)" >> "${LOG_DIR}/queue.log"
        break
    fi
    sleep 60
done

bash "${SCRIPT_DIR}/train_SYSU-CD-256.sh"
echo "[SYSU-DONE-ON-GPU1] $(date)" >> "${LOG_DIR}/queue.log"
