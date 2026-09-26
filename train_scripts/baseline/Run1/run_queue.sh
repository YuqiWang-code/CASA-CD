#!/usr/bin/env bash
set -uo pipefail

# ChangeViT-T baseline / Run1 queue (GPU0, 2 waves x 2 datasets in parallel)
# Wave 1: CDD + LEVIR   Wave 2: SYSU + WHU

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

bash "${SCRIPT_DIR}/train_CDD-CD-256.sh" &
P1=$!
bash "${SCRIPT_DIR}/train_LEVIR-CD-256.sh" &
P2=$!
wait $P1 $P2

bash "${SCRIPT_DIR}/train_SYSU-CD-256.sh" &
P3=$!
bash "${SCRIPT_DIR}/train_WHU-CD-256.sh" &
P4=$!
wait $P3 $P4

echo "[QUEUE-DONE] all 4 datasets finished"
