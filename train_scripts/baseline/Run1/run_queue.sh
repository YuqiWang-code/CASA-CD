#!/usr/bin/env bash
set -uo pipefail

# ChangeViT-T baseline / Run1 queue (GPU0, strictly sequential, official batch 16)
# ChangeViT-T batch16 needs ~15.7 GB (cross-attn over full c2 tokens), two parallel
# runs exceed the 32 GB of one RTX 5090 -> run one at a time.
# Order: CDD -> LEVIR -> SYSU -> WHU

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=/home/yqwang/outputs/CASA-CD/baseline/Run1

for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    echo "[QUEUE-START ${DS}] $(date)" >> "${LOG_DIR}/queue.log"
    bash "${SCRIPT_DIR}/train_${DS}.sh"
    echo "[QUEUE-END   ${DS}] $(date)" >> "${LOG_DIR}/queue.log"
done

echo "[QUEUE-DONE] all 4 datasets finished" >> "${LOG_DIR}/queue.log"
