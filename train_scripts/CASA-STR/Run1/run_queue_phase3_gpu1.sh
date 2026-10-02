#!/usr/bin/env bash
set -uo pipefail

# CASA-STR Run1 Phase3 GPU1 顺序队列（前一个完成/3 次 retry 耗尽后才启动下一个）
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

RUN_DIR=/home/yqwang/projects/CASA-CD/train_scripts/CASA-STR/Run1
bash "${RUN_DIR}/C1_FULLATTN_PLAIN/train_SYSU-CD-256.sh"
bash "${RUN_DIR}/C1_FULLATTN_PLAIN/train_WHU-CD-256.sh"
bash "${RUN_DIR}/C2_CONTENT_SAA_PLAIN/train_SYSU-CD-256.sh"
bash "${RUN_DIR}/C2_CONTENT_SAA_PLAIN/train_WHU-CD-256.sh"

echo "[PHASE3-GPU1] queue finished"
