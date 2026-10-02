#!/usr/bin/env bash
set -uo pipefail

# CASA-STR Run1 Phase2 GPU1 顺序队列（前一个完成/3 次 retry 耗尽后才启动下一个）
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

RUN_DIR=/home/yqwang/projects/CASA-CD/train_scripts/CASA-STR/Run1
bash "${RUN_DIR}/A1_CASAA_PLAIN/train_SYSU-CD-256.sh"
bash "${RUN_DIR}/A1_CASAA_PLAIN/train_WHU-CD-256.sh"
bash "${RUN_DIR}/A2_STR_ONLY/train_SYSU-CD-256.sh"
bash "${RUN_DIR}/A2_STR_ONLY/train_WHU-CD-256.sh"

echo "[PHASE2-GPU1] queue finished"
