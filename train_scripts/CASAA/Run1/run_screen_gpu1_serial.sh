#!/usr/bin/env bash
set -uo pipefail

# CASAA Run1 筛选阶段——GPU1 串行版（GPU0 被共享租户长期占用时的退路）。
# 一张 RTX 5090 同时最多一个任务（batch16 需 ~16-20 GB）。
#   GPU1: A1_SAA_LEVIR -> A2_CASAA_LEVIR -> A1_SAA_SYSU -> A2_CASAA_SYSU
# 启动：nohup bash run_screen_gpu1_serial.sh > /dev/null 2>&1 &

export RUN_GPU=1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG=/home/yqwang/outputs/CASA-CD/CASAA/Run1/queue_screen.log

for TASK in train_A1_SAA_LEVIR-CD-256.sh \
            train_A2_CASAA_LEVIR-CD-256.sh \
            train_A1_SAA_SYSU-CD-256.sh \
            train_A2_CASAA_SYSU-CD-256.sh; do
    echo "[TASK-START ${TASK}] $(date)" >> "${LOG}"
    bash "${SCRIPT_DIR}/${TASK}"
    echo "[TASK-END   ${TASK}] $(date)" >> "${LOG}"
done

echo "[SCREEN-DONE] all GPU1-serial screening runs finished $(date)" >> "${LOG}"
