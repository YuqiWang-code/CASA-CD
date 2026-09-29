#!/usr/bin/env bash
set -uo pipefail

# CASAA Run2 队列（GPU1 专属，GPU0 留给其它项目；每卡同时最多一个任务）
#   顺序：A1_LEVIR -> A3_LEVIR -> A1_SYSU -> A3_SYSU（同数据集 A1 在前做机制对照）
# 启动：nohup bash run_queue.sh > /dev/null 2>&1 &

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG=/home/yqwang/outputs/CASA-CD/CASAA/Run2/queue.log

for TASK in train_A1_SAA_FROZEN_LEVIR-CD-256.sh \
            train_A3_ORACLE_FROZEN_LEVIR-CD-256.sh \
            train_A1_SAA_FROZEN_SYSU-CD-256.sh \
            train_A3_ORACLE_FROZEN_SYSU-CD-256.sh; do
    echo "[TASK-START ${TASK}] $(date)" >> "${LOG}"
    bash "${SCRIPT_DIR}/${TASK}"
    echo "[TASK-END   ${TASK}] $(date)" >> "${LOG}"
done

echo "[QUEUE-DONE] all Run2 runs finished $(date)" >> "${LOG}"
