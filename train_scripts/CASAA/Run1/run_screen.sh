#!/usr/bin/env bash
set -uo pipefail

# CASAA Run1 筛选阶段：A1/A2 × LEVIR/SYSU，双卡各一串、每卡同时最多一个任务。
#   GPU0: A1_SAA_LEVIR -> A2_CASAA_LEVIR
#   GPU1: A1_SAA_SYSU  -> A2_CASAA_SYSU
# 启动：nohup bash run_screen.sh > /dev/null 2>&1 &

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG=/home/yqwang/outputs/CASA-CD/CASAA/Run1/queue_screen.log

{
    echo "[GPU0-START] $(date)"
    bash "${SCRIPT_DIR}/train_A1_SAA_LEVIR-CD-256.sh"
    bash "${SCRIPT_DIR}/train_A2_CASAA_LEVIR-CD-256.sh"
    echo "[GPU0-DONE ] $(date)"
} >> "${LOG}" 2>&1 &

{
    echo "[GPU1-START] $(date)"
    bash "${SCRIPT_DIR}/train_A1_SAA_SYSU-CD-256.sh"
    bash "${SCRIPT_DIR}/train_A2_CASAA_SYSU-CD-256.sh"
    echo "[GPU1-DONE ] $(date)"
} >> "${LOG}" 2>&1 &

wait
echo "[SCREEN-DONE] all screening runs finished $(date)" >> "${LOG}"
