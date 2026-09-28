#!/usr/bin/env bash
set -uo pipefail

# CASAA Run1 补全阶段（筛选通过后再启动）：A2 CASAA × CDD/WHU，双卡并行。
#   GPU0: A2_CASAA_CDD
#   GPU1: A2_CASAA_WHU
# 启动：nohup bash run_full_casaa.sh > /dev/null 2>&1 &

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG=/home/yqwang/outputs/CASA-CD/CASAA/Run1/queue_full.log

{
    echo "[GPU0-START] $(date)"
    bash "${SCRIPT_DIR}/train_A2_CASAA_CDD-CD-256.sh"
    echo "[GPU0-DONE ] $(date)"
} >> "${LOG}" 2>&1 &

{
    echo "[GPU1-START] $(date)"
    bash "${SCRIPT_DIR}/train_A2_CASAA_WHU-CD-256.sh"
    echo "[GPU1-DONE ] $(date)"
} >> "${LOG}" 2>&1 &

wait
echo "[FULL-DONE] all complement runs finished $(date)" >> "${LOG}"
