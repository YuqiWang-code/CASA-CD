#!/usr/bin/env bash
set -uo pipefail

# CASA-TViM Run2 全训练：3 波 x 4 数据集并行（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发、batch 32）。
# 一个脚本串完全部 12 个 80K（不分步）；每波 wait 全部完成后进入下一波。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

RUN_DIR=/home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run2
echo "===== WAVE 1 ====="
bash "${RUN_DIR}/E1_CP_CAACP/train_CDD-CD-256.sh" > /tmp/wave1_P0_0.log 2>&1 & P0_0=$!
bash "${RUN_DIR}/E1_CP_CAACP/train_LEVIR-CD-256.sh" > /tmp/wave1_P0_1.log 2>&1 & P0_1=$!
bash "${RUN_DIR}/E1_CP_CAACP/train_SYSU-CD-256.sh" > /tmp/wave1_P0_2.log 2>&1 & P0_2=$!
bash "${RUN_DIR}/E1_CP_CAACP/train_WHU-CD-256.sh" > /tmp/wave1_P0_3.log 2>&1 & P0_3=$!
_wave_fail=0
wait $P0_0 || _wave_fail=1
wait $P0_1 || _wave_fail=1
wait $P0_2 || _wave_fail=1
wait $P0_3 || _wave_fail=1
if [ "$_wave_fail" -ne 0 ]; then echo "[WAVE-ABORT] wave 1 has failed runs; aborting"; exit 1; fi
echo "===== WAVE 1 done ====="

echo "===== WAVE 2 ====="
bash "${RUN_DIR}/E2_FRH/train_CDD-CD-256.sh" > /tmp/wave2_P0_0.log 2>&1 & P0_0=$!
bash "${RUN_DIR}/E2_FRH/train_LEVIR-CD-256.sh" > /tmp/wave2_P0_1.log 2>&1 & P0_1=$!
bash "${RUN_DIR}/E2_FRH/train_SYSU-CD-256.sh" > /tmp/wave2_P0_2.log 2>&1 & P0_2=$!
bash "${RUN_DIR}/E2_FRH/train_WHU-CD-256.sh" > /tmp/wave2_P0_3.log 2>&1 & P0_3=$!
_wave_fail=0
wait $P0_0 || _wave_fail=1
wait $P0_1 || _wave_fail=1
wait $P0_2 || _wave_fail=1
wait $P0_3 || _wave_fail=1
if [ "$_wave_fail" -ne 0 ]; then echo "[WAVE-ABORT] wave 2 has failed runs; aborting"; exit 1; fi
echo "===== WAVE 2 done ====="

echo "===== WAVE 3 ====="
bash "${RUN_DIR}/E3_CP_FRH/train_CDD-CD-256.sh" > /tmp/wave3_P0_0.log 2>&1 & P0_0=$!
bash "${RUN_DIR}/E3_CP_FRH/train_LEVIR-CD-256.sh" > /tmp/wave3_P0_1.log 2>&1 & P0_1=$!
bash "${RUN_DIR}/E3_CP_FRH/train_SYSU-CD-256.sh" > /tmp/wave3_P0_2.log 2>&1 & P0_2=$!
bash "${RUN_DIR}/E3_CP_FRH/train_WHU-CD-256.sh" > /tmp/wave3_P0_3.log 2>&1 & P0_3=$!
_wave_fail=0
wait $P0_0 || _wave_fail=1
wait $P0_1 || _wave_fail=1
wait $P0_2 || _wave_fail=1
wait $P0_3 || _wave_fail=1
if [ "$_wave_fail" -ne 0 ]; then echo "[WAVE-ABORT] wave 3 has failed runs; aborting"; exit 1; fi
echo "===== WAVE 3 done ====="

echo "[RUN-ALL] all 12 runs finished"
