#!/usr/bin/env bash
set -uo pipefail

# CASA-TViM Run4 全训练：2 波 x 4 数据集并行（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发、batch 32）。
# 先 M1_R4CTRL（同期对照）四库全部结束，再 E6_FET1（唯一正式主实验）四库；某波失败立即中止，不启动下一波。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

RUN_DIR=/home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run4
PROJ=/home/yqwang/projects/CASA-CD

# --- 预检 1：所有正式 ckpt/log 路径必须不存在或严格空目录（除非 RESUME_SAME_RUN=1） ---
if [ "${RESUME_SAME_RUN:-0}" != "1" ]; then
    for v in M1_R4CTRL E6_FET1; do
        for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
            ck=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4/${v}/${d}
            lg=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/${v}/${d}
            if [ -d "${ck}" ] && [ -n "$(ls -A "${ck}" 2>/dev/null)" ]; then
                echo "[ABORT] non-empty ckpt dir: ${ck}"; exit 3
            fi
            if [ -d "${lg}" ] && [ -n "$(ls -A "${lg}" 2>/dev/null)" ]; then
                echo "[ABORT] non-empty log dir: ${lg}"; exit 3
            fi
        done
    done
else
    echo "[RESUME_SAME_RUN=1] skipping the empty-dir precheck (certified same-run resume only)"
fi

# --- 预检 2：每 GPU 并发 2 个 batch32 进程的显存 probe（R8）。任一 OOM 停止并发，不改 batch/协议 ---
echo "[PREFLIGHT] batch32 x2 per GPU concurrency memory probe"
_probe_fail=0
for g in 0 1; do
    for k in 1 2; do
        CUDA_VISIBLE_DEVICES=${g} python "${RUN_DIR}/gpu_concurrency_probe.py" \
            --variant E6_FET1 --device cuda:0 --batch-size 32 --steps 2 \
            > /tmp/run4_probe_g${g}_${k}.log 2>&1 & _p=$!
        eval "PID_${g}_${k}=$_p"
    done
done
for g in 0 1; do
    for k in 1 2; do
        eval "p=\$PID_${g}_${k}"
        wait "$p" || _probe_fail=1
    done
done
if [ "${_probe_fail}" -ne 0 ]; then
    echo "[ABORT] batch32 x2/GPU probe failed (likely OOM). Keep batch32; switch to same-GPU serial + cross-GPU parallel; do NOT change the training protocol."
    for f in /tmp/run4_probe_g*.log; do echo "--- $f"; tail -5 "$f"; done
    exit 4
fi
echo "[PREFLIGHT] concurrency probe OK"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

echo "===== WAVE 1 ====="
bash "${RUN_DIR}/M1_R4CTRL/train_CDD-CD-256.sh" > /tmp/wave1_P0_0.log 2>&1 & P0_0=$!
bash "${RUN_DIR}/M1_R4CTRL/train_LEVIR-CD-256.sh" > /tmp/wave1_P0_1.log 2>&1 & P0_1=$!
bash "${RUN_DIR}/M1_R4CTRL/train_SYSU-CD-256.sh" > /tmp/wave1_P0_2.log 2>&1 & P0_2=$!
bash "${RUN_DIR}/M1_R4CTRL/train_WHU-CD-256.sh" > /tmp/wave1_P0_3.log 2>&1 & P0_3=$!
_wave_fail=0
wait $P0_0 || _wave_fail=1
wait $P0_1 || _wave_fail=1
wait $P0_2 || _wave_fail=1
wait $P0_3 || _wave_fail=1
if [ "$_wave_fail" -ne 0 ]; then echo "[WAVE-ABORT] wave 1 has failed runs; aborting"; exit 1; fi
echo "===== WAVE 1 done ====="

echo "===== WAVE 2 ====="
bash "${RUN_DIR}/E6_FET1/train_CDD-CD-256.sh" > /tmp/wave2_P0_0.log 2>&1 & P0_0=$!
bash "${RUN_DIR}/E6_FET1/train_LEVIR-CD-256.sh" > /tmp/wave2_P0_1.log 2>&1 & P0_1=$!
bash "${RUN_DIR}/E6_FET1/train_SYSU-CD-256.sh" > /tmp/wave2_P0_2.log 2>&1 & P0_2=$!
bash "${RUN_DIR}/E6_FET1/train_WHU-CD-256.sh" > /tmp/wave2_P0_3.log 2>&1 & P0_3=$!
_wave_fail=0
wait $P0_0 || _wave_fail=1
wait $P0_1 || _wave_fail=1
wait $P0_2 || _wave_fail=1
wait $P0_3 || _wave_fail=1
if [ "$_wave_fail" -ne 0 ]; then echo "[WAVE-ABORT] wave 2 has failed runs; aborting"; exit 1; fi
echo "===== WAVE 2 done ====="

echo "[RUN-ALL] all 8 runs finished"
