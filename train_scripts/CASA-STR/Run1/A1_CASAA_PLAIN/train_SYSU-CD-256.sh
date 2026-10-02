#!/usr/bin/env bash
set -uo pipefail

# CASA-STR Run1 A1_CASAA_PLAIN — 创新一消融（change CASAA, plain） / SYSU-CD-256（正式 80K，GPU1；自动 retry 上限 3）
GPU=1
DATASET=SYSU-CD-256
VARIANT=A1_CASAA_PLAIN
ATTN_MODE=change
REP_MODE=plain
MAX_STEPS=80000
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-STR/Run1/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASA-STR/Run1/${VARIANT}/${DATASET}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
cd "${MODELS}"

for attempt in $(seq 1 3); do
    python train.py \
        --dataset "${DATASET}" \
        --dataset_root "${DS_ROOT}" \
        --train_list "${DS_ROOT}/list/train.txt" \
        --test_list "${DS_ROOT}/list/test.txt" \
        --pretrained_weight_path "${PROJ}/pretrained_weight/shvit_s1.pth" \
        --ckpt_dir "${CKPT_DIR}" \
        --max_steps "${MAX_STEPS}" \
        --batch_size "${BATCH}" \
        --test_batch_size 16 \
        --inWidth 256 \
        --inHeight 256 \
        --model_type tiny \
        --mode baseline \
        --arch casa_str \
        --attn_mode "${ATTN_MODE}" \
        --rep_mode "${REP_MODE}" \
        --backbone_lr_ratio 0.1 \
        --str_dim 160 \
        --casaa_keep_ratio 0.25 \
        --casaa_change_share 0.5 \
        --lr 2e-4 \
        --lr_mode poly \
        --num_workers 4 \
        --seed 16 \
        --gpu_id 0 \
        >> "${LOG_DIR}/train_log.txt" 2>&1
    rc=$?
    if [ ${rc} -eq 0 ]; then
        echo "[DONE] training+test finished successfully" >> "${LOG_DIR}/train_log.txt"
        break
    fi
    echo "[RETRY ${attempt}/3] crashed with exit ${rc}, resuming from last.pth in 10s" >> "${LOG_DIR}/train_log.txt"
    sleep 10
done
