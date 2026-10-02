#!/usr/bin/env bash
set -uo pipefail

# Run11 TASS M1_TASS（spatial_mode=tass） / CDD-CD-256 — 正式 80K 从头训练，GPU0；自动 retry 上限 3
GPU=0
DATASET=CDD-CD-256
VARIANT=M1_TASS
MAX_STEPS=80000
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/STR-Fusion/Run2_TASS/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/STR-Fusion/Run2_TASS/${VARIANT}/${DATASET}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
cd "${MODELS}"

for attempt in $(seq 1 3); do
    python train.py \
        --dataset "${DATASET}" \
        --dataset_root "${DS_ROOT}" \
        --train_list "${DS_ROOT}/list/train.txt" \
        --test_list "${DS_ROOT}/list/test.txt" \
        --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
        --ckpt_dir "${CKPT_DIR}" \
        --max_steps "${MAX_STEPS}" \
        --batch_size "${BATCH}" \
        --test_batch_size 16 \
        --inWidth 256 \
        --inHeight 256 \
        --model_type tiny \
        --mode baseline \
        --arch str_tass \
        --str_dim 160 \
        --spatial_mode tass \
        --freeze_vit 1 \
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
    echo "[RETRY ${attempt}/3] crashed with exit ${rc}, resuming from own last.pth in 10s" >> "${LOG_DIR}/train_log.txt"
    sleep 10
done