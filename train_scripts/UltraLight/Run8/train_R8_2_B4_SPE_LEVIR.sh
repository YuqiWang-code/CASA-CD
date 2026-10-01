#!/usr/bin/env bash
set -uo pipefail

# Run8 R8-2 — B4_SPE / LEVIR（R8-1 SYSU 全过才跑；结构/超参完全冻结）
# Gate：F1>=91.50 & IoU>=84.33、params<=2.10M、FLOPs<=2.00G。
GPU=1
DATASET=LEVIR-CD-256
VARIANT=R8_1_B4_SPE
MAX_STEPS=80000
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run8/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/UltraLight/Run8/${VARIANT}/${DATASET}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
cd "${MODELS}"

for attempt in $(seq 1 200); do
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
        --vit_depth 4 \
        --detail_mode none_b4 \
        --head_mode b4_spe \
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
    echo "[RETRY ${attempt}] crashed with exit ${rc}, resuming from last.pth in 10s" >> "${LOG_DIR}/train_log.txt"
    sleep 10
done
