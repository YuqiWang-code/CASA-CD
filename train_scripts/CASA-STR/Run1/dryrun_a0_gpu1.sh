#!/usr/bin/env bash
set -uo pipefail

# CASA-STR Run1 dry-run：A0_BASE_PLAIN 短训（2 epochs，验证 LR-GROUPS 0.1x 与完整 TEST 链路）
GPU=1
DATASET=SYSU-CD-256
VARIANT=_dryrun/A0_BASE_PLAIN
MAX_STEPS=750
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
    --attn_mode none \
    --rep_mode plain \
    --backbone_lr_ratio 0.1 \
    --str_dim 160 \
    --lr 2e-4 \
    --lr_mode poly \
    --num_workers 4 \
    --seed 16 \
    --gpu_id 0 \
    >> "${LOG_DIR}/train_log.txt" 2>&1
