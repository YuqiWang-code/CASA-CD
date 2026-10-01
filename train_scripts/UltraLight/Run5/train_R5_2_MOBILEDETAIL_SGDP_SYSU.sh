#!/usr/bin/env bash
set -uo pipefail

# Run5 R5-2 — MOBILEDETAIL_SGDP / SYSU（唯一变量 = head：旧 FI+decoder → SGDP）
# MobileDetail-P3 原始 16/16/24 输出直接进 SGDP（adapters 彻底移除）；
# ViT4 frozen、80K、seed16。终 gate：F1>=82.30 & IoU>=69.92（绝对）、
# ΔF1>=-0.25 & ΔIoU>=-0.40（相对 R5-1）、effective<=2.11M、FLOPs<=2.0G。
# 任一不过 → SGDP 永久停止，Run5 结束。
GPU=1
DATASET=SYSU-CD-256
VARIANT=R5_2_MOBILEDETAIL_SGDP
MAX_STEPS=80000
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run5/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/UltraLight/Run5/${VARIANT}/${DATASET}

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
        --detail_mode mobile_p3 \
        --head_mode sgdp \
        --mobile_pretrained_weight_path "${PROJ}/pretrained_weight/mobilenet_v3_small-047dcff4.pth" \
        --resnet_pretrained 1 \
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
