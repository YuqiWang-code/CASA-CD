#!/usr/bin/env bash
set -uo pipefail

# CASAA Run3 — A4-D detail-only（可部署救援实验，冻结 ViT）/ SYSU-CD-256 (GPU 1)
# Audit 结果：fused 未过 gate（PR-AUC +0.014 / Spearman -0.052），
# 但 detail-only 明显优于 ViT-only（PR-AUC +0.059 / Top32 precision +0.231 /
# coverage +0.213 / ROC +0.041）→ 按决策树 §34 只做这 1 个 SYSU detail-only 80K。
# score = rank(1 - cos(d̄1,d̄2))，detail 1/8（resnet.layer3 32×32 → pool → 16×16）。
GPU=1
DATASET=SYSU-CD-256
VARIANT=A4_DETAIL_FROZEN
MODE=casaa
ROUTER=detail
MAX_STEPS=80000
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run3/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASAA/Run3/${VARIANT}/${DATASET}

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
        --mode "${MODE}" \
        --casaa_layers 8,9,10,11 \
        --casaa_keep_ratio 0.25 \
        --casaa_change_share 0.50 \
        --casaa_router "${ROUTER}" \
        --freeze_vit 1 \
        --vit_lr_ratio 0.1 \
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
