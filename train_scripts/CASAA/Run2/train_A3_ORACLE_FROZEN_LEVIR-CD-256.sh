#!/usr/bin/env bash
set -uo pipefail

# CASAA Run2 — A3 Oracle routing（DIAGNOSTIC-ONLY，冻结 ViT）/ LEVIR-CD-256 (GPU 1)
# change score 用 GT patch occupancy（机制上界，不可部署）；ViT 冻结保证路由作用在
# 健康的 pretrained ViT 上。其余与 A2 一致：K=64、Kc=32、Kb=32、blocks 8-11。
GPU=1
DATASET=LEVIR-CD-256
VARIANT=A3_ORACLE_FROZEN
MODE=casaa
ROUTER=oracle
MAX_STEPS=80000
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run2/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASAA/Run2/${VARIANT}/${DATASET}

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
