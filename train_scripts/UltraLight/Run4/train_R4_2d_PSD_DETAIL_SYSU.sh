#!/usr/bin/env bash
set -uo pipefail

# Run4 R4-2d — PSD_DETAIL / SYSU（R4-D0 主推荐路径）
# 唯一变量：detail branch 由 LightDetail(32/64/128)+adapters 换成
# PSD-Detail 0.078M（ImageNet pretrained stem + residual depthwise pyramid，
# 输出 64/128/256 与 FI 直接兼容，无 adapters）。
# 其余全部不变：ViT4 frozen、原 FI、原 decoder、80K、seed16。
# Gate：F1 >= 82.47 且 IoU >= 70.11（相对 R4-1 82.77 / 70.61，不降低标准）。
# 不过 gate → 停止当前轻量 detail 路线（不启动 SABI/DFPD）。
GPU=1
DATASET=SYSU-CD-256
VARIANT=R4_2d_PSD_DETAIL
MAX_STEPS=80000
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/UltraLight/Run4/${VARIANT}/${DATASET}

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
        --detail_mode psd \
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
