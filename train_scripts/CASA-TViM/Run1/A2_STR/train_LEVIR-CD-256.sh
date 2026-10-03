#!/usr/bin/env bash
set -uo pipefail

# CASA-TViM Run1 A2_STR — caacp=0 rep=full / LEVIR-CD-256（正式 80K，batch 32，GPU0；自动 retry 上限 3）
GPU=0
DATASET=LEVIR-CD-256
VARIANT=A2_STR
CAACP=0
REP=full
MAX_STEPS=80000
BATCH=32

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# selective_scan_cuda_oflex(.so) 链接 torch 动态库所需（自建 kernel 无内嵌 rpath）
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run1/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run1/${VARIANT}/${DATASET}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
cd "${MODELS}"

for attempt in $(seq 1 3); do
    python train.py \
        --dataset "${DATASET}" \
        --dataset_root "${DS_ROOT}" \
        --train_list "${DS_ROOT}/list/train.txt" \
        --test_list "${DS_ROOT}/list/test.txt" \
        --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
        --tinyvim_pretrained_weight_path "${PROJ}/pretrained_weight/tinyvim_s_1000e.pth" \
        --ckpt_dir "${CKPT_DIR}" \
        --max_steps "${MAX_STEPS}" \
        --batch_size "${BATCH}" \
        --test_batch_size 16 \
        --inWidth 256 \
        --inHeight 256 \
        --model_type tiny \
        --mode baseline \
        --arch casa_tvim_str \
        --caacp "${CAACP}" \
        --rep_mode "${REP}" \
        --backbone_lr_ratio 0.1 \
        --str_dim 96 \
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
