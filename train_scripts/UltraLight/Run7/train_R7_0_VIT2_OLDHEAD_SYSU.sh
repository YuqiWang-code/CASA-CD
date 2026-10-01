#!/usr/bin/env bash
set -uo pipefail

# Run7 R7-0 — VIT2_OLDHEAD / SYSU（Source Causality：唯一变量 = ViT depth 4→2）
# 其余与 R4-1 完全一致：ResNet detail + 原 FI + 原 decoder，frozen ViT2、80K、seed16。
# Gate：F1 >= 82.47（drop<=0.30）且 IoU >= 70.26（drop<=0.35）。
# 不过 → 「token ranking 好 ≠ 最终 dense 表示好」，方向停止（不实现 CSDP head）。
GPU=1
DATASET=SYSU-CD-256
VARIANT=R7_0_VIT2_OLDHEAD
MAX_STEPS=80000
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run7/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/UltraLight/Run7/${VARIANT}/${DATASET}

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
        --vit_depth 2 \
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
