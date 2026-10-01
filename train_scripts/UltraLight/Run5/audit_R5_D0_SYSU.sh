#!/usr/bin/env bash
set -uo pipefail

# Run5 R5-D0 — postmortem（H1/H2/H3）+ MobileDetail-P3 raw gate（零训练，GPU1）
# 读 [MOBILE-GATE] PASS/FAIL：FAIL → Run5 立即停止（不再试其它 backbone prefix）。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
LOG_DIR=/home/yqwang/outputs/CASA-CD/UltraLight/Run5/R5_D0_DETAIL_AUDIT
mkdir -p "${LOG_DIR}"
cd "${PROJ}"

python analyse/run5_postmortem_and_mobile_audit.py \
    --dataset_root /share_datasets/CD/SYSU-CD-256 \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --mobile_pretrained_weight_path "${PROJ}/pretrained_weight/mobilenet_v3_small-047dcff4.pth" \
    --ckpt_r41_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
    --ckpt_r42_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_2_VIT4_LIGHTDETAIL/SYSU-CD-256 \
    --ckpt_r42d_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_2d_PSD_DETAIL/SYSU-CD-256 \
    --gpu_id 0 --batch_size 16 --num_workers 4 \
    > "${LOG_DIR}/audit_log.txt" 2>&1
echo "[audit-exit] $?"
