#!/usr/bin/env bash
set -uo pipefail

# Run6 R6-D0 — Semantic / Patch-Token Audit（零训练，GPU1，完整 SYSU test）
# 读 [R6-D0-GATE] PASS/FAIL；FAIL → Run6 路线永久停止（不实现/不训练 R6-1）。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
LOG_DIR=/home/yqwang/outputs/CASA-CD/UltraLight/Run6/R6_D0_SEMANTIC_TOKEN_AUDIT
mkdir -p "${LOG_DIR}"
cd "${PROJ}"

python analyse/run6_semantic_token_audit.py \
    --dataset_root /share_datasets/CD/SYSU-CD-256 \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_r41_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
    --ckpt_r40_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_0_A0_FULL12_FROZEN/SYSU-CD-256 \
    --gpu_id 0 --batch_size 16 --num_workers 4 \
    > "${LOG_DIR}/audit_report.txt" 2>&1
echo "[r6d0-exit] $?"
