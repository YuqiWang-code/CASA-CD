#!/usr/bin/env bash
set -uo pipefail

# Run7 R7-D0 — 四数据集 Change-Sensitive Depth Audit（零训练，GPU1）
# 读 [R7-D0-GATE] PASS/FAIL：任一条件 <3/4 数据集 → CSDP-CD 停止。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
LOG_DIR=/home/yqwang/outputs/CASA-CD/UltraLight/Run7/R7_D0_DEPTH_AUDIT
mkdir -p "${LOG_DIR}"
cd "${PROJ}"

python analyse/run7_depth_source_audit.py \
    --data_root /share_datasets/CD \
    --datasets CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256 \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_r41_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
    --gpu_id 0 --batch_size 16 --num_workers 4 \
    > "${LOG_DIR}/audit_report.txt" 2>&1
echo "[r7d0-exit] $?"
