#!/usr/bin/env bash
set -uo pipefail

# Run9 R9-D0 — O-PRE Complementarity Audit（零训练，GPU1，四数据集完整 test）
# 读 [R9-D0-GATE] PASS/FAIL：FAIL → Run9 停止，0 个 80K。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
LOG_DIR=/home/yqwang/outputs/CASA-CD/UltraLight/Run9/R9_D0_OPRE_AUDIT
mkdir -p "${LOG_DIR}"
cd "${PROJ}"

python analyse/run9_overlap_reembedding_audit.py \
    --data_root /share_datasets/CD \
    --datasets CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256 \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_r41_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
    --gpu_id 0 --batch_size 16 --num_workers 4 \
    > "${LOG_DIR}/audit_report.txt" 2>&1
echo "[r9d0-exit] $?"
