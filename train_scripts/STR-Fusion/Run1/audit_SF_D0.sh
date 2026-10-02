#!/usr/bin/env bash
set -uo pipefail

# STRFusion Run1 SF-D0 — 深度即尺度接口零训练 gate（四数据集完整 test，GPU1）
# 读 [SF-D0-GATE] PASS/FAIL：FAIL → 不启动任何 80K，融合主线按预注册终止。
# 返回码：0=PASS / 2=FAIL / 3=AUDIT_INVALID。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
LOG_DIR=/home/yqwang/outputs/CASA-CD/diagnostics/STR-Fusion/Run1/SF_D0_INTERFACE_AUDIT
mkdir -p "${LOG_DIR}"
cd "${PROJ}"

python analyse/run1_strfusion_interface_audit.py \
    --data_root /share_datasets/CD \
    --datasets CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256 \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_r41_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
    --gpu_id 0 --batch_size 16 --num_workers 4 \
    > "${LOG_DIR}/audit_report.txt" 2>&1
echo "[sf-d0-exit] $?"
