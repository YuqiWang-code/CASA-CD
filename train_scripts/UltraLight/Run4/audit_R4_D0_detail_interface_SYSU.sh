#!/usr/bin/env bash
set -uo pipefail

# Run4 R4-D0 — Detail Interface Audit（零训练成本，只用 GPU1 前向）
# 用 R4-1 / R4-2 / R4-2b best checkpoint 在完整 SYSU test 上对比
# ResNet direct vs Light raw vs Light adapted（1/2、1/4、1/8 三尺度），
# 按预注册规则（决策文档 §4）判定：A → R4-2c ADAPTER_ALIGN；B/C → R4-2d PSD_DETAIL。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
LOG_DIR=/home/yqwang/outputs/CASA-CD/UltraLight/Run4/R4_D0_DETAIL_AUDIT
mkdir -p "${LOG_DIR}"
cd "${PROJ}"

python analyse/run4_detail_interface_audit.py \
    --dataset_root /share_datasets/CD/SYSU-CD-256 \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_r41_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
    --ckpt_r42_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_2_VIT4_LIGHTDETAIL/SYSU-CD-256 \
    --ckpt_r42b_dir /share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/R4_2b_VIT4_LIGHTDETAIL48/SYSU-CD-256 \
    --gpu_id 0 --batch_size 16 --num_workers 4 \
    --out "${LOG_DIR}/audit_report.txt" \
    > "${LOG_DIR}/audit_log.txt" 2>&1
echo "[audit-exit] $?"
