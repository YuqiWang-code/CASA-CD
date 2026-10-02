#!/usr/bin/env bash
set -uo pipefail

# Run11 TASS-D0 — raw source feasibility gate（零训练，GPU1，四数据集完整 test）
# 读 [TASS-D0-GATE] PASS/FAIL：FAIL → 不实现/不训练 TASS，转方向 (c)。
# 返回码：0=PASS / 2=FAIL / 3=AUDIT_INVALID。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
LOG_DIR=/home/yqwang/outputs/CASA-CD/diagnostics/STR-Fusion/Run2_TASS/TASS_D0_SOURCE_GATE
mkdir -p "${LOG_DIR}"
cd "${PROJ}"

python analyse/run11_tass_source_gate.py \
    --data_root /share_datasets/CD \
    --datasets CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256 \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --out_dir "${LOG_DIR}" \
    --gpu_id 0 --batch_size 16 --num_workers 4 \
    > "${LOG_DIR}/audit_report.txt" 2>&1
echo "[tass-d0-exit] $?"
