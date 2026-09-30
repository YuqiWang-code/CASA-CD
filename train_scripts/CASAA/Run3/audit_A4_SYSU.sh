#!/usr/bin/env bash
set -uo pipefail

# CASAA Run3 — A4 Detail Router Audit（无训练）/ SYSU
# primary checkpoint: Run2 A1_SAA_FROZEN（detail branch 未被 GT 路由污染）
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export CUDA_VISIBLE_DEVICES=1

PROJ=/home/yqwang/projects/CASA-CD
DS=SYSU-CD-256
CK=$(ls /share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run2/A1_SAA_FROZEN/${DS}/best_F1=*.pth | head -1)
OUT=/home/yqwang/outputs/CASA-CD/CASAA/Run3/AUDIT/${DS}/router_audit.txt
mkdir -p "$(dirname ${OUT})"

cd ${PROJ}
python analyse/casaa_router_diagnostic.py \
    --dataset ${DS} --dataset_root /share_datasets/CD/${DS} \
    --pretrained_weight_path ${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth \
    --ckpt "${CK}" --ckpt_tag A1_SAA_FROZEN \
    --gpu_id 0 --batch_size 16 --num_workers 4 --out ${OUT}
