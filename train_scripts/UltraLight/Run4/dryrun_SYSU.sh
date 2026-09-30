#!/usr/bin/env bash
set -uo pipefail

# Run4 dry run — VIT4 真实 SYSU，~60 steps，独立 scratch 目录
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
DS_ROOT=/share_datasets/CD/SYSU-CD-256
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/Run4
LOG_DIR=/home/yqwang/outputs/CASA-CD/_dryrun/Run4
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

cd ${PROJ}/models
python train.py \
    --dataset SYSU-CD-256 --dataset_root "${DS_ROOT}" \
    --train_list "${DS_ROOT}/list/train.txt" --test_list "${DS_ROOT}/list/test.txt" \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_dir "${CKPT_DIR}" --max_steps 60 --batch_size 16 --test_batch_size 16 \
    --inWidth 256 --inHeight 256 --model_type tiny --mode baseline \
    --vit_depth 4 --freeze_vit 1 \
    --lr 2e-4 --lr_mode poly --num_workers 4 --seed 16 --gpu_id 0 \
    >> "${LOG_DIR}/train_log.txt" 2>&1
echo "[dryrun-exit] $?"
# ViT checksum：与预训练应完全一致（冻结）
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
python - <<'EOF'
import torch
cp = torch.load("/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/Run4/last.pth", map_location="cpu", weights_only=False)["state_dict"]
pre = torch.load("/home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth", map_location="cpu")["model"]
key = "encoder.vit.blocks.0.attn.qkv.weight"
print("viT qkv norm (ckpt):", round(cp[key].float().norm().item(), 4))
print("viT qkv norm (pretrained):", round(pre["blocks.0.attn.qkv.weight"].float().norm().item(), 4))
print("equal:", torch.equal(cp[key], pre["blocks.0.attn.qkv.weight"].float()))
EOF
