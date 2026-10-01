#!/usr/bin/env bash
set -uo pipefail

# Run7 R7-1 dry run — CSDP 真实 SYSU，~60 steps，独立 scratch 目录
# 验收：exit 0 / loss 有限 / 冻结 ViT2 checksum / 无 resnet / params≈1.178M。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
DS_ROOT=/share_datasets/CD/SYSU-CD-256
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/UltraLight/Run7/R7_1/SYSU-CD-256
LOG_DIR=/home/yqwang/outputs/CASA-CD/_dryrun/UltraLight/Run7/R7_1/SYSU-CD-256
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

cd ${PROJ}/models
python train.py \
    --dataset SYSU-CD-256 --dataset_root "${DS_ROOT}" \
    --train_list "${DS_ROOT}/list/train.txt" --test_list "${DS_ROOT}/list/test.txt" \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_dir "${CKPT_DIR}" --max_steps 60 --batch_size 16 --test_batch_size 16 \
    --inWidth 256 --inHeight 256 --model_type tiny --mode baseline \
    --vit_depth 2 --detail_mode depth_pyramid --head_mode csdp --freeze_vit 1 \
    --lr 2e-4 --lr_mode poly --num_workers 4 --seed 16 --gpu_id 0 \
    >> "${LOG_DIR}/train_log.txt" 2>&1
echo "[dryrun-exit] $?"
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
python - <<'EOF'
import torch
cp = torch.load("/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/UltraLight/Run7/R7_1/SYSU-CD-256/last.pth", map_location="cpu", weights_only=False)["state_dict"]
pre = torch.load("/home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth", map_location="cpu")["model"]
key = "encoder.vit.blocks.1.attn.qkv.weight"
print("frozen ViT2 equal:", torch.equal(cp[key], pre["blocks.1.attn.qkv.weight"].float()))
print("resnet present:", any(k.startswith("encoder.resnet.") for k in cp))
print("csdp head keys:", sum(1 for k in cp if k.startswith("decoder.")))
EOF
