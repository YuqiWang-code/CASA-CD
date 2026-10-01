#!/usr/bin/env bash
set -uo pipefail

# Run5 R5-1 dry run — MOBILEDETAIL_OLDHEAD 真实 SYSU，~120 steps，独立 scratch 目录
# 验收：exit 0 / loss 有限 / 冻结 ViT checksum / Mobile exact-load / adapters 注册。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
DS_ROOT=/share_datasets/CD/SYSU-CD-256
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/UltraLight/Run5/R5_1/SYSU-CD-256
LOG_DIR=/home/yqwang/outputs/CASA-CD/_dryrun/UltraLight/Run5/R5_1/SYSU-CD-256
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

cd ${PROJ}/models
python train.py \
    --dataset SYSU-CD-256 --dataset_root "${DS_ROOT}" \
    --train_list "${DS_ROOT}/list/train.txt" --test_list "${DS_ROOT}/list/test.txt" \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_dir "${CKPT_DIR}" --max_steps 120 --batch_size 16 --test_batch_size 16 \
    --inWidth 256 --inHeight 256 --model_type tiny --mode baseline \
    --vit_depth 4 --detail_mode mobile_p3 --head_mode legacy \
    --mobile_pretrained_weight_path "${PROJ}/pretrained_weight/mobilenet_v3_small-047dcff4.pth" \
    --resnet_pretrained 1 --freeze_vit 1 \
    --lr 2e-4 --lr_mode poly --num_workers 4 --seed 16 --gpu_id 0 \
    >> "${LOG_DIR}/train_log.txt" 2>&1
echo "[dryrun-exit] $?"
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
python - <<'EOF'
import torch
cp = torch.load("/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/UltraLight/Run5/R5_1/SYSU-CD-256/last.pth", map_location="cpu", weights_only=False)["state_dict"]
pre = torch.load("/home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth", map_location="cpu")["model"]
key = "encoder.vit.blocks.0.attn.qkv.weight"
print("frozen ViT equal:", torch.equal(cp[key], pre["blocks.0.attn.qkv.weight"].float()))
mob = torch.load("/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v3_small-047dcff4.pth", map_location="cpu")
k0 = "encoder.detail.f0.0.weight"
print("mobile f0 in ckpt:", k0 in cp, "| equal:", torch.equal(cp[k0], mob["features.0.0.weight"]))
print("adapters in ckpt:", any("detail_adapters" in k for k in cp))
EOF
