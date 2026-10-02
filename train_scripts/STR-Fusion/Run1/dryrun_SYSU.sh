#!/usr/bin/env bash
set -uo pipefail

# STRFusion Run1 dry run — C0_Plain 真实 SYSU，2 个完整 epoch（1500 steps），
# 独立 scratch 目录。验收：exit 0 / loss 有限 / 冻结 ViT4 checksum / 无 resnet key /
# arch.json 正确 / eval 架构不匹配被拒绝 / TEST RESULTS 区块给出真实 batch-16
# 训练态下的 [REPARAM-MAX-ABS-ERROR] 与 [REPARAM-ARGMAX-DISAGREE]（80K 前健康判据）。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
DS_ROOT=/share_datasets/CD/SYSU-CD-256
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/STR-Fusion/Run1/C0_Plain/SYSU-CD-256
LOG_DIR=/home/yqwang/outputs/CASA-CD/_dryrun/STR-Fusion/Run1/C0_Plain/SYSU-CD-256
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

cd ${PROJ}/models
python train.py \
    --dataset SYSU-CD-256 --dataset_root "${DS_ROOT}" \
    --train_list "${DS_ROOT}/list/train.txt" --test_list "${DS_ROOT}/list/test.txt" \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_dir "${CKPT_DIR}" --max_steps 1500 --batch_size 16 --test_batch_size 16 \
    --inWidth 256 --inHeight 256 --model_type tiny --mode baseline \
    --arch strfusion --str_dim 160 --str_rep_mode plain --freeze_vit 1 \
    --lr 2e-4 --lr_mode poly --num_workers 4 --seed 16 --gpu_id 0 \
    >> "${LOG_DIR}/train_log.txt" 2>&1
echo "[dryrun-exit] $?"
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
python - <<'EOF'
import torch, os, json
ckpt_dir = "/share_datasets/yqwang/checkpoints/CASA-CD/_dryrun/STR-Fusion/Run1/C0_Plain/SYSU-CD-256"
cp = torch.load(os.path.join(ckpt_dir, "last.pth"), map_location="cpu", weights_only=False)["state_dict"]
pre = torch.load("/home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth", map_location="cpu")["model"]
key = "encoder.encoder.vit.blocks.3.attn.qkv.weight"
print("frozen ViT4 equal:", torch.equal(cp[key], pre["blocks.3.attn.qkv.weight"].float()))
print("resnet present:", any(k.startswith("encoder.encoder.resnet.") for k in cp))
print("trainable count:", sum(1 for k in cp if "vit" not in k))
print("arch.json:", json.load(open(os.path.join(ckpt_dir, "arch.json"), encoding="utf-8")))
EOF
# eval 架构不匹配应被拒绝（故意给错 str_rep_mode）
cd ${PROJ}/models
python eval.py \
    --dataset SYSU-CD-256 --dataset_root "${DS_ROOT}" --test_list "${DS_ROOT}/list/test.txt" \
    --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
    --ckpt_dir "${CKPT_DIR}" --model_type tiny --mode baseline --gpu_id 0 \
    --arch strfusion --str_dim 160 --str_rep_mode full > "${LOG_DIR}/eval_mismatch.txt" 2>&1
echo "[eval-mismatch-exit] $?  (expect nonzero)"
grep -o "ARCH-MISMATCH.*refusing to eval" "${LOG_DIR}/eval_mismatch.txt" | head -1
