#!/usr/bin/env bash
set -u
REPO=/home/yqwang/projects/CASA-CD
cd "$REPO/models" || exit 1
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH="/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages:${PYTHONPATH:-}"
python run9_radius_probe.py \
  --dataset-root /share_datasets/CD/CDD-CD-256 \
  --train-list /share_datasets/CD/CDD-CD-256/list/train.txt \
  --vmamba-pretrained-weight-path "$REPO/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth" \
  --batch-size 16 --num-workers 4 2>&1 | grep -vE "UserWarning|warnings.warn"
