#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"
mkdir -p "$OUT"
for file in train_scripts/CASA-Pyramid/Run1/*.sh; do bash -n "$file"; done
python models/test_pyramid_context.py
python models/test_CASA.py
python models/test_CASA_resume.py
python train_scripts/CASA-Pyramid/Run1/test_pipeline.py
python models/profile_CASA.py --dataset-root /share_datasets/CD/LEVIR-CD-256 \
  --pretrained "$REPO/pretrained_weight/mobilenet_v2-b0353104.pth" --output "$OUT/profile.json"
echo '[VALIDATION-PASS]'
