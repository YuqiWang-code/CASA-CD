#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
