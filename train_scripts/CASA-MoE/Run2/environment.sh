#!/usr/bin/env bash
# Source this file before running an exploration command.
set -euo pipefail
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH="/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
