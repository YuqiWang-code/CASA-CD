#!/usr/bin/env bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
export CUDA_VISIBLE_DEVICES=1
