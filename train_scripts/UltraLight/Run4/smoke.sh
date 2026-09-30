#!/usr/bin/env bash
set -uo pipefail
# Run4 smoke：全部模式 + run4 深度测试
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export CUDA_VISIBLE_DEVICES=1
cd /home/yqwang/projects/CASA-CD/models
python smoke_test.py \
    --pretrained_weight_path /home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth \
    --gpu_id 0 --mode all 2>&1 | grep -vE 'UserWarning|warnings.warn|xFormers|fvcore|initialize|using MLP|missing_keys|unexpected_keys|model_type|checkpoint_path|Unsupported operator|submodules|encoder.resnet|trace|FutureWarning' | tail -60
