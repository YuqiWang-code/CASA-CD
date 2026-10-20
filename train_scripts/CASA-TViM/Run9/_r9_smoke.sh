#!/usr/bin/env bash
# Run9 HSAM 真实数据 smoke（服务器侧）。不训练、不写 ckpt。
set -u
REPO=/home/yqwang/projects/CASA-CD
OUT=/tmp/run9/smoke
mkdir -p "$OUT"
cd "$REPO/models" || exit 1
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH="/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages:${PYTHONPATH:-}"

echo "=== $(date '+%F %T') ==="
echo "--- GPU 现状（只看，不动）---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'

echo "--- CPU 回归门 ---"
python test_hsam_cpu.py 2>&1 | grep -E "^===|FAILED" | tail -3

echo "--- 真实数据 smoke（CDD）---"
python run9_smoke.py \
  --dataset-root /share_datasets/CD/CDD-CD-256 \
  --train-list /share_datasets/CD/CDD-CD-256/list/train.txt \
  --vmamba-pretrained-weight-path "$REPO/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth" \
  --output-dir "$OUT" \
  --steps 6 --batch-size 32 --num-workers 4 \
  2>&1 | grep -vE "UserWarning|warnings.warn" | tail -40
