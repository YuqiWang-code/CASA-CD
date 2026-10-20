#!/usr/bin/env bash
# 拆除回归门：在服务器上跑（GPU 空着时才有 C6）。
set -u
cd /home/yqwang/projects/CASA-CD/models || exit 1
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH="/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
echo "=== $(date '+%F %T')  python=$(command -v python) ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/  /'
LATTICE_TEST_CUDA=1 python test_hqkv_lattice_cpu.py 2>&1 | tail -45
echo "=== rc=$? ==="
