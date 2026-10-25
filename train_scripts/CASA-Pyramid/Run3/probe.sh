#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run3
if pgrep -f '[/]CASA-Pyramid/Run3/health_probe.py|[/]CASA-Pyramid/Run3/pipeline.py' >/dev/null; then
  echo 'CASA-Pyramid Run3 probes or queue already running' >&2
  exit 3
fi
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
probe() {
  local MODE=$1 GPU=$2 DATASET=$3 EPOCHS=$4 NAME=$5 DIR="$OUT/probes/$5"
  mkdir -p "$DIR"
  CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/train_scripts/CASA-Pyramid/Run3/health_probe.py" \
    --module "$MODE" --dataset "$DATASET" --epochs "$EPOCHS" --output-dir "$DIR" > "$DIR/probe.log" 2>&1 &
  local PID=$!
  echo "$PID" > "$DIR/probe.pid"
  echo "probe=$NAME pid=$PID gpu=$GPU"
  wait "$PID"
}
probe hyperbolic 0 LEVIR-CD-256 3 levir_hyperbolic &
H_LEVIR=$!
probe qkv 0 SYSU-CD-256 4 sysu_qkv &
E_SYSU=$!
probe qkv 1 LEVIR-CD-256 8 levir_qkv &
E_LEVIR=$!
probe hyperbolic 1 SYSU-CD-256 4 sysu_hyperbolic &
H_SYSU=$!
wait "$H_LEVIR"
wait "$E_SYSU"
wait "$E_LEVIR"
wait "$H_SYSU"
echo '[ALL-PROBES-PASS]'
