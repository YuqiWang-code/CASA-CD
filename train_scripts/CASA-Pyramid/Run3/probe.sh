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
worker() {
  local MODE=$1 GPU=$2 DATASET EPOCHS NAME DIR
  for spec in 'LEVIR-CD-256 3' 'SYSU-CD-256 4'; do
    read -r DATASET EPOCHS <<< "$spec"
    NAME="${DATASET%%-*}"
    NAME="${NAME,,}_$MODE"
    DIR="$OUT/probes/$NAME"
    mkdir -p "$DIR"
    CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/train_scripts/CASA-Pyramid/Run3/health_probe.py"       --module "$MODE" --dataset "$DATASET" --epochs "$EPOCHS" --output-dir "$DIR"       > "$DIR/probe.log" 2>&1 &
    local PID=$!
    echo "$PID" > "$DIR/probe.pid"
    echo "probe=$NAME pid=$PID gpu=$GPU"
    wait "$PID" || return $?
  done
}
lr_probe() {
  local MULT=$1 GPU=$2 NAME=$3 DIR="$OUT/probes/$3"
  mkdir -p "$DIR"
  CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/train_scripts/CASA-Pyramid/Run3/health_probe.py" \
    --module none --dataset SYSU-CD-256 --epochs 12 --encoder-lr-mult "$MULT" \
    --validation-every-epoch --output-dir "$DIR" > "$DIR/probe.log" 2>&1 &
  local PID=$!
  echo "$PID" > "$DIR/probe.pid"
  echo "probe=$NAME pid=$PID gpu=$GPU"
  wait "$PID"
}
lr_probe 1.0 0 sysu_ctrl_lr1 &
LR1=$!
lr_probe 0.1 1 sysu_ctrl_lr01 &
LR01=$!
worker hyperbolic 0 &
HYPER=$!
worker qkv 1 &
EUCLIDEAN=$!
wait "$HYPER"
wait "$EUCLIDEAN"
wait "$LR1"
wait "$LR01"
echo '[ALL-PROBES-PASS]'
