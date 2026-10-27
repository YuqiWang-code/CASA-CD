#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run5
if pgrep -f '[/]CASA-Pyramid/Run5/health_probe.py|[/]CASA-Pyramid/Run5/pipeline.py' >/dev/null; then
  echo 'CASA-Pyramid Run5 probes or queue already running' >&2
  exit 3
fi
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
PIDS=()
trap 'for PID in "${PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done' EXIT
start_probe() {
  local MODE=$1 GPU=$2 DATASET=$3 NAME=$4 EPOCHS=$5 DIR="$OUT/probes/$4"
  mkdir -p "$DIR"
  CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/train_scripts/CASA-Pyramid/Run5/health_probe.py" \
    --module "$MODE" --dataset "$DATASET" --epochs "$EPOCHS" --output-dir "$DIR" > "$DIR/probe.log" 2>&1 &
  local PID=$!
  PIDS+=("$PID")
  echo "$PID" > "$DIR/probe.pid"
  echo "probe=$NAME pid=$PID gpu=$GPU"
}
wait_probes() {
  while (( ${#PIDS[@]} )); do
    local FINISHED
    wait -n -p FINISHED "${PIDS[@]}"
    local REMAINING=()
    for PID in "${PIDS[@]}"; do
      if [[ "$PID" != "$FINISHED" ]]; then REMAINING+=("$PID"); fi
    done
    PIDS=("${REMAINING[@]}")
  done
}
start_probe hyperbolic 0 LEVIR-CD-256 levir_hyperbolic 6
start_probe qkv 1 LEVIR-CD-256 levir_qkv 6
wait_probes
start_probe hyperbolic 0 SYSU-CD-256 sysu_hyperbolic 4
start_probe qkv 1 SYSU-CD-256 sysu_qkv 4
wait_probes
echo '[ALL-PROBES-PASS]'
