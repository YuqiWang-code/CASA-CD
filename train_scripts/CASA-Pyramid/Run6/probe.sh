#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run6
if pgrep -f '[/]CASA-Pyramid/Run6/health_probe.py|[/]CASA-Pyramid/Run6/pipeline.py' >/dev/null; then
  echo 'Run6 probes or queue already running' >&2
  exit 3
fi
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
PIDS=()
trap 'for PID in "${PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done' EXIT
start_probe() {
  local DATASET=$1 NAME=$2 EPOCHS=$3 DIR="$OUT/probes/$2"
  mkdir -p "$DIR"
  python "$REPO/train_scripts/CASA-Pyramid/Run6/health_probe.py" --module hyperbolic \
    --dataset "$DATASET" --epochs "$EPOCHS" --output-dir "$DIR" > "$DIR/probe.log" 2>&1 &
  local PID=$!
  PIDS+=("$PID")
  echo "$PID" > "$DIR/probe.pid"
  echo "probe=$NAME pid=$PID gpu=0"
}
PARALLEL=$(python - <<'PY'
import json, subprocess
from pathlib import Path
p=json.loads(Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run6/profile.json').read_text())['hyperbolic']
free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
print(int(2*(p['peak_reserved_mib']+p['context_overhead_mib']) <= free-1024))
PY
)
start_probe LEVIR-CD-256 levir_hyperbolic 6
if [[ "$PARALLEL" == 0 ]]; then wait "${PIDS[0]}"; PIDS=(); fi
start_probe SYSU-CD-256 sysu_hyperbolic 4
for PID in "${PIDS[@]}"; do wait "$PID"; done
PIDS=()
echo '[ALL-PROBES-PASS]'
