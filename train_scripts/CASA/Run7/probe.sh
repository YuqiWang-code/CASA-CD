#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA/Run7
if pgrep -f '[/]CASA/Run7/health_probe.py|[/]CASA/Run7/pipeline.py' >/dev/null; then
  echo 'Run7 probes or formal queue are already running' >&2
  exit 3
fi
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
python - <<'PY'
import json, subprocess
from pathlib import Path
profile=json.loads(Path('/home/yqwang/outputs/CASA-CD/CASA/Run7/profile.json').read_text())
free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
available={int(row.split(',')[0]):int(row.split(',')[1])-1024 for row in free.splitlines()}
for gpu,mode in ((0,'hyperbolic'),(1,'qkv')):
    required=2*(profile[mode]['peak_reserved_mib']+profile[mode]['context_overhead_mib'])
    if required>available[gpu]:
        raise RuntimeError(f'GPU{gpu} lacks memory for two {mode} probes: {required}>{available[gpu]}MiB')
PY
for item in 'levir_hyperbolic hyperbolic LEVIR-CD-256 3 0' 'sysu_hyperbolic hyperbolic SYSU-CD-256 4 0' 'levir_qkv qkv LEVIR-CD-256 3 1' 'sysu_qkv qkv SYSU-CD-256 4 1'; do
  read -r NAME MODE DATASET EPOCHS GPU <<< "$item"
  DIR="$OUT/probes/$NAME"
  mkdir -p "$DIR"
  CUDA_VISIBLE_DEVICES="$GPU" setsid nohup python "$REPO/train_scripts/CASA/Run7/health_probe.py" \
    --module "$MODE" --dataset "$DATASET" --epochs "$EPOCHS" --output-dir "$DIR" \
    > "$DIR/probe.log" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$DIR/probe.pid"
  echo "probe=$NAME pid=$PID gpu=$GPU epochs=$EPOCHS"
done
