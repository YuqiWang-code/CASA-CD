#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9
if pgrep -f '[/]CASA-Pyramid/Run9/pipeline.py' >/dev/null; then
  echo 'CASA-Pyramid Run9 is already running' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=1
mkdir -p "$OUT"
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
python "$REPO/train_scripts/CASA-Pyramid/Run9/configure_schedule.py"
if [[ ! -f "$OUT/source_code.tar.gz" ]]; then
  tar -czf "$OUT/source_code.tar.gz" -C "$REPO" models train_scripts/CASA-Pyramid/Run9
  python - <<'PY'
import json,platform,torch
from pathlib import Path
record=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
            physical_gpu=1,gpu=torch.cuda.get_device_name(0),cudnn=torch.backends.cudnn.version())
Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9/environment.json').write_text(json.dumps(record,indent=2))
PY
fi
setsid nohup python "$REPO/train_scripts/CASA-Pyramid/Run9/pipeline.py" \
  >> "$OUT/pipeline.log" 2>&1 < /dev/null &
PID=$!
sleep 5
kill -0 "$PID"
echo "pipeline pid=$PID"
tail -8 "$OUT/pipeline.log"
