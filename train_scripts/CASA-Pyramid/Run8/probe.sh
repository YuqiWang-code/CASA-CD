#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
OUT=/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
grep -q '^\[VALIDATION-PASS\]' "$OUT/validation.log"
PIDS=()
trap 'for PID in "${PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done' EXIT
for DATASET in SYSU WHU; do
 for ARM in H_CLASS H_NONE; do
  RETRIEVAL=.1; if [[ "$ARM" == H_NONE ]]; then RETRIEVAL=0; fi
  DIR="$OUT/probes/$ARM/$DATASET"
  mkdir -p "$DIR"
  if [[ -f "$DIR/health_result.json" ]]; then
   python - "$DIR/health_result.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
assert value['health_passed'] and value['protocol']=='casa_pyramid_40k_attribution_v8'
PY
   continue
  fi
  python "$REPO/train_scripts/CASA-Pyramid/Run8/health_probe.py" --models-dir "$REPO/models" --module hyperbolic --dataset "$DATASET" --epochs 2 --retrieval-weight "$RETRIEVAL" --output-dir "$DIR" > "$DIR/probe.log" 2>&1 &
  PIDS+=("$!")
  echo "probe $ARM $DATASET pid=$!"
 done
 for PID in "${PIDS[@]}"; do wait "$PID"; done
 PIDS=()
done
python - <<'PY'
import json
from pathlib import Path
root=Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8')
reports={}
for arm in ('H_CLASS','H_NONE'):
 for dataset in ('SYSU','WHU'):
  path=root/'probes'/arm/dataset/'health_result.json'
  value=json.loads(path.read_text())
  assert value['health_passed'] and value['task_gradient_check'],str(path)
  reports[f'{arm}/{dataset}']=value
(root/'readiness.json').write_text(json.dumps(dict(protocol='casa_pyramid_40k_attribution_v8',phase='retrieval_loss',comparison='H_CLASS_vs_H_NONE',passed=True,criterion='finite losses and nonzero segmentation gradients; not an accuracy claim',probes=reports),indent=2))
PY
echo '[ALL-PROBES-PASS]'
