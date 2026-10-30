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
cd "$REPO"
for FILE in train_scripts/CASA-Pyramid/Run8/*.sh; do bash -n "$FILE"; done
python models/test_hyperbolic_context.py
python models/test_direct_geometry.py
python models/test_attribution.py
python models/test_CASA.py
python models/test_CASA_resume.py
python train_scripts/CASA-Pyramid/Run8/test_pipeline.py
until grep -q '^\[FULL-VAL-BRANCHES-COMPLETE\]' "$OUT/branch_analysis.log"; do
 if ! kill -0 "$(cat "$OUT/branch_analysis.pid")" 2>/dev/null; then echo 'Frozen branch analysis did not complete'; exit 3; fi
 sleep 10
done
for ARM in H_CLASS H_NONE; do
 RETRIEVAL=0.1; if [[ "$ARM" == H_NONE ]]; then RETRIEVAL=0; fi
 python models/profile_CASA.py --dataset-root /share_datasets/CD/SYSU-CD-256 --pretrained "$REPO/pretrained_weight/mobilenet_v2-b0353104.pth" --modules hyperbolic --guide-weight .1 --retrieval-weight "$RETRIEVAL" --refinement-beta 0 --output "$OUT/profile_$ARM.json"
done
python - <<'PY'
import json
from pathlib import Path
root=Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8')
result={arm:json.loads((root/f'profile_{arm}.json').read_text())['hyperbolic'] for arm in ('H_CLASS','H_NONE')}
(root/'profile.json').write_text(json.dumps(result,indent=2))
PY
echo '[VALIDATION-PASS]'
