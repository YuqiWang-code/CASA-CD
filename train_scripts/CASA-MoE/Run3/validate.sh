#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/environment.sh"
OUT=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run3

RUNTIME=$(python -c "import json; print(json.load(open('$OUT/runtime.json'))['directory'])")
cd "$RUNTIME"
for FILE in train_scripts/CASA-MoE/Run3/*.sh; do bash -n "$FILE"; done
python models_v1/check_candidates.py --output "$OUT/model_tests.json" > "$OUT/model_tests.log" 2>&1

python models_v1/profile_MOE.py --dataset-root /share_datasets/CD/LEVIR-CD-256 \
  --pretrained /home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth \
  --modules none static moe \
  --output "$OUT/profile.json" > "$OUT/profile.log" 2>&1
python -c "import json; from pathlib import Path; root=Path('$OUT'); (root/'validation_result.json').write_text(json.dumps(dict(passed=True, runtime='$RUNTIME', physical_gpu=0), indent=2))"
echo '[VALIDATION-PASS]'
