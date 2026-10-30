#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/environment.sh"
OUT=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run1
python "$HERE/prepare_runtime.py"
RUNTIME=$(python -c "import json; print(json.load(open('$OUT/runtime.json'))['directory'])")
cd "$RUNTIME"
for FILE in train_scripts/CASA-MoE/Run1/*.sh; do bash -n "$FILE"; done
python models_v1/check_MOE.py --output "$OUT/model_tests.json" > "$OUT/model_tests.log" 2>&1
python train_scripts/CASA-MoE/Run1/test_pipeline.py
python models_v1/profile_MOE.py --dataset-root /share_datasets/CD/LEVIR-CD-256 \
  --pretrained /home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth \
  --modules none hyperbolic qkv frequency equal moe no_hyperbolic no_qkv no_frequency \
  --output "$OUT/profile.json" > "$OUT/profile.log" 2>&1
python -c "import json; from pathlib import Path; root=Path('$OUT'); (root/'validation_result.json').write_text(json.dumps(dict(passed=True, runtime='$RUNTIME', physical_gpu=1), indent=2))"
echo '[VALIDATION-PASS]'
