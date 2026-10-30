#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/environment.sh"
OUT=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run2
RUNTIME=$(python -c "import json; print(json.load(open('$OUT/runtime.json'))['directory'])")
python "$RUNTIME/train_scripts/CASA-MoE/Run2/evaluate.py" \
  --variant "${1:?variant required}" --dataset "${2:?dataset required}"
