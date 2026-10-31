#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/environment.sh"
OUT=/home/yqwang/outputs/CASA-CD/CASA-MoE/Run3
RUNTIME=$(python -c "import json; print(json.load(open('$OUT/runtime.json'))['directory'])")
python "$RUNTIME/train_scripts/CASA-MoE/Run3/evaluate.py" \
  --variant "${1:?variant required}" --dataset "${2:?dataset required}"
