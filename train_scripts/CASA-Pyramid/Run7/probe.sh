#!/usr/bin/env bash
set -euo pipefail
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export CUDA_VISIBLE_DEVICES=0
cd /home/yqwang/projects/CASA-CD/train_scripts/CASA-Pyramid/Run7
python -c 'from pipeline import check_selection; decision, _ = check_selection(); print("[SELECTION-PASS] " + decision["selected_arm"])'
