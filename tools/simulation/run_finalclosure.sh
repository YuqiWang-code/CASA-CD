#!/bin/bash
# Re-run the corrected S9a / S8b simulations for the Run10 frozen checkpoints.
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10
HERE=$ROOT/analysis/simulation/FinalClosure
SCRIPTS=$HERE/scripts
source $ROOT/runtime/train_scripts/CASA-MinorGoal/Run10/environment.sh
export PYTHONPATH=$PYTHONPATH:$ROOT/runtime/models:$ROOT/runtime/train_scripts/CASA-MinorGoal/Run10
export CUDA_VISIBLE_DEVICES=
export S7B_THREADS=4
cd $SCRIPTS || exit 1
for CK in R_FULL-2334 R_FULL-2335; do
  for DS in SYSU-CD-256 LEVIR-CD-256; do
    echo "=== $CK $DS start $(date) ==="
    python s9a_fix.py --arm $CK --dataset $DS --images 48 --mode fit --out $HERE/S9a || exit 1
    python s9a_fix.py --arm $CK --dataset $DS --images 48 --mode arms --out $HERE/S9a || exit 1
    for GS in b224 b128 b64 b32; do
      python s9a_fix.py --arm $CK --dataset $DS --images 48 --mode grids --gridset $GS --out $HERE/S9a || exit 1
    done
    python s8b_fix.py --arm $CK --dataset $DS --images 12 --out $HERE/S8b || exit 1
    echo "=== $CK $DS done $(date) ==="
  done
done
echo "=== ALL DONE $(date) ==="
