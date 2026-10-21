#!/usr/bin/env bash
# Run10: one fresh 80K-step run. No automatic retry after numerical failure.
set -euo pipefail

VARIANT="${1:?variant required}"
DS="${2:?dataset required}"
GPU="${3:-0}"
case "$VARIANT" in
  S0_CTRL) HS=0 ;;
  S1_HSAM_LD) HS=1 ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac
case "$DS" in SYSU-CD-256|LEVIR-CD-256) ;; *) echo "unknown dataset: $DS" >&2; exit 2 ;; esac

REPO=/home/yqwang/projects/CASA-CD
DATA=/share_datasets/CD/$DS
LOG=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run10/$VARIANT/$DS
CKPT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run10/$VARIANT/$DS
PRET=$REPO/pretrained_weight
mkdir -p "$LOG" "$CKPT"
if [ -e "$LOG/train_log.txt" ] || [ -n "$(find "$CKPT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "refusing to overwrite existing Run10 artifacts: $VARIANT/$DS" >&2
  exit 3
fi
for p in "$DATA/list/train.txt" "$DATA/list/test.txt" "$PRET/deit_tiny_patch16_224-a1311bcf.pth" "$PRET/vssm_tiny_0230_ckpt_epoch_262.pth"; do
  [ -s "$p" ] || { echo "missing input: $p" >&2; exit 3; }
done

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PATH="/home/yqwang/miniforge3/envs/casacd/bin:$PATH"
export PYTHONPATH="/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO/models"
echo "[START] $VARIANT/$DS GPU=$GPU HSAM=$HS radius=local_dispersion" | tee "$LOG/train_log.txt"
set +e
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  --dataset "$DS" --dataset_root "$DATA" \
  --train_list "$DATA/list/train.txt" --test_list "$DATA/list/test.txt" \
  --pretrained_weight_path "$PRET/deit_tiny_patch16_224-a1311bcf.pth" \
  --vmamba_pretrained_weight_path "$PRET/vssm_tiny_0230_ckpt_epoch_262.pth" \
  --ckpt_dir "$CKPT" --max_steps 80000 --exact_max_steps 1 \
  --batch_size 32 --test_batch_size 16 --inWidth 256 --inHeight 256 \
  --model_type tiny --mode baseline --arch casa_vmamba_str_v2 \
  --hqkv 0 --hqkv_geometry lattice --hgsl 0 --hgsl_detach_score 1 \
  --hsam "$HS" --hsam_stages 2,3 --hsam_logit_norm std \
  --hsam_radius local_dispersion --hsam_feature_source pre_norm \
  --fs_tar 0 --rep_mode full --backbone_lr_ratio 0.1 --str_dim 192 \
  --lr 2e-4 --lr_mode poly --num_workers 4 --seed 16 --gpu_id 0 \
  >> "$LOG/train_log.txt" 2>&1
RC=$?
set -e
echo "[EXIT] $VARIANT/$DS rc=$RC" | tee -a "$LOG/train_log.txt"
[ "$RC" -eq 0 ] || exit "$RC"
grep -q '=== END TEST RESULTS ===' "$LOG/train_log.txt" || { echo "missing TEST block" >&2; exit 5; }
