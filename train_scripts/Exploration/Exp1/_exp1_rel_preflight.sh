#!/usr/bin/env bash
# STG_REL（方向 A）预检：epoch-0 逐位 + 参数/部署 + 3 步真实 dry-run（GPU1）
set -u
GPU="${1:-1}"
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
CA_ENV=/home/yqwang/miniforge3/envs/casacd
MAMBA_SITE=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate casacd 2>/dev/null || true; fi
export PATH="$CA_ENV/bin:${PATH}"
export PYTHONPATH="${MAMBA_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA_ENV/lib/python3.10/site-packages/torch/lib:$CA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU
REPO=/home/yqwang/projects/CASA-CD
V1=$REPO/models_v1
PRET=$REPO/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth
DATA=/share_datasets/CD
TMP=/tmp/exp1; mkdir -p "$TMP"
RC=0; ok(){ echo "[PASS] $*"; }; bad(){ echo "[FAIL] $*"; RC=1; }

echo "=== 1) 23 项数学单测（回归）==="
( cd "$V1" && python test_st_geodesic.py > "$TMP/unit_rel.log" 2>&1 )
grep -q "RESULT: ALL OK" "$TMP/unit_rel.log" && ok "单测全过" || { bad "单测"; tail -6 "$TMP/unit_rel.log"; }

echo "=== 2) epoch-0 与 CTRL_R8 逐位（rel 形态）==="
( cd "$V1" && python epoch0_check.py --mode dump --pkg-root "$V1" --arch stgeo \
    --stgeo rel --stgeo-geom hyperbolic --vmamba "$PRET" --out "$TMP/v1_rel.pt" ) \
    > "$TMP/d_rel.log" 2>&1 && ok "rel dump" || { bad "rel dump"; tail -4 "$TMP/d_rel.log"; }
if [ -f "$TMP/ref_ctrl.pt" ] && [ -f "$TMP/v1_rel.pt" ]; then
  python "$V1/epoch0_check.py" --mode cmp --a "$TMP/ref_ctrl.pt" --b "$TMP/v1_rel.pt" \
    && ok "CTRL_R8 == rel 逐位" || bad "rel 与 CTRL 不逐位"
else bad "缺 ref 或 rel dump"; fi

echo "=== 3) 3 步真实 dry-run（CDD 张量，正式只在 LEVIR/SYSU）==="
rm -rf "$TMP/dry_rel"; mkdir -p "$TMP/dry_rel"
( cd "$V1" && python train_stgeo.py \
    --dataset CDD-CD-256 --dataset_root "$DATA/CDD-CD-256" \
    --train_list "$DATA/CDD-CD-256/list/train.txt" \
    --test_list "$DATA/CDD-CD-256/list/test.txt" \
    --vmamba_pretrained_weight_path "$PRET" --ckpt_dir "$TMP/dry_rel" \
    --max_steps 3 --exact_max_steps 1 --batch_size 32 --test_batch_size 16 \
    --stgeo rel --stgeo_geom hyperbolic --stgeo_detail_only 1 \
    --num_workers 2 --seed 16 --gpu_id 0 ) > "$TMP/dry_rel.log" 2>&1
rc=$?; [ $rc -eq 0 ] && ok "dry-run rc=0" || { bad "dry-run rc=$rc"; tail -20 "$TMP/dry_rel.log"; }
grep -E "^\[(STGEO|PARAM-BREAKDOWN|PRETRAIN-LOAD|ACTUAL-OPT-STEPS)\]" "$TMP/dry_rel.log" | head -6
echo "=== REL PREFLIGHT: $([ $RC -eq 0 ] && echo PASS || echo FAIL) ==="
exit $RC
