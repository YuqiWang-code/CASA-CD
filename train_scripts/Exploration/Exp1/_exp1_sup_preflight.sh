#!/usr/bin/env bash
# B 形态（STG_SUP）预检：接线自检 + epoch-0 逐位 + 3 步 batch32 dry-run（GPU1）
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
RC=0
ok(){ echo "[PASS] $*"; }; bad(){ echo "[FAIL] $*"; RC=1; }

echo "=== 1) 数学单测（回归，23 项）==="
( cd "$V1" && python test_st_geodesic.py > "$TMP/unit_sup.log" 2>&1 )
grep -q "RESULT: ALL OK" "$TMP/unit_sup.log" && ok "23/23 单测" || { bad "单测未全过"; tail -8 "$TMP/unit_sup.log"; }

echo
echo "=== 2) 接线自检 S1–S4（参数/deploy 删除/梯度到编码器/3 步联合训练）==="
( cd "$V1" && python check_sup_wiring.py --pretrained "$PRET" ) 2>&1 | tee "$TMP/wiring.log" | grep -E "^\[S[0-9]|^\[RESULT\]"
grep -q "^\[RESULT\] PASS" "$TMP/wiring.log" && ok "接线自检 PASS" || bad "接线自检 FAIL"

echo
echo "=== 3) epoch-0 与 CTRL_R8 逐位 ==="
( cd "$V1" && python epoch0_check.py --mode dump --pkg-root "$V1" --arch stgeo \
    --stgeo sup --vmamba "$PRET" --out "$TMP/v1_sup.pt" ) > "$TMP/d_sup.log" 2>&1 \
    && ok "sup dump" || { bad "sup dump"; tail -4 "$TMP/d_sup.log"; }
if [ -f "$TMP/ref_ctrl.pt" ]; then
  python "$V1/epoch0_check.py" --mode cmp --a "$TMP/ref_ctrl.pt" --b "$TMP/v1_sup.pt" \
    && ok "CTRL_R8 == sup（前向逐位）" || bad "sup 前向与 CTRL 不逐位"
else
  bad "缺 $TMP/ref_ctrl.pt（先跑 _exp1_identity_only.sh）"
fi

echo
echo "=== 4) 3 步真实 dry-run（CDD 数据仅作张量来源；STG_SUP 正式只在 LEVIR/SYSU 跑）==="
rm -rf "$TMP/dry_sup"; mkdir -p "$TMP/dry_sup"
( cd "$V1" && python train_stgeo.py \
    --dataset CDD-CD-256 --dataset_root "$DATA/CDD-CD-256" \
    --train_list "$DATA/CDD-CD-256/list/train.txt" \
    --test_list "$DATA/CDD-CD-256/list/test.txt" \
    --vmamba_pretrained_weight_path "$PRET" --ckpt_dir "$TMP/dry_sup" \
    --max_steps 3 --exact_max_steps 1 --batch_size 32 --test_batch_size 16 \
    --stgeo sup --stgeo_sup_small 16 --stgeo_sup_lambda 0.2 \
    --num_workers 2 --seed 16 --gpu_id 0 ) > "$TMP/dry_sup.log" 2>&1
rc=$?
[ $rc -eq 0 ] && ok "3 步 dry-run rc=0" || { bad "dry-run rc=$rc"; tail -25 "$TMP/dry_sup.log"; }
grep -E "^\[(STGEO|STGEO-GRAD|STGEO-LOSS|PRETRAIN-LOAD|PARAM-BREAKDOWN|ACTUAL-OPT-STEPS)\]" "$TMP/dry_sup.log" | head -8

echo
echo "=== B PREFLIGHT RESULT: $([ $RC -eq 0 ] && echo PASS || echo FAIL) ==="
exit $RC
