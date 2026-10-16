#!/usr/bin/env bash
# Exp1 服务器预检（只读 + 3 步真实 dry-run；全部在 GPU1，绝不碰 GPU0 的 Run8）
#   bash _exp1_preflight.sh [GPU]       默认 GPU=1
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

REPO=/home/yqwang/projects/CASA-CD
V1=${REPO}/models_v1
REF=${REPO}/models
DATA=/share_datasets/CD
PRET=${REPO}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth
TMP=/tmp/exp1
mkdir -p "$TMP"
RC=0
ok(){ echo "[PASS] $*"; }
bad(){ echo "[FAIL] $*"; RC=1; }

echo "=== 0) 文件到位 / 行尾 ==="
for f in "$V1/train_stgeo.py" "$V1/model/casa_vmamba_stgeo_net.py" \
         "$V1/model/layers/st_geodesic.py" "$V1/epoch0_check.py"; do
  [ -s "$f" ] && ok "present $(basename "$f")" || bad "missing $f"
done
if file "$V1/train_stgeo.py" | grep -q CRLF; then bad "train_stgeo.py 是 CRLF"; else ok "LF 行尾"; fi

echo
echo "=== 1) 纯 CPU 数学单测（st_geodesic）==="
( cd "$V1" && python test_st_geodesic.py > "$TMP/unit.log" 2>&1 )
if grep -q "RESULT: ALL OK" "$TMP/unit.log"; then ok "20/20 单测通过"
else bad "单测未全过（见 $TMP/unit.log）"; tail -20 "$TMP/unit.log"; fi

echo
echo "=== 2) epoch-0 恒等性：Run8 CTRL 架构 vs 本探索架构(stgeo=none/inject) ==="
( cd "$REF" && CUDA_VISIBLE_DEVICES=$GPU python "$V1/epoch0_check.py" --mode dump \
    --pkg-root "$REF" --arch run8 --vmamba "$PRET" --out "$TMP/ref_ctrl.pt" ) \
    > "$TMP/d_ref.log" 2>&1 && ok "Run8 CTRL dump" || { bad "Run8 CTRL dump"; tail -5 "$TMP/d_ref.log"; }
( cd "$V1" && CUDA_VISIBLE_DEVICES=$GPU python epoch0_check.py --mode dump \
    --pkg-root "$V1" --arch stgeo --stgeo none --vmamba "$PRET" --out "$TMP/v1_none.pt" ) \
    > "$TMP/d_none.log" 2>&1 && ok "v1 stgeo=none dump" || { bad "v1 none dump"; tail -5 "$TMP/d_none.log"; }
( cd "$V1" && CUDA_VISIBLE_DEVICES=$GPU python epoch0_check.py --mode dump \
    --pkg-root "$V1" --arch stgeo --stgeo inject --vmamba "$PRET" --out "$TMP/v1_inj.pt" ) \
    > "$TMP/d_inj.log" 2>&1 && ok "v1 stgeo=inject dump" || { bad "v1 inject dump"; tail -5 "$TMP/d_inj.log"; }
grep -h "EPOCH0-DUMP" "$TMP"/d_*.log
python "$V1/epoch0_check.py" --mode cmp --a "$TMP/ref_ctrl.pt" --b "$TMP/v1_none.pt" \
  && ok "CTRL_R8 ≡ v1(stgeo=none) 逐位" || bad "CTRL_R8 与 v1(stgeo=none) 不逐位一致"
python "$V1/epoch0_check.py" --mode cmp --a "$TMP/v1_none.pt" --b "$TMP/v1_inj.pt" \
  && ok "v1(none) ≡ v1(inject) 逐位（γ=0 门控生效）" || bad "γ 零初始化未生效"

echo
echo "=== 3) 3 步真实 dry-run（CDD，GPU$GPU，ckpt 落 /tmp）==="
rm -rf "$TMP/dry_ckpt"; mkdir -p "$TMP/dry_ckpt"
( cd "$V1" && CUDA_VISIBLE_DEVICES=$GPU python train_stgeo.py \
    --dataset CDD-CD-256 --dataset_root "$DATA/CDD-CD-256" \
    --train_list "$DATA/CDD-CD-256/list/train.txt" \
    --test_list "$DATA/CDD-CD-256/list/test.txt" \
    --vmamba_pretrained_weight_path "$PRET" --ckpt_dir "$TMP/dry_ckpt" \
    --max_steps 3 --exact_max_steps 1 --batch_size 32 --test_batch_size 16 \
    --stgeo inject --num_workers 2 --seed 16 --gpu_id 0 ) > "$TMP/dry.log" 2>&1
rc=$?
[ $rc -eq 0 ] && ok "3 步 dry-run rc=0" || { bad "dry-run rc=$rc"; tail -25 "$TMP/dry.log"; }
grep -E "^\[(PRETRAIN-LOAD|STGEO|PARAM-BREAKDOWN|TOTAL-PARAMS|ACTUAL-OPT-STEPS)\]" "$TMP/dry.log" | head -12
python - "$TMP/dry.log" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
p = re.findall(r"\[PARAM-BREAKDOWN\][^\n]*", txt)
print(p[0] if p else "[PARAM-BREAKDOWN] (none)")
g = re.findall(r"\[STGEO\][^\n]*", txt)
print(g[0] if g else "[STGEO] (none)")
a = re.findall(r"\[ACTUAL-OPT-STEPS\][^\n]*", txt)
print(a[-1] if a else "[ACTUAL-OPT-STEPS] (none)")
PY

echo
echo "=== 4) 显存占用（batch32，真实 step）==="
grep -E "dry-run|CUDA out of memory|OutOfMemory" "$TMP/dry.log" | head -3
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

echo
echo "=== PREFLIGHT RESULT: $([ $RC -eq 0 ] && echo PASS || echo FAIL) ==="
exit $RC
