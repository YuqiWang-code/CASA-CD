#!/usr/bin/env bash
# 只做 epoch-0 身份核对（batch 4，约 3–4 GB，可在余量小的卡上跑）
set -u
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
CA_ENV=/home/yqwang/miniforge3/envs/casacd
MAMBA_SITE=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate casacd 2>/dev/null || true; fi
export PATH="$CA_ENV/bin:${PATH}"
export PYTHONPATH="${MAMBA_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA_ENV/lib/python3.10/site-packages/torch/lib:$CA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU="${1:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
REPO=/home/yqwang/projects/CASA-CD
V1=$REPO/models_v1
PRET=$REPO/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth
TMP=/tmp/exp1; mkdir -p "$TMP"
RC=0
ok(){ echo "[PASS] $*"; }; bad(){ echo "[FAIL] $*"; RC=1; }

# 参考（Run8 CTRL 架构）——若已存在就不再重跑
if [ ! -f "$TMP/ref_ctrl.pt" ]; then
  ( cd "$REPO/models" && python "$V1/epoch0_check.py" --mode dump --pkg-root "$REPO/models" \
      --arch run8 --stgeo none --vmamba "$PRET" --out "$TMP/ref_ctrl.pt" ) \
      > "$TMP/d2_ref.log" 2>&1 && ok "ref(CTRL_R8) dump" || { bad "ref dump"; tail -3 "$TMP/d2_ref.log"; }
else ok "ref(CTRL_R8) dump 已存在（复用）"; fi

for spec in "none:max_shape:0:v1_none" "inject:max_shape:0:v1_inj" "inject:topk_mean:1:v1_v2"; do
  IFS=':' read -r stg pool xs tag <<< "$spec"
  ( cd "$V1" && python epoch0_check.py --mode dump --pkg-root "$V1" --arch stgeo \
      --stgeo "$stg" --stgeo-pool "$pool" --stgeo-xscale "$xs" \
      --vmamba "$PRET" --out "$TMP/$tag.pt" ) > "$TMP/d2_$tag.log" 2>&1 \
      && ok "$tag dump ($stg/$pool/xs=$xs)" || { bad "$tag dump"; tail -3 "$TMP/d2_$tag.log"; }
done
grep -h "EPOCH0-DUMP" "$TMP"/d2_*.log 2>/dev/null | sed 's/^/  /'

if [ -f "$TMP/ref_ctrl.pt" ] && [ -f "$TMP/v1_none.pt" ]; then
  python "$V1/epoch0_check.py" --mode cmp --a "$TMP/ref_ctrl.pt" --b "$TMP/v1_none.pt" \
    && ok "CTRL_R8 == v1(none) 逐位" || bad "CTRL_R8 与 v1(none) 不逐位"
fi
if [ -f "$TMP/v1_none.pt" ] && [ -f "$TMP/v1_v2.pt" ]; then
  python "$V1/epoch0_check.py" --mode cmp --a "$TMP/v1_none.pt" --b "$TMP/v1_v2.pt" \
    && ok "v1(none) == v2(topk+xscale) 逐位（γ=0 门控在 v2 同样生效）" || bad "v2 γ 零初始化未生效"
fi
echo "=== IDENTITY RESULT: $([ $RC -eq 0 ] && echo PASS || echo FAIL) ==="
exit $RC
