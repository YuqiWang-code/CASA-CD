#!/usr/bin/env bash
# Run8b 单数据集编排器（`bash _r8b_run_one_ds.sh <VARIANT> <DATASET> <GPU>`）。
#
# 与 Run8 的 `_r8_run_one_ds.sh` 的唯一差别：
#   * `--arch casa_vmamba_str_v2`（抗自锁版）
#   * 新增 `--hqkv_gamma_init ${GAMMA_INIT}`（抗自锁的承重开关，**必须 > 0**）
#   * 日志 / ckpt 根目录换成 Run8b（**绝不写 Run8 的目录**）
#   * 协议 `run8b_vmamba_hqkv_antilock_exact80k_v2`
# 其余（80K / batch32 / seed16 / lr / 半径 / HGSL λ / 数据契约）与 Run8 **逐字相同**，
# 以保证 Run8b 与 Run8 的对比只差"抗自锁"这一个变量。
set -u

VARIANT="${1:?usage: $0 <VARIANT> <DATASET> <GPU>}"
DS="${2:?usage: $0 <VARIANT> <DATASET> <GPU>}"
GPU="${3:?usage: $0 <VARIANT> <DATASET> <GPU>}"

REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8b"
MODELS="$REPO/models"
DATA_ROOT=/share_datasets/CD
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8b
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
PRET="$REPO/pretrained_weight"

CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
CA_ENV=/home/yqwang/miniforge3/envs/casacd
MAMBA_SITE=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
source "$CONDA_SH" 2>/dev/null || true
conda activate casacd 2>/dev/null || true
export PATH="$CA_ENV/bin:${PATH}"
export PYTHONPATH="${MAMBA_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA_ENV/lib/python3.10/site-packages/torch/lib:$CA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
command -v python >/dev/null || { echo "[ABORT] python 不在 PATH" >&2; exit 3; }

MAX_STEPS=80000; BATCH=32; SEED=16; LR=2e-4
# 协议串必须与 train.py::PROTOCOL_VERSION 逐字一致（check_mapping.py 会查）；
# 它也会被写进日志的 TEST 区块，收口链据此判定 VALID。
PROTOCOL=run8b_vmamba_hqkv_antilock_exact80k_v2
BACKBONE_LR_RATIO=0.1; STR_DIM=192; HGSL_LAMBDA=0.2
HQKV_RANK=32; HQKV_DETAIL_W=2; HQKV_GLOBAL_W=4; HQKV_ATTN_STAGES=1,2,3
HQKV_R0=0.5; HQKV_A=0.5
# **抗自锁的承重开关**：>0 才有梯度；=0 会精确复现 Run8 v1 的自锁（仅作负对照）
GAMMA_INIT="${GAMMA_INIT:-0.5}"

# variant -> (hqkv, geometry, hgsl, hgsl_detach_score, inject_mode)
# 与 `models/run8b_dry_run.py::VARIANTS` 必须一致（check_mapping.py 会查）
variant_args() {
  case "$1" in
    CTRL_R8B)  echo "0 hyperbolic 0 1 none" ;;
    B1_HYP)    echo "1 hyperbolic 0 1 dcr1" ;;
    B2_EUC)    echo "1 euclidean  0 1 dcr1" ;;
    B3_HYP_GS) echo "1 hyperbolic 1 1 dcr1" ;;
    *) echo "[ABORT] unknown variant $1" >&2; return 2 ;;
  esac
}
read -r HQ GEOM GS GSD INJ <<< "$(variant_args "$VARIANT")" || exit 2

DS_ROOT="$DATA_ROOT/$DS"
CKPT="$CKPT_ROOT/$VARIANT/$DS"
LOGD="$LOG_ROOT/$VARIANT/$DS"
mkdir -p "$CKPT" "$LOGD"

if [ -s "${CKPT}/last.pth" ] && [ "${RESUME_SAME_RUN:-0}" != "1" ]; then
  echo "[ABORT] ${CKPT}/last.pth exists; set RESUME_SAME_RUN=1 only for certified same-run resume" >&2
  exit 3
fi
# 守卫必须忽略封存残留（*SUPERSEDED*）——Run8 踩过的坑
REAL_FILES="$(find "$CKPT" -maxdepth 1 -type f ! -name '*SUPERSEDED*' 2>/dev/null | head -1)"
if [ "${RESUME_SAME_RUN:-0}" != "1" ] && [ -n "$REAL_FILES" ]; then
  echo "[ABORT] ${CKPT} not empty (real artifacts); refusing to start a new experiment here" >&2
  ls -la "$CKPT" >&2; exit 3
fi
if [ -n "$(find "$CKPT" -maxdepth 1 -type f -name '*SUPERSEDED*' 2>/dev/null | head -1)" ]; then
  echo "[NOTE] ${CKPT} 含封存残留（*SUPERSEDED*），按纪律保留并放行"
fi

if [ -s "${LOGD}/train_log.txt" ] && ! grep -q "PROTOCOL-VERSION" "${LOGD}/train_log.txt"; then
  mv "${LOGD}/train_log.txt" "${LOGD}/train_log.txt.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_no_curve"
fi

echo "[START] ${VARIANT}/${DS} GPU=${GPU} hqkv=${HQ} geom=${GEOM} hgsl=${GS} inject=${INJ} gamma_init=${GAMMA_INIT} -> ${LOGD}/train_log.txt"
cd "$MODELS" || exit 1
for attempt in 1 2 3; do
  set +e
  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --dataset "${DS}" \
    --dataset_root "${DS_ROOT}" \
    --train_list "${DS_ROOT}/list/train.txt" \
    --test_list "${DS_ROOT}/list/test.txt" \
    --pretrained_weight_path "${PRET}/deit_tiny_patch16_224-a1311bcf.pth" \
    --vmamba_pretrained_weight_path "${PRET}/vssm_tiny_0230_ckpt_epoch_262.pth" \
    --ckpt_dir "${CKPT}" \
    --max_steps "${MAX_STEPS}" --exact_max_steps 1 \
    --batch_size "${BATCH}" --test_batch_size 16 \
    --inWidth 256 --inHeight 256 \
    --model_type tiny --mode baseline --arch casa_vmamba_str_v2 \
    --hqkv "${HQ}" --hqkv_geometry "${GEOM}" \
    --hqkv_inject_mode "${INJ}" \
    --hqkv_rank "${HQKV_RANK}" --hqkv_lite_dim "${STR_DIM}" \
    --hqkv_detail_window "${HQKV_DETAIL_W}" --hqkv_global_window "${HQKV_GLOBAL_W}" \
    --hqkv_attn_stages "${HQKV_ATTN_STAGES}" \
    --hqkv_r0 "${HQKV_R0}" --hqkv_a "${HQKV_A}" \
    --hqkv_gamma_init "${GAMMA_INIT}" \
    --hgsl "${GS}" --hgsl_detach_score "${GSD}" --hgsl_lambda "${HGSL_LAMBDA}" \
    --fs_tar 0 \
    --rep_mode full --backbone_lr_ratio "${BACKBONE_LR_RATIO}" --str_dim "${STR_DIM}" \
    --lr "${LR}" --lr_mode poly --num_workers 4 --seed "${SEED}" --gpu_id 0 \
    >> "${LOGD}/train_log.txt" 2>&1
  rc=$?
  set -e
  if [ ${rc} -eq 0 ]; then
    echo "[DONE] ${VARIANT}/${DS} attempt=${attempt}" >> "${LOGD}/train_log.txt"
    break
  fi
  # rc=3 是 fail-fast 主动中止（contrib_norm 过小）⇒ **不重试**，直接上报
  if [ ${rc} -eq 3 ]; then
    echo "[FAILFAST-ABORT] ${VARIANT}/${DS} 抗自锁检查未过（rc=3），不重试" >> "${LOGD}/train_log.txt"
    echo "[FAILFAST-ABORT] ${VARIANT}/${DS} rc=3"
    exit 3
  fi
  if [ ${attempt} -eq 3 ]; then
    echo "[FAIL] ${VARIANT}/${DS} 3 attempts exhausted (last rc=${rc})" >> "${LOGD}/train_log.txt"
    exit 1
  fi
  echo "[RETRY ${attempt}/3] rc=${rc}, resuming from own last.pth in 10s" >> "${LOGD}/train_log.txt"
  sleep 10
done

if grep -q '=== END TEST RESULTS ===' "${LOGD}/train_log.txt"; then
  echo "[GATE] ${VARIANT}/${DS} OK"
  exit 0
fi
echo "[GATE] ${VARIANT}/${DS} 无完整 TEST 区块"
exit 1
