#!/usr/bin/env bash
# Run8 单数据集编排器：`bash _r8_run_one_ds.sh <VARIANT> <DATASET> <GPU>`
#
# 为什么需要它：Run8 峰值显存实测 26.9–27.7 GB ⇒ **每张卡只能放 1 个 run**，
# 所以"两卡并行"= 同时跑 2 个 run；每个 run 由本脚本带自己的 GPU 与日志路径独立拉起，
# 互不干扰、可分别续训、可分别封存。
#
# 与 `run_all.sh` 的分工：
#   - `run_all.sh` 是**原始**的单卡串行流水线（保留不动，wave1 的 CDD 就是它跑完的）；
#   - 本脚本 + `_r8_pipeline_supervisor.sh` 是**双卡并行**流水线（wave1 余量 + wave2/3/4）。
#
# ⚠️ 披露（跨调度器一致性）：本脚本调用 `python train.py` 的**参数与 run_all.sh 逐字相同**
# （同一 `variant_args` 口径、同 seed/batch/80K/协议），差别**只在设备号与并发度**。
# 设备号不是 `run_manifest.json` 的冻结字段（Run7 已实测：OOM 后换卡续训，manifest 逐字段校验通过）。
#
# 用法：
#   bash _r8_run_one_ds.sh CTRL_R8 SYSU-CD-256 1
#   RESUME_SAME_RUN=1 bash _r8_run_one_ds.sh CTRL_R8 LEVIR-CD-256 0   # 仅认证过的同 run 续训
set -u

VARIANT="${1:?usage: $0 <VARIANT> <DATASET> <GPU>}"
DS="${2:?usage: $0 <VARIANT> <DATASET> <GPU>}"
GPU="${3:?usage: $0 <VARIANT> <DATASET> <GPU>}"

REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
MODELS="$REPO/models"
DATA_ROOT=/share_datasets/CD
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
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

PROTOCOL=run8_vmamba_hqkv_exact80k_v1
MAX_STEPS=80000; BATCH=32; SEED=16; LR=2e-4
BACKBONE_LR_RATIO=0.1; STR_DIM=192; HGSL_LAMBDA=0.2
HQKV_RANK=32; HQKV_DETAIL_W=2; HQKV_GLOBAL_W=4; HQKV_ATTN_STAGES=1,2,3
HQKV_R0=0.5; HQKV_A=0.5

# 与 run_all.sh::variant_args 完全同口径：(hqkv, geometry, hgsl, hgsl_detach_score, inject_mode)
variant_args() {
  case "$1" in
    CTRL_R8)   echo "0 hyperbolic 0 1 none" ;;
    H1_HYP)    echo "1 hyperbolic 0 1 dcr1" ;;
    H2_EUC)    echo "1 euclidean  0 1 dcr1" ;;
    H3_HYP_GS) echo "1 hyperbolic 1 1 dcr1" ;;
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
# ⚠️ "非空"守卫必须**忽略封存残留**（`*SUPERSEDED*`）。
# 2026-10-15 实测坑：某个编排器被打断前先写了 `run_manifest.json`，我按纪律把它改名为
# `run_manifest.json.SUPERSEDED_...`（不删除），但 `ls -A` **不看文件名** ⇒ 目录仍算"非空"
# ⇒ 之后每次启动都被 ABORT（rc=3），白等一个 GPU 空转窗口。
# 正确口径：只把**真实训练产物**（last.pth / best_*.pth / arch.json / 未封存的 manifest）
# 视为"非空"；`*SUPERSEDED*` 是历史残渣，必须放行。
REAL_FILES="$(find "$CKPT" -maxdepth 1 -type f ! -name '*SUPERSEDED*' 2>/dev/null | head -1)"
if [ "${RESUME_SAME_RUN:-0}" != "1" ] && [ -n "$REAL_FILES" ]; then
  echo "[ABORT] ${CKPT} not empty (real artifacts); refusing to start a new experiment here" >&2
  ls -la "$CKPT" >&2; exit 3
fi
if [ -n "$(find "$CKPT" -maxdepth 1 -type f -name '*SUPERSEDED*' 2>/dev/null | head -1)" ]; then
  echo "[NOTE] ${CKPT} 含封存残留（*SUPERSEDED*），按纪律保留并放行"
fi

# 空日志里若已有上一次启动残留，先改名封存（与 run_all.sh 同口径，绝不删除）
if [ -s "${LOGD}/train_log.txt" ] && ! grep -q "PROTOCOL-VERSION" "${LOGD}/train_log.txt"; then
  mv "${LOGD}/train_log.txt" "${LOGD}/train_log.txt.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_no_curve"
fi

echo "[START] ${VARIANT}/${DS} GPU=${GPU} hqkv=${HQ} geom=${GEOM} hgsl=${GS} inject=${INJ} -> ${LOGD}/train_log.txt"
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
    --model_type tiny --mode baseline --arch casa_vmamba_str \
    --hqkv "${HQ}" --hqkv_geometry "${GEOM}" --hqkv_branch both \
    --hqkv_inject_mode "${INJ}" \
    --hqkv_rank "${HQKV_RANK}" --hqkv_lite_dim "${STR_DIM}" \
    --hqkv_detail_window "${HQKV_DETAIL_W}" --hqkv_global_window "${HQKV_GLOBAL_W}" \
    --hqkv_attn_stages "${HQKV_ATTN_STAGES}" \
    --hqkv_r0 "${HQKV_R0}" --hqkv_a "${HQKV_A}" \
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
