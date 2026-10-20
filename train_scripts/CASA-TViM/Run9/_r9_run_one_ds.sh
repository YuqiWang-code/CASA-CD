#!/usr/bin/env bash
# Run9 单数据集编排器（`bash _r9_run_one_ds.sh <VARIANT> <DATASET> <GPU>`）。
#
# Run9 = Run8c 的 CTRL + **HSAM**（Poincaré 球双曲相似度注意力，零参数，论文 JSTARS 2024）。
# 与 Run8b 的编排器的差别：
#   * 协议 `run9_hsam_poincare_exact80k_v1`
#   * `--hqkv 0`（HQKV 全关，Run8c 的 lattice 模块不在本轮变量里）
#   * 新增 `--hsam ${HS} --hsam_stages ${HSAM_STAGES} --hsam_logit_norm ${HSAM_LOGIT_NORM}`
#   * 日志 / ckpt 根目录换成 Run9（**绝不写 Run8/Run8b 的目录**）
#   * **修掉 Run8b 的重试复活缺陷**：加 `trap`，`kill` 掉本脚本时重试循环随之退出
# 其余（80K / batch32 / seed16 / lr / data contract / str_dim）与 Run8b **逐字相同**，
# 以保证 Run9 的对比只差"HSAM 开/关"这一个变量。
set -u

VARIANT="${1:?usage: $0 <VARIANT> <DATASET> <GPU>}"
DS="${2:?usage: $0 <VARIANT> <DATASET> <GPU>}"
GPU="${3:?usage: $0 <VARIANT> <DATASET> <GPU>}"

REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run9"
MODELS="$REPO/models"
DATA_ROOT=/share_datasets/CD
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run9
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run9
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

# 被 kill 时不再重试（这是 Run8b 那次"杀掉又复活"的根因修复）
trap 'echo "[TRAP] received TERM/INT, aborting retry loop" >&2; exit 130' TERM INT

MAX_STEPS=80000; BATCH=32; SEED=16; LR=2e-4
# 协议串必须与 train.py::PROTOCOL_VERSION 逐字一致
PROTOCOL=run9_hsam_poincare_exact80k_v1
BACKBONE_LR_RATIO=0.1; STR_DIM=192
# 预注册冻结值（**不搜索**）
HSAM_STAGES="${HSAM_STAGES:-2,3}"
HSAM_LOGIT_NORM="${HSAM_LOGIT_NORM:-std}"

# variant -> hsam
# 与 `train_scripts/CASA-TViM/Run9/README.md` §2 与 SELECTION_LOCK.json 必须一致
variant_args() {
  case "$1" in
    S0_CTRL) echo "0" ;;
    S1_HSAM) echo "1" ;;
    *) echo "[ABORT] unknown variant $1" >&2; return 2 ;;
  esac
}
HS="$(variant_args "$VARIANT")" || exit 2

DS_ROOT="$DATA_ROOT/$DS"
CKPT="$CKPT_ROOT/$VARIANT/$DS"
LOGD="$LOG_ROOT/$VARIANT/$DS"
mkdir -p "$CKPT" "$LOGD"

if [ -s "${CKPT}/last.pth" ] && [ "${RESUME_SAME_RUN:-0}" != "1" ]; then
  echo "[ABORT] ${CKPT}/last.pth exists; set RESUME_SAME_RUN=1 only for certified same-run resume" >&2
  exit 3
fi
REAL_FILES="$(find "$CKPT" -maxdepth 1 -type f ! -name '*SUPERSEDED*' 2>/dev/null | head -1)"
if [ "${RESUME_SAME_RUN:-0}" != "1" ] && [ -n "$REAL_FILES" ]; then
  echo "[ABORT] ${CKPT} not empty (real artifacts); refusing to start a new experiment here" >&2
  ls -la "$CKPT" >&2; exit 3
fi

if [ -s "${LOGD}/train_log.txt" ] && ! grep -q "PROTOCOL-VERSION" "${LOGD}/train_log.txt"; then
  mv "${LOGD}/train_log.txt" "${LOGD}/train_log.txt.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_no_curve"
fi

echo "[START] ${VARIANT}/${DS} GPU=${GPU} hsam=${HS} stages=${HSAM_STAGES} logit_norm=${HSAM_LOGIT_NORM} -> ${LOGD}/train_log.txt"
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
    --hqkv 0 --hqkv_geometry lattice \
    --hgsl 0 --hgsl_detach_score 1 \
    --hsam "${HS}" --hsam_stages "${HSAM_STAGES}" --hsam_logit_norm "${HSAM_LOGIT_NORM}" \
    --fs_tar 0 \
    --rep_mode full --backbone_lr_ratio "${BACKBONE_LR_RATIO}" --str_dim "${STR_DIM}" \
    --lr "${LR}" --lr_mode poly --num_workers 4 --seed "${SEED}" --gpu_id 0 \
    >> "${LOGD}/train_log.txt" 2>&1
  rc=$?
  set +e
  if [ ${rc} -eq 0 ]; then
    echo "[DONE] ${VARIANT}/${DS} attempt=${attempt}" >> "${LOGD}/train_log.txt"
    break
  fi
  # rc=4 是 HSAM fail-fast 主动中止（NaN / 均匀注意力 / 半径退化）⇒ **不重试**
  if [ ${rc} -eq 4 ]; then
    echo "[HSAM-FAILFAST-ABORT] ${VARIANT}/${DS} rc=4，不重试" | tee -a "${LOGD}/train_log.txt"
    exit 4
  fi
  if [ ${attempt} -eq 3 ]; then
    echo "[FAIL] ${VARIANT}/${DS} 3 attempts exhausted (last rc=${rc})" >> "${LOGD}/train_log.txt"
    exit 1
  fi
  echo "[RETRY ${attempt}/3] rc=${rc}, resuming from own last.pth in 10s" >> "${LOGD}/train_log.txt"
  # 分段 sleep：收到 TERM 立刻退出，不会"睡完又拉起"
  for _ in 1 2 3 4 5 6 7 8 9 10; do sleep 1; done
done

if grep -q '=== END TEST RESULTS ===' "${LOGD}/train_log.txt"; then
  echo "[GATE] ${VARIANT}/${DS} OK"
  exit 0
fi
echo "[GATE] ${VARIANT}/${DS} 无完整 TEST 区块"
exit 1
