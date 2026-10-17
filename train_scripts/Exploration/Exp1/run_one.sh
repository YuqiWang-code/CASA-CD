#!/usr/bin/env bash
# 探索线 Exp1（SPD(2)/H² 结构张量测地证据）单 run 启动器
#   bash run_one.sh <VARIANT> <DATASET> <GPU>
#   VARIANT: STG_INJ（stgeo=inject，主实验）| STG_OFF（stgeo=none，本架构自对照）
# 纪律：只用自己的 ckpt/log 目录；非空目录拒绝启动（除非 RESUME_SAME_RUN=1）；
#       产物只改名封存（*SUPERSEDED*），绝不删除。
set -u

CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
CA_ENV=/home/yqwang/miniforge3/envs/casacd
MAMBA_SITE=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"; conda activate casacd 2>/dev/null || true
fi
export PATH="$CA_ENV/bin:${PATH}"
export PYTHONPATH="${MAMBA_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA_ENV/lib/python3.10/site-packages/torch/lib:$CA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
command -v python >/dev/null || { echo "[ABORT] python 不在 PATH" >&2; exit 3; }

REPO=/home/yqwang/projects/CASA-CD
CODE=${REPO}/models_v1
DATA_ROOT=/share_datasets/CD
PRET=${REPO}/pretrained_weight
EXP=Exploration/Exp1
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/${EXP}
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/${EXP}
PROTOCOL=exp1_stgeo_exact80k_v1

VARIANT="$1"; DS="$2"; GPU="${3:-1}"
case "$VARIANT" in
  STG_INJ) STGEO=inject; EXTRA="--stgeo_pool max_shape --stgeo_xscale 0" ;;  # v1（已完成）
  STG_V2)  STGEO=inject; EXTRA="--stgeo_pool topk_mean --stgeo_xscale 1" ;;  # v2：鲁棒池化+跨尺度门
  STG_OFF) STGEO=none;   EXTRA="--stgeo_pool max_shape --stgeo_xscale 0" ;;
  *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac

CKPT="${CKPT_ROOT}/${VARIANT}/${DS}"
LOGD="${LOG_ROOT}/${VARIANT}/${DS}"
mkdir -p "$CKPT" "$LOGD"
LOG="${LOGD}/train_log.txt"

if [ -s "${CKPT}/last.pth" ] && [ "${RESUME_SAME_RUN:-0}" != "1" ]; then
  echo "[ABORT] ${CKPT}/last.pth 已存在；只有认证过的同 run 续训才允许 RESUME_SAME_RUN=1" >&2
  exit 3
fi
LEFT=$(find "$CKPT" -maxdepth 1 -type f ! -name '*SUPERSEDED*' 2>/dev/null | wc -l)
if [ "${RESUME_SAME_RUN:-0}" != "1" ] && [ "$LEFT" -gt 0 ]; then
  echo "[ABORT] ${CKPT} 含非封存文件（$LEFT 个），拒绝在此启动新实验" >&2
  ls -la "$CKPT" >&2
  exit 3
fi
if [ -s "$LOG" ] && ! grep -q "PROTOCOL-VERSION" "$LOG"; then
  mv "$LOG" "${LOG}.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_no_curve"
  echo "[ARCHIVE] 未产生训练曲线的旧日志已改名封存"
fi

echo "[START] ${VARIANT}/${DS} gpu=${GPU} stgeo=${STGEO} -> ${LOG}"
cd "$CODE" || exit 1
for attempt in 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPU} python train_stgeo.py \
    --dataset "${DS}" --dataset_root "${DATA_ROOT}/${DS}" \
    --train_list "${DATA_ROOT}/${DS}/list/train.txt" \
    --test_list "${DATA_ROOT}/${DS}/list/test.txt" \
    --vmamba_pretrained_weight_path "${PRET}/vssm_tiny_0230_ckpt_epoch_262.pth" \
    --ckpt_dir "${CKPT}" --max_steps 80000 --exact_max_steps 1 \
    --batch_size 32 --test_batch_size 16 --inWidth 256 --inHeight 256 \
    --stgeo "${STGEO}" --stgeo_sigmas 1.0,2.0,4.0 --stgeo_mid 32 --stgeo_dim 192 \
    ${EXTRA} \
    --rep_mode full --backbone_lr_ratio 0.1 --str_dim 192 \
    --lr 2e-4 --lr_mode poly --num_workers 4 --seed 16 --gpu_id 0 \
    >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[DONE] ${VARIANT}/${DS} attempt=${attempt}" >> "$LOG"
    break
  fi
  if [ $attempt -eq 3 ]; then
    echo "[FAIL] ${VARIANT}/${DS} 3 次尝试耗尽 (last rc=${rc})" >> "$LOG"
    exit 1
  fi
  echo "[RETRY ${attempt}/3] rc=${rc}，10s 后从自身 last.pth 续训" >> "$LOG"
  sleep 10
done

python - "$LOG" "$PROTOCOL" <<'PY'
import re, sys
log, proto = sys.argv[1], sys.argv[2]
txt = open(log, encoding="utf-8", errors="replace").read()
blocks = re.findall(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", txt, re.S)
if not blocks:
    print(f"[GATE] 无完整 TEST 区块: {log}"); raise SystemExit(1)
b = blocks[-1]
ok = ("[ACTUAL-OPT-STEPS] 80000" in b) and (f"[PROTOCOL-VERSION] {proto}" in b)
print(f"[GATE] {log}: blocks={len(blocks)} last_ok={ok}")
raise SystemExit(0 if ok else 1)
PY
