#!/usr/bin/env bash
# CASA-TViM Run7 HFAE 训练编排（3 波 × 4 库；GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发、batch 32）。
#
#   bash run_all.sh --wave 1     # M1_R7CTRL 四库（同期对照，必须先跑完）
#   bash run_all.sh --wave 2     # E9_HFAE_A 四库（train-only 深监督，deploy 不变）
#   bash run_all.sh --wave 3     # E10_HFAE_B 四库（+ 零初始化注入 DCR）
#   bash run_all.sh --wave all   # 三波串行
#
# 设计说明（与 Run4 的差异）：Run4 用「VARIANT/train_<DS>.sh + check_run.py」的目录化布局；
# Run7 改为**自包含 runner**（同一份 CLI，参数化 variant），理由是减少 12 份近乎重复的脚本
# 与一个额外的 gate 工具，同一 wave 内四库调用完全同构、唯一变量只有 --hf_edge*。
# 收口门内联在 run_one()：只看**最后一个完整 TEST 区块**是否含
# [ACTUAL-OPT-STEPS] 80000 与 [PROTOCOL-VERSION] run7_hf_edge_exact80k_v1。
set -uo pipefail

RUN=Run7
PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
PRET=${PROJ}/pretrained_weight
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/${RUN}
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/${RUN}
DATA_ROOT=/share_datasets/CD
PROTOCOL=run7_hf_edge_exact80k_v1
MAX_STEPS=80000
BATCH=32
LAMBDA=0.2
TAPS=1_4_1_8

WAVE="${1:-}"
if [ "${WAVE}" = "--wave" ]; then WAVE="${2:-}"; fi
case "${WAVE}" in
  1|2|3|all) ;;
  *) echo "usage: bash run_all.sh --wave {1|2|3|all}"; exit 2 ;;
esac

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# variant -> (hf_edge, hf_edge_form) ；与 models/run7_dry_run.py::VARIANTS 必须一致
variant_args() {
  case "$1" in
    M1_R7CTRL)  echo "0 none" ;;
    E9_HFAE_A)  echo "1 aux_only" ;;
    E10_HFAE_B) echo "1 aux_inject" ;;
    *) echo "" ;;
  esac
}
wave_variant() {
  case "$1" in
    1) echo M1_R7CTRL ;;
    2) echo E9_HFAE_A ;;
    3) echo E10_HFAE_B ;;
  esac
}
# 数据集 -> GPU
gpu_of() {
  case "$1" in
    CDD-CD-256|LEVIR-CD-256) echo 0 ;;
    SYSU-CD-256|WHU-CD-256)  echo 1 ;;
  esac
}
DATASETS=(CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256)

# ---------------------------------------------------------------- 前置校验
if [ ! -f "${PRET}/tinyvim_s_1000e.pth" ]; then
  echo "[ABORT] missing ${PRET}/tinyvim_s_1000e.pth"; exit 3
fi
for f in model/str_hf_edge.py train.py eval.py run7_dry_run.py test_run7_hf_edge.py; do
  [ -f "${MODELS}/${f}" ] || { echo "[ABORT] missing models/${f}"; exit 3; }
done

# ---------------------------------------------------------------- 收口门（只看最后一个完整 TEST 区块）
check_test_block() {
  local log="$1" variant="$2" ds="$3"
  python - "$log" "$variant" "$ds" "$PROTOCOL" <<'PY'
import re, sys
log, variant, ds, proto = sys.argv[1:5]
try:
    txt = open(log, encoding="utf-8", errors="replace").read()
except FileNotFoundError:
    print(f"[GATE-FAIL] {variant}/{ds}: no log"); raise SystemExit(1)
blocks = re.findall(r"=== TEST RESULTS ===\n(.*?)\n=== END TEST RESULTS ===", txt, re.S)
if not blocks:
    print(f"[GATE-FAIL] {variant}/{ds}: no complete TEST RESULTS block"); raise SystemExit(1)
b = blocks[-1]
steps = re.search(r"\[ACTUAL-OPT-STEPS\]\s+(\d+)", b)
pv = re.search(r"\[PROTOCOL-VERSION\]\s+(\S+)", b)
f1 = re.search(r"F1=([0-9.]+)", b)
ok = bool(steps) and steps.group(1) == "80000" and bool(pv) and pv.group(1) == proto
print(f"[GATE] {variant}/{ds}: opt_steps={steps.group(1) if steps else None} "
      f"protocol={pv.group(1) if pv else None} F1={f1.group(1) if f1 else None} "
      f"-> {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PY
}

# ---------------------------------------------------------------- 单个 run
run_one() {
  local variant="$1" ds="$2"
  local gpu; gpu=$(gpu_of "$ds")
  read -r HF FORM <<< "$(variant_args "$variant")"
  local ds_root="${DATA_ROOT}/${ds}"
  local ckpt="${CKPT_ROOT}/${variant}/${ds}"
  local logd="${LOG_ROOT}/${variant}/${ds}"
  mkdir -p "$ckpt" "$logd"
  if [ -s "${ckpt}/last.pth" ] && [ "${RESUME_SAME_RUN:-0}" != "1" ]; then
    echo "[ABORT] ${ckpt}/last.pth exists; set RESUME_SAME_RUN=1 only for certified same-run resume" >&2
    return 3
  fi
  if [ "${RESUME_SAME_RUN:-0}" != "1" ] && [ -n "$(ls -A "$ckpt" 2>/dev/null)" ]; then
    echo "[ABORT] ${ckpt} not empty; refusing to start a new experiment here" >&2
    ls -la "$ckpt" >&2
    return 3
  fi
  echo "[START] ${variant}/${ds} GPU=${gpu} hf_edge=${HF} form=${FORM} -> ${logd}/train_log.txt"
  cd "${MODELS}" || return 1
  for attempt in 1 2 3; do
    set +e
    CUDA_VISIBLE_DEVICES=${gpu} python train.py \
      --dataset "${ds}" \
      --dataset_root "${ds_root}" \
      --train_list "${ds_root}/list/train.txt" \
      --test_list "${ds_root}/list/test.txt" \
      --pretrained_weight_path "${PRET}/deit_tiny_patch16_224-a1311bcf.pth" \
      --tinyvim_pretrained_weight_path "${PRET}/tinyvim_s_1000e.pth" \
      --ckpt_dir "${ckpt}" \
      --max_steps "${MAX_STEPS}" \
      --exact_max_steps 1 \
      --batch_size "${BATCH}" \
      --test_batch_size 16 \
      --inWidth 256 --inHeight 256 \
      --model_type tiny --mode baseline --arch casa_tvim_str \
      --caacp 1 --caacp_score_mode rank --caacp_residual_mode current \
      --frh 0 --fs_tar 0 --fine_tap 0 \
      --hf_edge "${HF}" --hf_edge_form "${FORM}" \
      --hf_edge_taps "${TAPS}" --hf_edge_lambda "${LAMBDA}" \
      --rep_mode full --backbone_lr_ratio 0.1 --str_dim 96 \
      --lr 2e-4 --lr_mode poly --num_workers 4 --seed 16 --gpu_id 0 \
      >> "${logd}/train_log.txt" 2>&1
    rc=$?
    set -e
    if [ ${rc} -eq 0 ]; then
      echo "[DONE] ${variant}/${ds} attempt=${attempt}" >> "${logd}/train_log.txt"
      break
    fi
    if [ ${attempt} -eq 3 ]; then
      echo "[FAIL] ${variant}/${ds} 3 attempts exhausted (last rc=${rc})" >> "${logd}/train_log.txt"
      return 1
    fi
    echo "[RETRY ${attempt}/3] rc=${rc}, resuming from own last.pth in 10s" >> "${logd}/train_log.txt"
    sleep 10
  done
  check_test_block "${logd}/train_log.txt" "$variant" "$ds"
}

run_wave() {
  local w="$1" variant; variant=$(wave_variant "$w")
  echo "===== WAVE ${w}: ${variant} ====="
  local pids=()
  for ds in "${DATASETS[@]}"; do
    run_one "$variant" "$ds" > "/tmp/run7_w${w}_${ds}.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  for ds in "${DATASETS[@]}"; do
    echo "--- wave${w} ${variant}/${ds} ---"; tail -6 "/tmp/run7_w${w}_${ds}.log"
  done
  if [ "$fail" -ne 0 ]; then
    echo "[WAVE-ABORT] wave ${w} has failed runs; not starting the next wave"; return 1
  fi
  echo "===== WAVE ${w} done ====="
}

if [ "${WAVE}" = "all" ]; then
  for w in 1 2 3; do run_wave "$w" || exit 1; done
else
  run_wave "${WAVE}" || exit 1
fi
echo "[RUN7] finished wave(s) ${WAVE}"
