#!/usr/bin/env bash
# CASA-CD Run8：CASA-VMamba-STRNet（VMamba-Tiny + CAACP-HQKV + HGSL）训练编排
#
#   四波 × 四库 = 16 run；同一波内四库调用完全同构，唯一变量只有 variant_args() 的开关。
#   波次内四库**串行**（GPU1 被同用户另一项目占用 ~24.5GB/100%，只剩 GPU0 可用；
#   Run7 波三曾因同卡 2 并发 OOM；Run8 实测 batch32 peak 26.9–27.7 GB，余量仅 ~5 GB）。
#
#   本轮**串行执行**（`--wave 1` 跑完才开始 `--wave 2`），不做自动督导：
#   Run7 的波次督导在 Run8 里没有必要，且少一个故障点（Run7 血泪：督导中途 ABORT 很难判读）。
#   逐个开波：bash _r8_launch.sh --wave N
#
#   bash run_all.sh --wave 1     # CTRL_R8   四库（同期对照，必须先跑完）
#   bash run_all.sh --wave 2     # H1_HYP    四库（双曲 + 双分支 + 注入）
#   bash run_all.sh --wave 3     # H2_EUC    四库（同式欧氏对照：曲率是否必要）
#   bash run_all.sh --wave 4     # H3_HYP_GS 四库（+ HGSL 几何监督损失）
#
# 门：每个 run 的 train_log.txt 必须含最后一个完整 TEST 区块，
#     内含 [ACTUAL-OPT-STEPS] 80000 与 [PROTOCOL-VERSION] run8_vmamba_hqkv_exact80k_v1。
set -u

# ---------------------------------------------------------------- 环境（必须！）
# ⚠️ 血泪坑（2026-10-15 首启实测）：本脚本由 `setsid nohup bash run_all.sh` 拉起，
# **不会继承交互式 shell 的 conda 激活**，因此 `python` 不在 PATH 上，
# 四个库会以 `python: command not found`（rc=127）连续失败 3 次后判 FAIL。
# 必须在这里显式 source + activate + 导出两个环境变量（与 `_r8_smoke.sh` 同口径）。
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
CA_ENV=/home/yqwang/miniforge3/envs/casacd
# mamba_base 的 site-packages 必须前置：casacd 自带的 selective_scan_cuda.so 是 10-03 编的
# **ABI 已坏**版本（undefined symbol），而 Run8 的 VMamba 需要它兜底（见 vmamba_kernel.py）。
MAMBA_SITE=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate casacd 2>/dev/null || true
fi
export PATH="$CA_ENV/bin:${PATH}"
export PYTHONPATH="${MAMBA_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA_ENV/lib/python3.10/site-packages/torch/lib:$CA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
command -v python >/dev/null || { echo "[ABORT] python 不在 PATH（conda 激活失败）" >&2; exit 3; }

RUN=Run8
PROTOCOL=run8_vmamba_hqkv_exact80k_v1
REPO=/home/yqwang/projects/CASA-CD
MODELS=${REPO}/models
DATA_ROOT=/share_datasets/CD
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/${RUN}
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/${RUN}
PRET=${REPO}/pretrained_weight
MAMBA_SITE=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages

MAX_STEPS=80000
BATCH=32
SEED=16
LR=2e-4
BACKBONE_LR_RATIO=0.1
STR_DIM=192
HGSL_LAMBDA=0.2
HQKV_RANK=32
HQKV_DETAIL_W=2
HQKV_GLOBAL_W=4
HQKV_ATTN_STAGES=1,2,3
HQKV_R0=0.5
HQKV_A=0.5

DATASETS=(CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256)

# variant -> (hqkv, geometry, hgsl, hgsl_detach_score, inject_mode)
# 与 models/_legacy/run8_dry_run.py::VARIANTS 必须一致（check_mapping.py 会查这个）
variant_args() {
  case "$1" in
CTRL_R8)   echo "0 hyperbolic 0 1 none" ;;
H1_HYP)    echo "1 hyperbolic 0 1 dcr1" ;;
H2_EUC)    echo "1 euclidean  0 1 dcr1" ;;
H3_HYP_GS) echo "1 hyperbolic 1 1 dcr1" ;;
    *) echo "unknown variant $1" >&2; return 2 ;;
  esac
}

wave_variant() {
  case "$1" in
    1) echo CTRL_R8 ;;
    2) echo H1_HYP ;;
    3) echo H2_EUC ;;
    4) echo H3_HYP_GS ;;
    *) echo "unknown wave $1" >&2; return 2 ;;
  esac
}

# 四库串行（GPU1 被占，只剩 GPU0）——留余量，不并发
gpu_of() { echo 0; }

check_test_block() {
  local log="$1"
  python - "$log" "$PROTOCOL" <<'PY'
import re, sys
log, proto = sys.argv[1], sys.argv[2]
txt = open(log, encoding="utf-8", errors="replace").read()
blocks = re.findall(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", txt, re.S)
if not blocks:
    print(f"[GATE] no complete TEST block in {log}"); raise SystemExit(1)
b = blocks[-1]
ok = ("[ACTUAL-OPT-STEPS] 80000" in b) and (f"[PROTOCOL-VERSION] {proto}" in b)
print(f"[GATE] {log}: blocks={len(blocks)} last_ok={ok}")
raise SystemExit(0 if ok else 1)
PY
}

# ---------------------------------------------------------------- 单个 run
run_one() {
  local variant="$1" ds="$2"
  local gpu; gpu=$(gpu_of "$ds")
  read -r HQ GEOM GS GSD INJ <<< "$(variant_args "$variant")"
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
  echo "[START] ${variant}/${ds} GPU=${gpu} hqkv=${HQ} geom=${GEOM} hgsl=${GS} inject=${INJ} -> ${logd}/train_log.txt"
  # 空日志里已有上一次启动残留的 [START]/[FAIL] 行；每库日志必须只对应一条曲线，
  # 否则 `check_test_block` 与收口解析会在多份记录上取"最后一个 TEST 区块"，
  # 造成跨尝试串味。这里把上一次的**空配置**日志改名封存（绝不删除），再开新的一份。
  if [ -s "${logd}/train_log.txt" ] && ! grep -q "PROTOCOL-VERSION" "${logd}/train_log.txt"; then
    mv "${logd}/train_log.txt" "${logd}/train_log.txt.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_no_curve"
    echo "[ARCHIVE] 上一次未产生训练曲线的日志已改名封存（绝不删除）"
  fi
  cd "${MODELS}" || return 1
  for attempt in 1 2 3; do
    set +e
    CUDA_VISIBLE_DEVICES=${gpu} \
    PYTHONPATH="${MAMBA_SITE}${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train.py \
      --dataset "${ds}" \
      --dataset_root "${ds_root}" \
      --train_list "${ds_root}/list/train.txt" \
      --test_list "${ds_root}/list/test.txt" \
      --pretrained_weight_path "${PRET}/deit_tiny_patch16_224-a1311bcf.pth" \
      --vmamba_pretrained_weight_path "${PRET}/vssm_tiny_0230_ckpt_epoch_262.pth" \
      --ckpt_dir "${ckpt}" \
      --max_steps "${MAX_STEPS}" \
      --exact_max_steps 1 \
      --batch_size "${BATCH}" \
      --test_batch_size 16 \
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
  local fail=0
  for ds in "${DATASETS[@]}"; do
    # 串行：一次一个库（GPU0 单卡，且要避开外部作业的余量窗口）
    run_one "$variant" "$ds" || fail=1
  done
  if [ "$fail" -ne 0 ]; then
    echo "[WAVE-ABORT] wave ${w} has failed runs; not starting the next wave"; return 1
  fi
  echo "===== WAVE ${w} done ====="
}

main() {
  local wave=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --wave) wave="$2"; shift 2 ;;
      *) echo "unknown arg $1" >&2; return 2 ;;
    esac
  done
  [ -n "$wave" ] || { echo "usage: $0 --wave {1|2|3|4|all}" >&2; return 2; }
  mkdir -p "${LOG_ROOT}" "${CKPT_ROOT}"
  if [ "$wave" = "all" ]; then
    for w in 1 2 3 4; do run_wave "$w" || return 1; done
  else
    run_wave "$wave" || return 1
  fi
  echo "ALL DONE"
}

main "$@"
