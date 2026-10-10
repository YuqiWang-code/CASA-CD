#!/usr/bin/env bash
# CASA-TViM Run-Diag / Diag1 分阶段诊断流水线（只读；不训练、不覆盖历史产物）
#
# 用法：
#   bash run_diag.sh [stage ...]
#   阶段：p0 d0 d1 d2 d3 d4 report all
#   默认 = p0 + d0(M1) + d1 + d2 + d4 + report（D3 需显式指定，因它依赖前置 Gate）
#
# Gate 纪律（文档 §11.2）：先跑 p0 与 d0(M1)；二者 PASS 才继续后续阶段；
#   D0(M1) FAIL 时脚本中止（不静默跳过）。D3 只在 d0/d1/d2 均 PASS 时运行。
set -uo pipefail

PROJECT=/home/yqwang/projects/CASA-CD
DIAG=/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1
DATASET=${DATASET:-SYSU-CD-256}
RUN=${RUN:-Run1}
VARIANT=${VARIANT:-M1_FULL}
COMPARE_RUN=${COMPARE_RUN:-Run1}
COMPARE_VARIANT=${COMPARE_VARIANT:-A2_STR}
DEVICE=${DEVICE:-cuda:0}
LIMIT=${LIMIT:-0}                  # >0 时仅为 smoke（结果标 dry_run，不可用于结论）
MAX_TRAIN=${MAX_TRAIN:-3000}       # D3 probe 拟合样本上限（0=全部 train）
STEPS=${STEPS:-1500}
ENC_INPUT=${ENC_INPUT:-concat}     # D3c 分层评价使用的 probe 目录口径：concat | absdiff
D5_DATASETS=${D5_DATASETS:-"LEVIR-CD-256 WHU-CD-256 CDD-CD-256"}

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
cd "$PROJECT"
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$DIAG/logs" "$DIAG/P0"

STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then STAGES=(p0 d0 d1 d2 d4 report); fi
has() { for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
gate_status() { python -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$1" 2>/dev/null || echo MISSING; }

echo "[Diag1] stages: ${STAGES[*]}"
echo "[Diag1] dataset=$DATASET run=$RUN variant=$VARIANT limit=$LIMIT device=$DEVICE"

# ---------------------------------------------------------------- P0
if has p0 || has all; then
  echo "===== P0: metric protocol + unit tests ====="
  python -m unittest discover -s analyse/tests -p 'test_tvim_*.py' -v 2>&1 | tail -5
  python analyse/tvim_diag_common.py protocol --out-dir "$DIAG/P0"
fi

# ---------------------------------------------------------------- D0
if has d0 || has all; then
  echo "===== D0: audit $VARIANT ====="
  python analyse/tvim_diag_common.py audit --dataset "$DATASET" --run "$RUN" --variant "$VARIANT" \
    --device "$DEVICE" --out-dir "$DIAG/D0/$VARIANT" $([ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT")
  st=$(gate_status "$DIAG/D0/$VARIANT/gate.json")
  echo "[Diag1] D0($VARIANT) gate=$st"
  if [ "$st" != "PASS" ]; then
    echo "[Diag1] D0 gate not PASS → 停止后续阶段（文档 §4.1/§11.2）"
    exit 1
  fi
  if has d0_compare || has all; then
    python analyse/tvim_diag_common.py audit --dataset "$DATASET" --run "$RUN" --variant "$COMPARE_VARIANT" \
      --device "$DEVICE" --out-dir "$DIAG/D0/$COMPARE_VARIANT"
    echo "[Diag1] D0($COMPARE_VARIANT) gate=$(gate_status "$DIAG/D0/$COMPARE_VARIANT/gate.json")"
  fi
fi

# ---------------------------------------------------------------- D1
if has d1 || has all; then
  echo "===== D1: small-object error audit ====="
  python analyse/tvim_small_error_audit.py --dataset "$DATASET" --run "$RUN" --variant "$VARIANT" \
    --compare-run "$COMPARE_RUN" --compare-variant "$COMPARE_VARIANT" --device "$DEVICE" \
    --out-dir "$DIAG/D1/$VARIANT" --examples $([ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT")
  echo "[Diag1] D1 gate=$(gate_status "$DIAG/D1/$VARIANT/gate.json")"
fi

# ---------------------------------------------------------------- D2
if has d2 || has all; then
  echo "===== D2: stage recoverability ====="
  python analyse/tvim_stage_recoverability.py --dataset "$DATASET" --run "$RUN" --variant "$VARIANT" \
    --device "$DEVICE" --out-dir "$DIAG/D2/$VARIANT" --verbose $([ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT")
  echo "[Diag1] D2 gate=$(gate_status "$DIAG/D2/$VARIANT/gate.json")"
fi

# ---------------------------------------------------------------- D3（需前置 Gate）
if has d3 || has all; then
  d1st=$(gate_status "$DIAG/D1/$VARIANT/gate.json"); d2st=$(gate_status "$DIAG/D2/$VARIANT/gate.json")
  if [ "$d1st" = "PASS" ] && [ "$d2st" = "PASS" ]; then
    echo "===== D3: frozen linear probe ====="
    python analyse/tvim_linear_probe.py --dataset "$DATASET" --run "$RUN" --variant "$VARIANT" \
      --device "$DEVICE" --out-dir "$DIAG/D3/$VARIANT" --max-train "$MAX_TRAIN" --steps "$STEPS"
    echo "[Diag1] D3 gate=$(gate_status "$DIAG/D3/$VARIANT/gate.json")"
  else
    echo "[Diag1] D3 跳过：D1=$d1st D2=$d2st（需均 PASS）"
  fi
fi

# ---------------------------------------------------------------- D4
if has d4 || has all; then
  echo "===== D4: CAACP beta counterfactual ====="
  python analyse/tvim_caacp_counterfactual.py --dataset "$DATASET" --run "$RUN" --variant "$VARIANT" \
    --device "$DEVICE" --out-dir "$DIAG/D4/$VARIANT" $([ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT")
  echo "[Diag1] D4 gate=$(gate_status "$DIAG/D4/$VARIANT/gate.json")"
fi

# ---------------------------------------------------------------- report
if has report || has all; then
  echo "===== report ====="
  python analyse/tvim_diag_report.py --root "$DIAG" --out "$DIAG/DIAGNOSIS_REPORT.md" --write-manifest
fi

# ---------------------------------------------------------------- D3c 分层 probe（用已保存权重）
if has d3c || has all; then
  PROBE_DIR="$DIAG/D3_${ENC_INPUT}/$VARIANT"
  [ -d "$PROBE_DIR" ] || PROBE_DIR="$DIAG/D3/$VARIANT"
  echo "===== D3c: stratified probe ($PROBE_DIR) ====="
  python analyse/tvim_probe_stratified.py --probe-dir "$PROBE_DIR" --dataset "$DATASET" \
    --run "$RUN" --variant "$VARIANT" --device "$DEVICE" \
    --out-dir "$DIAG/D3_stratified/$VARIANT"
  echo "[Diag1] D3c gate=$(gate_status "$DIAG/D3_stratified/$VARIANT/gate.json")"
fi

# ---------------------------------------------------------------- D2b margin 配对 CI（CPU）
if has d2b || has all; then
  echo "===== D2b: margin paired bootstrap CI ====="
  python analyse/tvim_stage_raw_stats.py --stage-raw "$DIAG/D2/$VARIANT/stage_raw.csv" \
    --out "$DIAG/D2/$VARIANT/margin_paired_ci.json"
fi

# ---------------------------------------------------------------- D1 8 连通敏感性
if has d1conn8 || has all; then
  echo "===== D1 sensitivity: 8-connectivity ====="
  python analyse/tvim_small_error_audit.py --dataset "$DATASET" --run "$RUN" --variant "$VARIANT" \
    --device "$DEVICE" --out-dir "$DIAG/D1_conn8/$VARIANT" --connectivity 8
fi

# ---------------------------------------------------------------- D5 跨数据集（收窄：D0+D1+D2）
if has d5 || has all; then
  for ds in $D5_DATASETS; do
    echo "===== D5: $ds (D0 + D1 + D2) ====="
    python analyse/tvim_diag_common.py audit --dataset "$ds" --run "$RUN" --variant "$VARIANT" \
      --device "$DEVICE" --out-dir "$DIAG/D5/$ds/D0"
    python analyse/tvim_small_error_audit.py --dataset "$ds" --run "$RUN" --variant "$VARIANT" \
      --device "$DEVICE" --out-dir "$DIAG/D5/$ds/D1" --no-compare
    python analyse/tvim_stage_recoverability.py --dataset "$ds" --run "$RUN" --variant "$VARIANT" \
      --device "$DEVICE" --out-dir "$DIAG/D5/$ds/D2"
    echo "[Diag1] D5 $ds: D0=$(gate_status "$DIAG/D5/$ds/D0/gate.json") \
D1=$(gate_status "$DIAG/D5/$ds/D1/gate.json") D2=$(gate_status "$DIAG/D5/$ds/D2/gate.json")"
  done
fi

echo "[Diag1] done. artifacts under $DIAG"
