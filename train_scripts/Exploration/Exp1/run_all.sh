#!/usr/bin/env bash
# 探索线 Exp1：逐库串行（GPU1，避开正在跑 Run8 的 GPU0）
#   bash run_all.sh STG_INJ 1                       # 默认 LEVIR + SYSU（用户 2026-10-16 收窄范围）
#   bash run_all.sh STG_INJ 1 LEVIR-CD-256 SYSU-CD-256 [更多库...]
# 说明：用户明确要求**只在 LEVIR 与 SYSU 上训练**——LEVIR 小目标多、SYSU 变化场景多且难度大，
#       本课题的硬目标缺口也集中在这两库（Run8 CTRL：LEVIR 91.21 / SYSU 81.88 vs 目标 92.5 / 85）。
#       CDD/WHU 不再消耗 GPU 时间。任一库失败即停（fail-closed）。
set -u
VARIANT="${1:?usage: run_all.sh <VARIANT> <GPU> [DATASET...]}"
GPU="${2:-1}"
shift 2 || true
if [ "$#" -gt 0 ]; then DATASETS=("$@"); else DATASETS=(LEVIR-CD-256 SYSU-CD-256); fi
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "===== EXP1 WAVE: ${VARIANT} on GPU${GPU} datasets=${DATASETS[*]} ====="
fail=0
for ds in "${DATASETS[@]}"; do
  free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | \
         awk -F, -v g="$GPU" '$1+0==g {gsub(/ MiB/,"",$2); print $2}')
  echo "[GPU${GPU}] before ${ds}: used=${free} MiB  ($(date '+%F %T'))"
  bash "${HERE}/run_one.sh" "${VARIANT}" "${ds}" "${GPU}" || fail=1
  [ "$fail" -eq 0 ] || { echo "[WAVE-ABORT] ${VARIANT}/${ds} 失败，停止本波"; exit 1; }
done
echo "===== EXP1 WAVE DONE: ${VARIANT} ($(date '+%F %T')) ====="
