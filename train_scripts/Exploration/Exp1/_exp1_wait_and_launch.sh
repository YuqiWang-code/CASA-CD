#!/usr/bin/env bash
# Exp1-v2 等卡督导：GPU1 一旦真正空出，先跑完整预检，PASS 后才启动 STG_V2（LEVIR → SYSU）。
#
# 纪律（源自 2026-10-15 的教训："一次快照里卡空着 ≠ 卡归我了"）：
#   · 连续 NEED_CONSEC 次（默认 2）快照都满足「显存 ≤ FREE_MB 且该卡无 compute 进程」才算空；
#   · 对同门/他人进程**只检测、绝不触碰**（从不 kill）；
#   · 若同门中途回来导致我们 OOM，train_stgeo 会从自身 last.pth 续训（有损但可恢复）；
#   · 全程只碰 GPU1。
set -u
GPU="${1:-1}"
VARIANT="${2:-STG_V2}"
DATASETS="${3:-LEVIR-CD-256 SYSU-CD-256}"
FREE_MB=4000            # 该卡已用 ≤ 4 GB 视为空（我们单 run 峰值 ≈27.8 GB，余量充足）
NEED_CONSEC=2
POLL=120
MAX_H=72                # 最多等 72 h
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE=/tmp/exp1; mkdir -p "$STATE"
LOG="$STATE/wait_${VARIANT}.log"
say(){ echo "[WAIT $(date '+%F %T')] $*" | tee -a "$LOG"; }

# 已有 run_all 在跑就直接退出（避免双开）
if [ -f "$STATE/run_all.pid" ] && kill -0 "$(cat "$STATE/run_all.pid")" 2>/dev/null; then
  say "run_all 已在运行 pid=$(cat "$STATE/run_all.pid")，本督导退出"; exit 3
fi
if pgrep -f "[t]rain_stgeo\.py" >/dev/null 2>&1; then
  say "已有 train_stgeo 进程在跑，本督导退出"; exit 3
fi

gpu_used_mb() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader |
  awk -F, -v g="$GPU" '$1+0==g {gsub(/ MiB/,"",$2); print $2+0}'; }
gpu_uuid() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
  awk -F', ' -v g="$GPU" '$1+0==g {print $2}'; }
gpu_apps() {  # 该卡上的 compute 进程数
  local u; u=$(gpu_uuid)
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader |
    awk -F', ' -v u="$u" '$1==u {c++} END{print c+0}'; }

say "arm: GPU=$GPU variant=$VARIANT datasets='$DATASETS' free_mb<=$FREE_MB consec=$NEED_CONSEC poll=${POLL}s max=${MAX_H}h"
consec=0; waited=0
while [ "$waited" -lt $((MAX_H*3600)) ]; do
  used=$(gpu_used_mb); apps=$(gpu_apps)
  if [ "${used:-99999}" -le "$FREE_MB" ] && [ "${apps:-9}" -eq 0 ]; then
    consec=$((consec+1))
    say "free snapshot ${consec}/${NEED_CONSEC}: used=${used}MB apps=${apps}"
    if [ "$consec" -ge "$NEED_CONSEC" ]; then
      say "GPU${GPU} 连续 ${consec} 次快照为空 ⇒ 先跑完整预检"
      if bash "$HERE/_exp1_preflight.sh" "$GPU" >> "$LOG" 2>&1; then
        say "预检 PASS ⇒ 启动 ${VARIANT}（$(echo $DATASETS)）"
        exec bash "$HERE/run_all.sh" "$VARIANT" "$GPU" $DATASETS >> "$LOG" 2>&1
      else
        say "预检 FAIL（见 $LOG）⇒ 不启动训练；督导退出，交人工处理"
        exit 4
      fi
    fi
  else
    if [ "$consec" -gt 0 ]; then say "回落：used=${used}MB apps=${apps}（同门回来了）"; fi
    consec=0
    say "occupied: used=${used}MB apps=${apps} —— 等待（不触碰他人进程）"
  fi
  sleep "$POLL"; waited=$((waited+POLL))
done
say "超时 ${MAX_H}h 仍无空位，退出"; exit 5
