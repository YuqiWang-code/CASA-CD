#!/usr/bin/env bash
# Run8 波次督导：**只在 GPU0 有余量时**依次接续波二/三/四。
#
# 设计原则（按用户 2026-10-15 的授权）：
#   「4 个数据集都可以上服务器跑，波二三四自动接续，只要服务器有空位
#     （但是不要打扰别人的训练进程）」。
#
# 因此本督导**不做**任何抢占、不 kill 任何非本项目的进程；它只做两件事：
#   1) 等前一波跑完（判据：正式日志里出现 `[WAVE k done]`，而不是看进程在不在）；
#   2) **探到 GPU0 真实余量 ≥ 阈值**才启动下一波。
#
# 为什么要探余量（实测依据）：
#   Run8 batch32 的峰值显存实测 **26.9–27.7 GB**（32.6 GB 卡，余量仅 ~5 GB）。
#   Run7 波三就是因为同卡压了两个重库 + 同用户别的项目作业而 **连续 OOM 3 次被判失败**。
#   所以「卡上有多少空闲」必须由 `nvidia-smi` 实测决定，不能假设。
#
# 判据（全部可复核，见 [SUP] 日志行）：
#   free = memory.total − memory.used（GPU0）
#   还要扣除 **本项目自己** 的残留占用（RESUME 时自己的 last.pth 会先占一点）
#   ⇒ 需要 free ≥ PEAK_MB + MARGIN_MB，且 GPU0 上没有**别人**的 compute 进程。
#
# 用法（脱离会话拉起）：
#   setsid nohup bash _r8_supervisor.sh > /tmp/run8/sup.stdout 2>&1 < /dev/null &
# 自检（只打印判断、不启动任何东西）：
#   bash _r8_supervisor.sh --dry
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
SUPLOG="$STATE/supervisor.log"

PEAK_MB=28000        # 实测 26.9–27.7 GB，取 28 GB 作为需要量
MARGIN_MB=2500       # 留给碎片化与别人的小波动（expandable_segments 下仍要留）
POLL_S=300           # 每 5 分钟探一次
# 死线必须 > 单波实测耗时，否则会把还在正常跑的波判死。
# 实测（2026-10-15，batch32、GPU0 串行）：step = 0.3227 s ⇒ 单库 80K steps ≈ **7.17 h**
# ⇒ 单波（四库）= **28.7 h**；四波 ≈ 109 h（4.5 天）。取 60 h 留 2× 余量。
DEADLINE_H=60

DRY=0
[ "${1:-}" = "--dry" ] && DRY=1
mkdir -p "$STATE"

log() { echo "[SUP] $(date '+%F %T') $*" | tee -a "$SUPLOG"; }

# 本项目自己的 GPU 进程 pid 集合（用于把"自己的残留"和"别人的作业"分开）
own_pids() {
  { pgrep -f '^python train\.py' 2>/dev/null; pgrep -f 'run_all\.sh' 2>/dev/null; } | sort -u
}

gpu_free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 \
                | awk -F, '{gsub(/ /,"");print $1-$2}'; }

# GPU0 上是否有**非本项目**的 compute 进程
foreign_on_gpu0() {
  local row pid
  local mine; mine="$(own_pids)"
  while read -r pid _rest; do
    pid="$(echo "$pid" | tr -d ' ')"
    [ -z "$pid" ] && continue
    if ! echo "$mine" | grep -qx "$pid"; then
      echo "$pid"
    fi
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i 0 2>/dev/null)
}

wait_for_slot() {
  local need=$(( PEAK_MB + MARGIN_MB ))
  while true; do
    local free foreign
    free="$(gpu_free_mb)"
    foreign="$(foreign_on_gpu0 | tr '\n' ' ')"
    if [ -z "$(printf '%s' "$foreign" | tr -d ' ')" ] && [ "${free:-0}" -ge "$need" ]; then
      log "SLOT-OK free=${free}MB need=${need}MB foreign=none"
      return 0
    fi
    log "SLOT-WAIT free=${free:-?}MB need=${need}MB foreign=[${foreign}] -> ${POLL_S}s 后再探"
    [ "$DRY" = "1" ] && return 7
    sleep "$POLL_S"
    if [ $(( $(date +%s) - T0 )) -gt $(( DEADLINE_H * 3600 )) ]; then
      log "DEADLINE 超过 ${DEADLINE_H}h，放弃接续（已跑完的波次不受影响）"
      return 8
    fi
  done
}

wave_done() {
  local w="$1"
  # 判据：本波四个正式日志都要有完整 TEST 区块（不是看进程在不在）
  local V
  case "$w" in
    1) V=CTRL_R8 ;; 2) V=H1_HYP ;; 3) V=H2_EUC ;; 4) V=H3_HYP_GS ;;
    *) log "unknown wave $w"; return 2 ;;
  esac
  local n=0
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    local F="$LOG_ROOT/$V/$DS/train_log.txt"
    [ -f "$F" ] && grep -q '=== END TEST RESULTS ===' "$F" && n=$((n+1))
  done
  [ "$n" -eq 4 ]
}

wait_wave_done() {
  local w="$1"
  while ! wave_done "$w"; do
    log "WAIT wave${w} 未完成（正式日志 TEST 区块 <4/4）-> ${POLL_S}s 后再探"
    [ "$DRY" = "1" ] && return 7
    sleep "$POLL_S"
    if [ $(( $(date +%s) - T0 )) -gt $(( DEADLINE_H * 3600 )) ]; then
      log "DEADLINE 超过 ${DEADLINE_H}h，停止等待 wave${w}"
      return 8
    fi
  done
  log "WAVE${w}-DONE 4/4 正式日志齐备"
}

start_wave() {
  local w="$1"
  if [ "$DRY" = "1" ]; then log "DRY 本应启动 wave${w}"; return 0; fi
  bash "$RUN_DIR/_r8_launch.sh" --wave "$w" 2>&1 | sed 's/^/  /' | tee -a "$SUPLOG"
}

T0=$(date +%s)
log "===== Run8 supervisor 启动 DRY=${DRY} PEAK=${PEAK_MB}MB MARGIN=${MARGIN_MB}MB POLL=${POLL_S}s ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /' | tee -a "$SUPLOG"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /' | tee -a "$SUPLOG"

for W in 2 3 4; do
  wait_wave_done $((W - 1)) || exit $?
  wait_for_slot || exit $?
  start_wave "$W"
  # 给新波 3 分钟落地，避免"刚启动就判 wave_done"
  sleep 180
done

log "===== 全部波次已启动完毕；等待最后一波完成 ====="
wait_wave_done 4 || exit $?
log "ALL-WAVES-DONE"
