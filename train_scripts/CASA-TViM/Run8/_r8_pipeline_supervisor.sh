#!/usr/bin/env bash
# Run8 双卡并行流水线督导（v2，重写版）。
#
# ## 它做什么
#   1. 按波次顺序（1=CTRL_R8, 2=H1_HYP, 3=H2_EUC, 4=H3_HYP_GS）遍历；
#   2. 每波把 4 个库配成 2 条通道：GPU0 ← CDD,WHU / GPU1 ← SYSU,LEVIR；
#   3. **每张卡同时只放 1 个 run**（峰值 27.5 GB × 2 > 32.6 GB ⇒ 两个必 OOM）；
#   4. 开跑前用 `nvidia-smi` 实测该卡余量，且**只认"别人的进程"为 foreign**
#      （我们自己的残留放行）；
#   5. 波末检查 4/4 完整 TEST 区块；不足则整波重试（最多 3 轮）；
#   6. 对别人的进程**只检测、从不抢占、从不 kill**。
#
# ## 为什么重写（v1 的三个致命 bug，全部由实测暴露）
#   ① **`T0: unbound variable`**：`T0=$(date +%s)` 写在函数定义之后，而通道是在
#      **fork 出来的子 shell** 里跑 `run_stage` ⇒ 子 shell 里 T0 未定义 ⇒ `set -u` 下
#      `wait_gpu_ready` 立刻失败 ⇒ `run_stage` 直接 return ⇒ **两个通道瞬间"结束"**
#      ⇒ 督导在 20 分钟内把四波全部"空转"完并打印 ALL-WAVES-DONE（波二/三/四一个都没跑）。
#      ⇒ v2 把 `T0` 放在文件最顶部，并且 `run_stage` 用**显式返回码**区分"拿不到卡"与"跑完"。
#   ② **波门不闭环**：v1 在 `WAVE1-GATE 1/4` 之后不回来补 WHU，而是直接跳去波二。
#      ⇒ v2 加**整波重试**（最多 3 轮）。
#   ③ **没有并发保护**：多个督导实例可能同时存活并抢同一张卡。
#      ⇒ v2 用 `flock` 给每张卡上硬锁（`/tmp/run8/.gpu<N>.lock`）。
#
# ## 用法
#   setsid nohup bash _r8_pipeline_supervisor.sh > /tmp/run8/pipe.stdout 2>&1 < /dev/null &
#   bash _r8_pipeline_supervisor.sh --dry     # 只打印计划，不启动任何东西
set -u

REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8
SUPLOG="$STATE/pipeline.log"
mkdir -p "$STATE"

POLL_S=300
DEADLINE_H=200          # 四波双卡 ≈ 50 h；给 200 h 余量（不轻易放弃）
NEED_MB=30500           # 单 run 27.5 GB + 余量
MAX_WAVE_RETRY=3
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"
# 每波两条通道的配库（把最慢的 SYSU 先起，让两张卡尽量同时收尾）
GPU0_SEQ="CDD-CD-256 WHU-CD-256"
GPU1_SEQ="SYSU-CD-256 LEVIR-CD-256"

DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

# ⚠️ T0 必须在**任何函数调用与 fork 之前**赋值（v1 的致命 bug 见文件头 ①）
T0=$(date +%s)

log() { echo "[PIPE] $(date '+%F %T') $*" | tee -a "$SUPLOG"; }

wave_variant() {
  case "$1" in
    1) echo CTRL_R8 ;;
    2) echo H1_HYP ;;
    3) echo H2_EUC ;;
    4) echo H3_HYP_GS ;;
    *) return 2 ;;
  esac
}

# ---- 本项目自己的进程：三重并集（宁可多认自己的，绝不多认别人的）
own_pids() {
  {
    pgrep -f '^python train\.py' 2>/dev/null
    pgrep -f '_r8_run_one_ds\.sh' 2>/dev/null
    pgrep -f '_r8_pipeline_supervisor\.sh' 2>/dev/null
    for p in $(pgrep -f 'python' 2>/dev/null); do
      cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
      case "$cmd" in *"/home/yqwang/projects/CASA-CD"*) echo "$p" ;; esac
    done
  } | sort -u
}

gpu_free_mb() {   # $1 = index
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" \
    | awk -F, '{gsub(/ /,"");print $1-$2}'
}

foreign_on_gpu() {   # $1 = index；打印非本项目的 compute 进程 pid
  # ⚠️ `--format=csv,noheader` 是 `pid, mem`（**带逗号**）⇒ 必须 IFS=,，
  # 否则整行进第一个字段，`grep -qx` 永远失配（v1 曾因此把自己在跑的 run 判成 foreign）。
  local idx="$1" pid mine
  mine="$(own_pids)"
  while IFS=, read -r pid _mem; do
    pid="$(echo "${pid:-}" | tr -d ' ')"
    [ -z "$pid" ] && continue
    echo "$mine" | grep -qx "$pid" || echo "$pid"
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$idx" 2>/dev/null)
}

wait_gpu_ready() {   # $1 = index；0=可用 7=DRY 8=死线
  local idx="$1"
  while true; do
    local free foreign
    free="$(gpu_free_mb "$idx")"
    foreign="$(foreign_on_gpu "$idx" | tr '\n' ' ' | tr -d ' ')"
    if [ -z "$foreign" ] && [ "${free:-0}" -ge "$NEED_MB" ]; then
      log "GPU${idx}-READY free=${free}MB foreign=none"
      return 0
    fi
    log "GPU${idx}-WAIT free=${free:-?}MB foreign=[${foreign}] -> ${POLL_S}s 后再探"
    [ "$DRY" = "1" ] && return 7
    sleep "$POLL_S"
    if [ $(( $(date +%s) - T0 )) -gt $(( DEADLINE_H * 3600 )) ]; then
      log "DEADLINE ${DEADLINE_H}h 到期，停止等待 GPU${idx}"
      return 8
    fi
  done
}

ds_done() {   # $1=variant $2=dataset
  local F="$LOG_ROOT/$1/$2/train_log.txt"
  [ -f "$F" ] && grep -q '=== END TEST RESULTS ===' "$F"
}

wave_gate() {   # $1=variant；打印并返回完成库数
  local V="$1" n=0
  for DS in $DATASETS; do ds_done "$V" "$DS" && n=$((n+1)); done
  echo "$n"
}

run_one_stage() {   # $1=variant $2=dataset $3=gpu；**调用者必须已持有该 GPU 的 flock**
  if ds_done "$1" "$2"; then log "SKIP $1/$2 已完成"; return 0; fi

  # 断点续训：有 last.pth 且未完成 ⇒ 用 RESUME_SAME_RUN=1 从自身续训
  local resume=0
  local last="$CKPT_ROOT/$1/$2/last.pth"
  if [ -s "$last" ] && [ "$(wc -c < "$last" 2>/dev/null || echo 0)" -gt 1000 ]; then
    resume=1
    log "RESUME $1/$2 检测到 last.pth($(wc -c < "$last")B)"
  fi
  local lf="$LOG_ROOT/$1/$2/train_log.txt"
  if [ "$resume" = "1" ] && [ -s "$lf" ] && ! grep -q '=== END TEST RESULTS ===' "$lf"; then
    mv "$lf" "$lf.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_before_resume"
    log "ARCHIVE $1/$2 未完成日志已封存（续训前）"
  fi

  if [ "$DRY" = "1" ]; then log "DRY 本应在 GPU$3 启动 $1/$2 (resume=$resume)"; return 0; fi

  RESUME_SAME_RUN=$resume bash "$RUN_DIR/_r8_run_one_ds.sh" "$1" "$2" "$3" \
      > "$STATE/run_$1_$2.log" 2>&1
  local rc=$?
  log "STAGE-DONE $1/$2 gpu=$3 rc=$rc"
  return 0     # 单库失败不阻断整波；波门会看到它没完成并触发整波重试
}

gpu_channel() {   # $1=variant $2=gpu $3=库序列
  local V="$1" G="$2" SEQ="$3"
  exec 9>"/tmp/run8/.gpu$G.lock"
  if ! flock -w 0 9; then
    log "GPU$G 已被别的通道占用，本通道等待 60s"
    [ "$DRY" = "1" ] && return 0
    sleep 60
    flock -w 0 9 || { log "GPU$G 仍忙，本通道放弃本轮"; return 1; }
  fi
  wait_gpu_ready "$G" || { flock -u 9; return 1; }
  for DS in $SEQ; do run_one_stage "$V" "$DS" "$G"; done
  flock -u 9
  return 0
}

log "===== Run8 双卡并行流水线 v2 启动 DRY=${DRY} NEED=${NEED_MB}MB POLL=${POLL_S}s ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /' | tee -a "$SUPLOG"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /' | tee -a "$SUPLOG"

for W in 1 2 3 4; do
  V="$(wave_variant "$W")" || { log "unknown wave $W"; exit 2; }
  n="$(wave_gate "$V")"
  if [ "$n" -eq 4 ]; then log "WAVE${W}(${V}) 已完成 4/4，跳过"; continue; fi
  log "===== WAVE ${W}: ${V}（当前 ${n}/4 完成）====="

  retry=0
  while [ "$n" -lt 4 ] && [ "$retry" -le "$MAX_WAVE_RETRY" ]; do
    [ "$retry" -gt 0 ] && log "WAVE${W}-RETRY ${retry}/${MAX_WAVE_RETRY}（仍缺 $((4-n)) 个库）"
    gpu_channel "$V" 0 "$GPU0_SEQ" &
    C0=$!
    gpu_channel "$V" 1 "$GPU1_SEQ" &
    C1=$!
    wait "$C0"; wait "$C1"
    log "WAVE${W} 两条通道均结束"
    n="$(wave_gate "$V")"
    log "WAVE${W}-GATE ${n}/4"
    retry=$((retry+1))
  done
  if [ "$n" -lt 4 ]; then
    log "WAVE${W}-INCOMPLETE 只有 ${n}/4（已重试 ${MAX_WAVE_RETRY} 轮）——继续下一波，收口链会据此判 INVALID/INCOMPLETE"
  fi
done

log "ALL-WAVES-DONE"
