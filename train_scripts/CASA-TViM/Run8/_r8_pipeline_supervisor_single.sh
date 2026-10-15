#!/usr/bin/env bash
# Run8 单卡串行流水线督导（v3）。
#
# ## 为什么从双卡改回单卡
# 2026-10-15 22:02 实测：**GPU1 被同门 `bqwang` 的作业占用**
# （2 × 12,362 MiB，`3dinvnet/bin/python study.py --arm reflectivity|gradient`，利用率 100%）。
# 本项目的单 run 峰值显存 **26.9–27.7 GB**，而 GPU1 只剩 ~7.8 GB ⇒ **放不下，也不该抢**。
# ⇒ 用户裁定：**等 WP0 上的 WHU 跑完后，本次 Run8 其余全部训练都在 GPU0 上串行完成**。
#
# ## 纪律（硬性）
#   1. **只使用 `GPU_ID`（默认 0）**。本脚本对每张卡都做"是否有别人作业"的检测，
#      但**只在 GPU0 上启动**；GPU1 上的同门作业**只统计、绝不触碰**（不 kill、不抢占）。
#   2. 单卡串行 ⇒ 一次只有 1 个 run，显存天然安全（无需 flock）。
#   3. 每波四个库**串行**跑完再做下一波（保持"每波唯一变量"的时间边界清晰）。
#   4. 波末门：4/4 有完整 TEST 区块；不足则整波重试（最多 3 轮）。
#   5. 断点续训：有 `last.pth` 且未完成 ⇒ `RESUME_SAME_RUN=1` 从自身续训。
#
# ## 与 v2（双卡并行）的关系
# v2 保留在仓库里（`_r8_pipeline_supervisor.sh`）以便将来双卡空闲时复用；
# **当前生效的是本脚本**。两者对 `python train.py` 的调用参数完全相同，
# 差别只有「设备号」与「并发度」——设备号不是 `run_manifest.json` 的冻结字段。
#
# ## 用法（脱离会话）
#   setsid nohup bash _r8_pipeline_supervisor_single.sh > /tmp/run8/pipe1.stdout 2>&1 < /dev/null &
#   bash _r8_pipeline_supervisor_single.sh --dry        # 只打印计划
set -u

REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8
SUPLOG="$STATE/pipeline_single.log"
mkdir -p "$STATE"

GPU_ID="${GPU_ID:-0}"
POLL_S=120              # 单卡串行，等待更频繁地探（前一个 run 一结束就接上）
DEADLINE_H=200
NEED_MB=30500
MAX_WAVE_RETRY=3
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"

DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

# ⚠️ T0 必须在任何函数调用/fork 之前（v1 曾因它在子 shell 里 unbound 而静默空转四波）
T0=$(date +%s)

log() { echo "[P1] $(date '+%F %T') $*" | tee -a "$SUPLOG"; }

wave_variant() {
  case "$1" in
    1) echo CTRL_R8 ;; 2) echo H1_HYP ;; 3) echo H2_EUC ;; 4) echo H3_HYP_GS ;;
    *) return 2 ;;
  esac
}

own_pids() {
  {
    pgrep -f '^python train\.py' 2>/dev/null
    pgrep -f '_r8_run_one_ds\.sh' 2>/dev/null
    pgrep -f '_r8_pipeline_supervisor' 2>/dev/null
    for p in $(pgrep -f 'python' 2>/dev/null); do
      cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
      case "$cmd" in *"/home/yqwang/projects/CASA-CD"*) echo "$p" ;; esac
    done
  } | sort -u
}

gpu_free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" \
                | awk -F, '{gsub(/ /,"");print $1-$2}'; }

foreign_on_gpu() {   # $1=index；只打印**别人**的 compute 进程 pid（我们的残留放行）
  local idx="$1" pid mine
  mine="$(own_pids)"
  while IFS=, read -r pid _mem; do
    pid="$(echo "${pid:-}" | tr -d ' ')"
    [ -z "$pid" ] && continue
    echo "$mine" | grep -qx "$pid" || echo "$pid"
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$idx" 2>/dev/null)
}

report_neighbours() {
  # 只统计、绝不动手（用户明确要求不打扰同门作业）
  for g in 0 1; do
    local f; f="$(foreign_on_gpu "$g" | tr '\n' ' ' | tr -d ' ')"
    if [ -n "$f" ]; then
      log "邻居检测 GPU$g：别人占用进程 [${f}]（只统计，不触碰）"
    fi
  done
}

wait_gpu_ready() {   # $1=index；0=可以开跑 7=DRY 8=死线
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
      log "DEADLINE ${DEADLINE_H}h 到期"; return 8
    fi
  done
}

ds_done() {   # $1=variant $2=dataset
  local F="$LOG_ROOT/$1/$2/train_log.txt"
  [ -f "$F" ] && grep -q '=== END TEST RESULTS ===' "$F"
}

wave_gate() { local V="$1" n=0; for DS in $DATASETS; do ds_done "$V" "$DS" && n=$((n+1)); done; echo "$n"; }

run_one_stage() {   # $1=variant $2=dataset（固定用 GPU_ID）
  if ds_done "$1" "$2"; then log "SKIP $1/$2 已完成"; return 0; fi

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

  if [ "$DRY" = "1" ]; then log "DRY 本应在 GPU$GPU_ID 启动 $1/$2 (resume=$resume)"; return 0; fi

  # 防御：绝不因为环境变量意外而跑到别的卡上
  local this_gpu="$GPU_ID"
  log "LAUNCH $1/$2 on GPU${this_gpu} (resume=$resume)"
  RESUME_SAME_RUN=$resume bash "$RUN_DIR/_r8_run_one_ds.sh" "$1" "$2" "$this_gpu" \
      > "$STATE/run_$1_$2.log" 2>&1
  local rc=$?
  log "STAGE-DONE $1/$2 rc=$rc"
  return 0
}

log "===== Run8 单卡串行流水线 v3 启动 DRY=${DRY} GPU=${GPU_ID} NEED=${NEED_MB}MB ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /' | tee -a "$SUPLOG"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /' | tee -a "$SUPLOG"
report_neighbours

for W in 1 2 3 4; do
  V="$(wave_variant "$W")" || { log "unknown wave $W"; exit 2; }
  n="$(wave_gate "$V")"
  if [ "$n" -eq 4 ]; then log "WAVE${W}(${V}) 已完成 4/4，跳过"; continue; fi
  log "===== WAVE ${W}: ${V}（当前 ${n}/4 完成）====="

  retry=0
  while [ "$n" -lt 4 ] && [ "$retry" -le "$MAX_WAVE_RETRY" ]; do
    [ "$retry" -gt 0 ] && log "WAVE${W}-RETRY ${retry}/${MAX_WAVE_RETRY}（仍缺 $((4-n)) 个库）"
    wait_gpu_ready "$GPU_ID" || { log "WAVE${W} 拿不到 GPU$GPU_ID，终止"; exit 8; }
    for DS in $DATASETS; do
      run_one_stage "$V" "$DS"
      # 前一个 run 刚退出时显存还没释放干净（实测会看到 free 只有 ~17 GB），
      # 直接进 wait_gpu_ready 会白等一个轮询周期 ⇒ 先给 20 s 让 CUDA 上下文回收。
      sleep 20
      # 每个 run 结束后都重探一次（同门若中途上卡也能看到）
      wait_gpu_ready "$GPU_ID" || { log "WAVE${W} 后续库拿不到 GPU$GPU_ID，终止"; exit 8; }
    done
    n="$(wave_gate "$V")"
    log "WAVE${W}-GATE ${n}/4"
    retry=$((retry+1))
  done
  if [ "$n" -lt 4 ]; then
    log "WAVE${W}-INCOMPLETE 只有 ${n}/4（已重试 ${MAX_WAVE_RETRY} 轮）——继续下一波，收口链会据此判 INVALID/INCOMPLETE"
  fi
done

log "ALL-WAVES-DONE"
