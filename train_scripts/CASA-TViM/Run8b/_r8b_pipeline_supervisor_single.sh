#!/usr/bin/env bash
# Run8b 单卡串行流水线督导（v3 模板，只换 Run8b 的路径/变体）。
#
# ## 与 Run8 的差别（只有这些）
#   * 变体名：CTRL_R8B / B1_HYP / B2_EUC / B3_HYP_GS
#   * 日志/ckpt 根：`…/Run8b/`（**绝不写 Run8 的目录**）
#   * 编排器：`_r8b_run_one_ds.sh`（带 `--arch casa_vmamba_str_v2 --hqkv_gamma_init`）
#   * 失败语义：编排器返回 **rc=3 = fail-fast（抗自锁检查未过）⇒ 整条链立即停止**，
#     而不是像 Run8 那样继续跑下一个库（Run8 的教训：自锁跑满 80K 才发现，白烧算力）。
#
# ## 硬纪律（继承 Run8，全部是实测教训）
#   1. **只用 GPU_ID（默认 0）**；GPU1 上是探索线（另一对话框），**只统计、绝不触碰**；
#   2. 每张卡同时只放 1 个 run（峰值 27.5 GB × 2 > 32.6 GB）；
#   3. `T0` 必须在任何函数调用/fork 之前赋值（v1 曾因它在子 shell 里 unbound 而静默空转四波）；
#   4. 波末 4/4 门；不足则整波重试（最多 3 轮）；
#   5. 断点续训（`last.pth` + `RESUME_SAME_RUN=1`），续训前封存未完成日志；
#   6. 每个 run 结束后 sleep 20s 让 CUDA 上下文释放。
set -u

REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8b"
STATE=/tmp/run8b
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8b
SUPLOG="$STATE/pipeline.log"
mkdir -p "$STATE"

GPU_ID="${GPU_ID:-0}"
POLL_S=120
DEADLINE_H=200
# 单 run 的显存**实测**峰值：Run8b batch32 = 28,371 MB（B1_HYP）/ 26,864 MB（CTRL）。
# 原阈值 30500 = 峰值 + ~2.1 GB 余量；但 2026-10-19 08:04 起 GPU0 上出现同门 `bqwang` 的
# 小作业（1,938 MiB，脚本名带 lowmem）⇒ free 只剩 30,495 MB，**差 5 MB** 卡在门外，
# 督导空转数小时。改为**可配置**（默认 30000 = 峰值 + 1.6 GB 余量），
# 并保留"卡上有别人的进程时只统计"的原则——我们用**更小的余量**去共享，而不是去抢。
NEED_MB="${NEED_MB:-30000}"
MAX_WAVE_RETRY=3
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"

DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

# T0 必须在任何函数调用与 fork 之前（Run8 的致命 bug，见上 ③）
T0=$(date +%s)

log() { echo "[P1B] $(date '+%F %T') $*" | tee -a "$SUPLOG"; }

wave_variant() {
  case "$1" in
    1) echo CTRL_R8B ;; 2) echo B1_HYP ;; 3) echo B2_EUC ;; 4) echo B3_HYP_GS ;;
    *) return 2 ;;
  esac
}

own_pids() {
  {
    pgrep -f '^python train\.py' 2>/dev/null
    pgrep -f '_r8b_run_one_ds\.sh' 2>/dev/null
    pgrep -f '_r8b_pipeline_supervisor' 2>/dev/null
    for p in $(pgrep -f 'python' 2>/dev/null); do
      cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
      case "$cmd" in *"/home/yqwang/projects/CASA-CD"*) echo "$p" ;; esac
    done
  } | sort -u
}

gpu_free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" \
                | awk -F, '{gsub(/ /,"");print $1-$2}'; }

foreign_on_gpu() {
  local idx="$1" pid mine
  mine="$(own_pids)"
  while IFS=, read -r pid _mem; do
    pid="$(echo "${pid:-}" | tr -d ' ')"
    [ -z "$pid" ] && continue
    echo "$mine" | grep -qx "$pid" || echo "$pid"
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$idx" 2>/dev/null)
}

report_neighbours() {
  for g in 0 1; do
    local f; f="$(foreign_on_gpu "$g" | tr '\n' ' ' | tr -d ' ')"
    [ -n "$f" ] && log "邻居检测 GPU$g：别人占用进程 [${f}]（只统计，不触碰）"
  done
}

wait_gpu_ready() {
  local idx="$1"
  while true; do
    local free foreign
    free="$(gpu_free_mb "$idx")"
    foreign="$(foreign_on_gpu "$idx" | tr '\n' ' ' | tr -d ' ')"
    # 判据：只看**真实余量**，不要求"卡上完全没有别人的进程"。
    # 理由（2026-10-19 08:04 实测）：同门在 GPU0 上跑了一个 1,938 MiB 的 lowmem 小作业，
    # 卡上于是永远有 foreign ⇒ 旧的 `-z "$foreign"` 条件会让督导**无限期空转**。
    # 本项目的单 run 峰值是实测的 28.4 GB，所以只要 free ≥ NEED_MB 就能安全共卡；
    # 我们**不**为了独占而等待，也**不**触碰别人的进程。
    if [ "${free:-0}" -ge "$NEED_MB" ]; then
      if [ -z "$foreign" ]; then
        log "GPU${idx}-READY free=${free}MB foreign=none"; return 0
      fi
      log "GPU${idx}-READY(shared) free=${free}MB foreign=[${foreign}]（只共存，不触碰）"; return 0
    fi
    log "GPU${idx}-WAIT free=${free:-?}MB need=${NEED_MB}MB foreign=[${foreign}] -> ${POLL_S}s 后再探"
    [ "$DRY" = "1" ] && return 7
    sleep "$POLL_S"
    if [ $(( $(date +%s) - T0 )) -gt $(( DEADLINE_H * 3600 )) ]; then
      log "DEADLINE ${DEADLINE_H}h 到期"; return 8
    fi
  done
}

ds_done() {
  local F="$LOG_ROOT/$1/$2/train_log.txt"
  [ -f "$F" ] && grep -q '=== END TEST RESULTS ===' "$F"
}

wave_gate() { local V="$1" n=0; for DS in $DATASETS; do ds_done "$V" "$DS" && n=$((n+1)); done; echo "$n"; }

run_one_stage() {   # 返回 0=完成 3=fail-fast（应整链停止）
  if ds_done "$1" "$2"; then log "SKIP $1/$2 已完成"; return 0; fi
  local resume=0 last="$CKPT_ROOT/$1/$2/last.pth"
  if [ -s "$last" ] && [ "$(wc -c < "$last" 2>/dev/null || echo 0)" -gt 1000 ]; then
    resume=1; log "RESUME $1/$2 检测到 last.pth($(wc -c < "$last")B)"
  fi
  local lf="$LOG_ROOT/$1/$2/train_log.txt"
  if [ "$resume" = "1" ] && [ -s "$lf" ] && ! grep -q '=== END TEST RESULTS ===' "$lf"; then
    mv "$lf" "$lf.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_before_resume"
    log "ARCHIVE $1/$2 未完成日志已封存（续训前）"
  fi
  if [ "$DRY" = "1" ]; then log "DRY 本应在 GPU$3 启动 $1/$2 (resume=$resume)"; return 0; fi
  RESUME_SAME_RUN=$resume bash "$RUN_DIR/_r8b_run_one_ds.sh" "$1" "$2" "$3" \
      > "$STATE/run_$1_$2.log" 2>&1
  local rc=$?
  log "STAGE-DONE $1/$2 rc=$rc"
  [ "$rc" = "3" ] && return 3
  return 0
}

log "===== Run8b 单卡串行流水线启动 DRY=${DRY} GPU=${GPU_ID} NEED=${NEED_MB}MB ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /' | tee -a "$SUPLOG"
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
      run_one_stage "$V" "$DS" "$GPU_ID"
      rc=$?
      if [ "$rc" = "3" ]; then
        log "WAVE${W} FAILFAST：$V/$DS 抗自锁检查未过 ⇒ **整条链立即停止**（不继续跑，节省算力）"
        exit 3
      fi
      sleep 20
      wait_gpu_ready "$GPU_ID" || { log "WAVE${W} 后续库拿不到 GPU$GPU_ID，终止"; exit 8; }
    done
    n="$(wave_gate "$V")"
    log "WAVE${W}-GATE ${n}/4"
    retry=$((retry+1))
  done
  if [ "$n" -lt 4 ]; then
    log "WAVE${W}-INCOMPLETE 只有 ${n}/4（已重试 ${MAX_WAVE_RETRY} 轮）"
  fi
done

log "ALL-WAVES-DONE"
