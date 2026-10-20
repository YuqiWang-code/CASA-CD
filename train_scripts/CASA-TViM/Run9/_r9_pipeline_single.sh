#!/usr/bin/env bash
# Run9 单卡串行督导（`bash _r9_pipeline_single.sh`）。
#
# 波次按**数据集**交错，而不是按变体整波跑完：
#   CDD: S0_CTRL -> S1_HSAM
#   LEVIR: S0_CTRL -> S1_HSAM
#   SYSU: S0_CTRL -> S1_HSAM
#   WHU: S0_CTRL -> S1_HSAM
# 这样每一个库都有**紧邻的同期 CTRL**（唯一变量 = HSAM），
# 并且在前两个库跑完（≈29 h）就能拿到 ΔF1 做预注册的早停判定。
#
# 相对 Run8b 督导的三处修正：
#   1. **GPU 判据只看余量**（不再要求"卡上没有别人的进程"，那会让督导空转）；
#   2. **停止时先杀重试宿主再杀训练**（见 `_r9_stop.sh`）；本脚本自身收到 TERM 立即退出；
#   3. 每拍决策写 `_r9_pipeline.log`，便于事后审计。
set -u

REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run9"
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run9
STATE=/tmp/run9
mkdir -p "$STATE" "$LOG_ROOT"
LOG="$STATE/pipeline.log"

GPU_ID="${GPU_ID:-0}"
NEED_MB="${NEED_MB:-30000}"      # 实测 HSAM 峰值 26,943 MiB；留 ~3 GB 余量
POLL_S="${POLL_S:-20}"
DRY="${DRY:-0}"
# 预注册早停：前两个库两边都跑完后，若 ΔF1 在两个库上都 ≤ 0 则停（见 README §6）
EARLY_GATE="${EARLY_GATE:-1}"

DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"
VARIANTS="S0_CTRL S1_HSAM"

T0=$(date +%s)
log() { echo "[P9] $(date '+%F %T') $*" | tee -a "$LOG"; }
el()  { echo "$(( ($(date +%s) - T0) / 60 ))m"; }

trap 'log "收到 TERM/INT：督导退出（训练进程由 _r9_stop.sh 单独处理）"; exit 130' TERM INT

# ---------------------------------------------------------------- GPU 余量
foreign_on_gpu() {
  local g="$1"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r uuid pid mem; do
        idx=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader \
              | awk -F', ' -v u="$(echo "$uuid" | tr -d ' ')" '$2==u{print $1; exit}')
        [ "${idx:-x}" = "$g" ] || continue
        echo "$(echo "$pid" | tr -d ' ')"
      done | grep -v '^$' || true
}

wait_gpu_ready() {
  local idx="$1"
  while true; do
    local free foreign
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
    foreign=$(foreign_on_gpu "$idx" | tr '\n' ' ' | tr -d ' ')
    if [ "${free:-0}" -ge "$NEED_MB" ]; then
      if [ -z "$foreign" ]; then
        log "GPU${idx}-READY free=${free}MB foreign=none"; return 0
      fi
      log "GPU${idx}-READY(shared) free=${free}MB foreign=[${foreign}]（只共存，不触碰）"; return 0
    fi
    log "GPU${idx}-WAIT free=${free:-?}MB need=${NEED_MB}MB foreign=[${foreign}] -> ${POLL_S}s 后再探"
    sleep "$POLL_S"
  done
}

# ---------------------------------------------------------------- 单库
run_one() {
  local V="$1" DS="$2"
  local lf="$LOG_ROOT/$V/$DS/train_log.txt"
  if [ -s "$lf" ] && grep -q '=== END TEST RESULTS ===' "$lf"; then
    log "SKIP $V/$DS 已完成"; return 0
  fi
  wait_gpu_ready "$GPU_ID" || { log "拿不到 GPU$GPU_ID"; return 8; }
  if [ "$DRY" = "1" ]; then log "DRY 本应在 GPU$GPU_ID 启动 $V/$DS"; return 0; fi
  log "START $V/$DS (elapsed $(el))"
  bash "$RUN_DIR/_r9_run_one_ds.sh" "$V" "$DS" "$GPU_ID" >> "$STATE/stage.log" 2>&1
  local rc=$?
  log "STAGE-DONE $V/$DS rc=$rc (elapsed $(el))"
  return $rc
}

done_count() {
  local V="$1" n=0
  for DS in $DATASETS; do
    local lf="$LOG_ROOT/$V/$DS/train_log.txt"
    [ -s "$lf" ] && grep -q '=== END TEST RESULTS ===' "$lf" && n=$((n+1))
  done
  echo "$n"
}

# 预注册早停判定：CDD/LEVIR 两个库都两边完成时，比较 ΔF1
early_gate() {
  [ "$EARLY_GATE" = "1" ] || return 0
  local ok=1
  for DS in CDD-CD-256 LEVIR-CD-256; do
    for V in $VARIANTS; do
      local lf="$LOG_ROOT/$V/$DS/train_log.txt"
      [ -s "$lf" ] && grep -q '=== END TEST RESULTS ===' "$lf" || ok=0
    done
  done
  [ "$ok" = "1" ] || return 0
  local res
  res=$(python3 "$RUN_DIR/_r9_delta.py" 2>/dev/null) || { log "EARLY-GATE 解析失败，继续跑"; return 0; }
  log "EARLY-GATE $res"
  case "$res" in
    *STOP*) log "EARLY-GATE => STOP（前两库 ΔF1 均 ≤ 0，按预注册停跑）"; return 9 ;;
    *) log "EARLY-GATE => CONTINUE"; return 0 ;;
  esac
}

# ---------------------------------------------------------------- 主流程
log "===== Run9 单卡串行督导启动 DRY=${DRY} GPU=${GPU_ID} NEED=${NEED_MB}MB EARLY_GATE=${EARLY_GATE} ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$LOG"
log "波次：按数据集交错（每库先 CTRL 再 HSAM）"

for DS in $DATASETS; do
  log "===== DATASET: $DS ====="
  for V in $VARIANTS; do
    run_one "$V" "$DS"; rc=$?
    if [ "$rc" = "4" ]; then log "HSAM-FAILFAST $V/$DS ⇒ 终止整轮（数值/退化问题，继续跑没意义）"; exit 4; fi
    [ "$rc" -ne 0 ] && log "WARN $V/$DS rc=$rc（继续下一个）"
  done
  early_gate; eg=$?
  [ "$eg" = "9" ] && exit 9
done

log "===== 全部完成：S0_CTRL $(done_count S0_CTRL)/4，S1_HSAM $(done_count S1_HSAM)/4，用时 $(el) ====="
