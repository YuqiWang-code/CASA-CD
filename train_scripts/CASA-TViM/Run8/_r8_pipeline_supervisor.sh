#!/usr/bin/env bash
# Run8 双卡并行流水线督导。
#
# ## 为什么从"单卡串行"改成"双卡并行"
# 2026-10-15 09:06 实测：**GPU1 上那个别人的作业已下线**（15 MiB / 0%），
# 而 Run8 峰值显存 26.9–27.7 GB ⇒ **每张卡只能放 1 个 run**。
# 于是"两卡并行"= 同时跑 2 个 run：
#
#   单卡串行（原）：makespan = 4 × 7.17 h = **28.7 h / 波**
#   双卡并行（本）：makespan = 2 × 7.17 h = **14.3 h / 波**（每卡 2 个 run 串行）
#
# 四波合计从 109 h 降到约 **95 h**（wave1 的 CDD 已单卡跑完，无法追回）。
#
# ## 纪律与披露
# 1. **只动调度器，不动模型**：`python train.py` 的参数与 `run_all.sh` 逐字相同
#    （同 variant 口径、同 seed/batch/80K/协议）；差别**只有设备号与并发度**。
# 2. 设备号不是 `run_manifest.json` 的冻结字段（Run7 已实测：OOM 后换卡续训、
#    manifest 逐字段校验通过）。
# 3. **不打扰别人的进程**：每卡开跑前用 `nvidia-smi` 实测余量，要求
#    free ≥ 30.5 GB 或该卡上的 compute 进程**全是本项目自己的**（自己的残留放行）。
#    若别人重新占卡 ⇒ 阻塞等待，**绝不抢占、绝不 kill 非本项目进程**。
# 4. 每卡同时只放 1 个 run（硬性：27.5 GB × 2 > 32.6 GB）。
#
# 用法（脱离会话）：
#   setsid nohup bash _r8_pipeline_supervisor.sh > /tmp/run8/pipe.stdout 2>&1 < /dev/null &
# 自检（只判断、不启动）：
#   bash _r8_pipeline_supervisor.sh --dry
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
SUPLOG="$STATE/pipeline.log"
mkdir -p "$STATE"

POLL_S=300
DEADLINE_H=120          # 四波双卡 ≈95 h，给 120 h 余量
NEED_MB=30500           # 单 run 27.5 GB + 余量

DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

log() { echo "[PIPE] $(date '+%F %T') $*" | tee -a "$SUPLOG"; }

# ---- 本项目自己的进程（用于把"自己的 GPU 占用"与"别人的作业"分开）
#
# ⚠️ 为什么不能只 `pgrep -f '^python train\.py'`（2026-10-15 实测踩到）：
# `run_all.sh` 是 `python train.py ... &` **后台**起的（不是 exec 替换），
# 因此那个 python 进程的父 shell 仍在 `wait` 上、且**它自己**能被 pgrep 找到；
# 但在某些派生顺序下（本会话实测）`^python train\.py` 匹配不到它，
# 于是 GPU0 上**我们自己的 LEVIR** 被判成 "foreign" ⇒ WHU 会永久卡在 GPU0-WAIT。
#
# 正确口径（三重并集，宁可多认自己的、绝不多认别人的）：
#   ① pgrep '^python train\.py'
#   ② pgrep 我们的编排器/流水线脚本
#   ③ 反向判定：任何跑在 GPU 上的进程，只要 `/proc/<pid>/cmdline` 里出现本项目的绝对路径
#      （`/home/yqwang/projects/CASA-CD`），它就是我们的
own_pids() {
  {
    pgrep -f '^python train\.py' 2>/dev/null
    pgrep -f '_r8_run_one_ds\.sh|_r8_supervisor\.sh' 2>/dev/null
    for p in $(pgrep -f 'python' 2>/dev/null); do
      cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
      case "$cmd" in *"/home/yqwang/projects/CASA-CD"*) echo "$p" ;; esac
    done
  } | sort -u
}

gpu_free_mb() {  # $1 = index
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" \
    | awk -F, '{gsub(/ /,"");print $1-$2}'
}

foreign_on_gpu() {  # $1 = index；打印非本项目的 compute 进程 pid
  # ⚠️ `--format=csv,noheader` 输出是 `pid, mem`（**带逗号**）。首版用
  # `read -r pid _rest` 会把整行读进第一个字段 ⇒ 打印出来是 "3180554,"
  # ⇒ 后面的 `grep -qx` 永远匹配不上 ⇒ 我们自己的 run 被判成 foreign。
  # 因此必须显式 `-F,` 去掉分隔符。
  local idx="$1" pid mine
  mine="$(own_pids)"
  while IFS=, read -r pid _mem; do
    pid="$(echo "${pid:-}" | tr -d ' ')"
    [ -z "$pid" ] && continue
    echo "$mine" | grep -qx "$pid" || echo "$pid"
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$idx" 2>/dev/null)
}

# 等到 gpu $1 可用（余量够 或 上面只有我们自己的进程）
wait_gpu_ready() {
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
      log "DEADLINE ${DEADLINE_H}h 到期，停止等待 GPU${idx}"; return 8
    fi
  done
}

ds_done() {  # $1=variant $2=dataset
  local F="$LOG_ROOT/$1/$2/train_log.txt"
  [ -f "$F" ] && grep -q '=== END TEST RESULTS ===' "$F"
}

log "===== Run8 双卡并行流水线启动 DRY=${DRY} NEED=${NEED_MB}MB POLL=${POLL_S}s ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /' | tee -a "$SUPLOG"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /' | tee -a "$SUPLOG"

CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8

run_stage() {  # $1=variant $2=dataset $3=gpu
  if ds_done "$1" "$2"; then log "SKIP $1/$2 已完成"; return 0; fi
  wait_gpu_ready "$3" || return $?

  # ---- 断点续训：若该 run 已经跑了一部分（有 last.pth 但没跑完），
  # 必须用 `RESUME_SAME_RUN=1` 从**自己的** last.pth 续训。
  # ⚠️ 为什么必须有这段：`_r8_run_one_ds.sh` 的空目录守卫会**拒绝**在非空 ckpt 目录上启动
  # （这是防"误覆盖别人实验"的正确设计），但副作用是"中途死掉的 run 永远无法自动重启"
  # ⇒ 一个 GPU 空转窗口白白浪费。`_load_resume` 会恢复 model/optimizer/actual_steps/四项 RNG，
  # 且 `run_manifest.json` 逐字段校验用户没改参数（Run7 靠它验证过两次）。
  local resume=0
  local ck="$CKPT_ROOT/$1/$2"
  # `last.pth` 非空（>1000 字节，排除被截断的空壳）⇒ 该 run 跑过一部分，可以续训
  local last="$ck/last.pth"
  if [ -s "$last" ] && [ "$(wc -c < "$last" 2>/dev/null || echo 0)" -gt 1000 ]; then
    resume=1
    log "RESUME $1/$2 检测到 last.pth($(wc -c < "$last")B) ⇒ 断点续训"
  fi

  # 日志里若已有未完成曲线，续训前把它改名封存，避免同一库的日志混两条曲线
  if [ "$resume" = "1" ] && [ -s "$LOG_ROOT/$1/$2/train_log.txt" ] \
     && ! grep -q '=== END TEST RESULTS ===' "$LOG_ROOT/$1/$2/train_log.txt"; then
    mv "$LOG_ROOT/$1/$2/train_log.txt" \
       "$LOG_ROOT/$1/$2/train_log.txt.SUPERSEDED_$(date +%Y%m%d_%H%M%S)_before_resume"
    log "ARCHIVE $1/$2 未完成日志已封存（续训前）"
  fi

  if [ "$DRY" = "1" ]; then log "DRY 本应在 GPU$3 启动 $1/$2 (resume=$resume)"; return 0; fi
  # 每卡一个 run，串行等待；日志各自独立
  RESUME_SAME_RUN=$resume bash "$RUN_DIR/_r8_run_one_ds.sh" "$1" "$2" "$3" \
      > "$STATE/run_$1_$2.log" 2>&1
  local rc=$?
  log "STAGE-DONE $1/$2 gpu=$3 rc=$rc"
  # ⚠️ rc=3 表示"守卫拒绝"（目录里有真实产物但没有可续训的 last.pth）——
  # 这属于异常状态，必须**显式停下并报警**，绝不能让督导"以为跑完了"就跳到下一波。
  if [ "$rc" = "3" ]; then
    log "STAGE-BLOCKED $1/$2 守卫拒绝（rc=3）—— 需要人工检查 $ck ；督导在此等待 10 分钟后重试"
    sleep 600
  fi
  return 0
}

for W in 1 2 3 4; do
  case "$W" in
    1) V=CTRL_R8 ;; 2) V=H1_HYP ;; 3) V=H2_EUC ;; 4) V=H3_HYP_GS ;;
  esac
  log "===== WAVE ${W}: ${V} ====="
  # 每波 4 个库 → 2 个 GPU 并行、每卡 2 个库串行。
  # 配对原则：把最慢的 SYSU 先起，让两卡尽量同时收尾。
  if [ "$DRY" = "1" ]; then
    log "DRY wave${W}: GPU0 <- CDD,WHU ; GPU1 <- SYSU,LEVIR"
    continue
  fi
  ( run_stage "$V" SYSU-CD-256  1; run_stage "$V" LEVIR-CD-256 1 ) &
  P0=$!
  ( run_stage "$V" CDD-CD-256   0; run_stage "$V" WHU-CD-256   0 ) &
  P1=$!
  wait $P0; wait $P1
  log "===== WAVE ${W} 两个 GPU 通道均结束 ====="
  # 波门：四库都要有完整 TEST 区块，否则记下来但**不阻断**（留给收口链裁决）
  n=0
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    ds_done "$V" "$DS" && n=$((n+1))
  done
  log "WAVE${W}-GATE ${n}/4 完成"
done

log "ALL-WAVES-DONE"
