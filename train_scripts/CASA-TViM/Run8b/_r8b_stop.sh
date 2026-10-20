#!/usr/bin/env bash
# Run8b 停训（幂等、保守、**能扛住 run_one_ds 的重试循环**）。
#   bash _r8b_stop.sh              # 停督导 + 停 run_one_ds（不会再起新 run），保留在跑的 train.py
#   bash _r8b_stop.sh --kill-run   # 上述 + 杀掉我们自己的在跑 train.py
#
# ⚠️ 2026-10-20 12:00 踩到的坑（必须记下来）：
#    只杀 `python train.py` 是不够的 —— `_r8b_run_one_ds.sh` 里有一个 **10 s 重试循环**，
#    父进程还活着时会把训练进程**重新拉起来**（实测：11:48 杀掉 716355，11:50 又出现 720375，
#    同一数据集 CDD、user=yqwang、在 GPU0 上重新吃显存）。
#    正确顺序：**先杀 run_one_ds（重试循环的宿主）→ 再杀 train.py → 再观察 20 s 确认没有复活**。
#
# 纪律：
#   * 只认 cmdline 精确匹配的本项目路径，绝不按 pid 盲杀；
#   * 绝不触碰别人的进程（GPU1 上 bqwang 的 673363/673364 等）；
#   * 不删任何产物（只记录，不清理）。
set -u
KILL_RUN=0
[ "${1:-}" = "--kill-run" ] && KILL_RUN=1

REPO=/home/yqwang/projects/CASA-CD
SUP="$REPO/train_scripts/CASA-TViM/Run8b/_r8b_pipeline_supervisor_single.sh"
RUNONE="$REPO/train_scripts/CASA-TViM/Run8b/_r8b_run_one_ds.sh"

hr() { echo "  ---- $* ----"; }
list_by_pat() {
  # $1 = 正则；打印 cmdline 含本仓库 Run8b 路径的 pid
  for p in $(pgrep -f "$1" 2>/dev/null || true); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
    case "$cmd" in
      *"/Run8b/"*) echo "$p" ;;
    esac
  done
}

echo "=== Run8b 停训 $(date '+%F %T')  KILL_RUN=${KILL_RUN} ==="
hr "停之前"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'

# ---------- 1) 督导 ----------
hr "1) 停督导"
for p in $(list_by_pat '_r8b_pipeline_supervisor_single\.sh'); do
  kill -TERM "$p" 2>/dev/null && echo "  TERM supervisor pid=$p"
done
sleep 2
for p in $(list_by_pat '_r8b_pipeline_supervisor_single\.sh'); do
  kill -KILL "$p" 2>/dev/null && echo "  KILL supervisor pid=$p"
done
echo "  督导剩余: $(list_by_pat '_r8b_pipeline_supervisor_single\.sh' | wc -l)"

# ---------- 2) run_one_ds（重试循环的宿主）----------
hr "2) 停 run_one_ds（**必须先于 train.py**，否则会被复活）"
for p in $(list_by_pat '_r8b_run_one_ds\.sh'); do
  kill -TERM "$p" 2>/dev/null && echo "  TERM run_one_ds pid=$p"
done
sleep 3
for p in $(list_by_pat '_r8b_run_one_ds\.sh'); do
  kill -KILL "$p" 2>/dev/null && echo "  KILL run_one_ds pid=$p"
done
echo "  run_one_ds 剩余: $(list_by_pat '_r8b_run_one_ds\.sh' | wc -l)"

# ---------- 3) 我们的 train.py ----------
hr "3) 我们的 train.py"
MINE=$(list_by_pat 'python train\.py')
if [ -z "$MINE" ]; then echo "  (无)"; else for p in $MINE; do
  echo "  pid=$p :: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)"; done; fi

if [ "$KILL_RUN" = "1" ] && [ -n "$MINE" ]; then
  for p in $MINE; do kill -TERM "$p" 2>/dev/null && echo "  TERM pid=$p"; done
  sleep 6
  for p in $MINE; do
    kill -0 "$p" 2>/dev/null && { kill -KILL "$p" 2>/dev/null && echo "  KILL pid=$p（无视 TERM）"; }
  done
elif [ -n "$MINE" ]; then
  echo "  ⇒ 保守策略：**保留**在跑的 train.py（已无宿主，不会再起新 run）"
  echo "     如需一并停止：bash _r8b_stop.sh --kill-run"
fi

# ---------- 4) 复活检查 ----------
hr "4) 复活检查（20 s）"
for i in 1 2 3 4; do
  sleep 5
  n_sup=$(list_by_pat '_r8b_pipeline_supervisor_single\.sh' | wc -l)
  n_run=$(list_by_pat '_r8b_run_one_ds\.sh' | wc -l)
  n_py=$(list_by_pat 'python train\.py' | wc -l)
  echo "  t+$((i*5))s supervisor=$n_sup run_one_ds=$n_run train.py=$n_py"
  [ "$n_sup" -eq 0 ] && [ "$n_run" -eq 0 ] || echo "  ⚠️ 仍有宿主进程，重试循环可能复活训练"
done

hr "停之后"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
hr "进度快照"
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
for V in CTRL_R8B B1_HYP B2_EUC B3_HYP_GS; do
  printf '  %-10s' "$V"
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    F="$ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf ' %s=--' "${DS%%-*}"; continue; fi
    if grep -q '=== END TEST RESULTS ===' "$F"; then printf ' %s=DONE' "${DS%%-*}"
    else printf ' %s=%s' "${DS%%-*}" "$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//;s#/.*##')"; fi
  done
  echo
done
