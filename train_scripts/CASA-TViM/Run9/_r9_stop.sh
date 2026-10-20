#!/usr/bin/env bash
# Run9 停训（幂等、保守）。**顺序是关键**，见下。
#   bash _r9_stop.sh              # 停督导 + 停 run_one_ds（不再起新 run），保留在跑的 train.py
#   bash _r9_stop.sh --kill-run   # 上述 + 杀掉我们自己的在跑 train.py
#
# ⚠️ Run8b 的教训（2026-10-20 实测）：只杀 `python train.py` 没用 ——
#    `_r9_run_one_ds.sh` 里有重试循环，宿主不死就会把训练**10 秒后重新拉起**。
#    正确顺序：**先杀重试宿主 run_one_ds → 再杀 train.py → 再观察 20 s 确认没有复活**。
set -u
KILL_RUN=0
[ "${1:-}" = "--kill-run" ] && KILL_RUN=1
REPO=/home/yqwang/projects/CASA-CD

hr() { echo "  ---- $* ----"; }
pids_by() {
  local pat="$1"
  for p in $(pgrep -f "$pat" 2>/dev/null || true); do
    local cmd
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
    case "$cmd" in *"/Run9/"*) echo "$p" ;; esac
  done
}

echo "=== Run9 停训 $(date '+%F %T')  KILL_RUN=${KILL_RUN} ==="
hr "停之前"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'

hr "1) 停督导"
for p in $(pids_by '_r9_pipeline_single\.sh'); do kill -TERM "$p" 2>/dev/null && echo "  TERM supervisor pid=$p"; done
sleep 2
for p in $(pids_by '_r9_pipeline_single\.sh'); do kill -KILL "$p" 2>/dev/null && echo "  KILL supervisor pid=$p"; done
echo "  督导剩余: $(pids_by '_r9_pipeline_single\.sh' | wc -l)"

hr "2) 停 run_one_ds（**必须先于 train.py**）"
for p in $(pids_by '_r9_run_one_ds\.sh'); do kill -TERM "$p" 2>/dev/null && echo "  TERM run_one_ds pid=$p"; done
sleep 3
for p in $(pids_by '_r9_run_one_ds\.sh'); do kill -KILL "$p" 2>/dev/null && echo "  KILL run_one_ds pid=$p"; done
echo "  run_one_ds 剩余: $(pids_by '_r9_run_one_ds\.sh' | wc -l)"

hr "3) 我们的 train.py"
MINE=$(pids_by 'python train\.py')
if [ -z "$MINE" ]; then echo "  (无)"; else
  for p in $MINE; do echo "  pid=$p :: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-110)"; done
fi
if [ "$KILL_RUN" = "1" ] && [ -n "$MINE" ]; then
  for p in $MINE; do kill -TERM "$p" 2>/dev/null && echo "  TERM pid=$p"; done
  sleep 6
  for p in $MINE; do kill -0 "$p" 2>/dev/null && { kill -KILL "$p" 2>/dev/null && echo "  KILL pid=$p（无视 TERM）"; }; done
elif [ -n "$MINE" ]; then
  echo "  ⇒ 保守策略：保留在跑的 train.py（已无宿主，不会起新 run）；如需一并停：--kill-run"
fi

hr "4) 复活检查（20 s）"
for i in 1 2 3 4; do
  sleep 5
  echo "  t+$((i*5))s supervisor=$(pids_by '_r9_pipeline_single\.sh' | wc -l)" \
       "run_one_ds=$(pids_by '_r9_run_one_ds\.sh' | wc -l)" \
       "train.py=$(pids_by 'python train\.py' | wc -l)"
done

hr "停之后"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
hr "进度"
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run9
for V in S0_CTRL S1_HSAM; do
  printf '  %-9s' "$V"
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    F="$LOG_ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf ' %s=--' "${DS%%-*}"; continue; fi
    if grep -q '=== END TEST RESULTS ===' "$F"; then printf ' %s=DONE' "${DS%%-*}"
    else printf ' %s=%s' "${DS%%-*}" "$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//;s#/.*##')"; fi
  done
  echo
done
