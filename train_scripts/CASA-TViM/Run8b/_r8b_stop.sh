#!/usr/bin/env bash
# Run8b 停训（幂等、保守）。
#   bash _r8b_stop.sh              # 只停督导（不会再起新 run），**保留**在跑的 WHU
#   bash _r8b_stop.sh --kill-run   # 停督导 + 杀掉我们自己的在跑 train.py
#
# 纪律：
#   * 只认 cmdline 精确匹配的本项目路径，绝不按 pid 盲杀；
#   * 绝不触碰 GPU1 上另一会话的进程（673363/673364 等）；
#   * 不删任何产物（只记录，不清理）。
set -u
KILL_RUN=0
[ "${1:-}" = "--kill-run" ] && KILL_RUN=1

REPO=/home/yqwang/projects/CASA-CD
SUP="$REPO/train_scripts/CASA-TViM/Run8b/_r8b_pipeline_supervisor_single.sh"
RUNONE="$REPO/train_scripts/CASA-TViM/Run8b/_r8b_run_one_ds.sh"

echo "=== Run8b 停训 $(date '+%F %T')  KILL_RUN=${KILL_RUN} ==="

echo "--- 停之前 ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'

# ---------- 1) 停督导 ----------
echo "--- 停督导 ---"
S=0
for p in $(pgrep -f '_r8b_pipeline_supervisor_single\.sh' 2>/dev/null || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in
    *"_r8b_pipeline_supervisor_single.sh"*) kill -TERM "$p" 2>/dev/null && echo "  TERM pid=$p" && S=$((S+1)) ;;
    *) echo "  SKIP pid=$p (cmdline 不匹配: ${cmd:0:80})" ;;
  esac
done
[ "$S" -eq 0 ] && echo "  (没有在跑的督导)"
sleep 3
LEFT=$(pgrep -f '_r8b_pipeline_supervisor_single\.sh' 2>/dev/null | wc -l)
if [ "$LEFT" -gt 0 ]; then
  for p in $(pgrep -f '_r8b_pipeline_supervisor_single\.sh' 2>/dev/null); do
    kill -KILL "$p" 2>/dev/null && echo "  KILL pid=$p"
  done
  sleep 1
  LEFT=$(pgrep -f '_r8b_pipeline_supervisor_single\.sh' 2>/dev/null | wc -l)
fi
echo "  督导剩余: $LEFT"

# ---------- 2) 在跑的 run ----------
echo "--- 我们的 run ---"
MINE=""
for p in $(pgrep -f 'python train\.py' 2>/dev/null || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in
    *"/Run8b/"*) MINE="$MINE $p"; echo "  pid=$p :: $(echo "$cmd" | cut -c1-140)" ;;
    *) echo "  (别人) pid=$p :: $(echo "$cmd" | cut -c1-90)" ;;
  esac
done
[ -z "$MINE" ] && echo "  (无在跑的 Run8b train.py)"

if [ "$KILL_RUN" = "1" ] && [ -n "$MINE" ]; then
  echo "--- 杀在跑的 run ---"
  for p in $MINE; do kill -TERM "$p" 2>/dev/null && echo "  TERM pid=$p"; done
  sleep 5
  for p in $MINE; do kill -0 "$p" 2>/dev/null && { kill -KILL "$p" 2>/dev/null && echo "  KILL pid=$p"; }; done
elif [ -n "$MINE" ] && [ "$KILL_RUN" = "0" ]; then
  echo "  ⇒ 按保守策略**保留**在跑的 run（不会再起新 run）；如需一并停止请加 --kill-run"
fi

# ---------- 3) 停之后 ----------
echo "--- 停之后 ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "--- 进度快照 ---"
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
echo "--- pipeline.log 尾 5 ---"
tail -5 /tmp/run8b/pipeline.log 2>/dev/null | sed 's/^/  /'
