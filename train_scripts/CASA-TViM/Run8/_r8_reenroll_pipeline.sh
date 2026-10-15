#!/usr/bin/env bash
# Run8 并行督导「幂等重装」：清掉所有重复实例，再干净地拉起**一个**。
#   bash _r8_reenroll_pipeline.sh
#
# 为什么需要它：本会话里因为 PowerShell 内联引号问题 + 正则误杀，反复出现过
#   ① 两个并行督导同时存在（会各自尝试起同样的 run）；② 一个都没有。
# 本脚本把两步做成原子操作，并且**只操作 `_r8_pipeline_supervisor.sh`**，
# 绝不触碰 `python train.py` / `_r8_run_one_ds.sh` / `run_all.sh`。
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
PIPEFILE="$STATE/pipeline_supervisor.pid"
mkdir -p "$STATE"

echo "=== 并行督导幂等重装 $(date '+%F %T') ==="

echo "--- 1) 停掉所有并行督导实例 ---"
N=0
for p in $(pgrep -f '_r8_pipeline_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  kill -TERM "$p" 2>/dev/null && { echo "  TERM pid=$p"; N=$((N+1)); } || true
done
[ "$N" -eq 0 ] && echo "  (无)"
sleep 4

echo "--- 2) 确认已清空 ---"
LEFT=0
for p in $(pgrep -f '_r8_pipeline_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  仍存在 pid=$p"; LEFT=$((LEFT+1))
done
[ "$LEFT" -eq 0 ] && echo "  (干净)"

echo "--- 3) 确认训练未受影响 ---"
for p in $(pgrep -f '^python train\.py' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  ds=$(echo "$cmd" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  cvd=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
  echo "  pid=$p ds=$ds GPU=$cvd"
done

echo "--- 4) 拉起唯一一个并行督导 ---"
cd "$REPO" || exit 1
setsid nohup bash "$RUN_DIR/_r8_pipeline_supervisor.sh" > "$STATE/pipe.stdout" 2>&1 < /dev/null &
echo "$!" > "$PIPEFILE"
sleep 10
PID=$(cat "$PIPEFILE")
if kill -0 "$PID" 2>/dev/null; then
  echo "  ALIVE pid=$PID"
else
  echo "  DIED —— stdout 尾部："; tail -20 "$STATE/pipe.stdout"
fi
echo "--- 5) pipeline.log 尾部 ---"
tail -8 "$STATE/pipeline.log" 2>/dev/null | sed 's/^/  /'
echo "--- 6) 当前 GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
