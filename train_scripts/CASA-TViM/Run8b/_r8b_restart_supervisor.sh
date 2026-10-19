#!/usr/bin/env bash
# Run8b 督导重启（幂等）：停旧实例 → 用新的 NEED_MB 重新拉起。
#   bash _r8b_restart_supervisor.sh [NEED_MB]
#
# 背景（2026-10-19 08:04）：GPU0 上出现同门 `bqwang` 的 1,938 MiB lowmem 作业，
# 旧的 `-z "$foreign"` 条件让督导**无限期空转**（free=30,495 差 5 MB 卡在 30500 阈值下）。
# 已改为：只看真实余量（NEED_MB 可配置，默认 30000 = 实测峰值 28.4 GB + 1.6 GB 余量），
# 卡上有别人的进程时**共存但不触碰**。
#
# ⚠️ 只操作 `_r8b_pipeline_supervisor_single.sh`；绝不触碰任何 `python train.py`
# （无论本项目还是别人的）与 `_r8b_run_one_ds.sh`。
set -u
NEED="${1:-30000}"
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8b"
STATE=/tmp/run8b
PIDFILE="$STATE/pipeline_single.pid"
mkdir -p "$STATE"

echo "=== 重启 Run8b 督导 NEED_MB=${NEED} $(date '+%F %T') ==="
echo "--- 当前 GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "--- 停旧督导 ---"
for p in $(pgrep -f '_r8b_pipeline_supervisor' 2>/dev/null || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  kill -TERM "$p" 2>/dev/null && echo "  TERM pid=$p" || true
done
sleep 4
LEFT=0
for p in $(pgrep -f '_r8b_pipeline_supervisor' 2>/dev/null || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  仍存在 pid=$p"; LEFT=$((LEFT+1))
done
[ "$LEFT" -eq 0 ] && echo "  (已清空)"

echo "--- 确认没有 Run8b 的 train.py 在跑（避免双开）---"
RUN8BT=0
for p in $(pgrep -f '^python train\.py' 2>/dev/null || true); do
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q '/Run8b/' && { echo "  pid=$p"; RUN8BT=$((RUN8BT+1)); }
done
[ "$RUN8BT" -eq 0 ] && echo "  (无)"

echo "--- 拉起新督导（NEED_MB=${NEED}）---"
cd "$REPO" || exit 1
GPU_ID=0 NEED_MB="$NEED" setsid nohup bash "$RUN_DIR/_r8b_pipeline_supervisor_single.sh" \
    > "$STATE/pipe1.stdout" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"
sleep 12
if kill -0 "$PID" 2>/dev/null; then echo "  ALIVE pid=$PID"; else
  echo "  DIED"; tail -20 "$STATE/pipe1.stdout"; exit 1
fi
echo "--- pipeline.log 尾 8 行 ---"
tail -8 "$STATE/pipeline.log" 2>/dev/null | sed 's/^/  /'
