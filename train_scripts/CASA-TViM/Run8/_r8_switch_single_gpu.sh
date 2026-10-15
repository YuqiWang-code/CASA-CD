#!/usr/bin/env bash
# 从「双卡并行督导」切换到「单卡串行督导」（GPU1 已被同门占用）。
#
# 只做三件事，**绝不触碰**任何 `python train.py`（正在跑的 WHU 必须继续）与 GPU1 上同门的作业：
#   1. 停掉所有 `_r8_pipeline_supervisor.sh`（v2 双卡版，只按精确脚本名匹配）；
#   2. 停掉所有 `_r8_pipeline_supervisor_single.sh` 的旧实例（幂等）；
#   3. 拉起唯一一个 `_r8_pipeline_supervisor_single.sh`（GPU_ID=0）。
# 注意：**不要**动 pid 3361540 那个 `_r8_run_one_ds.sh CTRL_R8 WHU-CD-256 0` ——
# 它是当前 WHU 的父编排器，杀了它会连带影响重试逻辑（虽然 train.py 本身不会死）。
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
mkdir -p "$STATE"

echo "=== 切换到单卡串行督导 $(date '+%F %T') ==="

echo "--- 0) 切换前的 GPU 与邻居（只统计）---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /'

echo "--- 1) 停掉双卡督导 v2 实例 ---"
N=0
for p in $(pgrep -f '_r8_pipeline_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  kill -TERM "$p" 2>/dev/null && { echo "  TERM pid=$p"; N=$((N+1)); } || true
done
[ "$N" -eq 0 ] && echo "  (无)"

echo "--- 2) 停掉单卡督导的旧实例（准备重装）---"
M=0
for p in $(pgrep -f '_r8_pipeline_supervisor_single\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  kill -TERM "$p" 2>/dev/null && { echo "  TERM pid=$p"; M=$((M+1)); } || true
done
[ "$M" -eq 0 ] && echo "  (无)"
sleep 4

echo "--- 3) 确认督导已清空（训练必须仍在）---"
LEFT=0
for p in $(pgrep -f '_r8_pipeline_supervisor' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  仍存在 pid=$p"; LEFT=$((LEFT+1))
done
[ "$LEFT" -eq 0 ] && echo "  (干净)"
echo "  正在跑的训练："
for p in $(pgrep -f '^python train\.py' || true); do
  ds=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  cvd=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
  echo "    pid=$p ds=$ds GPU=$cvd"
done
echo "  父编排器（必须保留）："
for p in $(pgrep -f '_r8_run_one_ds\.sh' || true); do
  echo "    pid=$p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-100)"
done

echo "--- 4) 拉起单卡串行督导（GPU_ID=0，脱离会话）---"
cd "$REPO" || exit 1
GPU_ID=0 setsid nohup bash "$RUN_DIR/_r8_pipeline_supervisor_single.sh" \
    > "$STATE/pipe1.stdout" 2>&1 < /dev/null &
echo "$!" > "$STATE/pipeline_single.pid"
sleep 10
PID=$(cat "$STATE/pipeline_single.pid")
if kill -0 "$PID" 2>/dev/null; then echo "  ALIVE pid=$PID"; else echo "  DIED"; tail -20 "$STATE/pipe1.stdout"; fi
echo "--- 5) pipeline_single.log 尾部 ---"
tail -10 "$STATE/pipeline_single.log" 2>/dev/null | sed 's/^/  /'
