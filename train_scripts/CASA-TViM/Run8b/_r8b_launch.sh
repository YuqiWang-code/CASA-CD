#!/usr/bin/env bash
# Run8b 开跑：单卡串行督导（GPU_ID=0），脱离会话。
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8b"
STATE=/tmp/run8b
PIDFILE="$STATE/pipeline_single.pid"
mkdir -p "$STATE"

echo "=== Run8b 开跑 $(date '+%F %T') ==="
echo "--- 开跑前 GPU（GPU1 是探索线，不动）---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /'

echo "--- 确认 Run8 的进程已全部停止（避免同卡争抢）---"
LEFT=$(pgrep -f '_r8_pipeline_supervisor|_r8_run_one_ds\.sh' 2>/dev/null | wc -l)
echo "  Run8 残留督导/编排器：$LEFT 个（应为 0）"
RUN8T=$(for p in $(pgrep -f '^python train\.py' 2>/dev/null); do
  tr '\0' ' ' < "/proc/$p/cmdline" | grep -q '/Run8/' && echo "$p"
done | wc -l)
echo "  Run8 残留 train.py：$RUN8T 个（应为 0）"
if [ "$RUN8T" != "0" ]; then
  echo "  ⚠️ Run8 仍在训练，先停它再开 Run8b（同卡 27.5GB x2 会 OOM）"
  exit 3
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "  Run8b 督导已在运行 pid=$(cat "$PIDFILE")，拒绝重复启动"
  exit 0
fi

cd "$REPO" || exit 1
GPU_ID=0 setsid nohup bash "$RUN_DIR/_r8b_pipeline_supervisor_single.sh" \
    > "$STATE/pipe1.stdout" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"
sleep 12
if kill -0 "$PID" 2>/dev/null; then echo "[LAUNCH] ALIVE pid=$PID"; else
  echo "[LAUNCH] DIED"; tail -20 "$STATE/pipe1.stdout"; exit 1
fi
echo "--- pipeline.log ---"
tail -8 "$STATE/pipeline.log" 2>/dev/null | sed 's/^/  /'
