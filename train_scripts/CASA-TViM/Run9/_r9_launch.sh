#!/usr/bin/env bash
# Run9 开跑（幂等：已有督导在跑就直接返回）。
set -u
REPO=/home/yqwang/projects/CASA-CD
RD="$REPO/train_scripts/CASA-TViM/Run9"
STATE=/tmp/run9
mkdir -p "$STATE"

echo "=== Run9 开跑 $(date '+%F %T') ==="
if pgrep -f '_r9_pipeline_single\.sh' >/dev/null 2>&1; then
  echo "  已有督导在跑：$(pgrep -f '_r9_pipeline_single\.sh' | tr '\n' ' ')"
  exit 0
fi
if pgrep -f '_r9_run_one_ds\.sh' >/dev/null 2>&1; then
  echo "[ABORT] 有 run_one_ds 残留，先 bash $RD/_r9_stop.sh --kill-run" >&2
  exit 3
fi

cd "$REPO" || exit 1
GPU_ID=0 NEED_MB=30000 EARLY_GATE=1 setsid nohup bash "$RD/_r9_pipeline_single.sh" \
  > "$STATE/pipe.stdout" 2>&1 < /dev/null &
PID=$!
sleep 10
if kill -0 "$PID" 2>/dev/null; then
  echo "  supervised launched pid=$PID"
else
  echo "  [FAIL] 督导 10 s 内退出"; tail -30 "$STATE/pipe.stdout"; exit 1
fi
echo "--- pipe.stdout 尾 8 ---"
tail -8 "$STATE/pipe.stdout" 2>/dev/null | sed 's/^/  /'
