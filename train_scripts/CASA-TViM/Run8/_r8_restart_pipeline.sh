#!/usr/bin/env bash
# 重启**并行**督导（只动 _r8_pipeline_supervisor.sh）。
#
# 起因：`own_pids()` 只按 `^python train\.py` 匹配，认不出 `run_all.sh` 后台起的 LEVIR
# ⇒ GPU0 上我们自己的 run 被判成 "foreign" ⇒ WHU 永久卡在 GPU0-WAIT。
# 已改为三重并集（pgrep train.py ∪ pgrep 我们的编排器 ∪ cmdline 含项目绝对路径）。
#
# ⚠️ 只操作 `_r8_pipeline_supervisor.sh`；**绝不触碰** `python train.py`
# 与 `_r8_run_one_ds.sh`（正在跑的 LEVIR / SYSU 必须原封不动）。
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
PIPEFILE="$STATE/pipeline_supervisor.pid"

echo "=== 重启并行督导 $(date '+%F %T') ==="
echo "--- 重启前的并行督导实例 ---"
for p in $(pgrep -f '_r8_pipeline_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  pid=$p  $cmd"
done
# 只 kill 名为 _r8_pipeline_supervisor.sh 的**进程本身**，以及它的两个 fork 子进程
for p in $(pgrep -f '_r8_pipeline_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  kill -TERM "$p" 2>/dev/null && echo "  TERM pid=$p" || true
done
sleep 4
echo "--- 确认已退出 ---"
LEFT=0
for p in $(pgrep -f '_r8_pipeline_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  仍存在 pid=$p"; LEFT=$((LEFT+1))
done
[ "$LEFT" -eq 0 ] && echo "  (干净)"

echo "--- 确认训练未受影响（必须仍有 LEVIR 在 GPU0、SYSU 在 GPU1）---"
for p in $(pgrep -f '^python train\.py' || true); do
  ds=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  cvd=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
  echo "  pid=$p ds=$ds CUDA_VISIBLE_DEVICES=$cvd"
done
for p in $(pgrep -f '_r8_run_one_ds\.sh' || true); do
  echo "  orchestrator pid=$p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-110)"
done

echo "--- 重新拉起并行督导 ---"
setsid nohup bash "$RUN_DIR/_r8_pipeline_supervisor.sh" > "$STATE/pipe.stdout" 2>&1 < /dev/null &
echo "$!" > "$PIPEFILE"
sleep 10
PID=$(cat "$PIPEFILE")
if kill -0 "$PID" 2>/dev/null; then echo "  ALIVE pid=$PID"; else echo "  DIED"; tail -20 "$STATE/pipe.stdout"; fi
echo "--- pipeline.log 尾部 ---"
tail -10 "$STATE/pipeline.log" 2>/dev/null | sed 's/^/  /'
