#!/usr/bin/env bash
# 清理重复的 Run8 督导实例：只保留 pidfile 里那一个，其余 kill -TERM。
# **绝不触碰**任何 `python train.py` 与 `run_all.sh`（正在跑的训练与波次必须不动）。
set -u
PIDFILE=/tmp/run8/supervisor.pid
KEEP=$(cat "$PIDFILE" 2>/dev/null || echo "")

echo "=== pidfile supervisor.pid = ${KEEP:-<空>} ==="
echo "--- 当前所有 _r8_supervisor.sh 实例 ---"
mapfile -t ALL < <(pgrep -f '_r8_supervisor\.sh' 2>/dev/null)
# 排除 pgrep 自身与本次 bash -c
CLEAN=()
for p in "${ALL[@]}"; do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in
    *bash\ -c*) continue ;;                 # 包装 shell，不是督导本体
  esac
  echo "  pid=$p  $cmd"
  CLEAN+=("$p")
done

echo "--- 处置 ---"
for p in "${CLEAN[@]}"; do
  if [ -n "$KEEP" ] && [ "$p" = "$KEEP" ]; then
    echo "  KEEP  pid=$p"
  else
    echo "  TERM  pid=$p (重复实例)"
    kill -TERM "$p" 2>/dev/null || echo "    (kill 失败，可能已退出)"
  fi
done
sleep 2
echo "--- 处置后 ---"
pgrep -af '_r8_supervisor\.sh' | grep -v 'bash -c' || echo "  (无)"
echo "--- 训练/波次进程（必须原封不动）---"
pgrep -af 'run_all\.sh --wave|^python train\.py' | head -4
echo "--- supervisor.log 尾部 ---"
tail -5 /tmp/run8/supervisor.log 2>/dev/null
