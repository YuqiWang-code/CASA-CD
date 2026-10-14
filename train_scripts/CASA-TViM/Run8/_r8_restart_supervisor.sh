#!/usr/bin/env bash
# 重启 Run8 督导：先按**进程名**找出真正的督导实例（不信 pidfile，它可能被写坏），
# 全部 kill -TERM，再重新拉起一个。
#
# ⚠️ 教训（本会话第 3 次踩）：**绝不在 ssh 参数里拼 `2>/dev/null` / `\$()` / 引号**，
# PowerShell 会把它们吃掉（`2>/dev/null` 变成 `F:\dev\null`、`\$()` 报路径错）。
# 一律写 .sh 文件再 `bash`。
#
# 只操作 `_r8_supervisor.sh`；**绝不触碰** `python train.py` 与 `run_all.sh`。
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8

echo "=== 重启督导 $(date '+%F %T') ==="
echo "--- 重启前的督导实例（按进程名，不信 pidfile）---"
FOUND=0
for p in $(pgrep -f '_r8_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  pid=$p  $cmd"
  kill -TERM "$p" 2>/dev/null && echo "    -> TERM 已发" || echo "    -> TERM 失败（可能已退出）"
  FOUND=$((FOUND+1))
done
[ "$FOUND" -eq 0 ] && echo "  (无)"
sleep 4
echo "--- 确认已全部退出 ---"
LEFT=0
for p in $(pgrep -f '_r8_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  仍存在: pid=$p"; LEFT=$((LEFT+1))
done
[ "$LEFT" -eq 0 ] && echo "  (干净)"

echo "--- 确认训练未受影响 ---"
pgrep -af 'run_all\.sh --wave' | head -2
pgrep -af '^python train\.py' | head -1 | cut -c1-100

echo "--- 重新拉起督导 ---"
bash "$RUN_DIR/_r8_launch_supervisor.sh"
