#!/usr/bin/env bash
# Run8 彻查 + 一次修好（只读为主，只做"封存改名"这一类安全操作）。
#
# ## 背景（2026-10-15 09:08 实测的一个连锁故障）
#   1. 我在 09:07:03 跑 `_r8_switch_dual_gpu.sh`，它的 kill 正则 `_r8_supervisor\.sh`
#      **同时匹配**了旧的 `_r8_supervisor.sh` 和新的 `_r8_pipeline_supervisor.sh`
#      ⇒ 把刚拉起的并行督导连同其 fork 一起 TERM 掉；
#   2. 被 TERM 的子 shell 已经在跑 `run_one_ds SYSU`，它**先写了 run_manifest.json**
#      就被打断 ⇒ 留下孤儿 manifest；
#   3. `_r8_run_one_ds.sh` 的守卫是 `[ -n "$(ls -A "$CKPT")" ]`（**不看文件名**），
#      所以连 `*.SUPERSEDED_*` 也算"非空" ⇒ 之后每次启动都被 ABORT（rc=3）；
#   4. 结果：GPU0/GPU1 都空，wave1 只剩 LEVIR 的前 669 步。
#
# ## 本脚本
#   A. 打印 LEVIR 死因（日志尾部）
#   B. 把 SYSU 的 `*.SUPERSEDED_*` 移到**同级 _ARCHIVE 子目录**（改名封存，绝不删除）
#      ⇒ 让 ckpt 目录真正为空
#   C. 打印最终状态
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8

echo "=== Run8 彻查 $(date '+%F %T') ==="

echo "--- A) LEVIR 死因：编排器日志 + 正式日志尾部 ---"
for f in /tmp/run8/run_CTRL_R8_LEVIR-CD-256.log; do
  echo "  # $f"; tail -6 "$f" 2>/dev/null | sed 's/^/      /'
done
echo "  # 正式日志尾部"
tail -12 "$LOG_ROOT/CTRL_R8/LEVIR-CD-256/train_log.txt" 2>/dev/null | sed 's/^/      /'

echo "--- B) 封存 SYSU 的 SUPERSEDED 残留（移到同级 _ARCHIVE 子目录）---"
SY="$CKPT_ROOT/CTRL_R8/SYSU-CD-256"
AR="$CKPT_ROOT/CTRL_R8/_ARCHIVE"
mkdir -p "$AR"
if [ -d "$SY" ]; then
  for f in "$SY"/*; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    case "$b" in
      *_SUPERSEDED_*)
        mv "$f" "$AR/SYSU-CD-256.$b"
        echo "  MOVED $b -> _ARCHIVE/SYSU-CD-256.$b"
        ;;
      *) echo "  KEEP  $b （不是封存残留，保留）" ;;
    esac
  done
  echo "  SYSU ckpt 目录现在: [$(ls -A "$SY" | tr '\n' ' ')]  （空=守卫会放行）"
fi

echo "--- C) 最终状态快照 ---"
bash "$RUN_DIR/_r8_state.sh" 2>&1 | sed -n '1,30p'
