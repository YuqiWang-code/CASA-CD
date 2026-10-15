#!/usr/bin/env bash
# 停掉**旧的单卡编排器** `run_all.sh --wave 1`（它已完成 CDD，剩下的事由并行督导接管）。
# 为什么要停：若两个调度器同时想续训同一个库，会出现"两个 train.py 抢同一个 ckpt 目录"的竞态。
# **绝不触碰** `python train.py`（正在跑的 LEVIR/SYSU 必须继续）与 `_r8_run_one_ds.sh`。
set -u
echo "=== 停旧单卡编排器 $(date '+%F %T') ==="
for p in $(pgrep -f 'run_all\.sh --wave' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  TERM pid=$p  $cmd"
  kill -TERM "$p" 2>/dev/null || true
done
sleep 4
echo "--- 剩余 run_all.sh ---"
LEFT=$(pgrep -f 'run_all\.sh --wave' | wc -l)
echo "  $LEFT 个"
echo "--- 确认训练仍在跑 ---"
for p in $(pgrep -f '^python train\.py' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  ds=$(echo "$cmd" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  cvd=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
  echo "  pid=$p ds=$ds GPU=$cvd"
done
echo "--- 进度 ---"
for V in CTRL_R8; do for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
  F=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8/$V/$DS/train_log.txt
  [ -f "$F" ] && printf '  %-12s %-14s %s\n' "$V" "$DS" "$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//')"
done; done
