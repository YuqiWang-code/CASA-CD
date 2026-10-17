#!/usr/bin/env bash
# 干净地终止 Run8 campaign（为 Run8b 让路）。
#
# 纪律：
#   * 先停**督导**（否则它会在波末重启）→ 再停**编排器** → 最后按数据集精确停 train.py；
#   * 产物**只改名封存，绝不删除**；
#   * **不触碰** GPU1 上的探索线（另一对话框负责）与任何非本项目进程。
#
# 为什么可以停：2026-10-17 实测 H1_HYP 与 H2_EUC **都完全自锁**
#   （`[HQKV-GAMMA] ≈ 5e-41`、`[HQKV-DIAG-*] contrib_norm = 0`），
#   两者与 CTRL 的 F1 逐位同级（CDD: 0.9776 / 0.9777 / 0.9777）⇒ 剩余 ~40 h 只会重复同一模型。
set -u
STATE=/tmp/run8
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8

echo "=== 终止 Run8 campaign $(date '+%F %T') ==="

echo "--- 1) 停督导（单卡 v3 + 双卡 v2 的所有实例）---"
for pat in '_r8_pipeline_supervisor_single\.sh' '_r8_pipeline_supervisor\.sh'; do
  for p in $(pgrep -f "$pat" || true); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
    case "$cmd" in *bash\ -c*) continue ;; esac
    kill -TERM "$p" 2>/dev/null && echo "  TERM 督导 pid=$p" || true
  done
done
sleep 3

echo "--- 2) 停编排器（_r8_run_one_ds.sh）---"
for p in $(pgrep -f '_r8_run_one_ds\.sh' || true); do
  kill -TERM "$p" 2>/dev/null && echo "  TERM 编排器 pid=$p" || true
done
sleep 3

echo "--- 3) 停本项目正在跑的 train.py（先 TERM，5 秒后仍在则 KILL）---"
for p in $(pgrep -f '^python train\.py' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  ds=$(echo "$cmd" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  case "$cmd" in *"/CASA-CD/"*) ;; *) echo "  SKIP pid=$p（不是本项目）"; continue ;; esac
  kill -TERM "$p" 2>/dev/null && echo "  TERM train.py pid=$p ds=$ds" || true
done
sleep 6
for p in $(pgrep -f '^python train\.py' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *"/CASA-CD/"*) ;; *) continue ;; esac
  kill -KILL "$p" 2>/dev/null && echo "  KILL train.py pid=$p（TERM 未生效）" || true
done
sleep 2

echo "--- 4) 确认已停在干净状态（last.pth 保留，可续训）---"
for p in $(pgrep -f '^python train\.py' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *"/CASA-CD/"*) echo "  ⚠️ 仍在跑 pid=$p";; esac
done
echo "  GPU0:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0 | sed 's/^/    /'
echo "  GPU1（探索线，不动）:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1 | sed 's/^/    /'

echo "--- 5) Run8 各 run 的完成度快照（供文档引用）---"
for V in CTRL_R8 H1_HYP H2_EUC H3_HYP_GS; do
  line="  $V:"
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    F="$LOG_ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then line="$line $DS=--"; continue; fi
    TB=$(grep -c '=== END TEST RESULTS ===' "$F")
    if [ "$TB" -gt 0 ]; then
      line="$line ${DS%%-*}=DONE"
    else
      CUR=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//' | sed 's#/.*##')
      line="$line ${DS%%-*}=${CUR:-0}/80000"
    fi
  done
  echo "$line"
done

echo "=== Run8 已停止；产物完整保留（未删除任何文件）==="
