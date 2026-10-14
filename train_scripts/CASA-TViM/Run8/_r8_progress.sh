#!/usr/bin/env bash
# Run8 进度一条命令：每变体 × 每库的累计 optimizer.step / 80000 + 是否有完整 TEST 区块。
# 判死活**不要**看督导日志（波内长时间静默是正常的），要看 pidfile + 子进程 + GPU 进程数。
set -u
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
STATE=/tmp/run8

echo "=== $(date '+%F %T') ==="
for V in CTRL_R8 H1_HYP H2_EUC H3_HYP_GS; do
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    F="$LOG_ROOT/$V/$DS/train_log.txt"
    if [ -f "$F" ]; then
      ST=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//')
      TG=$(grep -c '=== END TEST RESULTS ===' "$F")
      printf '  %-10s %-14s steps=%-12s test_blocks=%s\n' "$V" "$DS" "${ST:-0}" "$TG"
    else
      printf '  %-10s %-14s (未启动)\n' "$V" "$DS"
    fi
  done
done

echo "--- 波次进程 ---"
for W in 1 2 3 4 all; do
  P="$STATE/wave${W}.pid"
  if [ -f "$P" ] && kill -0 "$(cat "$P")" 2>/dev/null; then
    echo "  wave${W}: ALIVE pid=$(cat "$P")"
  elif [ -f "$P" ]; then
    echo "  wave${W}: 已退出 pid=$(cat "$P")"
  fi
done
echo "--- run_all / train 进程 ---"
pgrep -af 'run_all\.sh|^python train\.py' | head -12
echo "--- GPU 计算进程 ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
