#!/usr/bin/env bash
# Run9 逐库起止与速率（只读）。
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run9
echo "=== $(date '+%F %T') ==="
echo "--- 逐库 born -> mtime ---"
for V in S0_CTRL S1_HSAM; do
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    F="$ROOT/$V/$DS/train_log.txt"
    [ -f "$F" ] || continue
    born=$(stat -c%w "$F" 2>/dev/null | cut -d. -f1)
    mt=$(stat -c%y "$F" 2>/dev/null | cut -d. -f1)
    tb=$(grep -c '=== END TEST RESULTS ===' "$F" || true)
    cur=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//')
    printf '  %-9s %-14s born=%s mtime=%s tests=%s steps=%s\n' "$V" "$DS" "${born:-?}" "$mt" "$tb" "${cur:-0}"
  done
done
echo
echo "--- 当前在跑的 run 的速率 ---"
F=$(ls -t "$ROOT"/*/*/train_log.txt 2>/dev/null | head -1)
echo "  log=$F"
born_epoch=$(stat -c%W "$F" 2>/dev/null || echo 0)
steps=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//;s#/.*##')
now=$(date +%s)
if [ "$born_epoch" -gt 0 ] && [ -n "${steps:-}" ]; then
  el=$(( now - born_epoch ))
  echo "  elapsed=${el}s steps=${steps}"
  if [ "$steps" -gt 0 ] && [ "$el" -gt 0 ]; then
    python3 - "$steps" "$el" <<'PY'
import sys
s, e = int(sys.argv[1]), int(sys.argv[2])
rate = s / e
need = (80000 - s) / rate if rate > 0 else float("inf")
print(f"  step/s={rate:.4f}  剩余={need/3600:.2f} h  预计完成(该库)={need/60:.0f} min 后")
PY
  fi
fi
echo
echo "--- HSAM 完整读数（不截断）---"
for DS in CDD-CD-256 LEVIR-CD-256; do
  F="$ROOT/S1_HSAM/$DS/train_log.txt"
  [ -f "$F" ] || continue
  echo "  [$DS] 最后一条 DIAG："
  grep -hE '^\[HSAM-DIAG\]' "$F" | tail -2 | sed 's/^\[HSAM-DIAG\] //' | sed 's/^/    /'
done
echo
echo "--- 督导尾 8 ---"
tail -8 /tmp/run9/pipeline.log 2>/dev/null | sed 's/^/  /'
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
