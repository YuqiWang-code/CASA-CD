#!/usr/bin/env bash
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
CK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8b
echo "=== $(date '+%F %T') ==="
echo
echo "############ 各库起止（mtime 推断）############"
for V in CTRL_R8B B1_HYP B2_EUC B3_HYP_GS; do
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    L="$ROOT/$V/$DS/train_log.txt"
    [ -f "$L" ] || continue
    B=$(stat -c%y "$L" | cut -d. -f1)
    S=$(stat -c%w "$L" 2>/dev/null | cut -d. -f1)
    TB=$(grep -c '=== END TEST RESULTS ===' "$L" || true)
    CUR=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -1 | sed 's/.*cumulative=//')
    printf '  %-10s %-14s born=%s mtime=%s tests=%s steps=%s\n' "$V" "$DS" "${S:-?}" "$B" "$TB" "${CUR:-0}"
  done
done
echo
echo "############ B1_HYP/LEVIR 详细 ############"
L="$ROOT/B1_HYP/LEVIR-CD-256/train_log.txt"
if [ -f "$L" ]; then
  echo "size=$(stat -c%s "$L")  mtime=$(stat -c%y "$L" | cut -d. -f1)"
  grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -4
  grep -E '^Epoch ' "$L" | tail -3 | cut -c1-150
  echo "RESUME 行: $(grep -c '\[RESUME\]' "$L" || true)"
  echo "OOM 行: $(grep -ci 'out of memory' "$L" || true)"
else echo "  无日志"; fi
echo
echo "############ GPU ############"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | sed 's/^/  /'
echo
echo "############ 督导 ############"
tail -4 /tmp/run8b/pipeline.log | sed 's/^/  /'
echo "  supervisor: $(pgrep -f '_r8b_pipeline_supervisor_single' | tr '\n' ' ')"
