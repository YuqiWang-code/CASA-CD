#!/usr/bin/env bash
# Run9 开跑前预检（只读）。语法 / JSON / 关键参数存在性 / GPU 现状。
set -u
REPO=/home/yqwang/projects/CASA-CD
RD="$REPO/train_scripts/CASA-TViM/Run9"
echo "=== Run9 预检 $(date '+%F %T') ==="
echo "--- bash 语法 ---"
for f in "$RD/_r9_run_one_ds.sh" "$RD/_r9_pipeline_single.sh" "$RD/_r9_stop.sh" \
         "$RD/_r9_state.sh" "$RD/_r9_smoke.sh" "$RD/_r9_cert.sh"; do
  printf '  %-24s ' "$(basename "$f")"
  if bash -n "$f" 2>/dev/null; then echo OK; else echo "SYNTAX-FAIL"; bash -n "$f"; fi
done
echo "--- python 语法 ---"
for f in "$RD/_r9_delta.py" "$REPO/models/run9_smoke.py" "$REPO/models/run9_degeneracy_cert.py" \
         "$REPO/models/run9_radius_probe.py" "$REPO/models/test_hsam_cpu.py"; do
  printf '  %-26s ' "$(basename "$f")"
  if python3 -m py_compile "$f" 2>/dev/null; then echo OK; else echo "PY-FAIL"; fi
done
echo "--- JSON ---"
printf '  %-24s ' "SELECTION_LOCK.json"
python3 -c "import json,sys; json.load(open(sys.argv[1])); print('OK')" \
  "$RD/SELECTION_LOCK.json" || echo FAIL

echo "--- 关键参数是否真的进了 train.py ---"
for pat in "hsam_feature_source" "hsam_radius" "hsam_stages" "RANK_CORR_MAX_OK" "run9_hsam_poincare_exact80k_v1"; do
  n=$(grep -c "$pat" "$REPO/models/train.py" 2>/dev/null || echo 0)
  printf '  %-34s hits=%s\n' "$pat" "$n"
done
echo "--- 变体同步（run_one_ds vs README）---"
grep -oE 'S[01]_[A-Z_]+' "$RD/_r9_run_one_ds.sh" | sort -u | sed 's/^/  run_one_ds: /'
grep -oE '`S[01]_[A-Z_]+`' "$RD/README.md" | tr -d '`' | sort -u | sed 's/^/  README    : /'
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "--- 已有 Run9 产物（应为空）---"
ls -la /home/yqwang/outputs/CASA-CD/CASA-TViM/Run9 2>/dev/null | sed 's/^/  /' || echo "  (无)"
ls -la /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run9 2>/dev/null | sed 's/^/  /' || echo "  (无)"
