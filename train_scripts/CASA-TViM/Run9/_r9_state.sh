#!/usr/bin/env bash
# Run9 状态 + 结果（只读）。
set -u
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run9
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run9
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"
VARIANTS="S0_CTRL S1_HSAM"

echo "=== $(date '+%F %T') ==="
echo
echo "############ 进度矩阵（0 = 未开始）############"
for V in $VARIANTS; do
  printf '  %-9s' "$V"
  for DS in $DATASETS; do
    F="$LOG_ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf ' %s=--' "${DS%%-*}"; continue; fi
    if grep -q '=== END TEST RESULTS ===' "$F"; then printf ' %s=DONE' "${DS%%-*}"
    else printf ' %s=%s' "${DS%%-*}" "$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//;s#/.*##')"; fi
  done
  echo
done

echo
echo "############ HSAM 生效读数（最近一条 DIAG）############"
for DS in $DATASETS; do
  F="$LOG_ROOT/S1_HSAM/$DS/train_log.txt"
  [ -f "$F" ] || continue
  echo "  [$DS]"
  echo "     fail-fast: $(grep -hE '^\[HSAM-FAILFAST' "$F" | tail -1 | cut -c1-150)"
  grep -hE '^\[HSAM-DIAG\]' "$F" | tail -2 | cut -c1-165 | sed 's/^/     /'
done

echo
echo "############ 完整 TEST 区块（正式结果）############"
for V in $VARIANTS; do
  echo "  ################ $V ################"
  for DS in $DATASETS; do
    F="$LOG_ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf '    %-14s (无日志)\n' "$DS"; continue; fi
    if ! grep -q '=== END TEST RESULTS ===' "$F"; then
      CUR=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//')
      printf '    %-14s 未完成 steps=%s\n' "$DS" "${CUR:-0}"; continue
    fi
    echo "    $DS"
    awk '
      /^=== TEST RESULTS ===$/ { buf=""; inb=1; next }
      /^=== END TEST RESULTS ===$/ { last=buf; inb=0; next }
      inb { buf = buf $0 "\n" }
      END { printf "%s", last }' "$F" \
      | grep -E '^(Recall=|\[TOTAL-PARAMS\]|\[FLOPS\]|\[HSAM\]|\[HSAM-STAGES\]|\[HSAM-PARAMS\]|\[BEST-F1\])' \
      | sed 's/^/      /'
  done
done

echo
echo "############ Δ 表（HSAM − CTRL）############"
python3 /home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run9/_r9_delta.py 2>/dev/null | sed 's/^/  /' || true
