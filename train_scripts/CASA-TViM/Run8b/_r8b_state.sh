#!/usr/bin/env bash
# Run8b 全量状态 + 结果（只读）。
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
CKPT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8b
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"
VARIANTS="CTRL_R8B B1_HYP B2_EUC B3_HYP_GS"

echo "=== $(date '+%F %T') ==="
echo
echo "############ 进度矩阵 ############"
for V in $VARIANTS; do
  printf '  %-10s' "$V"
  for DS in $DATASETS; do
    F="$ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf ' %s=--' "${DS%%-*}"; continue; fi
    TB=$(grep -c '=== END TEST RESULTS ===' "$F")
    if [ "$TB" -gt 0 ]; then printf ' %s=DONE' "${DS%%-*}"; else
      CUR=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//' | sed 's#/.*##')
      printf ' %s=%s' "${DS%%-*}" "${CUR:-0}"
    fi
  done
  echo
done

echo
echo "############ 抗自锁证据（每个 HQKV run）############"
for V in B1_HYP B2_EUC B3_HYP_GS; do
  for DS in $DATASETS; do
    F="$ROOT/$V/$DS/train_log.txt"
    [ -f "$F" ] || continue
    FF=$(grep -E '^\[FAILFAST' "$F" | grep -vE 'at_step' | tail -1)
    [ -z "$FF" ] && FF="(尚未到 step1000)"
    echo "  [$V/$DS]"
    echo "     fail-fast: $FF"
    echo "     step-1 gamma_grad: $(grep -E '^\[HQKV-GRAD\]' "$F" | tail -1 | sed 's/.*gamma_grad=//' | cut -c1-110)"
    echo "     last DIAG: $(grep -E '^\[HQKV-DIAG\]' "$F" | tail -1 | cut -c1-150)"
  done
done

echo
echo "############ 完整 TEST 区块（正式结果）############"
for V in $VARIANTS; do
  echo "  ################ $V ################"
  for DS in $DATASETS; do
    F="$ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf '    %-14s (无日志)\n' "$DS"; continue; fi
    NB=$(grep -c '=== END TEST RESULTS ===' "$F")
    if [ "$NB" -eq 0 ]; then
      CUR=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//')
      printf '    %-14s 未完成 steps=%s\n' "$DS" "${CUR:-0}"; continue
    fi
    echo "    $DS (blocks=$NB)"
    awk '
      /^=== TEST RESULTS ===$/ { buf=""; inb=1; next }
      /^=== END TEST RESULTS ===$/ { last=buf; inb=0; next }
      inb { buf = buf $0 "\n" }
      END { printf "%s", last }' "$F" \
      | grep -E '^(Recall=|\[DEPLOY-PARAMS\]|\[DEPLOY-FLOPS\]|\[ACTUAL-OPT-STEPS\]|\[HQKV-GAMMA\]|\[HQKV-CONTRIB\]|\[HQKV-PART-SHARE\]|\[HQKV-GAMMA-INIT\])' \
      | sed 's/^/      /'
  done
done
