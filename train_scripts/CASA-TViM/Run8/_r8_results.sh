#!/usr/bin/env bash
# Run8 结果提取（只读）：从每个 run 的 train_log.txt 取**最后一个完整** TEST 区块。
# 只输出关键行，便于快速汇报；不做任何判定（判定留给 run8_verdict.py）。
set -u
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"
VARIANTS="${VARIANTS:-CTRL_R8 H1_HYP H2_EUC H3_HYP_GS}"

for V in $VARIANTS; do
  echo "################ $V ################"
  for DS in $DATASETS; do
    F="$LOG_ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf '  %-14s (无日志)\n' "$DS"; continue; fi
    NB=$(grep -c '=== END TEST RESULTS ===' "$F")
    if [ "$NB" -eq 0 ]; then
      CUR=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//')
      printf '  %-14s 未完成 steps=%s blocks=0\n' "$DS" "${CUR:-0}"
      continue
    fi
    # 用 awk 抓最后一个完整区块
    awk '
      /^=== TEST RESULTS ===$/ { buf=""; inb=1; next }
      /^=== END TEST RESULTS ===$/ { last=buf; inb=0; next }
      inb { buf = buf $0 "\n" }
      END {
        if (last == "") { print "  (解析失败)"; exit }
        n=split(last, L, "\n")
        for (i=1;i<=n;i++) {
          if (L[i] ~ /^Recall=/)  print "    " L[i]
          if (L[i] ~ /^\[DEPLOY-PARAMS\]/) print "    " L[i]
          if (L[i] ~ /^\[DEPLOY-FLOPS\]/) print "    " L[i]
          if (L[i] ~ /^\[ACTUAL-OPT-STEPS\]/) print "    " L[i]
          if (L[i] ~ /^\[PRETRAIN-LOAD\]/) print "    " L[i]
          if (L[i] ~ /^\[REPARAM-REAL-ARGMAX-DISAGREE\]/) print "    " L[i]
          if (L[i] ~ /^\[HQKV-GAMMA\]/) print "    " L[i]
          if (L[i] ~ /^\[HQKV-PARAMS\]/) print "    " L[i]
          if (L[i] ~ /^\[HGSL\]/) print "    " L[i]
        }
      }' "$F" | sed "1i\\  $DS (blocks=$NB)"
  done
done
