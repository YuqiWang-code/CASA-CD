#!/usr/bin/env bash
# Run8 深挖（只读）：HQKV 门值 / 非退化度 / 各库 TEST 关键行。
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"

extract_last_block() {   # $1 = log file
  awk '
    /^=== TEST RESULTS ===$/ { buf=""; inb=1; next }
    /^=== END TEST RESULTS ===$/ { last=buf; inb=0; next }
    inb { buf = buf $0 "\n" }
    END { printf "%s", last }' "$1"
}

for V in CTRL_R8 H1_HYP H2_EUC H3_HYP_GS; do
  echo "################ ${V} ################"
  for DS in $DATASETS; do
    F="$ROOT/$V/$DS/train_log.txt"
    if [ ! -f "$F" ]; then printf '  %-14s (无日志)\n' "$DS"; continue; fi
    NB=$(grep -c '=== END TEST RESULTS ===' "$F")
    if [ "$NB" -eq 0 ]; then
      CUR=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//')
      printf '  %-14s 未完成 steps=%s\n' "$DS" "${CUR:-0}"
      continue
    fi
    printf '  %-14s blocks=%s\n' "$DS" "$NB"
    extract_last_block "$F" | grep -E \
      '^(Recall=|\[DEPLOY-PARAMS\]|\[DEPLOY-FLOPS\]|\[ACTUAL-OPT-STEPS\]|\[REPARAM-REAL-ARGMAX-DISAGREE\]|\[HQKV-GAMMA\])' \
      | sed 's/^/      /'
    # HQKV-DIAG 是每 epoch 打印的，最后一个完整 TEST 区块里没有 ⇒ 从日志尾部取
    if grep -q '^\[HQKV-DIAG-stage' "$F"; then
      echo "      --- HQKV-DIAG 末次（训练末期）---"
      grep -E '^\[HQKV-DIAG-stage' "$F" | tail -4 \
        | sed -E 's/^\[HQKV-DIAG-(stage[0-9])\] /\1: /' \
        | sed -E 's/score_(detail|global)_(mean|max)=[0-9.e+-]+ //g' \
        | sed 's/^/        /'
    fi
  done
  echo
done
