#!/usr/bin/env bash
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
for V in CTRL_R8B B1_HYP; do
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    L="$ROOT/$V/$DS/train_log.txt"
    [ -f "$L" ] || continue
    grep -q '=== END TEST RESULTS ===' "$L" || continue
    echo "######## $V / $DS ########"
    awk '/^=== TEST RESULTS ===$/{f=1} f{print} /^=== END TEST RESULTS ===$/{f=0; exit}' "$L" | sed 's/^/  /'
  done
done
