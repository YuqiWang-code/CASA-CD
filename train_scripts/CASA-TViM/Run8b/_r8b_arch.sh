#!/usr/bin/env bash
set -u
ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b
for V in CTRL_R8B B1_HYP; do
  for DS in CDD-CD-256; do
    D="$ROOT/$V/$DS"
    echo "######## $D ########"
    ls -la "$D" | sed 's/^/  /'
    for f in arch.json manifest.json; do
      [ -f "$D/$f" ] && { echo "  --- $f ---"; sed 's/^/    /' "$D/$f" | head -40; }
    done
  done
done
echo "######## Run8b 目录 ########"
ls -la "$ROOT" | sed 's/^/  /'
