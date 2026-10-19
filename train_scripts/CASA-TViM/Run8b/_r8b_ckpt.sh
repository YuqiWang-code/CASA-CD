#!/usr/bin/env bash
set -u
CK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8b
echo "######## $CK ########"
ls -la "$CK" | sed 's/^/  /'
for V in CTRL_R8B B1_HYP; do
  for DS in CDD-CD-256; do
    D="$CK/$V/$DS"
    echo "######## $D ########"
    ls -la "$D" 2>/dev/null | sed 's/^/  /'
    for f in arch.json manifest.json; do
      [ -f "$D/$f" ] && { echo "  --- $f ---"; sed 's/^/    /' "$D/$f" | head -45; }
    done
  done
done
