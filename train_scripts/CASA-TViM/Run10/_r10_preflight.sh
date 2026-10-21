#!/usr/bin/env bash
# Fail closed before creating any Run10 training artifact.
set -euo pipefail
REPO=/home/yqwang/projects/CASA-CD
RD=$REPO/train_scripts/CASA-TViM/Run10
echo "[R10-PREFLIGHT] $(date '+%F %T')"
for script in "$RD"/*.sh; do bash -n "$script"; done
python3 -m json.tool "$RD/SELECTION_LOCK.json" >/dev/null
grep -q 'PROTOCOL_VERSION = "run10_hsam_local_dispersion_exact80k_v1"' "$REPO/models/train.py"
grep -q 'hsam_radius local_dispersion' "$RD/_r10_run_one.sh"
(cd "$REPO" && sha256sum -c "$RD/SOURCE_SHA256.txt")
for root in /home/yqwang/outputs/CASA-CD/CASA-TViM/Run10 \
            /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run10; do
  if [ -d "$root" ] && [ -n "$(find "$root" -mindepth 1 -print -quit)" ]; then
    echo "[R10-PREFLIGHT] existing artifacts under $root" >&2
    exit 3
  fi
done
echo '[R10-PREFLIGHT] GPU state:'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo '[R10-PREFLIGHT] PASS'
