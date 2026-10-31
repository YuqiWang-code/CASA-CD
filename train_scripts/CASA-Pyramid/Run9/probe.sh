#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
exec python "$REPO/train_scripts/CASA-Pyramid/Run9/probe.py"
