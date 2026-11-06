#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
exec python "$(dirname "$0")/run_one.py" "$@"
