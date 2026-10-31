#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$REPO"
for SCRIPT in train_scripts/CASA-Pyramid/Run9/*.sh; do bash -n "$SCRIPT"; done
python models/test_hyperbolic_context.py
python models/test_direct_geometry.py
python models/test_attribution.py
python models/test_refinement_health.py
python models/test_CASA.py
python models/test_CASA_resume.py
python train_scripts/CASA-Pyramid/Run9/test_pipeline.py
python train_scripts/CASA-Pyramid/Run9/profile_variants.py
python - <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'train_scripts/CASA-Pyramid/Run9')
from config import ROOT,PROTOCOL
(ROOT/'validation_result.json').write_text(json.dumps(dict(protocol=PROTOCOL,passed=True,checks=['geometry','attribution','refinement_task_gradients','model','resume','matrix_capacity','four_variant_memory']),indent=2))
PY
echo '[VALIDATION-PASS]'
