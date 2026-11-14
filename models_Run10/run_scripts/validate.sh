#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$REPO"
for SCRIPT in train_scripts/CASA-Pyramid/Run10/*.sh; do bash -n "$SCRIPT"; done
python models/test_direct_geometry.py
python models/test_classposterior_compression.py
python models/test_classposterior_health.py
python models/test_centered_readout.py
python models/test_posterior_supervision.py
python models/test_CASA.py
python models/test_CASA_resume.py
python train_scripts/CASA-Pyramid/Run10/test_pipeline.py
python train_scripts/CASA-Pyramid/Run10/profile_variants.py
python - <<'PY'
import json,sys
sys.path.insert(0,'train_scripts/CASA-Pyramid/Run10')
from config import ROOT,PROTOCOL
(ROOT/'validation_result.json').write_text(json.dumps(dict(protocol=PROTOCOL,passed=True,checks=['geometry','class_posterior_compression','posterior_task_gradients','centered_readout','posterior_supervision_budget','model','resume','matrix_capacity','three_variant_memory']),indent=2))
PY
echo '[VALIDATION-PASS]'
