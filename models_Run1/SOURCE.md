# Run1 frozen source

- archive: `/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1/source_code.tar.gz`
- sha256: `c4ee3ff2a27a7ed5c275d059304db0f6f7db44bd7c6f65bd92384489a5c2568a`
- models files: 16
- run scripts: 11
- unpacked by `outputs/fetch_runs.py`; `models/` keeps its internal layout,
  the run scripts sit under `run_scripts/`.

## Ablation note (Run15)

No patch: this version already selects on validation F1 and reports test once.
Its normalisation mean is stored in BGR order (`MEAN = [0.406, 0.456, 0.485] * 2`),
unlike Run4/Run10, so its numbers are only comparable within Run1.
