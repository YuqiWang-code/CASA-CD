# Run4 frozen source

- archive: `/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run4/source_code.tar.gz`
- sha256: `abca5bd341a43421446316ea4ce1c5ae41ea1ff1da243c74ff9d8155331ca9a2`
- models files: 20
- run scripts: 11
- unpacked by `outputs/fetch_runs.py`; `models/` keeps its internal layout,
  the run scripts sit under `run_scripts/`.

## Ablation patch (Run15)

`outputs/patch_run4.py` adds `--selection-split {test,val}` (default `test`,
i.e. the archived behaviour is unchanged) and routes the per-epoch selection
evaluation to `select_loader` = val when `val` is requested. The manifest,
`evaluation_manifest.json`, and `result.json` record the chosen split and
`independent_test`. Run15 passes `--selection-split val`.
