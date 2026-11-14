# Run10 frozen source

- archive: `/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run10/source_code.tar.gz`
- sha256: `0004ce13781eb1e8592f3d5d7d0165914d0574c823d3915d40353c527d068567`
- models files: 29
- run scripts: 21
- unpacked by `outputs/fetch_runs.py`; `models/` keeps its internal layout,
  the run scripts sit under `run_scripts/`.

## Ablation patch (Run15)

`outputs/patch_run10.py` adds `--selection-split {test,val}` (default `test`,
preserving the archived behaviour) and routes the per-epoch selection evaluation
to `select_loader` = val when requested; the checkpoint's evaluation generator
state follows the same loader. Run15 passes `--selection-split val` and
`--seed 2333 --context-seed 2333`.
