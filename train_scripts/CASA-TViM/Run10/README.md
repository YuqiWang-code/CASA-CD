# Run10: SYSU → LEVIR, CTRL → HSAM local dispersion

This is a **new four-run protocol** authorized on 2026-10-21 after Run9 was stopped. Run9 logs, checkpoints and sidecars remain untouched. Run10 writes to separate `Run10` output and checkpoint roots. Each dataset runs `S0_CTRL` followed immediately by `S1_HSAM_LD`, in the order SYSU then LEVIR. CDD and WHU are excluded. All four runs start fresh, 80,000 optimizer steps, batch 32, seed 16, learning rate 2e-4, same pretrained VMamba-Tiny and CASA-STR v2 settings as Run9. The only comparison variable within each dataset is HSAM off/on.

## Why this radius mode

The CVPR 2025 paper's Feature Clipping places an upper limit on Euclidean feature norm before Poincaré mapping (Eq. 7). It addresses proximity to the ball boundary. Run9 LEVIR's stage3 radius variation instead fell to `r_cv=0.004133`, with `rank_corr(HD,1-cos)=0.999206`; `1-r²_min=0.390`. Upper clipping does not address this. Run10 uses the previously implemented `local_dispersion` radius mode to decouple radius from token energy. It is exploratory: it may still degenerate or fail, and accuracy benefit is unproven.

## Code and safety gates

- `models/train.py` uses protocol `run10_hsam_local_dispersion_exact80k_v1` and runs the HSAM numerical/nondegeneration checks **after every validation epoch**, removing Run9's first-pass-only bug. A later `rank_corr > 0.999`, non-finite output, near-zero aggregation, collapsed radius or uniform attention exits with code 4.
- `models/model/layers/hsam.py` uses replicate padding for local variance, a positive variance floor for finite backward gradients, and a float32-safe `acosh` guard for coincident or near-coincident feature pairs. These are fixed before training. No Feature Clipping parameter is copied from the CVPR paper.
- Any run error stops the serial queue. There is no retry and no resume from Run9 or Run10 checkpoint. The pipeline uses only GPU0 and waits for at least 30,000 MiB free; GPU1 is not selected.
- `SELECTION_LOCK.json` freezes the run order, parameters and failure policy. `SOURCE_SHA256.txt` records local code and script hashes before launch; server hashes must match.

Start: `bash train_scripts/CASA-TViM/Run10/_r10_launch.sh`. Stop: `bash train_scripts/CASA-TViM/Run10/_r10_stop.sh`. Progress: `/tmp/run10/pipeline.log` and the four per-run `train_log.txt` files. A complete formal result requires `=== END TEST RESULTS ===`; incomplete logs are not metrics.
