# CASA / Run3

Independent Euclidean and hyperbolic change-context modules with MobileNetV2/HFEA, TAR skips and DCR decoding.

Protocol: `casa_40k_separateqkv_v3`. Batch 32, seed 2333, Adam lr=5e-4, betas=(0.9,0.99), eps=1e-8. Module parameters use weight decay 0; the main network uses 1e-4. The target 40,000 steps rounds to complete epochs. Validation selects the best F1; formal test follows deployment folding verification.

Variants: M1_HYPER_QKV, M2_CA_QKV, M0_CTRL. All four encoder levels use the selected processing module. Each 4x4 cell produces two aligned context prototypes. Q retains N spatial positions; K/V use N/8 tokens.

Priority: LEVIR → SYSU → WHU → CDD, three variants per dataset. Both GPUs are available. The scheduler reserves measured per-variant memory and starts the highest-priority run that fits, refilling across datasets. Batch size and model settings are fixed across concurrent runs.

Before formal execution, both modules complete two LEVIR epochs (444 steps) with the full 40,182-step learning-rate schedule. Real task gradients, branch outputs, residuals and hyperbolic radius variation are recorded in `probes/<module>`. These probes do not contribute formal metrics. `scheduling.json` uses the larger of the full-batch profile and probe peak reservations, plus measured CUDA process overhead, with 1 GiB free per GPU.

```bash
cd /home/yqwang/projects/CASA-CD
bash train_scripts/CASA/Run3/launch.sh
bash train_scripts/CASA/Run3/stop.sh
```

Logs: `/home/yqwang/outputs/CASA-CD/CASA/Run3`.

Checkpoints: `/share_datasets/yqwang/checkpoints/CASA-CD/CASA/Run3`.

Code archive: `source_code.tar.gz` alongside the round logs. Source file paths are recorded without checksums. Runtime GPU placement and memory reservations are recorded in `pipeline_status.json` and per-run `placement.json`.

Collect on the local workstation with `python train_scripts/CASA/Run3/collect_results.py`. Only complete TEST RESULTS blocks are formal. Existing metrics are retained, with a TXT snapshot in `docs/temporary/models_and_metrics_CASA_Run3.txt`.

Backbone, HFEA and training-recipe provenance is documented in `models/SOURCES.md`. Prior Run1/Run2 scripts are archived under `docs/temporary/CASA_Run1_Run2_scripts`.
