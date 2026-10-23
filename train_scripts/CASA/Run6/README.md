# CASA Run6

Protocol: `casa_40k_changevalues_v6`.

## Models

MobileNetV2/HFEA produces four paired 64-channel feature maps. Each stage uses the selected context module before TAR temporal skips and the DCR decoder. Q retains N spatial positions; K/V retains N/8 tokens through two shared soft prototypes in each 4x4 cell.

- M0_CTRL: context modules disabled.
- M1_HYPER_QKV: map the pair to the Poincare ball, compute aligned Mobius value displacements at every position, then geometrically compress those displacements. Dense hyperbolic Q retrieves compressed K by geodesic distance; retrieved displacement values are aggregated geometrically and Log-mapped for fusion with the local branch.
- M2_CA_QKV: dense Q and compressed K use the existing LayerNorm/temporal-difference assignments. Global V concatenates signed and absolute projected temporal differences. Exchanging dates negates the signed component and retains the magnitude component. The local signed-difference branch is unchanged.

Four-output BCE+Dice is the segmentation objective. All stage_modules parameters have weight decay0; the remaining parameters have weight decay1e-4. Batch32, seed2333, Adam lr5e-4/betas(0.9,0.99)/eps1e-8, poly^0.9 and the initial200-step warmup are unchanged. Target40K steps is rounded up to full epochs. All formal models start from fresh initialization and the same MobileNetV2 pretrained weights.

## Health validation and scheduling

Before formal training, each module completes3 full LEVIR epochs (666steps) and4 full SYSU epochs (1,500steps), using the full40K learning-rate schedule. The probe records mean/range of training loss, gradient norms, stage branch/residual magnitudes, task gradients, attention entropy, query radius CV/clipping and displacement clipping. A full validation pass is recorded at the end. Probe checkpoints are separate from formal initialization and never enter the metrics workbook.

The scheduler requires all four passed health results. Both physical GPUs use measured per-variant memory reservations with1GiB headroom. Priority is LEVIR → SYSU → WHU → CDD, with fitting-job backfill across datasets. Four datasets each run hyperbolic, ordinary and CTRL:12 formal experiments.

```bash
bash train_scripts/CASA/Run6/validate.sh
bash train_scripts/CASA/Run6/probe.sh
# Launch after all four probes have passed.
bash train_scripts/CASA/Run6/launch.sh
bash train_scripts/CASA/Run6/stop.sh
```

Logs: `/home/yqwang/outputs/CASA-CD/CASA/Run6`.
Checkpoints: `/share_datasets/yqwang/checkpoints/CASA-CD/CASA/Run6`.
The round's `source_code.tar.gz` stores the model and scripts used for training.

## Evaluation and collection

Validation F1 selects the checkpoint. Formal testing evaluates that checkpoint after strict FP32 deployment folding with tolerance2e-4; binary disagreement is recorded. Training checkpoints retain optimizer and RNG state.

```bash
python train_scripts/CASA/Run6/collect_results.py
```

Only complete TEST RESULTS blocks enter the workbook. Preserve all135 existing formal records;12 new results bring the total to147 unless other records are added. The collector directly compares old cells, updates README and writes `docs/temporary/models_and_metrics_CASA_Run6.txt`. Earlier frozen snapshots and source archives remain separate.
