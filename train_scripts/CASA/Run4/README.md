# CASA Run4

This directory records the completed Run4 protocol. Reproduce it with that round's `source_code.tar.gz`; the current `models/` directory implements the Run6 protocol. Historical logs, checkpoints and snapshots are retained separately.

Protocol: `casa_40k_supervisedkv_v4`. MobileNetV2/HFEA, TAR and DCR with independent Euclidean and hyperbolic context modules. Four datasets each run M1_HYPER_QKV, M2_CA_QKV and M0_CTRL from fresh initialization.

## Supervised context assignment

Each of four module stages predicts change logits with 1x1 convolution, SiLU, depthwise 3x3 convolution, SiLU and 1x1 convolution. Euclidean inputs are the temporal mean and absolute difference of LayerNorm features. Hyperbolic inputs are the mean and absolute difference of Log-mapped metric features, plus aligned hyperbolic distance. Inputs are symmetric under temporal exchange.

Sigmoid probabilities and their complements form two normalized assignments in every 4x4 feature cell. Q retains N positions, and KV has N/8 positions. Euclidean pooling and hyperbolic geometric centroids remain separate. The prediction head is used during inference to form assignments; inference requires images only.

Targets are area-averaged ground-truth change occupancies at 64x64, 32x32, 16x16 and 8x8. Each head uses BCE-with-logits plus soft Dice. The mean of the four auxiliary losses is weighted by 0.1 and added to the four-output main BCE+Dice sum. Auxiliary supervision updates the assignment heads using detached input features. The main change loss remains end-to-end through the original assignment computation, geometry and encoder. CTRL has no heads and no auxiliary loss.

| Variant | Train parameters | Assignment input channels |
|---|---:|---:|
| M0_CTRL | 3,100,404 | — |
| M1_HYPER_QKV | 3,144,192 | 65 |
| M2_CA_QKV | 3,160,960 | 128 |

## Training and placement

Priority: LEVIR → SYSU → WHU → CDD. Both GPUs use measured memory reservations and refill with the highest-priority run that fits. Each formal run uses batch32, seed2333, target40K rounded to full epochs, Adam lr5e-4, betas(0.9,0.99), eps1e-8, poly^0.9 and the initial 200-step warmup. All stage_modules parameters, including assignment heads, have weight decay0; the remaining parameters have weight decay1e-4.

Before formal training, both modules complete two full LEVIR epochs (444 optimizer steps) using the full 40,182-step schedule. Probe checks cover main-task gradients, assignment auxiliary gradients, branch/residual outputs, finite geometry and radius variation. Probe weights are not used for formal training. Scheduling memory uses the larger of the full-batch profile and probe reservation, plus CUDA context overhead, leaving 1GiB per GPU.

```bash
bash train_scripts/CASA/Run4/launch.sh
bash train_scripts/CASA/Run4/stop.sh
```

## Evaluation and collection

Select by validation F1; formal test uses the best validation checkpoint. Deployment verification uses `strict_fp32_fold_v1`, tolerance2e-4, and reports binary disagreement. Checkpoints include model, optimizer, best selection and RNG states.

Server logs: `/home/yqwang/outputs/CASA-CD/CASA/Run4`. Checkpoints: `/share_datasets/yqwang/checkpoints/CASA-CD/CASA/Run4`. Runtime source is saved in the round's `source_code.tar.gz`.

```bash
python train_scripts/CASA/Run4/collect_results.py
```

Only complete TEST RESULTS blocks enter `docs/experiment_metrics.xlsx`. All 123 earlier formal records must remain unchanged; 12 Run4 results bring the total to135 unless other formal records are added. The collector directly compares prior cells and restores the previous workbook if they change. The current snapshot is `docs/temporary/models_and_metrics_CASA_Run4.txt`. Run3's frozen snapshot and source archive preserve its earlier module design.

## Interface and resource validation

CPU checks passed for single-pixel occupancy preservation, an analytical BCE/Dice case, four-scale loss averaging, auxiliary gradients at zero gamma, temporal-exchange symmetry, identical-pair residual identity, geometry gradients, shared-path initialization and all three deployment graphs. Four-scale fold differences were 5.96e-8 with zero binary disagreement.

Full-batch profile: peak allocated memory CTRL8873MiB, hyperbolic14321MiB, Euclidean10924MiB. Single-process training step time was approximately0.078/0.133/0.085s respectively. Logs and measurements are in `validation.log` and `profile.json` under the round's output directory.

## LEVIR health probes

Both modules completed two full LEVIR epochs (444 steps) with the full 40,182-step schedule. All four stages retained nonzero main-task gradients, auxiliary head gradients, branch outputs and residuals. Hyperbolic radius CV ranged from0.238 to0.266. Probe results are separate from formal metrics, and formal runs start with fresh weights.

| Module / stage | Change occupancy weighted p | Background occupancy weighted p | Radius CV |
|---|---:|---:|---:|
| hyperbolic / stage1 | 0.0785 | 0.0791 | 0.2660 |
| hyperbolic / stage2 | 0.1090 | 0.1071 | 0.2489 |
| hyperbolic / stage3 | 0.1098 | 0.0971 | 0.2586 |
| hyperbolic / stage4 | 0.1236 | 0.1284 | 0.2377 |
| qkv / stage1 | 0.1931 | 0.0099 | — |
| qkv / stage2 | 0.4186 | 0.0249 | — |
| qkv / stage3 | 0.1655 | 0.0200 | — |
| qkv / stage4 | 0.0729 | 0.0244 | — |

Probability means describe the fixed validation probe batch at each feature scale, weighted by area occupancy. They are not final segmentation accuracy or dataset-wide calibration.

## Formal launch

The formal queue started at 2026-10-22 23:09 CST with source revision `2a1346f` and supervisor PID1117533. It uses fresh initialization, and each run performs formal testing after completing its full training budget. Five jobs run concurrently at launch, with two on GPU0 and three on GPU1. Measured memory budgets are30914/31072MiB; reservations are15132MiB for M1,11974MiB for M2 and9534MiB for CTRL.

| Physical GPU | Initial jobs |
|---|---|
| 0 | M1_HYPER_QKV/LEVIR-CD-256, M1_HYPER_QKV/SYSU-CD-256 |
| 1 | M2_CA_QKV/LEVIR-CD-256, M0_CTRL/LEVIR-CD-256, M0_CTRL/SYSU-CD-256 |

Initial utilization sample was99%/98%, with29.7/30.1GiB in use. Capacity is refilled automatically with the highest-priority pending run that fits.
