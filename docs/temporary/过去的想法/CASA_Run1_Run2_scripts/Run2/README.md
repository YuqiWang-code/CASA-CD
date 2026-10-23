# CASA-WDMF / Run2

Run2 stopped at 2026-10-22 11:41 CST with 9/12 formal tests completed. CDD artifacts are retained. The formal model uses protocol `wdmf_str_40k_modulewd0_v2` and code baseline `c9f621d`; the archived source and 111 metrics are in `docs/temporary/models_and_metrics_CASA-WDMF_Run2.txt`. Current local models use separate QKV protocol v3; the following architecture, probes and execution describe Run2. The health inspection utility follows the current module forward interface.

MobileNetV2 + HFEA → optional four-level change-context modules → TAR temporal skips → DCR decoder. Architecture, initialization, data augmentation, loss and selection follow Run1.

## Protocol

`wdmf_str_40k_modulewd0_v2`

| Parameter group | Adam weight decay |
|---|---:|
| MobileNetV2, HFEA, TAR, DCR, prediction head | 1e-4 |
| Four processing modules: gamma, Q/K/V and output projections | 0 |

Adam uses lr=5e-4, betas=(0.9,0.99), eps=1e-8. Poly^0.9 and the upstream 200-step warmup are unchanged. Batch 32, seed 2333, input 256×256, four equally weighted BCE+Dice losses. Validation selects the highest F1 checkpoint, with ties updating. Test runs once after training.

| Dataset | Full-epoch optimizer steps |
|---|---:|
| SYSU | 40,125 |
| LEVIR | 40,182 |
| WHU | 40,145 |
| CDD | 40,248 |

Variants: `M0_CTRL` (none), `M1_HYPER_QKV` (Poincaré-distance QKV), `M2_CA_QKV` (dot-product QKV). All four encoder outputs feed their module before TAR. Each formal experiment starts from the common initialization and MobileNet pretrained weights.

## Health probes

`health_probe.py` trains each attention variant for eight complete SYSU epochs (3,000 optimizer steps), batch 32, with the formal 40,125-step learning-rate schedule. A fixed pair of validation images measures each stage's attention, radius variation, projection weights, branch/residual RMS and task gradients after each epoch. The probes have no test evaluation or formal performance metrics.

`pipeline.py` requires both `probes/{hyperbolic,qkv}/health_result.json` files to report success before starting formal training. The gate checks finite, nonzero branch outputs and task gradients through the original collapse window. Probe artifacts remain separate from the 12 formal experiments.

## SYSU health results

Both variants completed eight full SYSU epochs (3,000 steps) with the 40,125-step schedule. All four stages retained finite, nonzero branch outputs and task gradients at every inspection after the first epoch. These are optimization diagnostics from two fixed validation pairs, not test metrics.

| Module | Stage | Gamma | Radius CV | Branch RMS | Minimum task-gradient RMS |
|---|---:|---:|---:|---:|---:|
| hyperbolic | 1 | 0.1246 | 0.1681 | 1.5624 | 1.465e-04 |
| hyperbolic | 2 | 0.1005 | 0.1137 | 0.6878 | 2.109e-04 |
| hyperbolic | 3 | 0.1189 | 0.1487 | 0.9435 | 4.881e-04 |
| hyperbolic | 4 | -0.1386 | 0.2175 | 1.1155 | 3.585e-04 |
| qkv | 1 | -0.1447 | — | 2.8831 | 5.574e-05 |
| qkv | 2 | 0.0977 | — | 0.5675 | 4.182e-04 |
| qkv | 3 | 0.1243 | — | 0.8707 | 1.061e-03 |
| qkv | 4 | -0.1436 | — | 1.3344 | 7.317e-04 |

Reports: `outputs/CASA-WDMF/Run2/probes/{hyperbolic,qkv}/{health_history,health_result}.json`.

## Execution

```bash
cd /home/yqwang/projects/CASA-CD
bash train_scripts/CASA-WDMF/Run2/launch.sh 4
bash train_scripts/CASA-WDMF/Run2/stop.sh
```

GPU0 and GPU1 each have two training slots, for four concurrent runs. Dataset priority is SYSU → LEVIR → WHU → CDD; within a dataset the pending order is M1 → M2 → CTRL. A free slot starts the next pending run without a dataset completion barrier. The scheduler adopts an already running training PID during handover and records physical GPU placement in `placement.json` and `pipeline_status.json`.

The four-slot schedule started at 2026-10-22 08:53 CST: the existing LEVIR CTRL and WHU M1 run on GPU0, while WHU M2 and CTRL run on GPU1. CDD fills subsequent free slots.

| Artifact | Directory |
|---|---|
| Logs and queue state | `/home/yqwang/outputs/CASA-CD/CASA-WDMF/Run2/` |
| Formal checkpoints | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-WDMF/Run2/` |
| Probe reports | `/home/yqwang/outputs/CASA-CD/CASA-WDMF/Run2/probes/` |
| Local logs and reports | `outputs/CASA-WDMF/Run2/` |

Local collection:

```powershell
python train_scripts/CASA-WDMF/Run2/collect_results.py
```

The collector downloads Run1/Run2 text and JSON artifacts, updates README, and runs the existing spreadsheet and TXT tools on new complete TEST blocks. It also collects the retained Run1 CTRL as an independent historical result. A completed Run2 contributes 12 formal records; the retained Run1 CTRL contributes one additional record.

## Deployment verification and continuation

`strict_fp32_fold_v1` checks the original and folded models with cuDNN benchmarking and TF32 disabled. The main-output maximum absolute difference must remain at most `2e-4`. Four-scale differences and binary prediction disagreements are recorded in `deploy_check.json`. The original backend settings are restored before the formal test.

Launching the queue skips runs with complete TEST blocks. An incomplete run with `last.pth` uses `--resume`, restoring its model, optimizer, epoch, step and validation-selected checkpoint. A fully trained run proceeds directly to deployment verification and testing. Logs append to their existing files.

New checkpoints include Python, NumPy, CPU Torch and visible-device CUDA random states. `continuations.json` records each continuation's starting budget, source identity and random-state policy; `evaluation_manifest.json` records the evaluation source and precision policy. The original training manifest is retained. Timing for resumed invocations is labeled `current_session`; the reported peak memory also covers only that invocation.

## Limitations

Eight-epoch probes establish early optimization health. Long-term stability and accuracy require the full experiments. Shallow HSAM may amplify noise, and the compressed QKV geometry differs from HES-Net's dense attention. Compare the three variants within Run2 for module effects.

The original LEVIR checkpoints contain no random-generator state. M1 continues from step 39,294 with its saved optimizer and seed 2333 for the remaining 888 steps; its trajectory differs from an uninterrupted run. M2 has all 40,182 steps and requires no additional optimizer update. Pre-continuation artifacts are retained in the log `archive/20261022_075845` and checkpoint `recovery_backups/20261022_075845` directories.
