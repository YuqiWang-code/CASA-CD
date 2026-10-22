# CASA Run5

Protocol: `casa_40k_pairnorm_v5`. MobileNetV2/HFEA, TAR and DCR with independent Euclidean and hyperbolic context modules. Four datasets each run M1_HYPER_QKV, M2_CA_QKV and M0_CTRL from fresh initialization.

## Hyperbolic paired normalization

For each image pair, subtract the two dates' shared channel-wise spatial mean from their tangent vectors, then divide both by the same root mean squared token norm. Their joint mean squared tangent norm is one. Apply this operation before the embedding exponential map and after the Q/K and V tangent linear projections.

Relative spatial norms and temporal differences are retained. The transformation removes common offsets and global feature/weight gain, uses the same per-image-pair statistics during training and inference, and adds no parameters or running statistics. The original tangent limit2.3 remains for outliers. Unit mean squared norm bounds the fraction at this limit by1/2.3²≈18.9%. Full-resolution Q, geometric KV centroids and Mobius change values remain unchanged.

Diagnostics include radius mean/CV, query clipping fraction, normalized attention entropy and mean resultant length of unit query directions. A direction resultant near one indicates concentrated directions.

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

Before formal training, the hyperbolic module completes16 full SYSU epochs (6,000 steps). Hyperbolic and ordinary modules each complete2 full LEVIR epochs (444 steps). All probes use each dataset's full40K schedule. Every epoch from the second onward checks finite outputs, nonzero branch/residual outputs, main-task gradients and assignment auxiliary gradients. Hyperbolic stages require radius CV≥0.02, query clipping fraction≤0.20 and direction resultant≤0.95. The scheduler requires all three passed probe results. Formal runs use fresh initialization.

Scheduling memory uses the largest profile/probe reservation for each variant plus CUDA context overhead, leaving1GiB per GPU.

```bash
bash train_scripts/CASA/Run5/launch.sh
bash train_scripts/CASA/Run5/stop.sh
```

## Evaluation and collection

Select by validation F1; formal test uses the best validation checkpoint. Deployment verification uses `strict_fp32_fold_v1`, tolerance2e-4, and reports binary disagreement. Checkpoints include model, optimizer, best selection and RNG states.

Server logs: `/home/yqwang/outputs/CASA-CD/CASA/Run5`. Checkpoints: `/share_datasets/yqwang/checkpoints/CASA-CD/CASA/Run5`. Runtime source is saved in the round's `source_code.tar.gz`.

```bash
python train_scripts/CASA/Run5/collect_results.py
```

Only complete TEST RESULTS blocks enter `docs/experiment_metrics.xlsx`. All 123 earlier formal records must remain unchanged; 12 Run5 results bring the total to135 unless other formal records are added. The collector directly compares prior cells and restores the previous workbook if they change. The current snapshot is `docs/temporary/models_and_metrics_CASA_Run5.txt`. Run3's frozen snapshot and source archive preserve its earlier module design.

## Interface and resource validation

Server checks passed for shared-date centering, feature/metric scale invariance, retained relative radii, the clipping bound, temporal symmetry, identical-pair residual identity, geometry gradients, auxiliary gradient routing and all three deployment graphs. Four-scale folding differences were5.96e-8 with zero binary disagreement.

Full-batch peak allocated memory was8873/14448/10924MiB for CTRL/hyperbolic/ordinary QKV; single-process step time was approximately0.076/0.138/0.085s. Measurements are in validation.log and profile.json under the round's output directory.

## Completed health training

SYSU hyperbolic health training completed16 full epochs (6,000 steps) with live task/auxiliary gradients and residuals at all four stages. Both LEVIR probes completed444 steps.

| SYSU stage | Radius mean | Radius CV | Query clipping | Direction resultant | Attention entropy |
|---|---:|---:|---:|---:|---:|
| stage1 | 0.6217 | 0.3608 | 1.87% | 0.1743 | 0.9323 |
| stage2 | 0.6284 | 0.3667 | 0.59% | 0.1035 | 0.9117 |
| stage3 | 0.6824 | 0.2335 | 1.27% | 0.0672 | 0.9699 |
| stage4 | 0.6559 | 0.2855 | 2.34% | 0.0990 | 0.9483 |

The6,000-step probe checkpoint reached80.02% F1 on the full4,000-image SYSU validation split. This validation result is separate from formal test metrics.

## Formal launch

Started2026-10-23 00:28:16 CST, source revision `2675fd9`. Initial memory budgets: {'0': 30914, '1': 31072}. Variant reservations: {'M1_HYPER_QKV': 15260, 'M2_CA_QKV': 11976, 'M0_CTRL': 9534}.

| Physical GPU | Initial jobs |
|---|---|
| 0 | M1_HYPER_QKV/LEVIR-CD-256, M1_HYPER_QKV/SYSU-CD-256 |
| 1 | M2_CA_QKV/LEVIR-CD-256, M0_CTRL/LEVIR-CD-256, M0_CTRL/SYSU-CD-256 |
