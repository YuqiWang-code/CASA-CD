# CASA Run6

Protocol: `casa_40k_changevalues_v6`.

## Models

MobileNetV2/HFEA produces four paired 64-channel feature maps. Each stage uses the selected context module before TAR temporal skips and the DCR decoder. Q retains N spatial positions; K/V retains N/8 tokens through two shared soft prototypes in each 4x4 cell.

- M0_CTRL: context modules disabled.
- M1_HYPER_QKV: map the pair to the Poincare ball, compute aligned Mobius value displacements at every position, then geometrically compress those displacements. Dense hyperbolic Q retrieves compressed K by geodesic distance; retrieved displacement values are aggregated geometrically and Log-mapped for fusion with the local branch.
- M2_CA_QKV: dense Q and compressed K use the existing LayerNorm/temporal-difference assignments. Global V concatenates signed and absolute projected temporal differences. Exchanging dates negates the signed component and retains the magnitude component. The local signed-difference branch is unchanged.

Four-output BCE+Dice is the segmentation objective. All stage_modules parameters have weight decay 0; the remaining parameters have weight decay 1e-4. Batch 32, seed 2333, Adam lr 5e-4/betas(0.9,0.99)/eps 1e-8, poly^0.9 and the initial 200-step warmup are unchanged. Target 40K steps is rounded up to full epochs. All formal models start from fresh initialization and the same MobileNetV2 pretrained weights.

## Health validation and scheduling

Before formal training, each module completes 3 full LEVIR epochs (666 steps) and 4 full SYSU epochs (1,500 steps), using the full 40K learning-rate schedule. The probe records mean/range of training loss, gradient norms, stage branch/residual magnitudes, task gradients, attention entropy, query radius CV/clipping and displacement clipping. A full validation pass is recorded at the end. Probe checkpoints are separate from formal initialization and never enter the metrics workbook.

The scheduler requires all four passed health results. Both physical GPUs use measured per-variant memory reservations with 1 GiB headroom. Priority is LEVIR → SYSU → WHU → CDD, with fitting-job backfill across datasets. Four datasets each run hyperbolic, ordinary and CTRL: 12 formal experiments.

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

Validation F1 selects the checkpoint. Formal testing evaluates that checkpoint after strict FP32 deployment folding with tolerance 2e-4; binary disagreement is recorded. Training checkpoints retain optimizer and RNG state.

```bash
python train_scripts/CASA/Run6/collect_results.py
```

Only complete TEST RESULTS blocks enter the workbook. Preserve all 135 existing formal records; 12 new results bring the total to 147 unless other records are added. The collector directly compares old cells, updates README and writes `docs/temporary/models_and_metrics_CASA_Run6.txt`. Earlier frozen snapshots and source archives remain separate.

The Run6 snapshot uses the model sources in `source_code.tar.gz`; the `current` snapshot records the working tree.

## Initial observations

Source revision: `a946963`. Formal queue started 2026-10-23 11:54:12 CST; supervisor PID 1160295.

| 数据集 / 模块 | epoch / step | 逐轮平均训练损失 | 短训验证F1 (%) |
|---|---:|---|---:|
| LEVIR / 双曲 | 3 / 666 | 2.110 → 1.172 → 1.068 | 87.87 |
| LEVIR / 普通 | 3 / 666 | 2.109 → 1.175 → 1.061 | 88.41 |
| SYSU / 双曲 | 4 / 1500 | 2.592 → 1.959 → 1.723 → 1.582 | 79.50 |
| SYSU / 普通 | 4 / 1500 | 2.594 → 1.945 → 1.710 → 1.567 | 78.27 |

四项均通过真实数据健康检查，四层分支、残差及任务梯度均非零。双曲查询半径CV最低0.167，查询裁剪比例最高0.66%。短训验证结果不作为正式测试指标。

| 初始物理GPU | 正式任务 |
|---|---|
| 0 | LEVIR双曲、SYSU双曲 |
| 1 | LEVIR普通、LEVIR CTRL、SYSU CTRL |

Formal runs use fresh initialization. Full-batch measured allocated memory: CTRL 8873 MiB, hyperbolic 14057 MiB, ordinary 10737 MiB. Reservations including CUDA context: 9534/14846/11582 MiB. Initial five-way usage sampled 29.2/29.9 GiB with GPU utilization 97/100%.

| 正式实验 | 已完成epoch数 | 初期平均训练损失 |
|---|---:|---|
| LEVIR / `M0_CTRL` | 9 | 2.108076 → 1.182811 → 1.042988 → 1.015425 |
| LEVIR / `M1_HYPER_QKV` | 8 | 2.114929 → 1.177570 → 1.054548 → 0.985084 |
| LEVIR / `M2_CA_QKV` | 9 | 2.115384 → 1.188654 → 1.051330 → 1.006890 |
| SYSU / `M0_CTRL` | 5 | 2.596308 → 1.937525 → 1.715062 → 1.571029 |
| SYSU / `M1_HYPER_QKV` | 4 | 2.593634 → 1.945122 → 1.715477 → 1.564393 |

### Limitations

Hyperbolic attention remains near uniform in several early stages. In the health samples, SYSU stage1 displacement clipping reached 13.01%; formal validation diagnostic samples reached 28.45%. Query clipping remained low and all health-probe task gradients were live. These observations describe early optimization and do not establish final accuracy gains.
