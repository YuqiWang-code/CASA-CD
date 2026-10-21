# CASA-WDMF / Run1

MobileNetV2 + HFEA uses the WDMF-Net encoder. Four paired 64-channel features feed optional per-scale processing, then TAR temporal skip fusion and the DCR decoder. The decoder exposes four scales with a shared 1×1 prediction head for WDMF-Net's four-output BCE+Dice supervision.

| Variant | Per-scale processing | Routing |
|---|---|---|
| `M0_CTRL` | None | HFEA → TAR → DCR |
| `M1_HYPER_QKV` | Change-weighted compressed QKV with Poincaré-distance attention | All four HFEA levels → module → TAR → DCR |
| `M2_CA_QKV` | Change-weighted compressed QKV with dot-product attention | All four HFEA levels → module → TAR → DCR |

Each 4×4 cell produces one context token, weighted by the shared cross-time absolute difference. Full-resolution Q attends to compressed K/V from the opposite time. All scales use 32-dimensional Q/K/V. Module one uses exp-map with c=1 and bounded tangent norm 2.3, stable `2 asinh(sqrt(·))` hyperbolic distance, and HSAM's positive distance convention. Both attention variants standardize their logits, aggregate V, project back to 64 channels and add a learnable scalar residual initialized to zero. They have identical parameter counts. The prior VMamba HSAM/HQKV modules are absent from this network's forward path.

## Training protocol

Upstream: https://github.com/LYT-Works/WDMF-Net at `453c5493b2aab88c908ced9135b677285059ec70`.

- Adam, lr=5e-4, betas=(0.9,0.99), eps=1e-8, weight decay=1e-4; polynomial power 0.9; upstream 200-step warmup.
- Batch 32, 256×256, train drop_last=True, seed 2333; resize, random crop-resize, vertical/horizontal flips, temporal exchange; upstream normalization and tensor conversion.
- Four equally weighted BCE+Dice terms. Validate after every epoch except epoch 0, select maximum validation F1 (ties update), test that selected model after training.
- As in upstream, target 40,000 steps is rounded to full epochs and the scheduler uses the rounded budget: SYSU **40,125**; LEVIR **40,182**; WHU **40,145**; CDD **40,248**.
- Server dataset layout is adapted to `A/B/label/list`; labels use `gray>=128` so CDD JPG compression noise is not foreground. CDD extends the four-dataset protocol with the same 40K target. Python/NumPy generators are also seeded 2333 for reproducible augmentation.
- MobileNet pretrained weight: `/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth`.

## Execution and collection

GPU0 only, **two concurrent jobs**. The three SYSU variants finish before any LEVIR variant starts; WHU follows LEVIR and CDD follows WHU. Full-batch profiles measured peaks of 8,873 / 11,931 / 10,150 MiB (CTRL / module one / module two); three processes exceed the 32 GB capacity after process overhead.

```bash
cd /home/yqwang/projects/CASA-CD
bash train_scripts/CASA-WDMF/Run1/launch.sh 2
bash train_scripts/CASA-WDMF/Run1/stop.sh
```

Logs and status: `/home/yqwang/outputs/CASA-CD/CASA-WDMF/Run1/`. Checkpoints: `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-WDMF/Run1/`. Numerical failure stops the dataset wave and the queue. Complete metrics require the final `=== TEST RESULTS ===` block.

Local collection: `python train_scripts/CASA-WDMF/Run1/collect_results.py`. It downloads text/JSON artifacts into `outputs/CASA-WDMF/Run1`, updates README's result table, and calls the existing metric-table and snapshot tools when new formal results appear.

## Known limitations

HES-Net found that shallow HSAM can amplify noise; this experiment explicitly evaluates all four levels. The geometric attention is a compressed QKV adaptation of HSAM, rather than the paper's dense HW×HW operator. Module one and module two use TAR skips to preserve paired multiscale information before decoding. Results from this 40K, validation-selected MobileNet protocol are compared within this run; earlier 80K VMamba results have a different backbone and selection protocol.
