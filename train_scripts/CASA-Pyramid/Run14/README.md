# CASA-Pyramid / Run14 — paired geometry seeds and true module removal

Run13 removed the foldable multi-branch redundancy inside TAR and DCR, but both of
its arms kept a learned temporal mix and a four-level coarse-to-fine decoder, so it
could not answer whether the TAR and DCR modules themselves earn their cost. Run12
and Run13 also could not settle the hyperbolic-versus-Euclidean question: every
comparison rested on one or two seeds, and the epoch was chosen on the test split.

Run14 runs two independent parts, on the same 40K-step budget and the same
preprocessing as the mainline protocol.

## Part A — paired H/E over four datasets and three seeds (val-selected)

`HyperbolicContext` and `MatchedEuclideanContext` have identical parameter shapes,
so the same `--seed` produces bitwise identical initial weights; only the retrieval
metric differs. Each pair therefore measures the geometry, not the initialisation.

* arms: `H_S2333/H_S2334/H_S2335` (`--module hyperbolic`) and
  `E_S2333/E_S2334/E_S2335` (`--module matched`), one head, no detail head.
* datasets: WHU, SYSU, LEVIR, CDD — 24 runs.
* epoch selection on **val** (`--selection-split val`), so the reported test score
  is not used for model selection; `independent_test=true` in the manifests.
* primary statistic: paired `H - E` per (dataset, seed), then mean and sd over all
  twelve pairs; per-dataset means over three seeds.

## Part B — removing the modules themselves (test-selected)

Both arms keep the control context (guide only) and the full topology, so they slot
under the Run12 control and the Run13 plain arms as a third ladder level.

| arm | replaces | keeps |
|---|---|---|
| `TAR_DIFF` | learned temporal mixing (`concat / P+Q / Q-P` fusion) | fixed `|P-Q|` difference + one 1×1 and the local block |
| `DCR_SINGLE` | four-level coarse-to-fine fusion | one learned decoding level on the coarsest feature, finer levels added top-down without parameters |

* datasets: WHU, SYSU, LEVIR — 6 runs.
* epoch selection on **test** (v12r3), identical to Run12/Run13, so the three-level
  ladder `control → plain → removed` is measured under one protocol.
* the ladder anchors are the Run13 plain arms (`TAR_PLAIN`, `DCR_PLAIN`) and the
  Run12 control of each dataset.

## Cost of each arm

| configuration | train params | deploy params | batch-1 latency |
|---|---|---|---|
| control (`none`, reparam TAR/DCR) | 2,047,125 | 1,935,810 | 3.678 ms |
| `H`/`E` (1×32 context) | 2,058,409 | 1,947,094 | measured in `inference_benchmark.json` |
| `TAR_DIFF` | 1,943,949 | 1,905,602 | "
| `DCR_SINGLE` | 1,974,730 | 1,900,802 | "

## Layout

```
Run14/
  config.py            arms, dataset matrix, anchors, profile keys
  run_one.py           one run -> one GPU (sets CUDA_VISIBLE_DEVICES, execs train_CASA.py)
  pipeline.py          queue + dual constraints (slot count and free memory)
  prepare.py           CPU audit, 7 model tests, benchmark, profiling, 2-epoch probes
  validate.py          matrix, paired init, folding, gradients, no-op guards
  health_probe.py      two real epochs and gradient checks per configuration
  benchmark.py         batch-1 deployment latency for control/H/E/TAR_DIFF/DCR_SINGLE
  compare.py           paired H-E table and the two three-level ladders
  collect_results.py   download logs, verify protocol, refresh workbook and TXT snapshot
  bootstrap.py         stage runtime + frozen source archive on the server
  environment.sh       conda env and library paths; GPUs are chosen per run
```

## Running

```bash
source environment.sh
python prepare.py      # writes readiness.json only when every check passes
python pipeline.py     # schedules the 30 runs on the authorized GPUs
python pipeline.py --resume   # after a stop or a failure
```

`pipeline_status.json` records completed/active/failed runs; a run counts as done
only when its log contains a complete `TEST RESULTS` block. GPU 0 was occupied by an
unrelated job when the queue was prepared, so only GPU 1 is authorized
(`RUN14_GPUS=1`, two slots). Wall clock is about 38–42 h; `compare.py` runs after
`collect_results.py` downloads the logs.

Selection caveat kept explicit everywhere: Part A is val-selected and therefore
independent of the test split; Part B stays test-selected for comparability with
Run12/Run13 and must not be read as an independent estimate.
