# CASA-MinorGoal / Run5: GT edge supervision on the bridge and the retrieval

## Question

Run4 gave the bridge a signed, exposure-normalised difference stream and gave the
retrieval manifold-valued aggregation.  Both are still trained only by the change
mask, so the boundary of a changed region is a side effect rather than a target.
Run5 asks one question: does an explicit ground-truth edge target on the two
places where change evidence lives buy accuracy, and at what price.

## Arms (same seed, same schedule, paired inside the run)

| arm | module | tar | edge | train (M) | deploy (M) |
|---|---|---|---|---|---|
| `CG_HRA` | manifold | cg | — | 2.1113 | 2.0617 |
| `CG_HRA_EDGE` | manifold | cg | 0.1 | 2.1114 | 2.0617 |

Two 65-parameter 1x1 heads read the propagated change evidence of the bridge and
the write-back of the retrieval; both exist only while training, so the deployed
graph, its parameter count, its latency and the fold check are identical between
the two arms.

## Library order

`SYSU` → `LEVIR` → `WHU` → `CDD` (two slots on GPU1, eight jobs).  SYSU first
because the small-object library is where the last two runs moved most.

## Edge target

`sobel_edge_map(gt)`: replicate-pad, unpadded 3x3 Sobel, magnitude in [0, 1],
then a 1-pixel dilation.  Replicate padding matters: zero padding treats the
outside of the crop as background and invents a frame edge wherever a changed
region touches the border (a full 256x256 foreground mask yields 2,032 false
ring pixels), while the boundary metric used for evaluation ignores the frame.
The supervised loss is a positive-weighted BCE (positive share clamped at 1/64)
plus soft Dice, applied at every scale of both streams.

## Run

```bash
source environment.sh
python validate.py && python prepare.py
MINOR_GPUS=1 python pipeline.py          # --resume to continue
```

Collect with `python collect_results.py` after the queue reports complete; it
runs `compare.py`, whose gate is `CG_HRA_EDGE - CG_HRA >= 0.36 pp` (seed noise
0.18 pp on WHU, 0.32 pp on SYSU; CDD stays exploratory).
