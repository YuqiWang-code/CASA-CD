"""Run11: HRA, split into three independently switchable geometric axes.

Run7 showed that the parabolic HRA (all-hyperbolic) helps, and a single pooling
ablation (POOL_EUCLID vs POOL_LORENTZ, 83.212 vs 82.824 on SYSU) hinted that the
gain is not one geometric monolith.  Run11 replaces the one switch with three:

* ``member_pool``     WHERE the class prototypes (K/V) are averaged: on the
                      hyperboloid (Lorentz barycentre) or in the ambient chart.
* ``distance``        HOW a query is compared with a prototype: Lorentz geodesic
                      or squared Euclidean in the ambient chart.
* ``value_aggregate`` HOW the retrieved values are combined: Lorentz barycentre
                      or Euclidean weighted mean.

Everything else is the Run10 trunk: MobileNetV2, the cg competition bridge,
change-evidence K/V from the propagated difference stream, the learned-share
prior, the gamma write-back, and the DCR + FineRefine head.  E-HRA and H-HRA are
therefore the same parameter tensor, only three flags differ, which is what
makes an attribution claim possible.

Arms (32 prototypes = (2,2,2,2) everywhere unless stated; all four query scales):

    CTRL     cg + none, guide on the difference   matched no-HRA control
    LOCAL    cg + local only                       TAR + guide + detail branch
    H32      cg + HRA, H/H/H                       the Run10 geometry at 32
    E32      cg + HRA, E/E/E                       matched Euclidean HRA
    E32_HV   cg + HRA, E/E/H                       only the value axis is Lorentz
    H224     cg + HRA, H/H/H, (8,4,4,4)            the 224-prototype reference

Gates are paired within the run (same seeds, same data order).  The pre-run
noise of this trunk is sigma = 0.39 pp, so the "adds something" gate stays
max(0.36, 1.5 sigma) = 0.59 pp; the single-axis value gate uses one noise unit.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run11"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run11"))
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run11_hra_v1"
# this round is authorized on physical GPU 0 only
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "0").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEEDS = (2334, 2335)
SELECTION = "val"
MAX_STEPS = 40000
BATCH_SIZE = 32

# SYSU first, then LEVIR.
DATASETS = ("SYSU-CD-256", "LEVIR-CD-256")

COMPACT = (2, 2, 2, 2)
FLAT = (8, 4, 4, 4)

BASE_ARMS = {
    # no retrieval at all, but the guide is computed from the same difference
    # stream the HRA arms use (the matched control for LOCAL and the HRA arms)
    "CTRL": dict(module="none", tar="cg", guide_source="diff", support="mass",
                 hra_pool="lorentz", hra_distance="lorentz", hra_value="lorentz",
                 hra_grids=COMPACT),
    # TAR + guide + the local depthwise detail branch, without retrieve/write-back
    "LOCAL": dict(module="local", tar="cg", guide_source="tar", support="mass",
                  hra_pool="lorentz", hra_distance="lorentz", hra_value="lorentz",
                  hra_grids=COMPACT),
    "H32": dict(module="hra", tar="cg", guide_source="tar", support="mass",
                hra_pool="lorentz", hra_distance="lorentz", hra_value="lorentz",
                hra_grids=COMPACT),
    "E32": dict(module="hra", tar="cg", guide_source="tar", support="mass",
                hra_pool="euclidean", hra_distance="euclidean",
                hra_value="euclidean", hra_grids=COMPACT),
    "E32_HV": dict(module="hra", tar="cg", guide_source="tar", support="mass",
                   hra_pool="euclidean", hra_distance="euclidean",
                   hra_value="lorentz", hra_grids=COMPACT),
    "H224": dict(module="hra", tar="cg", guide_source="tar", support="mass",
                 hra_pool="lorentz", hra_distance="lorentz", hra_value="lorentz",
                 hra_grids=FLAT),
}
BASE_ORDER = ("CTRL", "LOCAL", "H32", "E32", "E32_HV", "H224")


def arm_name(base, seed):
    return f"{base}-{seed}"


def base_of(arm):
    return arm.rsplit("-", 1)[0]


def seed_of(arm):
    return int(arm.rsplit("-", 1)[1])


ARMS = {arm_name(base, seed): dict(BASE_ARMS[base], seed=seed,
                                   detail=False, edge_weight=0.0,
                                   edge_detach="retrieval", edge_scales="all")
        for base in BASE_ORDER for seed in SEEDS}
ARM_ORDER = tuple(sorted(ARMS, key=lambda name: (BASE_ORDER.index(base_of(name)), seed_of(name))))

JOBS = tuple((arm, dataset) for dataset in DATASETS for arm in ARM_ORDER)

# Measured by the local CPU audit (validate.py) and re-checked by prepare.py.
_TRAINABLE = {"CTRL": 2072404, "LOCAL": 2077204, "H32": 2090099, "E32": 2090099,
              "E32_HV": 2090099, "H224": 2090099}
_DEPLOY = {"CTRL": 2025795, "LOCAL": 2030595, "H32": 2043490, "E32": 2043490,
           "E32_HV": 2043490, "H224": 2043490}
TRAINABLE_PARAMS = {arm: _TRAINABLE[base_of(arm)] for arm in ARMS}
DEPLOY_PARAMS = {arm: _DEPLOY[base_of(arm)] for arm in ARMS}
# 32-token retrieval and the two non-retrieval arms sit near the Run10 D_* peak
# (~9.3 GiB) plus the small retrieval head; H224 is the Run10 R_FULL peak (~13.0).
DEFAULT_RESERVATIONS = {arm: (13600 if base_of(arm) == "H224" else
                              (10200 if ARMS[arm]["module"] == "hra" else 9800))
                        for arm in ARMS}
BENCH = {arm: base_of(arm) for arm in ARMS}
EDGE_WEIGHT = 0.0
EDGE_DILATE = 1
REFERENCE = {}

# Paired, one factor at a time.  ``minimum`` is the effect that counts as real;
# a negative minimum means "must not lose more than this".
GATES = {
    "LOCAL": dict(versus="CTRL", minimum=0.59, noise=0.39,
                  note="guide from D + the local detail branch, before any retrieval"),
    "H32": dict(versus="LOCAL", minimum=0.59, noise=0.39,
                note="H-HRA retrieval adds something on top of the local path"),
    "E32": dict(versus="LOCAL", minimum=0.59, noise=0.39,
                note="E-HRA retrieval adds something on top of the local path"),
    "E32_HV": dict(versus="E32", minimum=0.36, noise=0.39,
                   note="single axis: Lorentz value aggregation vs Euclidean mean"),
    "H224": dict(versus="H32", minimum=-0.36, noise=0.39,
                 note="32 prototypes must not lose more than one noise unit vs 224"),
}

# The attribution question of the round is H32 vs E32; it is not a gate because
# a tie is an informative result (the three axes are not one geometry).
PRIMARY_CONTRAST = ("H32", "E32")


def experiments():
    """Every (arm, dataset) pair in scheduling order."""
    return list(JOBS)


def run_dir(arm, dataset):
    return ROOT / arm / dataset


def checkpoint_dir(arm, dataset):
    return CHECKPOINTS / arm / dataset


def train_script():
    return RUNTIME / "models" / "train_CASA.py"


def context_kwargs(arm):
    """Constructor options for CASANet, shared by prepare/validate/benchmark."""
    spec = ARMS[arm]
    return dict(module=spec["module"], tar_variant=spec["tar"],
                decoder_variant="reparam", support=spec["support"],
                guide_source=spec["guide_source"], detail_head=spec["detail"],
                edge_supervision=spec["edge_weight"] > 0,
                edge_detach=spec["edge_detach"],
                hra_pool=spec["hra_pool"], hra_distance=spec["hra_distance"],
                hra_value=spec["hra_value"], hra_grids=spec["hra_grids"])


def model_options(arm):
    spec = ARMS[arm]
    options = ["--module", spec["module"], "--context-heads", "1",
               "--tar-variant", spec["tar"], "--decoder-variant", "reparam",
               "--support", spec["support"], "--selection-split", SELECTION,
               "--ds-weights", "1,0.5,0.25,0.125"]
    if spec["module"] == "hra":
        options += ["--hra-pool", spec["hra_pool"],
                    "--hra-distance", spec["hra_distance"],
                    "--hra-value", spec["hra_value"],
                    "--hra-grids", ",".join(str(g) for g in spec["hra_grids"]),
                    "--retrieval-scales", "0,1,2,3"]
    elif spec["module"] == "none" and spec["guide_source"] == "diff":
        options += ["--guide-source", "diff"]
    return options


def seed_for(arm):
    return ARMS[arm]["seed"]


def reservation_for(arm):
    return DEFAULT_RESERVATIONS[arm]


FORECAST_HOURS = {(arm, dataset): (2.4 if dataset.startswith("SYSU") else 3.3)
                  for arm, dataset in JOBS}


def reservation_mib(arm):
    return DEFAULT_RESERVATIONS[arm]


def steps_for(arm):
    return MAX_STEPS


def batch_for(arm):
    return BATCH_SIZE


def pretrained_for(arm):
    return PRETRAINED
