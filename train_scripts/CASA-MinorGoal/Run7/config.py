"""Run7: pool the KV prototypes inside the hyperbolic space, and drop the beta scale.

Until now the prototypes were averaged in the ambient Euclidean space and only
afterwards lifted to the Lorentz model; the manifold therefore shaped the query
side and the distance, never the knowledge that was being compressed.  Run7 moves
the compression itself onto the manifold: every member pixel of a cell is
projected, scaled by ``1/sqrt(d)``, lifted, and averaged with a weighted Lorentz
(Klein) barycentre, so both K and V are manifold objects before retrieval.  The
change stream in TAR also loses its learnable scale ``beta``: the difference
enters the fused content with unit weight.

The matrix is the requested 2x2 over the two modules, on SYSU first and LEVIR
after, with the edge term switched off so the two factors are the only
difference:

CTRL       reparam + none                 neither module
TAR_ONLY    cg      + none                 the bridge alone
HRA_ONLY    diff    + manifold (lorentz)   the retrieval alone
TAR_HRA     cg      + manifold (lorentz)   both
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run7"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run7"))
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run7_manifold_pooling_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEED = 2334
SELECTION = "val"

# SYSU first, then LEVIR, as requested.
DATASETS = ("SYSU-CD-256", "LEVIR-CD-256")

ARMS = {
    "CTRL": dict(module="none", tar="reparam", decoder="reparam", seed=2334,
                 support="mass", detail=False, edge_weight=0.0,
                 edge_detach="retrieval", edge_scales="all", aggregate="lorentz",
                 prototype_pooling="euclidean"),
    "TAR_ONLY": dict(module="none", tar="cg", decoder="reparam", seed=2334,
                     support="mass", detail=False, edge_weight=0.0,
                     edge_detach="retrieval", edge_scales="all", aggregate="lorentz",
                     prototype_pooling="euclidean"),
    "HRA_ONLY": dict(module="manifold", tar="diff", decoder="reparam", seed=2334,
                     support="mass", detail=False, edge_weight=0.0,
                     edge_detach="retrieval", edge_scales="all", aggregate="lorentz",
                     prototype_pooling="lorentz"),
    "TAR_HRA": dict(module="manifold", tar="cg", decoder="reparam", seed=2334,
                    support="mass", detail=False, edge_weight=0.0,
                    edge_detach="retrieval", edge_scales="all", aggregate="lorentz",
                    prototype_pooling="lorentz"),
}
ARM_ORDER = ("CTRL", "TAR_ONLY", "HRA_ONLY", "TAR_HRA")

JOBS = tuple((arm, dataset) for dataset in DATASETS for arm in ARM_ORDER)

TRAINABLE_PARAMS = {"CTRL": 2047125, "TAR_ONLY": 2093461,
                    "HRA_ONLY": 1961771, "TAR_HRA": 2111283}
DEPLOY_PARAMS = {"CTRL": 1935810, "TAR_ONLY": 2043842,
                 "HRA_ONLY": 1923424, "TAR_HRA": 2061664}
DEFAULT_RESERVATIONS = {"CTRL": 9800, "TAR_ONLY": 10400,
                        "HRA_ONLY": 12400, "TAR_HRA": 12800}
BENCH = {arm: arm for arm in ARM_ORDER}
EDGE_WEIGHT = 0.0
EDGE_DILATE = 1

# No frozen external reference: the control arm is trained inside this run, so the
# only fair differences are the within-run contrasts below.
REFERENCE = {}

# Differences are reported against the same-run control, one factor at a time.
GATES = {
    "TAR_ONLY": dict(versus="CTRL", minimum=0.36, noise=0.32,
                     note="change-gated bridge alone, against the plain 1x1 bridge"),
    "HRA_ONLY": dict(versus="CTRL", minimum=0.36, noise=0.32,
                     note="manifold-pooled retrieval alone, against no retrieval"),
    "TAR_HRA": dict(versus="CTRL", minimum=0.47, noise=0.32,
                    note="both modules against the control"),
}


def experiments():
    """Every (arm, dataset) pair in scheduling order."""
    return list(JOBS)


def run_dir(arm, dataset):
    return ROOT / arm / dataset


def checkpoint_dir(arm, dataset):
    return CHECKPOINTS / arm / dataset


def train_script():
    return RUNTIME / "models" / "train_CASA.py"


def model_options(arm):
    spec = ARMS[arm]
    options = ["--module", spec["module"], "--context-heads", "1",
               "--tar-variant", spec["tar"], "--decoder-variant", spec["decoder"],
               "--support", spec["support"], "--selection-split", SELECTION]
    if spec["edge_weight"] > 0:
        options += ["--edge-weight", str(spec["edge_weight"]),
                    "--edge-dilate", "1",
                    "--edge-detach", spec["edge_detach"],
                    "--edge-scales", spec["edge_scales"]]
    if spec["module"] == "manifold":
        options += ["--aggregate", spec["aggregate"],
                    "--prototype-pooling", spec["prototype_pooling"]]
    return options


def seed_for(arm):
    return ARMS[arm]["seed"]


def reservation_for(arm):
    return DEFAULT_RESERVATIONS[arm]


FORECAST_HOURS = {(arm, dataset): (2.9 if dataset.startswith("SYSU") else 2.7)
                  for arm, dataset in JOBS}


def reservation_mib(arm):
    return DEFAULT_RESERVATIONS[arm]
