"""Run8: decouple the change feature, then decouple the object size (CD2-TAR).

The bridge now answers two questions in order. First *what changed*: an
illumination normalised signed difference produces ``Delta`` and a separate
concat 1x1 produces the scene content ``M``; the two only meet at the fusion
point. Second *how large is the change*: ``Delta`` is split into a lossless
three-band pyramid ``S = Delta - P3``, ``M = P3 - P9``, ``L = P9`` with
``S + M + L == Delta``, each band gets its own 1x1, and a pixel-wise router
weighs them. Only the small band receives the 128^2 stem detail and a dilated
depthwise context, so small objects stop being averaged into large regions.

The matrix is the requested 2x2 over the two modules, SYSU first and LEVIR
after, with the edge term switched off so the two factors are the only
difference:

CTRL       reparam + none                 neither module
TAR_ONLY    cd2     + none                 the decoupling bridge alone
HRA_ONLY    diff    + manifold (lorentz)   the retrieval alone
TAR_HRA     cd2     + manifold (lorentz)   both
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run8"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run8"))
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run8_cd2_v1"
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
    "TAR_ONLY": dict(module="none", tar="cd2", decoder="reparam", seed=2334,
                     support="mass", detail=False, edge_weight=0.0,
                     edge_detach="retrieval", edge_scales="all", aggregate="lorentz",
                     prototype_pooling="euclidean"),
    "HRA_ONLY": dict(module="manifold", tar="diff", decoder="reparam", seed=2334,
                     support="mass", detail=False, edge_weight=0.0,
                     edge_detach="retrieval", edge_scales="all", aggregate="lorentz",
                     prototype_pooling="lorentz"),
    "TAR_HRA": dict(module="manifold", tar="cd2", decoder="reparam", seed=2334,
                    support="mass", detail=False, edge_weight=0.0,
                    edge_detach="retrieval", edge_scales="all", aggregate="lorentz",
                    prototype_pooling="lorentz"),
}
ARM_ORDER = ("CTRL", "TAR_ONLY", "HRA_ONLY", "TAR_HRA")

JOBS = tuple((arm, dataset) for dataset in DATASETS for arm in ARM_ORDER)

TRAINABLE_PARAMS = {"CTRL": 2047125, "TAR_ONLY": 2122209,
                    "HRA_ONLY": 1961771, "TAR_HRA": 2140031}
DEPLOY_PARAMS = {"CTRL": 1935810, "TAR_ONLY": 2072590,
                 "HRA_ONLY": 1923424, "TAR_HRA": 2090412}
DEFAULT_RESERVATIONS = {"CTRL": 9800, "TAR_ONLY": 10600,
                        "HRA_ONLY": 12400, "TAR_HRA": 13000}
BENCH = {arm: arm for arm in ARM_ORDER}
EDGE_WEIGHT = 0.0
EDGE_DILATE = 1

# No frozen external reference: the control arm is trained inside this run, so the
# only fair differences are the within-run contrasts below.
REFERENCE = {}

GATES = {
    "TAR_ONLY": dict(versus="CTRL", minimum=0.36, noise=0.32,
                     note="decoupling bridge alone, against the plain 1x1 bridge"),
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
