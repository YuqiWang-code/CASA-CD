"""Run8: does the compression have to happen on the manifold?

Run8 packed the two modules into a 2x2 and both arms that carried retrieval also
carried the manifold pooling, so the run cannot say whether the barycentre itself
buys anything.  Run8 fixes the architecture (manifold retrieval on the change
stream, ratio-free bridge, edge term off) and switches one axis only:

POOL_EUCLID    prototypes averaged in the tangent space, then lifted
POOL_LORENTZ   prototypes averaged on the manifold (weighted Klein barycentre)

Same seed, same schedule, SYSU only: this is the single factor that Run8 left
unmeasured.  The two arms have identical parameter counts by construction, so any
difference is geometry, not capacity.
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

PROTOCOL = "casa_minor_run8_pooling_axis_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEED = 2334
SELECTION = "val"

# Single library: the only place the retrieval arms ever moved.
DATASETS = ("SYSU-CD-256",)

ARMS = {
    "POOL_EUCLID": dict(module="manifold", tar="cg", decoder="reparam", seed=2334,
                        support="mass", detail=False, edge_weight=0.0,
                        edge_detach="retrieval", edge_scales="all", aggregate="lorentz",
                        prototype_pooling="euclidean"),
    "POOL_LORENTZ": dict(module="manifold", tar="cg", decoder="reparam", seed=2334,
                         support="mass", detail=False, edge_weight=0.0,
                         edge_detach="retrieval", edge_scales="all", aggregate="lorentz",
                         prototype_pooling="lorentz"),
}
ARM_ORDER = ("POOL_EUCLID", "POOL_LORENTZ")

JOBS = tuple((arm, dataset) for dataset in DATASETS for arm in ARM_ORDER)

# Pooling is a pure geometric operation: both arms carry the same weights.
TRAINABLE_PARAMS = {"POOL_EUCLID": 2111283, "POOL_LORENTZ": 2111283}
DEPLOY_PARAMS = {"POOL_EUCLID": 2061664, "POOL_LORENTZ": 2061664}
DEFAULT_RESERVATIONS = {"POOL_EUCLID": 12800, "POOL_LORENTZ": 12800}
BENCH = {arm: arm for arm in ARM_ORDER}
EDGE_WEIGHT = 0.0
EDGE_DILATE = 1

# No frozen external reference: the control arm is trained inside this run, so the
# only fair differences are the within-run contrasts below.
REFERENCE = {}

# Differences are reported against the same-run control, one factor at a time.
GATES = {
    "POOL_LORENTZ": dict(versus="POOL_EUCLID", minimum=0.36, noise=0.32,
                         note="averaging on the manifold against averaging in the tangent space"),
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
