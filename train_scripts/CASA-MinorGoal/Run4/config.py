"""Run4: redesign the temporal bridge and the hyperbolic retrieval together.

Two modules, one question each, on the two libraries where the change signal is
strongest (WHU: large-area change, SYSU: small objects):

TAR_CG    single content 1x1 + signed, exposure-normalised difference + change
          gate + coarse-to-fine evidence propagation. Compared with CTRL it
          answers "does the redesigned bridge beat the plain one".
HRA_SPP   manifold-valued retrieval (Klein barycenter readout, per-scale
          temperature, independent support head, level path, position-aware
          gate) on top of the SPP bridge. Compared with Run3 ARM_C - same
          bridge, same seed, only the readout changes.
CG_HRA    both modules together: does the pair compose.

CTRL is not re-run: Run2 CTRL at seed 2334 is the matched control
(WHU 93.6438, SYSU 82.0917), the same seed every Run4 arm uses.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run4"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run4"))
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run4_manifold_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "3"))
HEADROOM = 1024
SEED = 2334
SELECTION = "val"
DATASETS = ("WHU-CD-256", "SYSU-CD-256")

ARMS = {
    "TAR_CG": dict(module="none", tar="cg", decoder="reparam",
                   seed=2334, support="cell", detail=False),
    "HRA_SPP": dict(module="manifold", tar="spp", decoder="reparam",
                    seed=2334, support="mass", detail=False),
    "CG_HRA": dict(module="manifold", tar="cg", decoder="reparam",
                   seed=2334, support="mass", detail=False),
}
ARM_ORDER = ("TAR_CG", "HRA_SPP", "CG_HRA")

JOBS = tuple((arm, dataset) for arm in ARM_ORDER for dataset in DATASETS)

TRAINABLE_PARAMS = {"TAR_CG": 2093465, "HRA_SPP": 2136115, "CG_HRA": 2111287}
DEPLOY_PARAMS = {"TAR_CG": 2043846, "HRA_SPP": 2013536, "CG_HRA": 2061668}
DEFAULT_RESERVATIONS = {"TAR_CG": 10400, "HRA_SPP": 12800, "CG_HRA": 12800}
BENCH = {"CTRL": "CTRL", "TAR_CG": "TAR_CG", "HRA_SPP": "HRA_SPP", "CG_HRA": "CG_HRA"}

# Matched references. CTRL is Run2 seed 2333; ARM_C is Run3 seed 2334 (hyperbolic
# retrieval on the SPP bridge) and is the reference for HRA_SPP's readout change.
REFERENCE = {
    ("CTRL", "WHU-CD-256", 2334): 93.6438,
    ("CTRL", "SYSU-CD-256", 2334): 82.0917,
    ("ARM_C", "WHU-CD-256", 2334): 93.5646,
    ("ARM_C", "SYSU-CD-256", 2334): 83.3700,
}

# Pre-registered gates (pp). `noise` is the Run2 same-configuration seed spread.
GATES = {
    "TAR_CG": dict(versus="CTRL", minimum=0.36, noise=0.18,
                   note="redesigned bridge against the plain bridge"),
    "HRA_SPP": dict(versus="ARM_C", minimum=0.20, noise=0.18,
                    note="manifold readout against the hyperbolic readout on the same bridge"),
    "CG_HRA": dict(versus="CTRL", minimum=0.47, noise=0.32,
                   note="both modules against the plain bridge"),
}


def experiments():
    return list(JOBS)


def model_options(arm):
    spec = ARMS[arm]
    options = ["--module", spec["module"], "--context-heads", "1",
               "--tar-variant", spec["tar"], "--decoder-variant", spec["decoder"],
               "--support", spec["support"], "--selection-split", SELECTION]
    if spec["detail"]:
        options.append("--detail-head")
    return options


def seed_for(arm):
    return ARMS[arm]["seed"]


def run_dir(arm, dataset):
    return ROOT / arm / dataset


def checkpoint_dir(arm, dataset):
    return CHECKPOINTS / arm / dataset


def train_script():
    return RUNTIME / "models" / "train_CASA.py"


def reservation_key(arm):
    return arm


def reservation_mib(arm, table=None):
    return (table or DEFAULT_RESERVATIONS)[arm]
