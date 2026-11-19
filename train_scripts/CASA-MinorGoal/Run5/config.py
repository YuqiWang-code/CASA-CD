"""Run5: does GT edge supervision on the bridge and the retrieval pay off.

One pair of arms, four libraries, one seed, val-selected and threshold 0.5:

CG_HRA       the Run4 pair (change-gated bridge + manifold retrieval) without
             edge supervision, re-run here so the pair is same-seed, same
             schedule and same code revision.
CG_HRA_EDGE  the same model with a Sobel edge target on the ground truth and
             0.1 x (balanced BCE + Dice) on two 1x1 heads: one on the
             propagated change evidence of the bridge, one on the retrieval
             write-back.  The heads are training-only, so the deployed graph and
             its parameter count are identical to CG_HRA.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run5"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run5"))
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run5_edge_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEED = 2334
SELECTION = "val"

# order of DATASETS is also the queue order: small-object SYSU first, then LEVIR,
# then the two large-change libraries
DATASETS = ("SYSU-CD-256", "LEVIR-CD-256", "WHU-CD-256", "CDD-CD-256")

ARMS = {
    "CG_HRA": dict(module="manifold", tar="cg", decoder="reparam",
                   seed=2334, support="mass", detail=False, edge_weight=0.0),
    "CG_HRA_EDGE": dict(module="manifold", tar="cg", decoder="reparam",
                        seed=2334, support="mass", detail=False, edge_weight=0.1),
}
ARM_ORDER = ("CG_HRA", "CG_HRA_EDGE")

JOBS = tuple((arm, dataset) for dataset in DATASETS for arm in ARM_ORDER)

TRAINABLE_PARAMS = {"CG_HRA": 2111287, "CG_HRA_EDGE": 2111417}
DEPLOY_PARAMS = {"CG_HRA": 2061668, "CG_HRA_EDGE": 2061668}
DEFAULT_RESERVATIONS = {"CG_HRA": 12800, "CG_HRA_EDGE": 12800}
BENCH = {"CG_HRA": "CG_HRA", "CG_HRA_EDGE": "CG_HRA_EDGE"}

# No external reference: the two arms are paired inside this run, so every gate
# is "edge arm minus its own same-seed, same-schedule control".
REFERENCE = {}

# Pre-registered gates (pp).  `noise` is the same-configuration seed spread
# measured on Run2 (WHU 0.18, SYSU 0.32); LEVIR and CDD stay exploratory.
GATES = {
    "CG_HRA_EDGE": dict(versus="CG_HRA", minimum=0.36, noise=0.18,
                        note="edge supervision against the same model without it"),
}

EDGE_WEIGHT = 0.1
EDGE_DILATE = 1


def experiments():
    return list(JOBS)


def model_options(arm):
    spec = ARMS[arm]
    options = ["--module", spec["module"], "--context-heads", "1",
               "--tar-variant", spec["tar"], "--decoder-variant", spec["decoder"],
               "--support", spec["support"], "--selection-split", SELECTION]
    if spec["detail"]:
        options.append("--detail-head")
    if spec["edge_weight"] > 0:
        options += ["--edge-weight", f"{spec['edge_weight']:g}",
                    "--edge-dilate", str(EDGE_DILATE)]
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
