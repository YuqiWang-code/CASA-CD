"""Run1 of the minor-goal line: multi-scale temporal difference pooling and
difference-conditioned retrieval.

Arms
----
CTRL : trained baseline, no context module, foldable bi-temporal mixing.
SPP  : same topology, the local refinement now also receives a 1x1/3x3/5x5/7x7
       depthwise-pooled absolute temporal difference (foldable into one 7x7 DW).
DIFF : SPP plus a retrieval whose keys and values are pooled from the temporal
       difference stream instead of the fused feature.

Every arm keeps four scales, four decoder levels and a single head, uses the
validation split for model selection and the same optimizer budget, so the
differences are attributable to the two design changes only.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run1"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run1"))
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_spp_diff_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEED = 2333
SELECTION = "val"
DATASETS = ("WHU-CD-256", "SYSU-CD-256", "LEVIR-CD-256")

ARMS = {
    "CTRL": dict(module="none", tar="reparam", decoder="reparam"),
    "SPP": dict(module="none", tar="spp", decoder="reparam"),
    "DIFF": dict(module="difference", tar="spp", decoder="reparam"),
}
ARM_ORDER = ("CTRL", "SPP", "DIFF")
TRAINABLE_PARAMS = {"CTRL": 2047125, "SPP": 2118293, "DIFF": 2129705}
DEPLOY_PARAMS = {"CTRL": 1935810, "SPP": 1995714, "DIFF": 2007126}
DEFAULT_RESERVATIONS = {"CTRL": 9500, "SPP": 10200, "DIFF": 12600}
BENCH = {"CTRL": "CTRL", "SPP": "SPP", "DIFF": "DIFF"}


def experiments():
    return [(arm, dataset) for arm in ARM_ORDER for dataset in DATASETS]


def model_options(arm):
    spec = ARMS[arm]
    return ["--module", spec["module"], "--context-heads", "1",
            "--tar-variant", spec["tar"], "--decoder-variant", spec["decoder"],
            "--selection-split", SELECTION]


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
