"""Run6: does the retrieval really have to predict edges itself.

Run6 showed that 0.1 x Sobel edge supervision lifts SYSU by +0.47 pp, but the
loss was reaching two places at once: the change evidence of the bridge and the
retrieval write-back ``U_s``.  The write-back is the suspicious one, because a
whole-image global term enters it, so a 1x1 edge head on top of it can only
shift the score up or down, not localise an outline.

This run separates the two taps with one arm, on SYSU only, same seed and same
schedule as Run6:

EDGE_FULL     edge supervision on both heads (identical to Run6 CG_HRA_EDGE, so
              it also serves as a reproduction check of 82.8559).
EDGE_TAP_CUT  the same two heads and the same edge loss, but the retrieval head
              reads ``U_s.detach()``.  The head still learns; the edge gradient
              simply stops at the write-back and only reaches the bridge.

If EDGE_TAP_CUT keeps the gain, the retrieval does not need the edge signal and
the next step is scale selection on the bridge.  If it loses the gain, the
retrieval edge tap is doing the work and the mechanism deserves a closer look.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run6"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run6"))
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run6_edge_tap_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEED = 2334
SELECTION = "val"

# SYSU only: it is the library where the edge term moved the needle.
DATASETS = ("SYSU-CD-256",)

ARMS = {
    "EDGE_FULL": dict(module="manifold", tar="cg", decoder="reparam", seed=2334,
                      support="mass", detail=False, edge_weight=0.1, edge_detach="none"),
    "EDGE_TAP_CUT": dict(module="manifold", tar="cg", decoder="reparam", seed=2334,
                         support="mass", detail=False, edge_weight=0.1,
                         edge_detach="retrieval"),
}
ARM_ORDER = ("EDGE_FULL", "EDGE_TAP_CUT")

JOBS = tuple((arm, dataset) for dataset in DATASETS for arm in ARM_ORDER)

TRAINABLE_PARAMS = {"EDGE_FULL": 2111417, "EDGE_TAP_CUT": 2111417}
DEPLOY_PARAMS = {"EDGE_FULL": 2061668, "EDGE_TAP_CUT": 2061668}
DEFAULT_RESERVATIONS = {"EDGE_FULL": 12800, "EDGE_TAP_CUT": 12800}
BENCH = {"EDGE_FULL": "EDGE_FULL", "EDGE_TAP_CUT": "EDGE_TAP_CUT"}

# Same-seed, same-schedule reference from Run6 (val-selected, threshold 0.5).
REFERENCE = {
    ("EDGE_FULL", "SYSU-CD-256", 2334): 82.8559,
    ("CG_HRA_EDGE", "SYSU-CD-256", 2334): 82.8559,
}

# Pre-registered gate (pp).  noise is the SYSU same-configuration seed spread
# measured on Run2 (0.32 pp).
GATES = {
    "EDGE_TAP_CUT": dict(versus="EDGE_FULL", minimum=0.36, noise=0.32,
                         note="cutting the retrieval edge tap against the full tap"),
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
    if spec.get("edge_detach", "none") != "none":
        options += ["--edge-detach", spec["edge_detach"]]
    return options


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


def module_for(arm):
    return ARMS[arm]["module"]


def seed_for(arm):
    return ARMS[arm]["seed"]


def selection_for(arm):
    return SELECTION


def profile_for(arm):
    return arm


def reservation_keys():
    return {arm: arm for arm in ARMS}
