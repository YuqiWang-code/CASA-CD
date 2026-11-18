"""Run3: four attribution runs on the single authorized GPU.

Every job here answers one pre-registered question with a frozen-weight or
freshly trained arm, and every arm has a matched reference in Run1/Run2:

ARM_C/WHU/2334     hyperbolic context on top of the SPP pooling, seed 2334.
                   Compared with Run2 SPP/WHU/2334 it isolates "does the module
                   itself add anything".
ARM_C/SYSU/2334    same arm on SYSU, seed 2334.
SUPP_PIX/SYSU/2334 ARM_C plus a per-scale pixel-area support prior instead of the
                   pooled participation mass (``--support pixel``); the delta
                   against ARM_C/SYSU is the second lever.
DETAIL/LEVIR/2333  DIFF plus the boundary detail head (``--detail-head``), seed
                   2333 because LEVIR has an exact paired Run1 DIFF/SPP control
                   at that seed (90.8535 / 91.0612).

The first three use seed 2334 so their Run2 controls already exist. The GPU is
GPU1 only (GPU0 is retired after Run2), two slots, sequential fallback.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run3"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run3"))
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run3_attribution_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEED = 2334
SELECTION = "val"
DATASETS = ("WHU-CD-256", "SYSU-CD-256", "LEVIR-CD-256")

ARMS = {
    "ARM_C": dict(module="hyperbolic", tar="spp", decoder="reparam",
                  seed=2334, support="mass", detail=False),
    "SUPP_PIX": dict(module="hyperbolic", tar="spp", decoder="reparam",
                     seed=2334, support="pixel", detail=False),
    "DETAIL": dict(module="difference", tar="spp", decoder="reparam",
                   seed=2333, support="mass", detail=True),
}
ARM_ORDER = ("ARM_C", "SUPP_PIX", "DETAIL")

# The run matrix is not a Cartesian product: two arms are measured on the
# dataset where their control already exists.
JOBS = (
    ("ARM_C", "WHU-CD-256"),
    ("ARM_C", "SYSU-CD-256"),
    ("SUPP_PIX", "SYSU-CD-256"),
    ("DETAIL", "LEVIR-CD-256"),
)

TRAINABLE_PARAMS = {"ARM_C": 2129577, "SUPP_PIX": 2129577, "DETAIL": 2132394}
DEPLOY_PARAMS = {"ARM_C": 2006998, "SUPP_PIX": 2006998, "DETAIL": 2009815}
DEFAULT_RESERVATIONS = {"ARM_C": 12600, "SUPP_PIX": 12600, "DETAIL": 12600}
BENCH = {"ARM_C": "ARM_C", "SUPP_PIX": "SUPP_PIX", "DETAIL": "DETAIL"}

# Matched references, all at the seed each arm actually uses. The three seed-2334
# rows are the *same-seed* Run2 runs: WHU 93.8540 and SYSU 82.2689 are
# SPP_S2334, not the 2333 numbers (93.6925 / 82.7915) that share the arm name.
REFERENCE = {
    ("SPP", "WHU-CD-256", 2334): 93.8540,
    ("SPP", "SYSU-CD-256", 2334): 82.2689,
    ("SPP", "LEVIR-CD-256", 2333): 91.0612,
    ("DIFF", "LEVIR-CD-256", 2333): 90.8535,
}
# Run1 spatial diagnostics on LEVIR (boundary <= 0 px / interior), seed 2333.
LEVIR_BOUNDARY = dict(diff=59.486, spp=60.110, interior_diff=96.349, positive_share=0.3141)

# Pre-registered gates. `minimum` is the smallest delta (pp) that counts as an
# effect; `noise` is the same-configuration seed spread measured in Run2, used
# only to label a delta as "inside the noise" rather than as a gate.
GATES = {
    "ARM_C": dict(versus="SPP", minimum=0.36, noise=0.18,
                  note="module presence at the same seed; WHU"),
    "SUPP_PIX": dict(versus="ARM_C", minimum=0.47, noise=0.32,
                     note="second lever: pixel support prior over pooled mass"),
    "DETAIL": dict(versus="DIFF", minimum=0.36, noise=0.18,
                   note="boundary branch on LEVIR; also gated on boundary/interior F1"),
}
LEVIR_DETAIL_GATES = dict(total=90.8535 + 0.36,
                          boundary=59.486 + 3.0,
                          interior=96.349 - 0.18,
                          positive_share=0.3141)


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
