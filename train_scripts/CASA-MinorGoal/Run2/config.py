"""Run2 of the minor-goal line: seed replication plus the two isolating arms.

Run1 left three questions open, and this run answers each with a matched arm:

CTRL_S2334 / CTRL_S2335   same baseline as Run1 at two more seeds, so every
                          delta in this run is a per-seed paired difference.
SPP_S2334 / SPP_S2335     replication of the multi-scale difference pooling.
DIFF_S2334 / DIFF_S2335   replication of the difference-conditioned retrieval.
SPP1                      same branch structure as SPP but the pooling window set
                          is {1x1} only, so the gain of widening the window to
                          3/5/7 is read off directly instead of inferred.
DIFF_SPP1                 difference retrieval on top of the 1x1-only pooling,
                          the matching pooling-width control for DIFF.
DIFF_NOSPP                difference retrieval fed by the plain |P-Q| mixing
                          (no SPP stage), i.e. the retrieval decoupled from the
                          multi-scale pooling branch.

All arms keep four scales, four decoder levels and one head, use the validation
split for model selection, and share the optimizer budget; every comparison in
this run is therefore within a seed unless it is explicitly a cross-seed one.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run2"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run2"))
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_spp_diff_seeds_v2"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEED = 2333
SELECTION = "val"
DATASETS = ("WHU-CD-256", "SYSU-CD-256")

ARMS = {
    "CTRL_S2334": dict(module="none", tar="reparam", decoder="reparam", seed=2334),
    "CTRL_S2335": dict(module="none", tar="reparam", decoder="reparam", seed=2335),
    "SPP_S2334": dict(module="none", tar="spp", decoder="reparam", seed=2334),
    "SPP_S2335": dict(module="none", tar="spp", decoder="reparam", seed=2335),
    "DIFF_S2334": dict(module="difference", tar="spp", decoder="reparam", seed=2334),
    "DIFF_S2335": dict(module="difference", tar="spp", decoder="reparam", seed=2335),
    "SPP1": dict(module="none", tar="spp1", decoder="reparam", seed=2333),
    "DIFF_SPP1": dict(module="difference", tar="spp1", decoder="reparam", seed=2333),
    "DIFF_NOSPP": dict(module="difference", tar="diff", decoder="reparam", seed=2333),
}
ARM_ORDER = ("CTRL_S2334", "SPP_S2334", "DIFF_S2334",
             "CTRL_S2335", "SPP_S2335", "DIFF_S2335",
             "SPP1", "DIFF_SPP1", "DIFF_NOSPP")

TRAINABLE_PARAMS = {
    "CTRL_S2334": 2047125, "CTRL_S2335": 2047125, "SPP1": 2095509,
    "SPP_S2334": 2118293, "SPP_S2335": 2118293,
    "DIFF_S2334": 2129705, "DIFF_S2335": 2129705,
    "DIFF_SPP1": 2106921, "DIFF_NOSPP": 1955361,
}
DEPLOY_PARAMS = {
    "CTRL_S2334": 1935810, "CTRL_S2335": 1935810, "SPP1": 1983426,
    "SPP_S2334": 1995714, "SPP_S2335": 1995714,
    "DIFF_S2334": 2007126, "DIFF_S2335": 2007126,
    "DIFF_SPP1": 1994838, "DIFF_NOSPP": 1917014,
}
DEFAULT_RESERVATIONS = {
    "CTRL_S2334": 9500, "CTRL_S2335": 9500, "SPP1": 10200,
    "SPP_S2334": 10200, "SPP_S2335": 10200,
    "DIFF_S2334": 12600, "DIFF_S2335": 12600,
    "DIFF_SPP1": 12600, "DIFF_NOSPP": 9500,
}
BENCH = {"CTRL_S2334": "CTRL", "SPP1": "SPP1", "SPP_S2334": "SPP",
         "DIFF_S2334": "DIFF", "DIFF_SPP1": "DIFF_SPP1", "DIFF_NOSPP": "DIFF_NOSPP"}

# Run1 values at seed 2333, used as the reference column in the summary table.
RUN1_REFERENCE = {
    ("CTRL", "WHU-CD-256"): 93.6234, ("CTRL", "SYSU-CD-256"): 81.7290,
    ("SPP", "WHU-CD-256"): 93.6925, ("SPP", "SYSU-CD-256"): 82.7915,
    ("DIFF", "WHU-CD-256"): 94.3468, ("DIFF", "SYSU-CD-256"): 82.6630,
}


def experiments():
    return [(arm, dataset) for arm in ARM_ORDER for dataset in DATASETS]


def model_options(arm):
    spec = ARMS[arm]
    # The seed is supplied once by run_one.py from seed_for(arm); keeping it out of
    # the model options avoids passing the same flag twice.
    return ["--module", spec["module"], "--context-heads", "1",
            "--tar-variant", spec["tar"], "--decoder-variant", spec["decoder"],
            "--selection-split", SELECTION]


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
