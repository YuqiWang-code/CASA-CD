"""Run10 stage A: recipe only, structure frozen at the Run8 control.

Run9 showed the encoder is not the lever (frozen blocks collapse the retrieval
arms, and ViT-B full fine-tuning loses 4.15 pp on SYSU), so the backbone goes
back to MobileNetV2 and every arm here keeps the Run8 control architecture:

    module none + reparam bridge + reparam decoder, ~2.05 M / 1.94 M params.

Two switches are crossed, both cheap and both aimed at the two things the first
nine rounds never controlled:

    steps      40k vs 60k poly      -- was the 20k budget of Run9 a handicap?
    photometric on / off            -- |P-Q| currently absorbs exposure drift

Each (steps, photometric) cell runs two seeds (2334, 2335) on SYSU, LEVIR and
WHU, 24 jobs.  The paired same-config difference between the two seeds is the
noise floor sigma that stage B gates on; no arm in stage A is declared better.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run10"))
WEIGHTS = Path(os.environ.get(
    "MINOR_WEIGHTS", "/home/yqwang/projects/CASA-CD/pretrained_weight"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run10_hierarchy_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "0,1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEED = 2334
SELECTION = "val"
EDGE_DILATE = 1

PRETRAINED_BY_BACKBONE = {
    "mobilenet_v2": WEIGHTS / "mobilenet_v2-b0353104.pth",
}

BACKBONES = {"MB": dict(backbone="mobilenet_v2", label="MobileNetV2")}
FREEZE = {"T": False}

# SYSU first (hardest), then LEVIR, then WHU.
DATASETS = ("SYSU-CD-256", "LEVIR-CD-256", "WHU-CD-256")

SEEDS = (2334, 2335)

# steps, photometric, label
RECIPES = {
    "A_CTRL": dict(steps=40000, photometric=False, label="40k poly, no photometric"),
    "A_PHOTO": dict(steps=40000, photometric=True, label="40k poly + photometric"),
    "A_60K": dict(steps=60000, photometric=False, label="60k poly, no photometric"),
    "A_60K_PHOTO": dict(steps=60000, photometric=True, label="60k poly + photometric"),
}

ARM_ORDER = tuple(f"{recipe}-{seed}" for recipe in
                  ("A_CTRL", "A_PHOTO", "A_60K", "A_60K_PHOTO") for seed in SEEDS)

ARMS = {}
for _name in ARM_ORDER:
    _recipe, _seed = _name.rsplit("-", 1)
    ARMS[_name] = dict(
        backbone="mobilenet_v2", backbone_key="MB", freeze=False, freeze_key="T",
        module="none", tar="reparam", decoder="reparam", support="mass",
        prototype_pooling="euclidean", aggregate="lorentz", detail=False,
        edge_weight=0.0, edge_detach="retrieval", edge_scales="all",
        ablation=_recipe, recipe=_recipe, photometric=RECIPES[_recipe]["photometric"],
        steps=RECIPES[_recipe]["steps"], seed=int(_seed), stage="A")

JOBS = tuple((arm, dataset) for dataset in DATASETS for arm in ARM_ORDER)

# Filled by the smoke test at prepare time; stage A declares nothing better.
TRAINABLE_PARAMS = {}
DEPLOY_PARAMS = {}
BENCH = {}

MAX_STEPS = 60000

# MobileNetV2 at bs16 peaks near 6 GiB; two slots per card leaves room for the
# eval graph and the fold check.
DEFAULT_RESERVATIONS = {"A_CTRL": 6000, "A_PHOTO": 6000, "A_60K": 6000, "A_60K_PHOTO": 6000}

# Stage A measures the noise floor, it does not test a hypothesis.
GATES = {}

CROSS_BLOCKS = {}


def experiments():
    """Every (arm, dataset) pair in scheduling order."""
    return list(JOBS)


def run_dir(arm, dataset):
    return ROOT / arm / dataset


def checkpoint_dir(arm, dataset):
    return CHECKPOINTS / arm / dataset


def train_script():
    return RUNTIME / "models" / "train_CASA.py"


def pretrained_for(arm):
    return PRETRAINED_BY_BACKBONE[ARMS[arm]["backbone"]]


def steps_for(arm):
    return ARMS[arm]["steps"]


def batch_for(arm):
    return 16


def model_options(arm):
    spec = ARMS[arm]
    options = ["--module", spec["module"], "--context-heads", "1",
               "--tar-variant", spec["tar"], "--decoder-variant", spec["decoder"],
               "--support", spec["support"], "--selection-split", SELECTION,
               "--backbone", spec["backbone"]]
    if spec["photometric"]:
        options.append("--photometric")
    return options


def seed_for(arm):
    return ARMS[arm]["seed"]


def reservation_for(arm):
    return DEFAULT_RESERVATIONS[ARMS[arm]["ablation"]]


FORECAST_HOURS = {(arm, dataset): (3.1 if ARMS[arm]["steps"] > 40000 else 2.1)
                  for arm, dataset in JOBS}


def reservation_mib(arm):
    return reservation_for(arm)
