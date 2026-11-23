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

# Stage B: the prototype tree, on one carrier so the tree is the only difference.
#
# All three arms run the ARM_C carrier (hyperbolic retrieval whose keys come from
# the fused content) because that is the only retrieval variant that ever cleared
# a gate on SYSU; the tree is then the single variable:
#
#   H_FLAT     no tree: one shared radius, all 224 prototypes visible
#   H_TREE     per-scale radii + neighbours-and-ancestors pruning, no hinge
#   H_TREE_E   the same, plus the child-inside-parent cone hinge (weight .05)
#
# Rows stay seed-paired against each other and, for the record, against the
# stage-A recipe that won on each dataset.
STAGE_B = {
    "H_FLAT": dict(module="hyperbolic", tree="none", entailment=0.0,
                   label="flat hyperbolic: every prototype visible"),
    "H_TREE": dict(module="hyperbolic", tree="tree", entailment=0.05,
                   label="tree + hyperbolic: radii, ancestor pruning, cone hinge"),
    "H_TREE_E": dict(module="matched", tree="tree", entailment=0.05,
                     label="tree + Euclidean: same tree, no curvature, log-radius prior"),
}
STAGE_B_ARMS = ("H_FLAT", "H_TREE", "H_TREE_E")
STAGE_B_SEEDS = (2334, 2335, 2336)
STAGE_B_DATASETS = ("SYSU-CD-256", "LEVIR-CD-256")
STAGE_B_ARM_ORDER = tuple(f"{arm}-{seed}" for arm in STAGE_B_ARMS
                          for seed in STAGE_B_SEEDS)
STAGE_B_JOBS = tuple((arm, dataset) for dataset in STAGE_B_DATASETS
                     for arm in STAGE_B_ARM_ORDER)

MAX_STEPS = 60000

# MobileNetV2 at bs16 peaks near 6 GiB; two slots per card leaves room for the
# eval graph and the fold check.
DEFAULT_RESERVATIONS = {"A_CTRL": 6000, "A_PHOTO": 6000, "A_60K": 6000, "A_60K_PHOTO": 6000,
                        "H_FLAT": 6000, "H_TREE": 6000, "H_TREE_E": 6000}

# Stage A measures the noise floor, it does not test a hypothesis.
GATES = {}

CROSS_BLOCKS = {}


def experiments():
    """Stage A pairs first, then the stage-B tree arms once their gates are locked."""
    return list(JOBS) + list(STAGE_B_JOBS)


def stage_b_arms():
    """Arm specs for stage B, sharing the stage-A carrier and budget."""
    carrier = RECIPES["A_CTRL"]
    arms = {}
    for name in STAGE_B_ARM_ORDER:
        base, seed = name.rsplit("-", 1)
        spec = STAGE_B[base]
        arms[name] = dict(
            backbone="mobilenet_v2", backbone_key="MB", freeze=False, freeze_key="T",
            module=spec["module"], tar="reparam", decoder="reparam", support="mass",
            prototype_pooling="euclidean", aggregate="lorentz", detail=False,
            edge_weight=0.0, edge_detach="retrieval", edge_scales="all",
            ablation=base, recipe="A_CTRL", photometric=False,
            steps=carrier["steps"], seed=int(seed), stage="B",
            tree=spec["tree"], entailment=spec["entailment"])
    return arms


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
    if spec.get("tree", "none") != "none":
        options += ["--prototype-tree", spec["tree"],
                    "--entailment-weight", str(spec.get("entailment", 0.0))]
    return options


def seed_for(arm):
    return ARMS[arm]["seed"]


def reservation_for(arm):
    return DEFAULT_RESERVATIONS.get(ARMS[arm]["ablation"], 6000)


FORECAST_HOURS = {(arm, dataset): (3.1 if ARMS[arm]["steps"] > 40000 else 2.1)
                  for arm, dataset in JOBS}


def reservation_mib(arm):
    return reservation_for(arm)


# Stage-B specs join the same registry so every helper below works unchanged;
# they stay out of `JOBS`, so the stage-A queue is unaffected.
ARMS.update(stage_b_arms())
