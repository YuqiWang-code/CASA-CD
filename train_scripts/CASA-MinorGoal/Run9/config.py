"""Run9: swap the encoder for VMamba-Tiny and a 4-layer ViT-L, frozen and not.

The task is fixed at the Run8 optimum (CD2 bridge + manifold retrieval) and the
only thing that changes is the encoder:

    backbone  x  freeze  x  ablation arm  x  dataset

with four ablation arms per (backbone, freeze):

    CTRL      reparam bridge + no retrieval
    TAR_ONLY  CD2 bridge   + no retrieval
    HRA_ONLY  diff bridge  + manifold retrieval
    TAR_HRA   CD2 bridge   + manifold retrieval

SYSU runs first, LEVIR after, exactly as requested.  The encoder is the only
difference between the two backbones; everything downstream keeps its Run8
configuration so the comparison is a clean swap.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("MINOR_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("MINOR_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run9"))
RUNTIME = ROOT / "runtime"
DATA = Path(os.environ.get("MINOR_DATA", "/share_datasets/CD"))
CHECKPOINTS = Path(os.environ.get(
    "MINOR_CHECKPOINTS", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run9"))
WEIGHTS = Path(os.environ.get(
    "MINOR_WEIGHTS", "/home/yqwang/projects/CASA-CD/pretrained_weight"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run9_backbone_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "0,1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEED = 2334
SELECTION = "val"

PRETRAINED_BY_BACKBONE = {
    "mobilenet_v2": WEIGHTS / "mobilenet_v2-b0353104.pth",
    "vmamba_tiny": WEIGHTS / "vssm_tiny_0230_ckpt_epoch_262.pth",
    "vit_base4": WEIGHTS / "vit_base_patch16_224_augreg_in21k_ft_in1k.pth",
}

# SYSU first, then LEVIR.
DATASETS = ("SYSU-CD-256", "LEVIR-CD-256")

BACKBONES = {
    "VM": dict(backbone="vmamba_tiny", label="VMamba-Tiny"),
    "VIT": dict(backbone="vit_base4", label="ViT-B/16 / 4 blocks"),
}
FREEZE = {"F": True, "T": False}

ABLATION = {
    "CTRL": dict(module="none", tar="reparam", prototype_pooling="euclidean"),
    "TAR_ONLY": dict(module="none", tar="cd2", prototype_pooling="euclidean"),
    "HRA_ONLY": dict(module="manifold", tar="diff", prototype_pooling="lorentz"),
    "TAR_HRA": dict(module="manifold", tar="cd2", prototype_pooling="lorentz"),
}

ARM_ORDER = tuple(f"{key}_{freeze}_{arm}"
                  for key in ("VM", "VIT")
                  for freeze in ("F", "T")
                  for arm in ("CTRL", "TAR_ONLY", "HRA_ONLY", "TAR_HRA"))

ARMS = {}
for _name in ARM_ORDER:
    _backbone_key, _freeze_key, _arm = _name.split("_", 2)
    _spec = dict(ABLATION[_arm])
    _spec.update(backbone=BACKBONES[_backbone_key]["backbone"],
                 backbone_key=_backbone_key, freeze=FREEZE[_freeze_key],
                 freeze_key=_freeze_key, ablation=_arm, seed=SEED,
                 support="mass", detail=False,
                 edge_weight=0.0, edge_detach="retrieval", edge_scales="all",
                 decoder="reparam", aggregate="lorentz")
    ARMS[_name] = _spec

JOBS = tuple((arm, dataset) for dataset in DATASETS for arm in ARM_ORDER)

# Measured at prepare time; the table stays empty so nothing can silently drift.
# left empty on purpose: the 32 arms have not been measured yet, the smoke test
# records the numbers it sees and validate only compares entries that exist.
TRAINABLE_PARAMS = {}
DEPLOY_PARAMS = {}

# step budget: 20k keeps the 32-job matrix inside the 24 h GPU window; the
# schedule is decayed over the same budget for every arm, so the comparison
# stays internally fair.
MAX_STEPS = int(os.environ.get("MINOR_MAX_STEPS", "20000"))

# Measured at batch 16 on a 31.35 GiB card: the cd2+manifold arm peaks near
# 15 GiB, the cd2 arm near 12 GiB, and the two light arms near 6 GiB, so a card
# holds at most two jobs and a third one OOMs.
DEFAULT_RESERVATIONS = {
    "CTRL": 6000, "TAR_ONLY": 12000, "HRA_ONLY": 6000, "TAR_HRA": 15500,
}
BENCH = {}

# The only fair contrasts are inside one (backbone, freeze) block.
GATES = {}
for _name in ARM_ORDER:
    _spec = ARMS[_name]
    _prefix = f"{_spec['backbone_key']}_{_spec['freeze_key']}"
    if _spec["ablation"] == "TAR_ONLY":
        GATES[_name] = dict(versus=f"{_prefix}_CTRL", minimum=0.36, noise=0.32,
                            note="CD2 bridge inside this backbone block")
    elif _spec["ablation"] == "HRA_ONLY":
        GATES[_name] = dict(versus=f"{_prefix}_CTRL", minimum=0.36, noise=0.32,
                            note="manifold retrieval inside this backbone block")
    elif _spec["ablation"] == "TAR_HRA":
        GATES[_name] = dict(versus=f"{_prefix}_CTRL", minimum=0.47, noise=0.32,
                            note="both modules inside this backbone block")

# Cross-block references are reported, never gated: frozen vs trainable and
# VMamba vs ViT are different estimators, not nested models.
CROSS_BLOCKS = {
    "freeze_effect": ("T", "F"),
    "backbone_effect": ("VM", "VIT"),
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


def pretrained_for(arm):
    return PRETRAINED_BY_BACKBONE[ARMS[arm]["backbone"]]


def steps_for(arm):
    return MAX_STEPS


# The 16 below was chosen for VMamba's 26.8 GiB at batch 32, not for accuracy.
# MINOR_BATCH lets the matrix be retrained at 32 without editing the suite, which
# is what a re-run of this protocol means; reservations must be re-measured at
# whatever batch is used (see the note above DEFAULT_RESERVATIONS).
BATCH_SIZE = int(os.environ.get("MINOR_BATCH", "16"))


def batch_for(arm):
    return BATCH_SIZE


def model_options(arm):
    spec = ARMS[arm]
    options = ["--module", spec["module"], "--context-heads", "1",
               "--tar-variant", spec["tar"], "--decoder-variant", spec["decoder"],
               "--support", spec["support"], "--selection-split", SELECTION,
               "--backbone", spec["backbone"]]
    if spec["freeze"]:
        options.append("--freeze-backbone")
    if spec["module"] == "manifold":
        options += ["--aggregate", spec["aggregate"],
                    "--prototype-pooling", spec["prototype_pooling"]]
    return options


def seed_for(arm):
    return ARMS[arm]["seed"]


def reservation_for(arm):
    return DEFAULT_RESERVATIONS[ARMS[arm]["ablation"]]


FORECAST_HOURS = {(arm, dataset): (4.5 if ARMS[arm]["backbone_key"] == "VIT" else 3.4)
                  for arm, dataset in JOBS}


def reservation_mib(arm):
    return reservation_for(arm)
