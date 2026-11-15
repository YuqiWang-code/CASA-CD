"""Run15: one-protocol ablation of the three archived CASA-Pyramid versions.

Run1 / Run4 / Run10 each keep their own source tree under
``runtime/models_<RunN>`` and are trained on WHU and SYSU with the same
loader, budget, and *validation* model selection, so the module conclusions
can be compared across versions and against Run14's val-selected pairs.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("RUN15_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("RUN15_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run15"))
RUNTIME = ROOT / "runtime"
OUTPUT_ROOT = ROOT
DATA = Path("/share_datasets/CD")
CHECKPOINTS = Path("/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run15")
PRETRAINED = Path("/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth")

PROTOCOL = "casa_pyramid_40k_valselect_versions_v1"
GPUS = tuple(int(x) for x in os.environ.get("RUN15_GPUS", "1").split(","))
SLOTS = int(os.environ.get("RUN15_SLOTS", "2"))
HEADROOM = 1024
SEED = 2333

DATASETS = ("WHU-CD-256", "SYSU-CD-256")

VERSIONS = {
    "Run1": dict(
        model_dir="models_Run1",
        arms=("none", "hyperbolic", "qkv"),
        labels={"none": "CTRL", "hyperbolic": "H_POINCARE", "qkv": "E_RUN1"},
        seed_flags=(),
        selection_flag=None,
        selection="val",
        note="Poincare-ball hyperbolic context; selects on validation F1 natively",
        protocol="casa_pyramid_40k_guide02_v1",
    ),
    "Run4": dict(
        model_dir="models_Run4",
        arms=("none", "hyperbolic", "qkv"),
        labels={"none": "CTRL", "hyperbolic": "H_LORENTZ", "qkv": "E_RUN4"},
        seed_flags=(),
        selection_flag="--selection-split",
        selection="val",
        note="direct Lorentz geometry; test-selection patched to val for this ablation",
        protocol="casa_pyramid_40k_relative_adamwctx_v4",
    ),
    "Run10": dict(
        model_dir="models_Run10",
        arms=("none", "hyperbolic", "matched", "qkv"),
        labels={"none": "CTRL", "hyperbolic": "H_LORENTZ", "matched": "E_MATCHED", "qkv": "E_RUN10"},
        seed_flags=("--seed", "--context-seed"),
        selection_flag="--selection-split",
        selection="val",
        note="posterior-compression line; matched Euclidean is the isomorphic twin",
        protocol="casa_pyramid_40k_supervisedposterior_v10",
    ),
}

TRAINABLE_PARAMS = {
    "Run1/none": 2047125,
    "Run1/hyperbolic": 2063581,
    "Run1/qkv": 2063581,
    "Run4/none": 2047125,
    "Run4/hyperbolic": 2064537,
    "Run4/qkv": 2064537,
    "Run10/none": 2047125,
    "Run10/hyperbolic": 2058410,
    "Run10/matched": 2058410,
    "Run10/qkv": 2064537,
}
DEPLOY_PARAMS = {
    "Run1/none": 1935810,
    "Run1/hyperbolic": 1952266,
    "Run1/qkv": 1952266,
    "Run4/none": 1935810,
    "Run4/hyperbolic": 1953222,
    "Run4/qkv": 1953222,
    "Run10/none": 1935810,
    "Run10/hyperbolic": 1947095,
    "Run10/matched": 1947095,
    "Run10/qkv": 1953222,
}

# Fallback reservations before prepare.py measures the real peaks (MiB).
DEFAULT_RESERVATIONS = {"none": 9500, "hyperbolic": 12500, "matched": 11500, "qkv": 11500}


def tag(version, arm):
    return f"{version}/{arm}"


def model_dir(version):
    return VERSIONS[version]["model_dir"]


def arms(version):
    return VERSIONS[version]["arms"]


def label(version, arm):
    return VERSIONS[version]["labels"][arm]


def experiments():
    return [(version, arm, dataset) for version in VERSIONS for arm in arms(version) for dataset in DATASETS]


def run_dir(version, arm, dataset):
    return ROOT / version / arm / dataset


def checkpoint_dir(version, arm, dataset):
    return CHECKPOINTS / version / arm / dataset


def model_options(version, arm):
    options = ["--module", arm]
    selection_flag = VERSIONS[version]["selection_flag"]
    if selection_flag:
        options += [selection_flag, VERSIONS[version]["selection"]]
    for flag in VERSIONS[version]["seed_flags"]:
        options += [flag, str(SEED)]
    return options


def train_script(version):
    return RUNTIME / model_dir(version) / "train_CASA.py"


def reservation_key(version, arm):
    return f"{version}/{arm}"


def reservation_mib(version, arm, reservations=None):
    table = reservations or DEFAULT_RESERVATIONS
    return table.get(reservation_key(version, arm), table.get(arm, 12500))
