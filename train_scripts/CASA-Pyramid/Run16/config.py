"""Run16: module-removal arms replayed on the three archived versions.

Run15 showed that the hyperbolic geometry is not where the accuracy comes from.
This round asks the other half of the question: do the TAR temporal mixing and
the DCR multi-level decoder keep their measured contribution on Run1 / Run4 /
Run10, whose ``str_tar.py`` and ``str_dcr.py`` are byte-identical to the v12r3
files Run13/Run14 ablated?

Every arm keeps the 4-scale -> 4-level -> single-head topology and only swaps
the learnable content of one stage, so a delta isolates that stage:

``TAR_DIFF``    learned temporal mixing (concat / P+Q / Q-P)  -> fixed |P-Q| + 1x1
``DCR_SINGLE``  multi-level coarse-to-fine decoder             -> single-level upsampling

Both are selected on validation F1 so they are directly comparable with the
Run15 view-selected controls living in ``BASELINE_ROOT``.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

Path = _PlatformPath if os.name == "posix" or os.environ.get("RUN16_LOCAL_IO") else _PosixPath

ROOT = Path(os.environ.get("RUN16_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run16"))
RUNTIME = ROOT / "runtime"
OUTPUT_ROOT = ROOT
DATA = Path("/share_datasets/CD")
CHECKPOINTS = Path("/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run16")
PRETRAINED = Path("/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth")
BASELINE_ROOT = Path(os.environ.get("RUN15_ROOT", "/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run15"))

PROTOCOL = "casa_pyramid_40k_valselect_module_removal_v1"
GPUS = tuple(int(x) for x in os.environ.get("RUN16_GPUS", "1").split(","))
SLOTS = int(os.environ.get("RUN16_SLOTS", "2"))
HEADROOM = 1024
SEED = 2333

DATASETS = ("WHU-CD-256", "SYSU-CD-256")

VERSIONS = {
    "Run1": dict(
        model_dir="models_Run1",
        seed_flags=(),
        selection_flag=None,
        selection="val",
        note="Poincare-ball line; selects on validation F1 natively",
        protocol="casa_pyramid_40k_guide02_v1",
    ),
    "Run4": dict(
        model_dir="models_Run4",
        seed_flags=(),
        selection_flag="--selection-split",
        selection="val",
        note="direct Lorentz line",
        protocol="casa_pyramid_40k_relative_adamwctx_v4",
    ),
    "Run10": dict(
        model_dir="models_Run10",
        seed_flags=("--seed", "--context-seed"),
        selection_flag="--selection-split",
        selection="val",
        note="posterior-compression line",
        protocol="casa_pyramid_40k_supervisedposterior_v10",
    ),
}

ARMS = {
    "TAR_DIFF": dict(module="none", tar="diff", decoder="reparam",
                     note="fixed |P-Q| + 1x1 instead of learned temporal mixing"),
    "DCR_SINGLE": dict(module="none", tar="reparam", decoder="none",
                       note="single-level upsampling instead of the coarse-to-fine decoder"),
}

TRAINABLE_PARAMS = {"TAR_DIFF": 1943949, "DCR_SINGLE": 1974730}
DEPLOY_PARAMS = {"TAR_DIFF": 1905602, "DCR_SINGLE": 1900802}

# Fallback reservations before prepare.py measures the real peaks (MiB); both
# arms carry the full context-free backbone, which peaks near the CTRL arm.
DEFAULT_RESERVATIONS = {"TAR_DIFF": 10000, "DCR_SINGLE": 9500}


def tag(version, arm):
    return f"{version}/{arm}"


def model_dir(version):
    return VERSIONS[version]["model_dir"]


def arms(version=None):
    return tuple(ARMS)


def label(version, arm):
    return arm


def experiments():
    return [(version, arm, dataset) for version in VERSIONS for arm in ARMS for dataset in DATASETS]


def run_dir(version, arm, dataset):
    return ROOT / version / arm / dataset


def checkpoint_dir(version, arm, dataset):
    return CHECKPOINTS / version / arm / dataset


def baseline_dir(version, dataset):
    """Run15 ``none`` control for the same version and dataset."""
    return BASELINE_ROOT / version / "none" / dataset


def model_options(version, arm):
    options = ["--module", ARMS[arm]["module"],
               "--tar-variant", ARMS[arm]["tar"],
               "--decoder-variant", ARMS[arm]["decoder"]]
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
    return table.get(reservation_key(version, arm), table.get(arm, 10000))

