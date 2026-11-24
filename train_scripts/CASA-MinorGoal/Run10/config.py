"""Run10: make the three contributions (TAR, HRA, DCR) defensible one at a time.

Run7 measured the three modules together; Run10 rebuilds the trunk so each one
carries a mechanism that can be stated in one sentence, then isolates them:

* DCR  - the head now refines at 128x128 with the encoder stem before the final
         upsampling (the 64x64 logit was what capped the boundary band on LEVIR),
         and the four deep-supervision levels are weighted 1/.5/.25/.125 instead
         of equally.  This is the trunk, so it is inside every arm.
* TAR  - the change gate became a *competition*: ``g = sigma(1x1(positive -
         negative))``, ``fused = g * content + (1 - g) * delta``, both paths
         channel-normalised, gate zero-initialised (g = 0.5 at step 0).  The
         contrast kernel follows the scale (3/5/7/7) instead of 7 everywhere.
* HRA  - centring is a learnable gate (``mean_gate``, zero-initialised: the mean
         is kept at start), the retrieval can run on the two coarse scales only,
         the prototypes can be compacted 224 -> 32, and the class prior can be
         the confidence ``|2p - 1|`` instead of the participation mass.

Arms, all on the new trunk:

    D_CTRL      reparam + none                      trunk only
    D_CG        cg      + none                      new bridge alone
    R_FULL      cg      + manifold, 0-3, 224, mass  retrieval alone, on top of cg
    R_RES       cg      + manifold, 2-3, 224, mass  retrieval only on 16^2/8^2
    R_COMPACT   cg      + manifold, 0-3, 32, confidence
    R_LEAN      cg      + manifold, 2-3, 32, confidence

Two seeds (2334/2335) because the paired re-run noise of this trunk measured
sigma = 0.39 pp; the pre-registered gate is max(0.36, 1.5 * sigma) = 0.59 pp for
"this module adds something", and -0.36 pp for "this simplification does not
cost" (one noise unit).  SYSU first, then LEVIR, as requested.
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
PRETRAINED = Path(os.environ.get(
    "MINOR_PRETRAINED", "/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth"))
SCRIPTS = _PlatformPath(__file__).resolve().parent

PROTOCOL = "casa_minor_run10_lean_v1"
GPUS = tuple(int(item) for item in os.environ.get("MINOR_GPUS", "1").split(","))
SLOTS = int(os.environ.get("MINOR_SLOTS", "2"))
HEADROOM = 1024
SEEDS = (2334, 2335)
SELECTION = "val"
MAX_STEPS = 40000
BATCH_SIZE = 32

# SYSU first, then LEVIR, as requested.
DATASETS = ("SYSU-CD-256", "LEVIR-CD-256")

_RETRIEVAL = dict(aggregate="lorentz", prototype_pooling="lorentz")

BASE_ARMS = {
    "D_CTRL": dict(module="none", tar="reparam", support="mass",
                   retrieval_scales=(0, 1, 2, 3), compact=False, **_RETRIEVAL),
    "D_CG": dict(module="none", tar="cg", support="mass",
                 retrieval_scales=(0, 1, 2, 3), compact=False, **_RETRIEVAL),
    "R_FULL": dict(module="manifold", tar="cg", support="mass",
                   retrieval_scales=(0, 1, 2, 3), compact=False, **_RETRIEVAL),
    "R_RES": dict(module="manifold", tar="cg", support="mass",
                  retrieval_scales=(2, 3), compact=False, **_RETRIEVAL),
    "R_COMPACT": dict(module="manifold", tar="cg", support="confidence",
                      retrieval_scales=(0, 1, 2, 3), compact=True, **_RETRIEVAL),
    "R_LEAN": dict(module="manifold", tar="cg", support="confidence",
                   retrieval_scales=(2, 3), compact=True, **_RETRIEVAL),
}
BASE_ORDER = ("D_CTRL", "D_CG", "R_FULL", "R_RES", "R_COMPACT", "R_LEAN")


def arm_name(base, seed):
    return f"{base}-{seed}"


def base_of(arm):
    return arm.rsplit("-", 1)[0]


def seed_of(arm):
    return int(arm.rsplit("-", 1)[1])


ARMS = {arm_name(base, seed): dict(BASE_ARMS[base], seed=seed,
                                   detail=False, edge_weight=0.0,
                                   edge_detach="retrieval", edge_scales="all")
        for base in BASE_ORDER for seed in SEEDS}
ARM_ORDER = tuple(sorted(ARMS, key=lambda name: (BASE_ORDER.index(base_of(name)), seed_of(name))))

JOBS = tuple((arm, dataset) for dataset in DATASETS for arm in ARM_ORDER)

# Filled in by prepare.py from the smoke test; placeholders keep the audit strict.
TRAINABLE_PARAMS = {arm: 0 for arm in ARMS}
DEPLOY_PARAMS = {arm: 0 for arm in ARMS}
# measured peaks: D_* ~9.3 GiB, R_* (224 prototypes, four scales) ~13.0 GiB
DEFAULT_RESERVATIONS = {arm: (9800 if arm.startswith("D_") else 13600) for arm in ARMS}
BENCH = {arm: base_of(arm) for arm in ARMS}
EDGE_WEIGHT = 0.0
EDGE_DILATE = 1
REFERENCE = {}

# Same-run control, one factor at a time.  ``minimum`` is the effect that would
# count as real; a negative minimum means "must not lose more than this".
GATES = {
    "D_CG": dict(versus="D_CTRL", minimum=0.59, noise=0.39,
                 note="competition gate + per-scale contrast on top of the new trunk"),
    "R_FULL": dict(versus="D_CG", minimum=0.59, noise=0.39,
                   note="retrieval adds something once the bridge is competitive"),
    "R_RES": dict(versus="R_FULL", minimum=-0.36, noise=0.39,
                  note="running retrieval on 16^2/8^2 only must not cost"),
    "R_COMPACT": dict(versus="R_FULL", minimum=-0.36, noise=0.39,
                      note="32 prototypes + confidence prior must not cost"),
    "R_LEAN": dict(versus="R_FULL", minimum=-0.36, noise=0.39,
                   note="the combined lean configuration must not cost"),
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


def context_kwargs(arm):
    """Constructor options for CASANet, shared by prepare/validate/benchmark."""
    spec = ARMS[arm]
    return dict(module=spec["module"], tar_variant=spec["tar"],
                decoder_variant="reparam", support=spec["support"],
                detail_head=spec["detail"],
                edge_supervision=spec["edge_weight"] > 0,
                edge_detach=spec["edge_detach"],
                aggregate_mode=spec["aggregate"],
                prototype_pooling=spec["prototype_pooling"],
                retrieval_scales=tuple(spec["retrieval_scales"]),
                compact_prototypes=bool(spec["compact"]))


def model_options(arm):
    spec = ARMS[arm]
    options = ["--module", spec["module"], "--context-heads", "1",
               "--tar-variant", spec["tar"], "--decoder-variant", "reparam",
               "--support", spec["support"], "--selection-split", SELECTION,
               "--ds-weights", "1,0.5,0.25,0.125"]
    if spec["module"] == "manifold":
        options += ["--aggregate", spec["aggregate"],
                    "--prototype-pooling", spec["prototype_pooling"],
                    "--retrieval-scales", ",".join(str(s) for s in spec["retrieval_scales"])]
        if spec["compact"]:
            options += ["--compact-prototypes"]
    return options


def seed_for(arm):
    return ARMS[arm]["seed"]


def reservation_for(arm):
    return DEFAULT_RESERVATIONS[arm]


FORECAST_HOURS = {(arm, dataset): (2.4 if dataset.startswith("SYSU") else 3.3)
                  for arm, dataset in JOBS}


def reservation_mib(arm):
    return DEFAULT_RESERVATIONS[arm]


def steps_for(arm):
    return MAX_STEPS


def batch_for(arm):
    return BATCH_SIZE


def pretrained_for(arm):
    return PRETRAINED
