"""Run13: single-factor ablations of the TAR temporal fusion and the DCR decoder.

Both arms replace exactly one design decision with a degenerate version of the
same topology, keep the control context (guide only), one head and no detail
head, so every difference against the Run12 control belongs to that decision.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

# The server paths are POSIX; the local Windows helper only needs them as strings.
Path = _PlatformPath if os.name == 'posix' else _PosixPath


ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run13')
RUNTIME = ROOT / 'runtime'
DATA = Path('/share_datasets/CD')
CHECKPOINTS = Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run13')
BASELINE_ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run12')
PRETRAINED = '/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth'
PROTOCOL = 'casa_pyramid_40k_testselect_heads_detail_v12r3'
GPUS = (0, 1)
SLOTS = 2
HEADROOM = 1024
SEED = 2333
DATASETS = ('WHU-CD-256', 'SYSU-CD-256', 'LEVIR-CD-256')
VARIANTS = {
    'TAR_PLAIN': ('none', 1, False, 'plain', 'reparam'),
    'DCR_PLAIN': ('none', 1, False, 'reparam', 'plain'),
}
# Run12 control run of each dataset and the exact Run12 F1 used as the comparison anchor.
BASELINES = {
    'WHU-CD-256': ('CTRL_ANCHOR', 0.9464031561599745),
    'SYSU-CD-256': ('CTRL_REPEAT', 0.8255118631878211),
    'LEVIR-CD-256': ('DETAIL_OFF', 0.9104546789155755),
}
# Same-configuration reruns in Run12, used as the seed noise scale of the anchor.
REPEATS = {
    'WHU-CD-256': (('CTRL_ANCHOR', 0.9464031561599745), ('CTRL_REPEAT', 0.948173472354866),
                   ('DETAIL_OFF', 0.9478322557759873)),
    'SYSU-CD-256': (('CTRL_ANCHOR', 0.828698501504864), ('CTRL_REPEAT', 0.8255118631878211)),
    'LEVIR-CD-256': (('DETAIL_OFF', 0.9104546789155755),),
}
# Training parameters counted locally for the two arms; deployment is 1.935810 M for all.
TRAINABLE_PARAMS = {'TAR_PLAIN': 1974157, 'DCR_PLAIN': 2010250}
MATRIX = {dataset: tuple(VARIANTS) for dataset in DATASETS}
BENCH = {'CONTROL': ('none', 1, False, 'reparam', 'reparam'), **VARIANTS}
PROFILE_ARMS = tuple(VARIANTS)
PROBES = tuple((name, DATASETS[0]) for name in VARIANTS)


def experiments():
    return [f'{variant}/{dataset}' for dataset, variants in MATRIX.items() for variant in variants]


def model_options(variant):
    _, heads, detail, tar, decoder = VARIANTS[variant]
    options = ['--context-heads', str(heads), '--tar-variant', tar, '--decoder-variant', decoder]
    if detail:
        options += ['--detail-head', '--detail-target', detail_target(variant)]
    return options


def detail_target(variant):
    return 'nearest-exact' if variant.endswith('_HARD') else 'area'


def seed_for(variant):
    return 2334 if variant.endswith('_S2334') else SEED
