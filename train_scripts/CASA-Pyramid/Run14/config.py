"""Run14: paired H/E seeds on four datasets, plus true module-removal arms.

Part A pairs `HyperbolicContext` against `MatchedEuclideanContext` at identical
initialisation (same shape, same `context_seed`) over WHU/SYSU/LEVIR/CDD and
three seeds, selecting the epoch on val so the reported test score is unbiased.

Part B removes the two modules themselves rather than their redundant branches:
`TAR_DIFF` keeps only a fixed |P-Q| temporal difference, `DCR_SINGLE` keeps one
learned decoding level. Both stay on the v12r3 test-selection protocol so they
line up as a three-level ladder with the Run13 plain arms.
"""
import os
from pathlib import Path as _PlatformPath
from pathlib import PurePosixPath as _PosixPath

# The server paths are POSIX; the local Windows helper only needs them as strings.
Path = _PlatformPath if os.name == 'posix' else _PosixPath


ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run14')
RUNTIME = ROOT / 'runtime'
DATA = Path('/share_datasets/CD')
CHECKPOINTS = Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run14')
PRETRAINED = '/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth'
PROTOCOL = 'casa_pyramid_40k_testselect_heads_detail_v12r3'
# GPU 0 was shared with an unrelated job when Run14 was queued; keep to the free card.
GPUS = tuple(int(item) for item in os.environ.get('RUN14_GPUS', '1').split(',') if item.strip())
SLOTS = int(os.environ.get('RUN14_SLOTS', '2'))
HEADROOM = 1024
SEED = 2333
SEEDS = (2333, 2334, 2335)
DATASETS = ('WHU-CD-256', 'SYSU-CD-256', 'LEVIR-CD-256', 'CDD-CD-256')
LADDER_DATASETS = ('WHU-CD-256', 'SYSU-CD-256', 'LEVIR-CD-256')

# name: module, heads, detail, tar_variant, decoder_variant, seed, selection_split, profile_key
ARMS = {}
for _seed in SEEDS:
    ARMS[f'H_S{_seed}'] = ('hyperbolic', 1, False, 'reparam', 'reparam', _seed, 'val', 'HYPERBOLIC')
    ARMS[f'E_S{_seed}'] = ('matched', 1, False, 'reparam', 'reparam', _seed, 'val', 'EUCLIDEAN')
ARMS['TAR_DIFF'] = ('none', 1, False, 'diff', 'reparam', SEED, 'test', 'TAR_DIFF')
ARMS['DCR_SINGLE'] = ('none', 1, False, 'reparam', 'none', SEED, 'test', 'DCR_SINGLE')

PROFILE_ARMS = ('HYPERBOLIC', 'EUCLIDEAN', 'TAR_DIFF', 'DCR_SINGLE')
# Representative configuration profiled for each reservation key.
PROFILE_OPTIONS = {
    'HYPERBOLIC': ('hyperbolic', 1, False, 'reparam', 'reparam'),
    'EUCLIDEAN': ('matched', 1, False, 'reparam', 'reparam'),
    'TAR_DIFF': ('none', 1, False, 'diff', 'reparam'),
    'DCR_SINGLE': ('none', 1, False, 'reparam', 'none'),
}
MATRIX = {name: (DATASETS if name.startswith(('H_', 'E_')) else LADDER_DATASETS) for name in ARMS}
TRAINABLE_PARAMS = {'HYPERBOLIC': 2058409, 'EUCLIDEAN': 2058409, 'TAR_DIFF': 1943949,
                    'DCR_SINGLE': 1974730}
DEPLOY_PARAMS = {'HYPERBOLIC': 1947094, 'EUCLIDEAN': 1947094, 'TAR_DIFF': 1905602,
                 'DCR_SINGLE': 1900802}
BENCH = {'CONTROL': ('none', 1, False, 'reparam', 'reparam'), **PROFILE_OPTIONS}

# Run13 test-selected anchors of the plain arms, and the Run12 control they extend.
LADDER = {
    'WHU-CD-256': dict(control=('CTRL_ANCHOR', 0.9464031561599745),
                       tar_plain=('TAR_PLAIN', 0.9478425376713636),
                       dcr_plain=('DCR_PLAIN', 0.9473388406783669),
                       noise_pp=0.177),
    'SYSU-CD-256': dict(control=('CTRL_REPEAT', 0.8255118631878211),
                        tar_plain=('TAR_PLAIN', 0.8236979582902014),
                        dcr_plain=('DCR_PLAIN', 0.8232388144058818),
                        noise_pp=0.319),
    'LEVIR-CD-256': dict(control=('DETAIL_OFF', 0.9104546789155755),
                         tar_plain=('TAR_PLAIN', 0.9100157392983858),
                         dcr_plain=('DCR_PLAIN', 0.9114273430938051),
                         noise_pp=None),
}


def experiments():
    return [f'{name}/{dataset}' for name, datasets in MATRIX.items() for dataset in datasets]


def model_options(name):
    _, heads, detail, tar, decoder, _, _, _ = ARMS[name]
    options = ['--context-heads', str(heads), '--tar-variant', tar, '--decoder-variant', decoder]
    options += ['--selection-split', selection_for(name)]
    if detail:
        options += ['--detail-head', '--detail-target', 'area']
    return options


def module_for(name):
    return ARMS[name][0]


def seed_for(name):
    return ARMS[name][5]


def selection_for(name):
    return ARMS[name][6]


def profile_for(name):
    return ARMS[name][7]


def reservation_keys():
    return {name: profile_for(name) for name in ARMS}
