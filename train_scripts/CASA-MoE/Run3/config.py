"""Dense heterogeneous-expert exploration: paired ablations on GPU0."""
from pathlib import Path

PROTOCOL = 'casa_moe_40k_candidate_evidence_v3'
ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-MoE/Run3')
REPO = Path('/home/yqwang/projects/CASA-CD')
CHECKPOINTS = Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MoE/Run3')
PRETRAINED = REPO / 'pretrained_weight/mobilenet_v2-b0353104.pth'
DATASETS = ['LEVIR-CD-256', 'SYSU-CD-256']
VARIANTS = {'M_D_DYNAMIC': 'moe', 'M_T_TAR': 'none', 'M_S_STATIC': 'static'}


def experiments():
    return [f'{variant}/{dataset}' for dataset in DATASETS for variant in VARIANTS]


def runtime_root():
    import json
    return Path(json.loads((ROOT / 'runtime.json').read_text())['directory'])
