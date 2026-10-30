"""Dense heterogeneous-expert exploration: paired ablations on GPU1."""
from pathlib import Path

PROTOCOL = 'casa_moe_40k_dense_v1'
ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-MoE/Run1')
REPO = Path('/home/yqwang/projects/CASA-CD')
CHECKPOINTS = Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MoE/Run1')
PRETRAINED = REPO / 'pretrained_weight/mobilenet_v2-b0353104.pth'
DATASETS = ['LEVIR-CD-256', 'SYSU-CD-256', 'WHU-CD-256', 'CDD-CD-256']
VARIANTS = {
    'M4_MOE': 'moe', 'M0_CTRL': 'none', 'M4_EQUAL': 'equal',
    'M1_HYPER_QKV': 'hyperbolic', 'M2_CA_QKV': 'qkv', 'M3_FREQUENCY': 'frequency',
    'M4_NO_HYPER': 'no_hyperbolic', 'M4_NO_QKV': 'no_qkv', 'M4_NO_FREQUENCY': 'no_frequency',
}


def experiments():
    return [f'{variant}/{dataset}' for dataset in DATASETS
            for variant in (VARIANTS if dataset in DATASETS[:2]
                            else ('M4_MOE', 'M0_CTRL', 'M4_EQUAL'))]


def runtime_root():
    import json
    return Path(json.loads((ROOT / 'runtime.json').read_text())['directory'])
