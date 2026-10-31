"""Run9 four-dataset refinement matrix and fixed shared training settings."""
from pathlib import Path

ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9')
PROTOCOL = 'casa_pyramid_40k_refinement_v9'
DATASETS = ['LEVIR-CD-256', 'SYSU-CD-256', 'WHU-CD-256', 'CDD-CD-256']
VARIANTS = {
    'H_REFINE': dict(module='hyperbolic', beta=1.0),
    'E_REFINE': dict(module='matched', beta=1.0),
    'M0_CTRL': dict(module='none', beta=0.0),
    'E_BASE': dict(module='matched', beta=0.0),
}
PROBES = [('H_REFINE', 'SYSU'), ('E_REFINE', 'SYSU'),
          ('H_REFINE', 'WHU'), ('E_REFINE', 'WHU'),
          ('E_BASE', 'SYSU'), ('M0_CTRL', 'SYSU')]
COMPARISON = 'shared_guide_refinement_and_matched_geometry'


def experiments():
    return [f'{variant}/{dataset}' for dataset in DATASETS for variant in VARIANTS
            if variant != 'E_BASE' or dataset in ('SYSU-CD-256', 'WHU-CD-256')]


def training_options(variant):
    spec = VARIANTS[variant]
    return ['--module', spec['module'], '--refinement-beta', str(spec['beta']),
            '--guide-weight', '0.1', '--retrieval-weight', '0',
            '--seed', '2333', '--context-seed', '2333']
