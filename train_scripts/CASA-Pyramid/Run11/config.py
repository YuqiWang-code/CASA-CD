"""Paired geometry and interior-loss factorial on four RGB datasets."""
from pathlib import Path

ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run11')
RUNTIME = ROOT / 'runtime'
PROTOCOL = 'casa_pyramid_40k_linearpool_interior_v11'
DATASETS = ['LEVIR-CD-256', 'SYSU-CD-256', 'WHU-CD-256', 'CDD-CD-256']
VARIANTS = {
    'H_SIMPLE': dict(module='hyperbolic', interior_positive_weight=1.),
    'H_INTERIOR': dict(module='hyperbolic', interior_positive_weight=1.25),
    'E_SIMPLE': dict(module='matched', interior_positive_weight=1.),
    'E_INTERIOR': dict(module='matched', interior_positive_weight=1.25),
    'M0_CTRL': dict(module='none', interior_positive_weight=1.),
    'CTRL_INTERIOR': dict(module='none', interior_positive_weight=1.25),
}
PROBES = [(v, 'SYSU-CD-256') for v in VARIANTS] + [
    ('H_SIMPLE', 'WHU-CD-256'), ('H_INTERIOR', 'WHU-CD-256')]
PRETRAINED = '/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth'


def experiments():
    return [f'{v}/{d}' for d in DATASETS for v in VARIANTS]


def training_options(variant):
    spec = VARIANTS[variant]
    return ['--module', spec['module'], '--interior-positive-weight', str(spec['interior_positive_weight']),
            '--guide-weight', '.1', '--retrieval-weight', '0', '--seed', '2333', '--context-seed', '2333']
