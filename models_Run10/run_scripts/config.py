"""Formal centered-readout and supervised-posterior Run10 matrix."""
from pathlib import Path

ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run10')
PROTOCOL = 'casa_pyramid_40k_supervisedposterior_v10'
GUIDE_OBJECTIVE = 'half_raw_half_posterior_area_bce'
TEMPERATURE = .5
DATASETS = ['LEVIR-CD-256', 'SYSU-CD-256', 'WHU-CD-256', 'CDD-CD-256']
VARIANTS = {'H_POSTERIOR': dict(module='hyperbolic'),
            'E_POSTERIOR': dict(module='matched'), 'M0_CTRL': dict(module='none')}
COMPARISON = 'centered_readout_supervised_class_posterior'


def experiments():
    return [f'{variant}/{dataset}' for dataset in DATASETS for variant in VARIANTS]


def training_options(variant):
    result = ['--module', VARIANTS[variant]['module'], '--guide-weight', '.1',
              '--retrieval-weight', '0', '--seed', '2333', '--context-seed', '2333']
    if VARIANTS[variant]['module'] != 'none':
        result.extend(['--compression-temperature', str(TEMPERATURE)])
    return result
