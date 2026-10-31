"""Fixed-temperature centered-readout paired probes on physical GPU1."""
import json
from pathlib import Path

RUN_ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run10')
ROOT = RUN_ROOT / 'selection/centered'
PROFILE_PATH = RUN_ROOT / 'selection/classposterior/profile.json'
DATASETS = ('LEVIR', 'SYSU')
SEED = 2333
TEMPERATURE = .5
VARIANTS = {
    'H_CENTER_RAW': dict(module='hyperbolic', models=ROOT/'raw/models',
        protocol='casa_pyramid_40k_centeredposterior_v10', objective='raw_area_bce', profile='H_POSTERIOR'),
    'H_CENTER_SUP': dict(module='hyperbolic', models=ROOT/'supervised/models',
        protocol='casa_pyramid_40k_supervisedposterior_v10', objective='half_raw_half_posterior_area_bce', profile='H_POSTERIOR'),
    'E_CENTER_SUP': dict(module='matched', models=ROOT/'supervised/models',
        protocol='casa_pyramid_40k_supervisedposterior_v10', objective='half_raw_half_posterior_area_bce', profile='E_POSTERIOR'),
}
PROBES = [(arm, dataset) for dataset in DATASETS for arm in VARIANTS]


def valid_probe(report, arm, dataset):
    spec = VARIANTS[arm]
    return (report['health_passed'] and report['task_gradient_check'] and report['posterior_gradient_check']
            and report['protocol'] == spec['protocol'] and report['guide_objective'] == spec['objective']
            and report['module'] == spec['module'] and report['dataset'] == dataset+'-CD-256'
            and report['epochs'] == 4 and report['seed'] == SEED
            and report['compression_temperature_init'] == TEMPERATURE and report['retrieval_weight'] == 0)


def control_references(run_root=RUN_ROOT):
    paths = {'H_POSTERIOR_T05': run_root/'selection/classposterior/probes/H_POSTERIOR_T05',
             'E_POSTERIOR_T05': run_root/'selection/classposterior/probes/E_POSTERIOR_T05',
             'M0_CTRL': run_root/'selection/neighborhood/probes/M0_CTRL'}
    references = {}
    for arm, parent in paths.items():
        for dataset in DATASETS:
            directory=parent/dataset
            health=json.loads((directory/'health_result.json').read_text())
            manifest=json.loads((directory/'run_manifest.json').read_text())
            if (not health['health_passed'] or health['epochs'] != 4 or manifest['seed'] != SEED
                    or manifest['train_loader_seed'] != SEED or manifest['validation_loader_seed'] != SEED+1
                    or manifest['guide_weight'] != .1 or manifest['retrieval_weight'] != 0
                    or manifest['evaluation_split'] != 'val' or manifest['batch_size'] != 32
                    or (arm != 'M0_CTRL' and manifest['compression_temperature_init'] != TEMPERATURE)):
                raise RuntimeError(f'control does not match the paired protocol: {arm}/{dataset}')
            references[f'{arm}/{dataset}']=dict(path=str(directory),health=health,manifest=manifest)
    return references
