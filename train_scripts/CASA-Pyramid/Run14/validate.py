"""Exercise the Run14 arms, selection wiring and queue paths before GPU training."""
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from config import (ARMS, DATASETS, DEPLOY_PARAMS, LADDER, LADDER_DATASETS, MATRIX, PROFILE_OPTIONS,
                    RUNTIME, TRAINABLE_PARAMS, experiments, model_options, profile_for, reservation_keys,
                    seed_for, selection_for)
from pipeline import pending_runs

sys.path.insert(0, str(RUNTIME/'models'))
from model.casa_net import CASANet


def check_cli():
    help_text = subprocess.run([sys.executable, str(RUNTIME/'models/train_CASA.py'), '--help'],
                               check=True, capture_output=True, text=True).stdout
    for flag in ('--selection-split {test,val}', '--tar-variant {reparam,plain,diff}',
                 '--decoder-variant {reparam,plain,none}'):
        assert flag in help_text, flag


def main():
    torch.set_num_threads(4)
    assert len(experiments()) == len(set(experiments())) == 30
    assert set(ARMS) == {f'{arm}_S{seed}' for arm in ('H', 'E') for seed in (2333, 2334, 2335)} | {
        'TAR_DIFF', 'DCR_SINGLE'}
    assert sum(len(MATRIX[arm]) for arm in ARMS) == 30
    for arm in ARMS:
        seed = seed_for(arm)
        options = model_options(arm)
        assert options[options.index('--selection-split')+1] == selection_for(arm)
        assert seed_for(arm) == ARMS[arm][5]
        assert profile_for(arm) in PROFILE_OPTIONS
        if arm.startswith('H_'):
            assert ARMS[arm][0] == 'hyperbolic' and selection_for(arm) == 'val'
        if arm.startswith('E_'):
            assert ARMS[arm][0] == 'matched' and selection_for(arm) == 'val'
        if arm == 'TAR_DIFF':
            assert ARMS[arm][3] == 'diff' and selection_for(arm) == 'test'
        if arm == 'DCR_SINGLE':
            assert ARMS[arm][4] == 'none' and selection_for(arm) == 'test'
    # Paired H/E arms must use identical seeds so the initialisation is shared.
    for seed in (2333, 2334, 2335):
        assert seed_for(f'H_S{seed}') == seed_for(f'E_S{seed}') == seed
        assert ARMS[f'H_S{seed}'][1:5] == ARMS[f'E_S{seed}'][1:5]
    for dataset in LADDER_DATASETS:
        assert dataset in MATRIX['TAR_DIFF'] and dataset in MATRIX['DCR_SINGLE']
        assert set(LADDER[dataset]) == {'control', 'tar_plain', 'dcr_plain', 'noise_pp'}
    with TemporaryDirectory() as directory:
        complete, stopped, unstarted = experiments()[:3]
        path = Path(directory)/complete
        path.mkdir(parents=True)
        (path/'train_log.txt').write_text('=== TEST RESULTS ===\nF1=0.9\n=== END TEST RESULTS ===\n')
        with patch('pipeline.ROOT', Path(directory)):
            pending, finished = pending_runs({'stopped': [stopped]})
            assert finished == [complete] and pending[0] == stopped and unstarted in pending
    with TemporaryDirectory() as directory:
        with patch('pipeline.ROOT', Path(directory)):
            assert pending_runs({})[1] == []
    for dataset in DATASETS:
        directory = Path('/share_datasets/CD')/dataset/'list'
        if not directory.is_dir():
            continue
        train, val, test = [(directory/f'{split}.txt').read_text().split() for split in ('train', 'val', 'test')]
        assert all(len(x) == len(set(x)) for x in (train, val, test))
        assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    check_cli()

    image = torch.randn(1, 6, 256, 256)
    control = reference_control(image)
    for profile, (module, heads, detail, tar, decoder) in PROFILE_OPTIONS.items():
        torch.manual_seed(2333)
        model = CASANet(module=module, context_heads=heads, detail_head=detail,
                        tar_variant=tar, decoder_variant=decoder).eval()
        assert sum(p.numel() for p in model.parameters()) == TRAINABLE_PARAMS[profile], profile
        folded = CASANet(module=module, context_heads=heads, detail_head=detail,
                         tar_variant=tar, decoder_variant=decoder).eval()
        folded.load_state_dict(model.state_dict())
        folded.switch_to_deploy()
        assert sum(p.numel() for p in folded.parameters()) == DEPLOY_PARAMS[profile], profile
        with torch.no_grad():
            outputs = model(image[:, :3], image[:, 3:])
            deployed = folded.predict(image[:, :3], image[:, 3:])
        assert len(outputs) == 4 and outputs[0].shape == (1, 1, 256, 256)
        assert torch.allclose(outputs[0], deployed, atol=2e-4)
        assert all(torch.isfinite(out).all() for out in outputs)
        logits, aux = model(image[:, :3], image[:, 3:], return_aux=True)
        logits[0].mean().backward()
        gradients = {name: param.grad for name, param in model.named_parameters()}
        required = {'HYPERBOLIC': ['context.qk.weight', 'context.out.weight'],
                    'EUCLIDEAN': ['context.qk.weight', 'context.out.weight'],
                    'TAR_DIFF': ['tar.stage1.temporal.proj.weight', 'tar.stage2.temporal.proj.weight'],
                    'DCR_SINGLE': ['decoder.proj.weight', 'decoder.block.dw.weight']}[profile]
        for name in required:
            assert gradients[name] is not None and gradients[name].norm() > 0, (profile, name)
        # Removing a module must change behaviour, never silently fall through.
        if profile != 'CONTROL':
            assert not torch.allclose(outputs[0], control, atol=1e-6), profile
    # Paired initialisation: identical states for H and E at the same seed.
    torch.manual_seed(2333)
    hyperbolic = CASANet(module='hyperbolic', context_seed=2333)
    torch.manual_seed(2333)
    euclidean = CASANet(module='matched', context_seed=2333)
    for (name, left), (_, right) in zip(hyperbolic.state_dict().items(), euclidean.state_dict().items()):
        assert torch.equal(left, right), name
    assert reservation_keys()['H_S2333'] == 'HYPERBOLIC' and reservation_keys()['E_S2333'] == 'EUCLIDEAN'
    print('Run14 matrix, paired initialisation, folding and gradient paths PASS', flush=True)


def reference_control(image):
    torch.manual_seed(2333)
    control = CASANet(module='none', tar_variant='reparam', decoder_variant='reparam').eval()
    with torch.no_grad():
        return control(image[:, :3], image[:, 3:])[0]


if __name__ == '__main__':
    main()
