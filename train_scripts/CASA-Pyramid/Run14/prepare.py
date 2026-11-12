"""Validate models, benchmark, profile, and run healthy real-data probes for Run14."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from config import (DATA, DEPLOY_PARAMS, GPUS, HEADROOM, LADDER, PRETRAINED, PROFILE_ARMS, PROFILE_OPTIONS,
                    PROTOCOL, ROOT, RUNTIME, TRAINABLE_PARAMS, model_options, selection_for)
from pipeline import fit, free_memory, stop, write_json

HERE = Path(__file__).resolve().parent


def command(*args, env=None):
    subprocess.run([sys.executable, '-u', *map(str, args)], check=True, env=env)


def profile():
    reports = {}
    directory = ROOT / 'profiles'
    directory.mkdir(exist_ok=True)
    for name in PROFILE_ARMS:
        module, heads, detail, tar, decoder = PROFILE_OPTIONS[name]
        file = directory / f'{name}.json'
        options = ['--context-heads', str(heads), '--tar-variant', tar, '--decoder-variant', decoder]
        command(RUNTIME/'models/profile_CASA.py', '--dataset-root', DATA/'WHU-CD-256',
                '--pretrained', PRETRAINED, '--output', file, '--modules', module, *options)
        reports[name] = json.loads(file.read_text())[module]
    write_json(ROOT/'profile.json', reports)
    return {name: math.ceil(report['process_memory_mib'] + 256) for name, report in reports.items()}


def probe(budget, requirements):
    pending = list(PROFILE_ARMS)
    active, reports = {}, {}
    try:
        while pending or active:
            for pid, (name, process, _) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[pid]
                if code:
                    raise RuntimeError(f'probe failed: {name}; see probes/{name}/probe.log')
                report = json.loads((ROOT/'probes'/name/'health_result.json').read_text())
                if not report['passed'] or report['protocol'] != PROTOCOL or report['epochs'] != 2:
                    raise RuntimeError(f'invalid probe report: {name}')
                reports[name] = report
                print('[PROBE-DONE]', name, flush=True)
            profile = fit(pending, active, budget, requirements, GPUS[0])
            if profile:
                pending.remove(profile)
                out = ROOT/'probes'/profile
                out.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPUS[0])}
                with (out/'probe.log').open('w') as stream:
                    child = subprocess.Popen([sys.executable, '-u', str(HERE/'health_probe.py'), profile,
                                              'WHU-CD-256'],
                                             stdout=stream, stderr=subprocess.STDOUT, env=env,
                                             start_new_session=True)
                active[child.pid] = (profile, child, requirements[profile])
                print('[PROBE-START]', profile, child.pid, flush=True)
            time.sleep(3)
    finally:
        stop([job[1] for job in active.values()])
    return reports


def end_to_end():
    # 36 train-list pairs in a temporary split: one step of both selection protocols.
    reports = {}
    for arm in ('H_S2333', 'DCR_SINGLE'):
        target = ROOT/'smoke'/arm
        if target.exists():
            # A previous preflight may have been interrupted after the smoke step; these
            # artifacts are disposable, so rebuild them instead of refusing to continue.
            shutil.rmtree(target)
        data = target/'data'
        (data/'list').mkdir(parents=True, exist_ok=True)
        source = DATA/'WHU-CD-256'
        images = (source/'list/train.txt').read_text().splitlines()[:36]
        for name in ('A', 'B', 'label'):
            link = data/name
            if not link.exists():
                link.symlink_to(source/name, target_is_directory=True)
        for split, names in (('train', images[:32]), ('val', images[32:34]), ('test', images[34:])):
            (data/f'list/{split}.txt').write_text('\n'.join(names)+'\n')
        key = 'HYPERBOLIC' if arm.startswith('H_') else 'DCR_SINGLE'
        module, heads, detail, tar, decoder = PROFILE_OPTIONS[key]
        args = (RUNTIME/'models/train_CASA.py', '--dataset-root', data, '--output-dir', target/'logs',
                '--checkpoint-dir', target/'weights', '--pretrained', PRETRAINED, '--module', module,
                *model_options(arm), '--max-steps', 1, '--num-workers', '0')
        command(*args)
        result = json.loads((target/'logs/result.json').read_text())
        manifest = json.loads((target/'logs/run_manifest.json').read_text())
        point = json.loads((target/'logs/operating_point.json').read_text())
        selection = selection_for(arm)
        assert result['optimizer_steps'] == 1 and result['selection_split'] == selection
        assert result['independent_test'] == (selection == 'val')
        assert manifest['tar_variant'] == tar and manifest['decoder_variant'] == decoder
        assert manifest['selection_split'] == selection
        assert abs(result['total_params_m'] - TRAINABLE_PARAMS[key] / 1e6) < 1e-6
        assert abs(result['deploy_params_m'] - DEPLOY_PARAMS[key] / 1e6) < 1e-6
        assert point['selection_split'] == selection and point['calibration_split'] == 'val'
        assert len(point['validation']['curve']) == 121
        assert point['test_calibrated']['threshold'] == point['validation']['best']['threshold']
        # One step from initialisation leaves logits at about zero, so a float-level fold
        # difference can flip a stray pixel across the 0.5 boundary; bound the rate.
        assert result['deploy_max_diff'] <= 2e-4
        pixels = 256 * 256
        assert result['deploy_binary_disagreement'] <= pixels * 1e-3, result['deploy_binary_disagreement']
        reports[arm] = dict(optimizer_steps=1, selection_split=selection, calibration_split='val',
                            tar_variant=manifest['tar_variant'], decoder_variant=manifest['decoder_variant'],
                            total_params_m=result['total_params_m'],
                            deploy_params_m=result['deploy_params_m'], fold=result['deploy_max_diff'])
        write_json(target/'readiness.json', reports[arm])
    return reports


def main():
    if (ROOT/'readiness.json').exists() or (ROOT/'pipeline_status.json').exists():
        raise RuntimeError('Run14 preflight or formal queue already exists')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    command(HERE/'validate.py', env=env)
    for test in ('test_ablation_variants.py', 'test_direct_geometry.py', 'test_centered_readout.py',
                 'test_hyperbolic_context.py', 'test_context_geometry.py', 'test_CASA_resume.py',
                 'test_data_pipeline.py'):
        command(RUNTIME/'models'/test, env=env)
    command(HERE/'benchmark.py')
    reservations = profile()
    budgets = {gpu: free_memory(gpu) - HEADROOM for gpu in GPUS}
    if max(reservations.values()) > min(budgets.values()):
        raise RuntimeError('a measured Run14 job does not fit on one GPU')
    write_json(ROOT/'scheduling.json', dict(protocol=PROTOCOL,
               gpu_budget_mib={str(gpu): budget for gpu, budget in budgets.items()},
               variant_reserved_mib=reservations, slots_per_gpu=2, free_headroom_mib=HEADROOM))
    smoke = end_to_end()
    probes = probe(budgets[GPUS[0]], reservations)
    write_json(ROOT/'readiness.json', dict(
        protocol=PROTOCOL, passed=True, arms=list(PROFILE_OPTIONS),
        trainable_params=TRAINABLE_PARAMS, deployment_params=DEPLOY_PARAMS,
        selection={'pairing_arms': 'val', 'module_removal_arms': 'test'},
        baselines=LADDER, end_to_end=smoke, probes=probes, calibration_split='val',
        test_primary_threshold=.5))
    print('[READY] Run14 CPU audit and GPU preflight passed', flush=True)


if __name__ == '__main__':
    main()
