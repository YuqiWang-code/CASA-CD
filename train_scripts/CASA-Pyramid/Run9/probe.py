"""Six two-epoch train/val probes with the exact formal loss and beta settings."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from config import ROOT, PROTOCOL, VARIANTS, PROBES, COMPARISON


def passed(path, arm):
    report = json.loads(path.read_text())
    spec = VARIANTS[arm]
    return (report['health_passed'] and report['task_gradient_check']
            and report['protocol'] == PROTOCOL and report['module'] == spec['module']
            and report['refinement_beta'] == spec['beta'] and report['retrieval_weight'] == 0
            and report['epochs'] == 2)


def main():
    validation = json.loads((ROOT / 'validation_result.json').read_text())
    if not validation['passed'] or validation['protocol'] != PROTOCOL:
        raise RuntimeError('complete model validation and profiling before probes')
    profiles = json.loads((ROOT / 'profile.json').read_text())
    reservations = {arm: math.ceil(item['peak_reserved_mib'] + item['context_overhead_mib'])
                    for arm, item in profiles.items()}
    budget = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free',
                                          '--format=csv,noheader,nounits'], text=True).strip()) - 1024
    pending = list(PROBES)
    active = []
    reports = {}
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'probe runner interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while pending or active:
            for job in active[:]:
                code = job['process'].poll()
                if code is None:
                    continue
                job['stream'].close()
                active.remove(job)
                result = job['directory'] / 'health_result.json'
                if code or not result.exists() or not passed(result, job['arm']):
                    raise RuntimeError(f"health probe failed: {job['arm']}/{job['dataset']}")
                reports[f"{job['arm']}/{job['dataset']}"] = json.loads(result.read_text())
            while pending and len(active) < 2:
                arm, dataset = pending[0]
                directory = ROOT / 'probes' / arm / dataset
                result = directory / 'health_result.json'
                if result.exists():
                    if not passed(result, arm):
                        raise RuntimeError(f'existing health probe has not passed: {arm}/{dataset}')
                    reports[f'{arm}/{dataset}'] = json.loads(result.read_text())
                    pending.pop(0)
                    continue
                used = sum(reservations[job['arm']] for job in active)
                if used + reservations[arm] > budget:
                    if not active:
                        raise RuntimeError(f'health probe exceeds GPU1 memory budget: {arm}')
                    break
                pending.pop(0)
                directory.mkdir(parents=True, exist_ok=True)
                stream = (directory / 'probe.log').open('a')
                spec = VARIANTS[arm]
                command = [sys.executable, str(here / 'health_probe.py'), '--models-dir',
                           str(repo / 'models'), '--module', spec['module'], '--dataset', dataset,
                           '--epochs', '2', '--retrieval-weight', '0', '--refinement-beta',
                           str(spec['beta']), '--output-dir', str(directory)]
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                active.append(dict(process=process,stream=stream,arm=arm,dataset=dataset,directory=directory))
                print(f'[PROBE-START] {arm}/{dataset} pid={process.pid}', flush=True)
            time.sleep(5)
    finally:
        for job in active:
            if job['process'].poll() is None:
                os.killpg(job['process'].pid, signal.SIGTERM)
            job['stream'].close()
    (ROOT / 'readiness.json').write_text(json.dumps(dict(
        protocol=PROTOCOL, phase='four_dataset_refinement', comparison=COMPARISON,
        passed=True, validation=validation,
        criterion='finite losses and nonzero segmentation gradients; not an accuracy claim',
        probes=reports), indent=2))
    print('[ALL-PROBES-PASS]', flush=True)


if __name__ == '__main__':
    main()
