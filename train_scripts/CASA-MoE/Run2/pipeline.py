"""Capacity-based, GPU1-only scheduling for 6 matched experiments."""
import fcntl
import json
import os
from pathlib import Path
import signal
import re
import math
import subprocess
import time

from config import DATASETS, PROTOCOL, ROOT, VARIANTS, experiments, runtime_root

active = {}
state = dict(status='running', protocol=PROTOCOL, completed=[], active={}, active_gpus={}, failed=[])


def pending_runs(completed, running):
    occupied = set(completed) | set(running)
    return [name for name in experiments() if name not in occupied]


def choose_run(pending, jobs, limits, requirements, free_mib=None):
    used = sum(job['reserved_mib'] for job in jobs.values())
    room = limits[1] - used
    if free_mib is not None:
        room = min(room, free_mib-1024)
    for name in pending:
        required = requirements[name.split('/')[0]]
        if required <= room:
            return name, 1, required
    return None


def complete(name):
    path = ROOT / name / 'train_log.txt'
    return path.exists() and '=== END TEST RESULTS ===' in path.read_text(errors='replace')


def save():
    state['active'] = {str(pid): job['name'] for pid, job in active.items()}
    state['active_gpus'] = {str(pid): 1 for pid in active}
    state['active_reserved_mib'] = {str(pid): job['reserved_mib'] for pid, job in active.items()}
    state['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (ROOT / 'pipeline_status.json').write_text(json.dumps(state, indent=2))


def stop_children():
    for job in active.values():
        if job['process'].poll() is None:
            os.killpg(job['process'].pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for job in active.values():
        try:
            job['process'].wait(timeout=max(.1, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(job['process'].pid, signal.SIGKILL)


def interrupted(signum, frame):
    state['status'] = 'stopped'
    state['stopped'] = [job['name'] for job in active.values()]
    stop_children()
    active.clear()
    save()
    raise SystemExit(128+signum)


def record_initial_health():
    path=ROOT/'initial_health.json'
    reports=json.loads(path.read_text()) if path.exists() else {}
    for job in active.values():
        name=job['name']
        if name in reports:
            continue
        log=ROOT/name/'train_log.txt'
        if not log.exists():
            continue
        text=log.read_text(errors='replace')
        lines=re.findall(r'^\[TRAIN-EPOCH\].*',text,re.M)[:3]
        evaluations=re.findall(r'^Epoch .*',text,re.M)[:3]
        if len(lines)==3 and len(evaluations)==3:
            losses=[float(re.search(r'main=([-+0-9.eE]+)',line)[1]) for line in lines]
            if not all(math.isfinite(value) for value in losses):
                raise FloatingPointError(f'{name}: non-finite initial loss')
            reports[name]=dict(train=lines,evaluation=evaluations,main_losses=losses,
                task_gradients=re.findall(r'^\[TASK-GRAD\].*',text,re.M)[:3],
                diagnostics=re.findall(r'^\[MODULE-DIAG\].*',text,re.M)[:3])
            path.write_text(json.dumps(reports,indent=2))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    previous_queue=json.loads(Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8/pipeline_status.json').read_text())
    if previous_queue['status'] != 'complete' or previous_queue.get('active'):
        raise RuntimeError('Run8 must finish before starting MoE Run2')
    lock = (ROOT / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configuration = json.loads((ROOT / 'scheduling.json').read_text())
    limits = {int(gpu): value for gpu, value in configuration['gpu_budget_mib'].items()}
    if set(limits) != {1} or configuration['protocol'] != PROTOCOL:
        raise RuntimeError('CASA-MoE Run2 requires physical GPU1 and its recorded protocol')
    requirements = configuration['variant_reserved_mib']
    runtime = runtime_root()
    if Path(__file__).resolve() != runtime / 'train_scripts/CASA-MoE/Run2/pipeline.py':
        raise RuntimeError('launch the pipeline from its frozen runtime source')
    validation = json.loads((ROOT / 'validation_result.json').read_text())
    if not validation['passed'] or validation['runtime'] != str(runtime):
        raise RuntimeError('this runtime has not passed structural validation and profiling')
    for dataset in DATASETS:
        for mode in VARIANTS.values():
            health=json.loads((ROOT/'probes'/f"{dataset.split('-')[0].lower()}_{mode}"/'health_result.json').read_text())
            batches=222 if dataset.startswith('LEVIR') else 375
            if (not health['passed'] or health['protocol'] != PROTOCOL or health['epochs'] != 3
                    or health['steps'] != batches*3 or health['dataset'] != dataset
                    or health['module'] != mode or health['runtime'] != str(runtime)):
                raise RuntimeError(f'incomplete real-data probe: {dataset}/{mode}')
    previous = json.loads((ROOT / 'pipeline_status.json').read_text()) if (ROOT / 'pipeline_status.json').exists() else {}
    if previous.get('failed') or previous.get('stopped') or previous.get('active'):
        raise RuntimeError('existing incomplete queue requires explicit recovery; retained all artifacts')
    state.update(gpu_budget_mib=limits, variant_reserved_mib=requirements,
                 dataset_priority=DATASETS, experiments=experiments(), runtime=str(runtime),
                 completed=[name for name in experiments() if complete(name)])
    pending = pending_runs(state['completed'], [])
    (ROOT / 'pipeline.pid').write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        while pending or active:
            record_initial_health()
            for pid, job in list(active.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                active.pop(pid)
                if code != 0 or not complete(job['name']):
                    state['failed'].append(dict(run=job['name'], gpu=1, exit_code=code))
                    raise RuntimeError(f"{job['name']} exited {code} without complete TEST results")
                state['completed'].append(job['name'])
                print(f"[DONE] {job['name']} gpu=1", flush=True)
                save()
            while pending:
                free = int(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free',
                                                    '--format=csv,noheader,nounits'], text=True).strip())
                selected = choose_run(pending, active, limits, requirements, free)
                if selected is None:
                    state['waiting_for_gpu_memory'] = dict(free_mib=free, pending=len(pending))
                    save()
                    break
                name, gpu, reserved = selected
                pending.remove(name)
                variant, dataset = name.split('/')
                directory = ROOT / name
                directory.mkdir(parents=True, exist_ok=True)
                process = subprocess.Popen(['bash', str(Path(__file__).parent / 'run_one.sh'),
                                            variant, dataset, str(gpu)], start_new_session=True)
                active[process.pid] = dict(name=name, reserved_mib=reserved, process=process)
                state.pop('waiting_for_gpu_memory', None)
                (directory / 'placement.json').write_text(json.dumps(dict(
                    pid=process.pid, physical_gpu=1, logical_gpu=0, reserved_mib=reserved,
                    runtime=str(runtime), started=time.strftime('%F %T')), indent=2))
                print(f'[START] {name} gpu=1 reserved={reserved}MiB', flush=True)
                save()
                time.sleep(3)
            time.sleep(5)
        state['status'] = 'complete'
        save()
        print('[COMPLETE] all 6 experiments', flush=True)
    except Exception:
        state['status'] = 'failed'
        state['stopped'] = [job['name'] for job in active.values()]
        stop_children()
        active.clear()
        save()
        raise


if __name__ == '__main__':
    main()
