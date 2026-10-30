"""Run CASA ablations on GPU1 using measured per-variant memory."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8')
HERE = Path(__file__).resolve().parent
DATASETS = ['SYSU-CD-256', 'WHU-CD-256']
VARIANTS = {'H_CLASS': 'hyperbolic', 'H_NONE': 'hyperbolic'}
PROTOCOL = 'casa_pyramid_40k_attribution_v8'
active = {}
state = dict(status='running', protocol=PROTOCOL, completed=[], active={}, active_gpus={}, failed=[], variants=list(VARIANTS))


def pending_runs(completed, running):
    occupied = set(completed) | set(running)
    return [f'{variant}/{dataset}' for dataset in DATASETS for variant in VARIANTS
            if f'{variant}/{dataset}' not in occupied]


def choose_run(pending, jobs, limits, requirements):
    used = {gpu: sum(job['reserved_mib'] for job in jobs.values() if job['gpu'] == gpu)
            for gpu in limits}
    for name in pending:
        required = requirements[name.split('/')[0]]
        candidates = [gpu for gpu in limits if used[gpu] + required <= limits[gpu]]
        if candidates:
            return name, min(candidates, key=lambda gpu: (used[gpu], gpu)), required
    return None


def alive(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.exists() and b'train_CASA.py' in path.read_bytes()


def complete(name):
    path = ROOT / name / 'train_log.txt'
    return path.exists() and '=== END TEST RESULTS ===' in path.read_text(errors='replace')


def check_readiness(root=ROOT):
    decision = json.loads((root / 'readiness.json').read_text())
    if (decision.get('passed') is not True or decision.get('phase') != 'retrieval_loss'
            or decision.get('protocol') != PROTOCOL):
        raise RuntimeError('Run8 requires a passed phase-one readiness record')
    profile = json.loads((root / 'profile.json').read_text())
    for mode in VARIANTS:
        if (profile[mode]['protocol'] != PROTOCOL or profile[mode]['peak_reserved_mib'] <= 0
                or profile[mode]['context_overhead_mib'] < 0):
            raise RuntimeError(f'invalid final-model memory profile: {mode}')
    return decision, profile


def save():
    state['active'] = {str(pid): job['name'] for pid, job in active.items()}
    state['active_gpus'] = {str(pid): job['gpu'] for pid, job in active.items()}
    state['active_reserved_mib'] = {str(pid): job['reserved_mib'] for pid, job in active.items()}
    state['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (ROOT / 'pipeline_status.json').write_text(json.dumps(state, indent=2))


def stop_children():
    for pid in active:
        if alive(pid):
            os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for pid, job in active.items():
        process = job['process']
        if process is not None:
            try:
                process.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
        else:
            while alive(pid) and time.monotonic() < deadline:
                time.sleep(.2)
            if alive(pid):
                os.killpg(pid, signal.SIGKILL)


def interrupted(signum, frame):
    state['status'] = 'stopped'
    state['stopped'] = [job['name'] for job in active.values()]
    stop_children()
    active.clear()
    save()
    raise SystemExit(128 + signum)


def main():
    decision, _ = check_readiness()
    config = json.loads((ROOT / 'scheduling.json').read_text())
    limits = {int(gpu): value for gpu, value in config['gpu_budget_mib'].items()}
    if set(limits) != {1}:
        raise RuntimeError('Run8 is authorized for physical GPU1 only')
    requirements = config['variant_reserved_mib']
    if config['protocol'] != PROTOCOL or config['comparison'] != decision['comparison']:
        raise RuntimeError('scheduling must match the selected Run8 model')
    state['comparison'] = decision['comparison']
    state['gpu_budget_mib'] = limits
    state['variant_reserved_mib'] = requirements
    state['dataset_priority'] = DATASETS
    state['completed'] = [name for name in pending_runs([], []) if complete(name)]
    prior = json.loads((ROOT / 'pipeline_status.json').read_text()) if (ROOT / 'pipeline_status.json').exists() else {}
    for pid_text, name in prior.get('active', {}).items():
        if name in state['completed']:
            continue
        pid = int(pid_text)
        if not alive(pid):
            raise RuntimeError(f'prior training process is absent: {name} ({pid})')
        active[pid] = dict(name=name, gpu=prior['active_gpus'][pid_text],
                           reserved_mib=requirements[name.split('/')[0]], process=None)
    pending = pending_runs(state['completed'], [job['name'] for job in active.values()])
    (ROOT / 'pipeline.pid').write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        while pending or active:
            for pid, job in list(active.items()):
                process = job['process']
                code = process.poll() if process is not None else (None if alive(pid) else (0 if complete(job['name']) else -1))
                if code is None:
                    continue
                name = active.pop(pid)['name']
                if code != 0 or not complete(name):
                    state['failed'].append(dict(run=name, gpu=job['gpu'], exit_code=code))
                    raise RuntimeError(f'{name} exited {code} without complete TEST results')
                state['completed'].append(name)
                print(f'[DONE] {name} gpu={job["gpu"]}', flush=True)
                save()
            while pending:
                selected = choose_run(pending, active, limits, requirements)
                if selected is None:
                    if not active:
                        raise RuntimeError('pending experiment cannot fit measured GPU budgets')
                    break
                name, gpu, reserved = selected
                pending.remove(name)
                variant, dataset = name.split('/')
                print(f'[START] {name} gpu={gpu} reserved={reserved}MiB {time.strftime("%F %T")}', flush=True)
                process = subprocess.Popen(['bash', str(HERE / 'run_one.sh'), variant, dataset, str(gpu)],
                                           start_new_session=True)
                active[process.pid] = dict(name=name, gpu=gpu, reserved_mib=reserved, process=process)
                directory = ROOT / name
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'placement.json').write_text(json.dumps(dict(
                    pid=process.pid, physical_gpu=gpu, logical_gpu=0,
                    reserved_mib=reserved, started=time.strftime('%F %T')), indent=2))
                save()
                time.sleep(2)
            time.sleep(5)
        state['status'] = 'complete'
        save()
        print('[COMPLETE] all 4 phase-one experiments', flush=True)
    except Exception:
        state['status'] = 'failed'
        state['stopped'] = [job['name'] for job in active.values()]
        stop_children()
        active.clear()
        save()
        raise


if __name__ == '__main__':
    main()
