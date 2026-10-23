"""Run CASA ablations on both GPUs using measured per-variant memory."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/yqwang/outputs/CASA-CD/CASA/Run6')
HERE = Path(__file__).resolve().parent
DATASETS = ['LEVIR-CD-256', 'SYSU-CD-256', 'WHU-CD-256', 'CDD-CD-256']
VARIANTS = {'M1_HYPER_QKV': 'hyperbolic', 'M2_CA_QKV': 'qkv', 'M0_CTRL': 'none'}
PROTOCOL = 'casa_40k_changevalues_v6'
active = {}
state = dict(status='running', protocol=PROTOCOL, completed=[], active={}, active_gpus={}, failed=[])


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
    config = json.loads((ROOT / 'scheduling.json').read_text())
    limits = {int(gpu): value for gpu, value in config['gpu_budget_mib'].items()}
    requirements = config['variant_reserved_mib']
    for dataset, epochs, steps in (('LEVIR-CD-256', 3, 666), ('SYSU-CD-256', 4, 1500)):
        for mode in ('hyperbolic', 'qkv'):
            name = f"{dataset.split('-')[0].lower()}_{mode}"
            health = json.loads((ROOT / 'probes' / name / 'health_result.json').read_text())
            if (not health['passed'] or health['protocol'] != PROTOCOL or health['steps'] != steps
                    or health['epochs'] != epochs or health['dataset'] != dataset or health['module'] != mode):
                raise RuntimeError(f'incomplete health probe: {name}')
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
        print('[COMPLETE] all 12 experiments', flush=True)
    except Exception:
        state['status'] = 'failed'
        state['stopped'] = [job['name'] for job in active.values()]
        stop_children()
        active.clear()
        save()
        raise


if __name__ == '__main__':
    main()
