"""Run CASA ablations on GPU1 using measured per-variant memory."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from config import ROOT, PROTOCOL, DATASETS, VARIANTS, PROBES, experiments

HERE = Path(__file__).resolve().parent
active = {}
state = dict(status='running', protocol=PROTOCOL, completed=[], active={}, active_gpus={}, failed=[], variants=list(VARIANTS))


def pending_runs(completed, running):
    occupied = set(completed) | set(running)
    return [name for name in experiments() if name not in occupied]


def choose_run(pending, jobs, limits, requirements):
    used = {gpu: sum(job['reserved_mib'] for job in jobs.values() if job['gpu'] == gpu)
            for gpu in limits}
    free = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())
    for name in pending:
        required = requirements[name.split('/')[0]]
        candidates = [gpu for gpu in limits if used[gpu] + required <= limits[gpu]]
        if candidates and required <= free - 1024:
            return name, min(candidates, key=lambda gpu: (used[gpu], gpu)), required
    return None


def alive(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.exists() and b'train_CASA.py' in path.read_bytes()


def complete(name):
    path = ROOT / name / 'train_log.txt'
    return path.exists() and '=== END TEST RESULTS ===' in path.read_text(errors='replace')


def check_readiness(root=ROOT):
    report = json.loads((root / 'readiness.json').read_text())
    if report['protocol'] != PROTOCOL or not report['passed']:
        raise RuntimeError('Run11 requires completed validation and probes')
    for variant, dataset in PROBES:
        item = json.loads((root / 'probes' / variant / dataset / 'health_result.json').read_text())
        if not item['passed'] or item['protocol'] != PROTOCOL or item['epochs'] != 2:
            raise RuntimeError(f'invalid short training: {variant}/{dataset}')
    return report


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
    pidfile = ROOT / 'pipeline.pid'
    if pidfile.exists():
        old = Path('/proc') / pidfile.read_text().strip() / 'cmdline'
        if old.exists() and (str(HERE / 'pipeline.py').encode() in old.read_bytes()
                             or (b'pipeline.py' in old.read_bytes() and old.with_name('cwd').resolve() == HERE)):
            raise RuntimeError('Run11 supervisor already active')
    check_readiness()
    config = json.loads((ROOT / 'scheduling.json').read_text())
    limits = {int(gpu): value for gpu, value in config['gpu_budget_mib'].items()}
    if set(limits) != {1}:
        raise RuntimeError('Run11 is authorized for physical GPU1 only')
    requirements = config['variant_reserved_mib']
    if config['protocol'] != PROTOCOL:
        raise RuntimeError('schedule protocol mismatch')
    state['gpu_budget_mib'] = limits
    state['variant_reserved_mib'] = requirements
    state['dataset_priority'] = DATASETS
    state['experiments'] = experiments()
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
        print('[COMPLETE] all 24 Run11 experiments', flush=True)
    except Exception:
        state['status'] = 'failed'
        state['stopped'] = [job['name'] for job in active.values()]
        stop_children()
        active.clear()
        save()
        raise


if __name__ == '__main__':
    main()
