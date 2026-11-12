"""Schedule the Run13 ablation matrix across both physical GPUs."""
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from config import DATASETS, GPUS, HEADROOM, PROTOCOL, ROOT, SLOTS, VARIANTS, experiments

HERE = Path(__file__).resolve().parent


def free_memory(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())


def fit(pending, active, budget, reservations, gpu):
    if not pending or len(active) >= SLOTS:
        return None
    available = min(budget - sum(job[2] for job in active.values()), free_memory(gpu) - HEADROOM)
    return next((name for name in pending if reservations[name.split('/')[0]] <= available), None)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def stop(active):
    for process in active:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in active:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def complete(name):
    path = ROOT/name/'train_log.txt'
    return path.is_file() and bool(re.search(r'^=== TEST RESULTS ===\n.*?\n=== END TEST RESULTS ===$',
                                             path.read_text(encoding='utf-8'), re.M | re.S))


def pending_runs(previous):
    completed = [name for name in experiments() if complete(name)]
    pending = [name for name in experiments() if name not in completed]
    priority = set(previous.get('stopped', [])) | {job['run'] for job in previous.get('failed', [])}
    pending.sort(key=lambda name: name not in priority)
    return pending, completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='continue existing Run13 artifacts')
    args = parser.parse_args()
    status_file = ROOT / 'pipeline_status.json'
    with (ROOT / 'pipeline.pid').open('a+') as pidfile:
        fcntl.flock(pidfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(status_file.read_text()) if status_file.exists() else {}
        if previous and not args.resume:
            raise RuntimeError('existing Run13 queue; use --resume to continue it')
        prior_children = dict(previous.get('active', {}))
        for name in experiments():
            placement = ROOT/name/'placement.json'
            if placement.is_file():
                payload = json.loads(placement.read_text())
                prior_children[str(payload['pid'])] = name
        for pid, name in prior_children.items():
            try:
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:
                continue
            if any(str(path).encode() in cmd for path in
                   (HERE/'run_one.py', ROOT/'runtime/models/train_CASA.py')):
                raise RuntimeError(f'existing Run13 child: {name} PID {pid}')
        readiness = json.loads((ROOT/'readiness.json').read_text())
        if readiness['protocol'] != PROTOCOL or not readiness['passed']:
            raise RuntimeError('Run13 requires passing server preflight')
        schedule = json.loads((ROOT/'scheduling.json').read_text())
        budgets = {gpu: schedule['gpu_budget_mib'][str(gpu)] for gpu in GPUS}
        reservations = schedule['variant_reserved_mib']
        pidfile.seek(0); pidfile.truncate(); pidfile.write(str(os.getpid())); pidfile.flush()
        pending, completed = pending_runs(previous)
        per_gpu = {gpu: {} for gpu in GPUS}
        failed, stopped = [], []

        def save(status):
            write_json(status_file, dict(status=status, protocol=PROTOCOL, completed=completed,
                       active={str(pid): job[0] for gpu in GPUS for pid, job in per_gpu[gpu].items()},
                       active_gpus={str(pid): gpu for gpu in GPUS for pid in per_gpu[gpu]},
                       failed=failed, stopped=stopped, variants=list(VARIANTS), experiments=experiments(),
                       dataset_priority=DATASETS, gpu_budget_mib=schedule['gpu_budget_mib'],
                       variant_reserved_mib=reservations, slots_per_gpu=SLOTS,
                       updated=time.strftime('%F %T')))

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        save('running')
        try:
            while pending or any(per_gpu.values()):
                for gpu in GPUS:
                    for pid, (name, process, _, _) in list(per_gpu[gpu].items()):
                        code = process.poll()
                        if code is None:
                            continue
                        del per_gpu[gpu][pid]
                        if not complete(name):
                            failed.append(dict(run=name, gpu=gpu, exit_code=code))
                            print(f'[FAILED] {name} gpu={gpu} exit={code}; continuing other arms', flush=True)
                        else:
                            completed.append(name)
                            print(f'[DONE] {name} gpu={gpu}', flush=True)
                        save('running')
                for gpu in GPUS:
                    name = fit(pending, per_gpu[gpu], budgets[gpu], reservations, gpu)
                    if name is None:
                        continue
                    pending.remove(name)
                    variant, dataset = name.split('/')
                    command = [sys.executable, str(HERE/'run_one.py'), variant, dataset, '--gpu', str(gpu)]
                    if args.resume:
                        command.append('--resume')
                    process = subprocess.Popen(command, start_new_session=True)
                    reserved = reservations[variant]
                    per_gpu[gpu][process.pid] = (name, process, reserved, gpu)
                    folder = ROOT / name
                    folder.mkdir(parents=True, exist_ok=True)
                    write_json(folder/'placement.json', dict(pid=process.pid, physical_gpu=gpu, logical_gpu=0,
                               reserved_mib=reserved, started=time.strftime('%F %T')))
                    print(f'[START] {name} gpu={gpu} reserved={reserved} MiB', flush=True)
                    save('running')
                time.sleep(3)
        except BaseException as error:
            stopped.extend(job[0] for gpu in GPUS for job in per_gpu[gpu].values())
            stop([job[1] for gpu in GPUS for job in per_gpu[gpu].values()])
            for gpu in GPUS:
                per_gpu[gpu].clear()
            save('stopped' if isinstance(error, KeyboardInterrupt) else 'failed')
            raise
        save('failed' if failed else 'complete')
        print(f'[QUEUE-END] {len(completed)} complete, {len(failed)} failed', flush=True)


if __name__ == '__main__':
    main()
