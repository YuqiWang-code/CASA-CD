"""Schedule the Run15 version matrix across the authorized GPUs."""
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

from config import (GPUS, HEADROOM, PROTOCOL, ROOT, SLOTS, experiments, reservation_key,
                    reservation_mib)

HERE = Path(__file__).resolve().parent


def free_memory(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())


def job_key(job):
    return '/'.join(job)


def fit(pending, active, budget, reservations, gpu):
    if not pending or len(active) >= SLOTS:
        return None
    available = min(budget - sum(item['reserved'] for item in active.values()), free_memory(gpu) - HEADROOM)
    for version, arm, _ in pending:
        if reservation_mib(version, arm, reservations) <= available:
            return version, arm
    return None


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def stop(active):
    for job in active.values():
        if job['process'].poll() is None:
            os.killpg(job['process'].pid, signal.SIGTERM)
    for job in active.values():
        try:
            job['process'].wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(job['process'].pid, signal.SIGKILL)
            job['process'].wait()


def complete(job):
    path = ROOT.joinpath(*job) / 'train_log.txt'
    return path.is_file() and bool(re.search(r'^=== TEST RESULTS ===\n.*?\n=== END TEST RESULTS ===$',
                                             path.read_text(encoding='utf-8'), re.M | re.S))


def pending_runs(previous):
    completed = [job for job in experiments() if complete(job)]
    pending = [job for job in experiments() if job not in completed]
    priority = {tuple(name.split('/')) for name in previous.get('stopped', [])}
    priority |= {tuple(job['run'].split('/')) for job in previous.get('failed', [])}
    pending.sort(key=lambda job: job not in priority)
    return pending, ['/'.join(job) for job in completed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='continue existing Run15 artifacts')
    args = parser.parse_args()
    status_file = ROOT / 'pipeline_status.json'
    with (ROOT / 'pipeline.pid').open('a+') as pidfile:
        fcntl.flock(pidfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(status_file.read_text()) if status_file.exists() else {}
        if previous and not args.resume:
            raise RuntimeError('existing Run15 queue; use --resume to continue it')
        for pid in previous.get('active', {}):
            try:
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:
                continue
            if b'run_one.py' in cmd and b'Run15' in cmd:
                raise RuntimeError(f'existing Run15 child: PID {pid}')
        readiness = json.loads((ROOT / 'readiness.json').read_text())
        if readiness['protocol'] != PROTOCOL or not readiness['passed']:
            raise RuntimeError('Run15 requires passing server preflight')
        schedule = json.loads((ROOT / 'scheduling.json').read_text())
        budgets = {gpu: schedule['gpu_budget_mib'][str(gpu)] for gpu in GPUS}
        reservations = schedule['variant_reserved_mib']
        pidfile.seek(0)
        pidfile.truncate()
        pidfile.write(str(os.getpid()))
        pidfile.flush()
        pending, completed = pending_runs(previous)
        per_gpu = {gpu: {} for gpu in GPUS}
        failed, stopped = [], []

        def save(status):
            write_json(status_file, dict(
                status=status, protocol=PROTOCOL, completed=completed,
                active={str(pid): job_key(item['job']) for gpu in GPUS for pid, item in per_gpu[gpu].items()},
                active_gpus={str(pid): gpu for gpu in GPUS for pid in per_gpu[gpu]},
                failed=failed, stopped=stopped,
                experiments=[job_key(job) for job in experiments()],
                gpu_budget_mib=schedule['gpu_budget_mib'],
                variant_reserved_mib=reservations, slots_per_gpu=SLOTS,
                updated=time.strftime('%F %T')))

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        save('running')
        try:
            while pending or any(per_gpu.values()):
                for gpu in GPUS:
                    for pid, item in list(per_gpu[gpu].items()):
                        code = item['process'].poll()
                        if code is None:
                            continue
                        del per_gpu[gpu][pid]
                        name = job_key(item['job'])
                        if not complete(item['job']):
                            failed.append(dict(run=name, gpu=gpu, exit_code=code))
                            print(f'[FAILED] {name} gpu={gpu} exit={code}; continuing', flush=True)
                        else:
                            completed.append(name)
                            print(f'[DONE] {name} gpu={gpu}', flush=True)
                        save('running')
                for gpu in GPUS:
                    chosen = fit(pending, per_gpu[gpu], budgets[gpu], reservations, gpu)
                    if chosen is None:
                        continue
                    version, arm = chosen
                    job = next(item for item in pending if item[:2] == (version, arm))
                    pending.remove(job)
                    dataset = job[2]
                    command = [sys.executable, str(HERE / 'run_one.py'), version, arm, dataset, '--gpu', str(gpu)]
                    if args.resume:
                        command.append('--resume')
                    process = subprocess.Popen(command, start_new_session=True)
                    reserved = reservation_mib(version, arm, reservations)
                    per_gpu[gpu][process.pid] = dict(job=job, process=process, reserved=reserved)
                    folder = ROOT.joinpath(*job)
                    folder.mkdir(parents=True, exist_ok=True)
                    write_json(folder / 'placement.json', dict(
                        pid=process.pid, physical_gpu=gpu, logical_gpu=0,
                        reserved_mib=reserved, started=time.strftime('%F %T')))
                    print(f'[START] {job_key(job)} gpu={gpu} reserved={reserved} MiB', flush=True)
                    save('running')
                time.sleep(3)
        except BaseException as error:
            stopped.extend(job_key(item['job']) for gpu in GPUS for item in per_gpu[gpu].values())
            stop({pid: item for gpu in GPUS for pid, item in per_gpu[gpu].items()})
            for gpu in GPUS:
                per_gpu[gpu].clear()
            save('stopped' if isinstance(error, KeyboardInterrupt) else 'failed')
            raise
        save('failed' if failed else 'complete')
        print(f'[QUEUE-END] {len(completed)} complete, {len(failed)} failed', flush=True)


if __name__ == '__main__':
    main()
