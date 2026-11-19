"""Schedule the four Run5 jobs on the single authorized GPU."""
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

from config import (GPUS, HEADROOM, PROTOCOL, ROOT, SLOTS, experiments, reservation_mib)

HERE = Path(__file__).resolve().parent
# Measured Run2 wall-clock per job, used only for the dry-run forecast.
FORECAST_HOURS = {"CG_HRA/SYSU-CD-256": 3.0, "CG_HRA_EDGE/SYSU-CD-256": 3.0,
                  "CG_HRA/LEVIR-CD-256": 2.7, "CG_HRA_EDGE/LEVIR-CD-256": 2.7,
                  "CG_HRA/WHU-CD-256": 2.6, "CG_HRA_EDGE/WHU-CD-256": 2.6,
                  "CG_HRA/CDD-CD-256": 3.2, "CG_HRA_EDGE/CDD-CD-256": 3.2}
# Attribution first: the two ARM_C legs answer the headline question and take the
# two slots at 01:00; the second-lever and boundary arms follow.
# SYSU first, then WHU: the smaller library finishes sooner, so its three arms
# give a complete row of results before the WHU legs start.
DATASET_ORDER = ("SYSU-CD-256", "LEVIR-CD-256", "WHU-CD-256", "CDD-CD-256")
ARM_ORDER = ("CG_HRA", "CG_HRA_EDGE")


def free_memory(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())


def job_key(job):
    return '/'.join(job)


def forecast(jobs):
    """Two slots, priority-first packing; returns per-job start/finish offsets."""
    slots = [0.0] * SLOTS
    plan = []
    for job in sorted(jobs, key=lambda item: (DATASET_ORDER.index(item[1])
                                              if item[1] in DATASET_ORDER else 9,
                                              ARM_ORDER.index(item[0])
                                              if item[0] in ARM_ORDER else 9)):
        index = min(range(SLOTS), key=lambda slot: slots[slot])
        start = slots[index]
        finish = start + FORECAST_HOURS.get(job_key(job), 3.0)
        slots[index] = finish
        plan.append((job_key(job), start, finish))
    return plan, max(slots)


def fit(pending, active, budget, reservations, gpu):
    """Pick the first pending job that fits *this* GPU's slots and memory.

    ``active`` is the per-GPU table (not the global one), so the slot limit holds
    per card even when MINOR_GPUS lists more than one GPU.
    """
    if not pending or len(active) >= SLOTS:
        return None
    available = min(budget - sum(item['reserved'] for item in active.values()), free_memory(gpu) - HEADROOM)
    for job in pending:
        if reservation_mib(job[0], reservations) <= available:
            return job
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
    def rank(job):
        # dataset first so SYSU finishes as one block, then the arm order
        dataset = DATASET_ORDER.index(job[1]) if job[1] in DATASET_ORDER else 9
        arm = ARM_ORDER.index(job[0]) if job[0] in ARM_ORDER else 9
        return (job not in priority, dataset, arm)

    pending.sort(key=rank)
    return pending, ['/'.join(job) for job in completed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='continue existing Run5 artifacts')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the matrix, forecast and per-job commands, then exit')
    args = parser.parse_args()

    plan, makespan = forecast(experiments())
    if args.dry_run:
        print(f'[DRY-RUN] protocol={PROTOCOL} gpus={GPUS} slots={SLOTS} headroom={HEADROOM} MiB')
        for name, start, finish in plan:
            arm, dataset = name.split('/')
            free = free_memory(GPUS[0]) if not args.dry_run else -1
            print(f'  {name:28s} start=+{start:4.1f}h finish=+{finish:4.1f}h '
                  f'reserved={reservation_mib(arm)} MiB')
        print(f'[DRY-RUN] makespan={makespan:.1f}h; run_one command for DETAIL:')
        print('  ' + ' '.join([sys.executable, str(HERE / 'run_one.py'), 'DETAIL', 'LEVIR-CD-256',
                               '--gpu', str(GPUS[0])]))
        return

    status_file = ROOT / 'pipeline_status.json'
    with (ROOT / 'pipeline.pid').open('a+') as pidfile:
        fcntl.flock(pidfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(status_file.read_text()) if status_file.exists() else {}
        if previous and not args.resume:
            raise RuntimeError('existing Run5 queue; use --resume to continue it')
        for pid in previous.get('active', {}):
            try:
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:
                continue
            if b'run_one.py' in cmd and b'CASA-MinorGoal' in cmd:
                raise RuntimeError(f'existing Run5 child: PID {pid}')
        readiness = json.loads((ROOT / 'readiness.json').read_text())
        if readiness['protocol'] != PROTOCOL or not readiness['passed']:
            raise RuntimeError('Run5 requires passing server preflight')
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
                    job = fit(pending, per_gpu[gpu], budgets[gpu], reservations, gpu)
                    if job is None:
                        continue
                    pending.remove(job)
                    arm, dataset = job
                    command = [sys.executable, str(HERE / 'run_one.py'), arm, dataset, '--gpu', str(gpu)]
                    if args.resume:
                        command.append('--resume')
                    process = subprocess.Popen(command, start_new_session=True)
                    reserved = reservation_mib(arm, reservations)
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
            stop({pid: item for gpu in GPUS for item in per_gpu[gpu].items()})
            for gpu in GPUS:
                per_gpu[gpu].clear()
            save('stopped' if isinstance(error, KeyboardInterrupt) else 'failed')
            raise
        save('failed' if failed else 'complete')
        print(f'[QUEUE-END] {len(completed)} complete, {len(failed)} failed', flush=True)


if __name__ == '__main__':
    main()
