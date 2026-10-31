"""Read formal CASA logs, queue state and GPU usage on the training server."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


DATASETS = ['LEVIR-CD-256', 'SYSU-CD-256', 'WHU-CD-256', 'CDD-CD-256']
VARIANTS = ['M1_HYPER_QKV', 'M2_CA_QKV', 'M0_CTRL']
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def values(line):
    return {key: float(value) for key, value in
            re.findall(r'([A-Za-z][A-Za-z0-9_]*)=(' + NUMBER + ')', line)}


def process_alive(pid, entry):
    path = Path(f'/proc/{pid}/cmdline')
    try:
        return entry.encode() in path.read_bytes()
    except FileNotFoundError:
        return False


def describe_run(root, name, state, now):
    directory = root / name
    log = directory / 'train_log.txt'
    placement = read_json(directory / 'placement.json')
    manifest = read_json(directory / 'run_manifest.json')
    active = {name: pid for pid, name in state.get('active', {}).items()}
    pid = active.get(name)
    summary = dict(name=name, placement=placement, active_pid=int(pid) if pid else None,
                   physical_gpu=state.get('active_gpus', {}).get(pid, placement.get('physical_gpu')),
                   alive=process_alive(pid, 'train_CASA.py') if pid else None,
                   complete_test=False, status='pending',
                   evaluation_split=manifest.get('selection_split', 'val'),
                   independent_test=manifest.get('independent_test', True))
    if not log.exists():
        if pid:
            summary['status'] = 'starting' if summary['alive'] else 'missing_process'
        return summary

    text = log.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    steps = re.findall(r'(?:step=|cumulative=)(\d+)/(\d+)', text)
    training = []
    validation = []
    diagnostics = []
    diagnostic_parse_errors = []
    for line in lines:
        if line.startswith('[TRAIN-EPOCH] '):
            training.append(values(line))
        match = re.match(r'Epoch (\d+)/(\d+) \| ', line)
        if match:
            validation.append(dict(epoch=int(match[1]), total_epochs=int(match[2]), **values(line)))
        if line.startswith('[MODULE-DIAG] '):
            try:
                diagnostics.append(json.loads(line.split(' ', 1)[1]))
            except json.JSONDecodeError:
                diagnostic_parse_errors.append(line[:160])

    blocks = re.findall(r'=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===', text, re.S)
    summary['complete_test'] = bool(blocks)
    if blocks:
        summary['status'] = 'complete'
    elif pid:
        summary['status'] = 'running' if summary['alive'] else 'missing_process'
    elif name in state.get('stopped', []):
        summary['status'] = 'stopped'
    else:
        summary['status'] = 'incomplete'
    summary['log_age_seconds'] = round(max(0, now - log.stat().st_mtime), 1)
    summary['steps'] = int(steps[-1][0]) if steps else 0
    summary['target_steps'] = int(steps[-1][1]) if steps else manifest.get('actual_steps')
    summary['train_batches'] = manifest.get('train_batches')
    summary['total_epochs'] = manifest.get('epochs', validation[-1]['total_epochs'] if validation else None)
    summary['finished_train_epochs'] = int(training[-1]['epoch']) + 1 if training else None
    summary['recent_train'] = training[-5:]
    summary['recent_validation'] = validation[-5:]
    summary['best_validation'] = max(validation, key=lambda row: (row.get('F1', -1), row['epoch'])) if validation else None
    durations = [row['t'] for row in validation[-20:] if 't' in row]
    summary['mean_epoch_seconds_last20'] = round(statistics.mean(durations), 3) if durations else None
    summary['last_module_diagnostics'] = diagnostics[-1] if diagnostics else {}
    summary['errors'] = [line for line in lines if re.search(
        r'Traceback|FloatingPointError|RuntimeError|OutOfMemoryError|\bnan\b|\binf\b', line, re.I)]
    if diagnostic_parse_errors:
        summary['diagnostic_parse_errors'] = diagnostic_parse_errors
    if summary['errors'] or summary['status'] in ('missing_process', 'stopped', 'incomplete'):
        summary['log_tail'] = lines[-15:]
    if blocks:
        summary['test_block'] = blocks[-1].strip()
        metric_line = next((line for line in blocks[-1].splitlines() if 'F1=' in line), '')
        summary['test_metrics'] = values(metric_line)
        summary['result'] = read_json(directory / 'result.json')
    return summary


def gpu_status():
    command = ['nvidia-smi', '--id=0,1',
               '--query-gpu=index,memory.used,memory.total,utilization.gpu',
               '--format=csv,noheader,nounits']
    try:
        process = subprocess.run(command, text=True, capture_output=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return dict(error=str(error))
    if process.returncode:
        return dict(error=process.stderr.strip(), exit_code=process.returncode)
    return [dict(zip(('index', 'memory_used_mib', 'memory_total_mib', 'utilization_percent'),
                     [int(value.strip()) for value in line.split(',')]))
            for line in process.stdout.splitlines() if line.strip()]


def collect(tag, run):
    root = Path('/home/yqwang/outputs/CASA-CD') / tag / run
    state = read_json(root / 'pipeline_status.json')
    if not state:
        raise FileNotFoundError(root / 'pipeline_status.json')
    datasets = state.get('dataset_priority', DATASETS)
    variants = state.get('variants', VARIANTS)
    names = state.get('experiments') or [f'{variant}/{dataset}' for dataset in datasets for variant in variants]
    now = time.time()
    runs = [describe_run(root, name, state, now) for name in names]
    pid_file = root / 'pipeline.pid'
    supervisor = int(pid_file.read_text().strip()) if pid_file.exists() else None
    return dict(observed=datetime.now().astimezone().isoformat(timespec='seconds'),
                tag=tag, run=run, log_root=str(root), queue=state,
                supervisor_pid=supervisor,
                supervisor_alive=process_alive(supervisor, 'pipeline.py') if supervisor else None,
                completed=sum(row['complete_test'] for row in runs), total=len(runs),
                runs=runs, gpus=gpu_status())


def readable(summary):
    print(f"{summary['observed']}  {summary['tag']}/{summary['run']}  "
          f"queue={summary['queue']['status']}  complete={summary['completed']}/{summary['total']}")
    print(f"Supervisor: pid={summary['supervisor_pid']} alive={summary['supervisor_alive']}")
    print('GPUs:', json.dumps(summary['gpus'], ensure_ascii=False))
    if summary['queue'].get('failed'):
        print('Failed:', json.dumps(summary['queue']['failed'], ensure_ascii=False))
    for row in summary['runs']:
        print(f"\n{row['name']}  {row['status']}  gpu={row['physical_gpu']} "
              f"pid={row['active_pid']} alive={row['alive']} "
              f"steps={row.get('steps', 0)}/{row.get('target_steps')} "
              f"epochs={row.get('finished_train_epochs')}/{row.get('total_epochs')} "
              f"evaluation_split={row['evaluation_split']} "
              f"recent_epoch_s={row.get('mean_epoch_seconds_last20')} "
              f"log_age_s={row.get('log_age_seconds')}")
        for key in ('recent_train', 'recent_validation', 'best_validation',
                    'last_module_diagnostics', 'test_metrics', 'errors', 'diagnostic_parse_errors'):
            if row.get(key):
                print(f"  {key}: {json.dumps(row[key], ensure_ascii=False)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='CASA-Pyramid')
    parser.add_argument('--run', default='Run2')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    summary = collect(args.tag, args.run)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        readable(summary)


if __name__ == '__main__':
    main()
