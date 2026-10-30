"""Download formal exploration results and preserve historical metric cells."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from config import ROOT as REMOTE, experiments

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'tools'))
from extract_metrics_to_excel import extract_test_block, parse_block
from openpyxl import load_workbook


def command(*args):
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def metric_rows(path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook['metrics'].values)
    workbook.close()
    return rows


def frozen_source(base):
    source = base / 'source_code.tar.gz'
    if not source.exists():
        spec = importlib.util.spec_from_file_location('project_ssh', REPO / '.codex/ssh.py')
        ssh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ssh)
        client = ssh._conn()
        try:
            with client.open_sftp() as sftp:
                sftp.get((REMOTE / 'source_code.tar.gz').as_posix(), str(source))
        finally:
            client.close()
    destination = base / 'runtime_source'
    with tarfile.open(source, 'r:gz') as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith('models_v1/'):
                path = destination / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    return destination / 'models_v1'


def main():
    command('tools/download_training_results.py', '--tag', 'CASA-MoE', '--run', 'Run2')
    base = REPO / 'outputs/CASA-MoE/Run2'
    status = json.loads((base / 'pipeline_status.json').read_text())
    prior_path = base / 'collection_status.json'
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {'formal_runs': []}
    active = set(status.get('active', {}).values())
    failed = {item['run'] for item in status.get('failed', [])}
    formal, rows = [], []
    for name in experiments():
        variant, dataset = name.split('/')
        path = base / name / 'train_log.txt'
        text = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
        block = extract_test_block(text)
        f1, iou = '—', '—'
        if block:
            metrics = parse_block(block)
            if 'F1' not in metrics or 'IoU' not in metrics:
                raise RuntimeError(f'incomplete metric fields: {name}')
            formal.append(name)
            progress = '完成'
            f1, iou = f"{metrics['F1']*100:.2f}", f"{metrics['IoU']*100:.2f}"
        elif name in active:
            progress = '训练中'
        elif name in failed:
            progress = '失败'
        else:
            progress = '已停止' if text else '待启动'
        rows.append(f"| {dataset.split('-')[0]} | `{variant}` | {progress} | {f1} | {iou} |")
    observed = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    section = f"状态更新：{observed} CST；队列 `{status['status']}`；正式结果 **{len(formal)}/6**。\n\n"
    section += '| 数据集 | 实验 | 状态 | F1 (%) | IoU (%) |\n|---|---|---|---:|---:|\n' + '\n'.join(rows)
    readme = Path(__file__).parent / 'README.md'
    text = readme.read_text(encoding='utf-8')
    begin, end = '<!-- CASA_MOE_RUN2_RESULTS_START -->', '<!-- CASA_MOE_RUN2_RESULTS_END -->'
    start, stop = text.index(begin)+len(begin), text.index(end)
    readme.write_text(text[:start]+'\n\n'+section+'\n\n'+text[stop:], encoding='utf-8', newline='\n')
    if set(formal)-set(prior['formal_runs']):
        workbook = REPO / 'docs/experiment_metrics.xlsx'
        original, before = workbook.read_bytes(), metric_rows(workbook)
        command('tools/extract_metrics_to_excel.py')
        after = {tuple(row[:4]): row for row in metric_rows(workbook)[1:]}
        if any(after.get(tuple(row[:4])) != row for row in before[1:]):
            workbook.write_bytes(original)
            raise RuntimeError('existing metric cells changed; original workbook restored')
        source = frozen_source(base).relative_to(REPO).as_posix()
        command('tools/models_to_txt.py', '--tag', 'CASA-MoE', '--run', 'Run2', '--models-dir', source)
    prior_path.write_text(json.dumps(dict(formal_runs=formal, pipeline_status=status['status']), indent=2))
    print(f"CASA-MoE Run2 formal={len(formal)}/6, status={status['status']}")


if __name__ == '__main__':
    main()
