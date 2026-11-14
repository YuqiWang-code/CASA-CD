"""Collect formal CASA-Pyramid Run4 tests and preserve all prior metric rows."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

HISTORICAL_RECORDS = 197
EXPECTED_RECORDS = HISTORICAL_RECORDS + 12

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from extract_metrics_to_excel import extract_test_block, parse_block
from openpyxl import load_workbook


def command(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def metric_rows(path):
    book = load_workbook(path, read_only=True, data_only=True)
    rows = list(book['metrics'].values)
    book.close()
    return rows


def training_source_dir(base):
    destination = base / 'runtime_source'
    with tarfile.open(base / 'source_code.tar.gz', 'r:gz') as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith('models/'):
                path = destination / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    return destination / 'models'


def main():
    command('tools/download_training_results.py', '--tag', 'CASA-Pyramid', '--run', 'Run4')
    base = ROOT / 'outputs/CASA-Pyramid/Run4'
    status = json.loads((base / 'pipeline_status.json').read_text())
    prior_path = base / 'collection_status.json'
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {'formal_runs': []}
    active = set(status.get('active', {}).values())
    failed = {item['run'] for item in status.get('failed', [])}
    rows, formal = [], []
    for dataset in ['LEVIR-CD-256', 'SYSU-CD-256', 'WHU-CD-256', 'CDD-CD-256']:
        for variant in ['M0_CTRL', 'M1_HYPER_QKV', 'M2_CA_QKV']:
            name = f'{variant}/{dataset}'
            path = base / name / 'train_log.txt'
            text = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
            block = extract_test_block(text)
            f1, iou = '—', '—'
            if block:
                metrics = parse_block(block)
                formal.append(name)
                progress = '完成'
                f1, iou = f"{metrics['F1']*100:.2f}", f"{metrics['IoU']*100:.2f}"
            elif name in active:
                steps = re.findall(r'(?:step=|cumulative=)(\d+/\d+)', text)
                progress = f'训练中：{steps[-1]}' if steps else '初始化'
            elif name in failed:
                progress = '失败'
            elif text:
                progress = '已停止'
            else:
                progress = '待启动'
            rows.append(f"| {dataset.split('-')[0]} | `{variant}` | {progress} | {f1} | {iou} |")
    observed = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    section = f"状态更新：{observed} CST；队列 `{status['status']}`；正式结果 **{len(formal)}/12**。\n\n"
    section += '| 数据集 | 实验 | 状态 / 步数 | F1 (%) | IoU (%) |\n|---|---|---|---:|---:|\n'
    section += '\n'.join(rows)
    path = ROOT / 'README.md'
    text = path.read_text(encoding='utf-8')
    begin, end = '<!-- CASA_PYRAMID_RUN4_RESULTS_START -->', '<!-- CASA_PYRAMID_RUN4_RESULTS_END -->'
    start, stop = text.index(begin) + len(begin), text.index(end)
    path.write_text(text[:start] + '\n\n' + section + '\n\n' + text[stop:], encoding='utf-8', newline='\n')
    if set(formal) - set(prior['formal_runs']):
        workbook = ROOT / 'docs/experiment_metrics.xlsx'
        original = workbook.read_bytes()
        before = metric_rows(workbook)
        if len(before) - 1 < HISTORICAL_RECORDS:
            raise RuntimeError(f'expected at least {HISTORICAL_RECORDS} historical records')
        command('tools/extract_metrics_to_excel.py')
        after = {tuple(row[:4]): row for row in metric_rows(workbook)[1:]}
        if len(formal) == 12 and len(after) < EXPECTED_RECORDS:
            workbook.write_bytes(original)
            raise RuntimeError(f'expected at least {EXPECTED_RECORDS} completed records; workbook restored')
        if any(after.get(tuple(row[:4])) != row for row in before[1:]):
            workbook.write_bytes(original)
            raise RuntimeError('prior metric cells changed; existing workbook restored')
        models_dir = training_source_dir(base).relative_to(ROOT).as_posix()
        command('tools/models_to_txt.py', '--tag', 'CASA-Pyramid', '--run', 'Run4',
                '--models-dir', models_dir)
    prior_path.write_text(json.dumps(dict(formal_runs=formal, pipeline_status=status['status']), indent=2))
    print(f"CASA-Pyramid Run4 formal={len(formal)}/12, status={status['status']}")


if __name__ == '__main__':
    main()
