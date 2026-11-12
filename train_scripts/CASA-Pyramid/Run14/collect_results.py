"""Download and publish only complete Run14 test records."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from config import ARMS, PROTOCOL, experiments, profile_for, selection_for

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
from extract_metrics_to_excel import METRIC_KEYS, extract_test_block, parse_block
from openpyxl import load_workbook


def command(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def workbook_rows(path):
    with path.open('rb') as stream:
        book = load_workbook(stream, read_only=True, data_only=True)
        try:
            rows = list(book['metrics'].values)
        finally:
            book.close()
    return {tuple(row[:4]): row for row in rows[1:]}


def main():
    base = ROOT/'outputs/CASA-Pyramid/Run14'
    command('tools/download_training_results.py', '--tag', 'CASA-Pyramid', '--run', 'Run14')
    state = json.loads((base/'pipeline_status.json').read_text())
    if state['protocol'] != PROTOCOL:
        raise RuntimeError('Run14 queue uses a different protocol')
    formal = {}
    for name in experiments():
        log = base/name/'train_log.txt'
        block = extract_test_block(log.read_text(encoding='utf-8')) if log.is_file() else None
        if block:
            manifest = json.loads(log.with_name('run_manifest.json').read_text(encoding='utf-8'))
            arm = name.split('/')[0]
            if manifest['protocol'] != PROTOCOL:
                raise RuntimeError(f'Run14 result uses a different protocol: {name}')
            if manifest['selection_split'] != selection_for(arm):
                raise RuntimeError(f'Run14 result uses a different selection split: {name}')
            if manifest['independent_test'] != (selection_for(arm) == 'val'):
                raise RuntimeError(f'Run14 independence flag does not match selection: {name}')
            expected = ARMS[arm][0], ARMS[arm][3], ARMS[arm][4]
            if (manifest['module'], manifest['tar_variant'], manifest['decoder_variant']) != expected:
                raise RuntimeError(f'Run14 configuration not recorded in manifest: {name}')
            values = parse_block(block)
            if not all(key in values for key in METRIC_KEYS):
                raise RuntimeError(f'incomplete metric line: {name}')
            formal[('CASA-Pyramid', 'Run14', *name.split('/'))] = tuple(
                round(100*values[key], 2) for key in METRIC_KEYS)
    if not formal:
        print('No complete Run14 TEST RESULTS')
        return
    if state['status'] == 'complete' and len(formal) != len(experiments()):
        raise RuntimeError('completed queue is missing formal logs')

    sys.path.insert(0, str(ROOT/'.codex'))
    from ssh import _conn
    with _conn() as client, client.open_sftp() as sftp:
        sftp.get('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run14/source_code.tar.gz',
                 str(base/'source_code.tar.gz'))
    frozen = base/'runtime_source'
    with tarfile.open(base/'source_code.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith('models/'):
                path = frozen/member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())

    path = ROOT/'docs/experiment_metrics.xlsx'
    original, prior = path.read_bytes(), workbook_rows(path)
    try:
        command('tools/extract_metrics_to_excel.py')
        rows = workbook_rows(path)
        if any(rows.get(key) != row for key, row in prior.items()):
            raise RuntimeError('existing metric cells changed')
        if {key for key in rows if key[:2] == ('CASA-Pyramid', 'Run14')} != formal.keys():
            raise RuntimeError('Run14 workbook keys do not match formal experiments')
        if any(rows[key][4:10] != metrics for key, metrics in formal.items()):
            raise RuntimeError('Run14 workbook metrics do not match the TEST blocks')
    except BaseException:
        path.write_bytes(original)
        raise
    command('tools/models_to_txt.py', '--tag', 'CASA-Pyramid', '--run', 'Run14', '--models-dir', frozen/'models')
    command('tools/models_to_txt.py', '--tag', 'CASA-Pyramid', '--run', 'current')
    command(Path(__file__).with_name('compare.py'))
    command('tools/export_training_records.py', '--tag', 'CASA-Pyramid', '--run', 'Run14')
    print(f'{len(formal)} Run14 tests, {len(rows)} workbook records')


if __name__ == '__main__':
    main()
