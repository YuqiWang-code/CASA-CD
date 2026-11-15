"""Stage the Run16 runtime (three archived model trees) on the training server."""
import json
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / '.codex'))
from ssh import _conn

from config import (ARMS, CHECKPOINTS, DATASETS, GPUS, PROTOCOL, ROOT as REMOTE_ROOT,
                    RUNTIME, VERSIONS)

SCRIPTS = HERE
SUFFIXES = {'.py', '.md', '.txt', '.sh'}


def payload():
    for version, spec in VERSIONS.items():
        tree = REPO / f"models_{version}"
        for path in sorted(tree.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py', '.md', '.txt'}:
                yield path, f'{spec["model_dir"]}/{path.relative_to(tree).as_posix()}'
    for path in sorted(SCRIPTS.iterdir()):
        if path.is_file() and path.suffix in SUFFIXES:
            yield path, f'train_scripts/CASA-Pyramid/{SCRIPTS.name}/{path.name}'


def add(tar, source, name):
    info = tar.gettarinfo(str(source), arcname=name)
    with source.open('rb') as stream:
        tar.addfile(info, stream)


def run(client, command):
    stdin, stdout, stderr = client.exec_command(command)
    out, err = stdout.read().decode(), stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(out.strip())
    if code:
        raise RuntimeError(f'remote command failed ({code}): {command}\n{err}')


def main():
    staging = REPO / 'outputs/CASA-Pyramid/Run16'
    staging.mkdir(parents=True, exist_ok=True)
    source = staging / 'source_code.tar.gz'
    with tarfile.open(source, 'w:gz') as tar:
        for path, name in payload():
            add(tar, path, name)
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip()
    manifest = dict(protocol=PROTOCOL, base_commit=commit, source_archive='source_code.tar.gz',
                    runtime=str(RUNTIME), authorized_physical_gpus=list(GPUS),
                    datasets=list(DATASETS), versions=list(VERSIONS),
                    experiments=len(ARMS) * len(VERSIONS) * len(DATASETS),
                    calibration_split='val', test_primary_threshold=.5, selection_split='val',
                    initialized_from='fresh MobileNetV2 pretrained weights',
                    source_provenance={
                        'Run1': 'archived Run1 source tree (Poincare context, validation selection already native)',
                        'Run4': 'archived Run4 source tree + --selection-split and TAR/DCR variant switches (defaults preserved)',
                        'Run10': 'archived Run10 source tree + --selection-split and TAR/DCR variant switches (defaults preserved)'})
    (staging / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    bundle = staging / 'runtime_bundle.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        for path, name in payload():
            add(tar, path, f'runtime/{name}')
        add(tar, source, 'source_code.tar.gz')
        add(tar, staging / 'source_manifest.json', 'source_manifest.json')
    with _conn() as client:
        run(client, f'test ! -f {REMOTE_ROOT}/readiness.json')
        with client.open_sftp() as sftp:
            for folder in (str(REMOTE_ROOT), str(RUNTIME), str(CHECKPOINTS)):
                try:
                    sftp.stat(folder)
                except IOError:
                    sftp.mkdir(folder)
            sftp.put(str(bundle), str(REMOTE_ROOT / 'runtime_bundle.tar.gz'))
        run(client, f'tar -xzf {REMOTE_ROOT}/runtime_bundle.tar.gz -C {REMOTE_ROOT} '
                    f'&& rm {REMOTE_ROOT}/runtime_bundle.tar.gz')
        run(client, f'cd {RUNTIME}/train_scripts/CASA-Pyramid/Run16 && '
                    f'bash -lc "source environment.sh && python environment_probe.py"')
        run(client, f'cd {RUNTIME}/train_scripts/CASA-Pyramid/Run16 && '
                    f'bash -lc "source environment.sh && python -c \'import config;'
                    f'print(len(config.experiments()), config.GPUS, config.VERSIONS.keys())\'"')
    print(f'[STAGED] {RUNTIME} from {commit}; bundle {bundle.stat().st_size} bytes')


if __name__ == '__main__':
    main()

