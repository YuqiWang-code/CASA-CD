"""Stage the Run13 runtime and frozen source archive on the training server.

The runtime mirrors Run12: an immutable copy of `models/` plus this run's own
scripts. Training reads only the copy, so later local edits cannot change a
running queue.
"""
import json
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO/'.codex'))
from ssh import _conn

from config import CHECKPOINTS, GPUS, PROTOCOL, ROOT as REMOTE_ROOT, RUNTIME

MODELS = REPO/'models'
SCRIPTS = HERE
SUFFIXES = {'.py', '.md', '.txt', '.sh'}


def payload():
    for path in sorted(MODELS.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py', '.md', '.txt'}:
            yield path, f'models/{path.relative_to(MODELS).as_posix()}'
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
    staging = REPO/'outputs/CASA-Pyramid/Run13'
    staging.mkdir(parents=True, exist_ok=True)
    source = staging/'source_code.tar.gz'
    with tarfile.open(source, 'w:gz') as tar:
        for path, name in payload():
            add(tar, path, name)
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip()
    manifest = dict(protocol=PROTOCOL, base_commit=commit, source_archive='source_code.tar.gz',
                    runtime=str(RUNTIME), authorized_physical_gpus=list(GPUS), experiments=6,
                    seeds=[2333], selection_split='test', calibration_split='val',
                    test_primary_threshold=.5, independent_test=False,
                    initialized_from='fresh MobileNetV2 pretrained weights',
                    ablation=dict(TAR_PLAIN='redundant multi-branch temporal fusion -> single 1x1 + BN',
                                  DCR_PLAIN='redundant concat/sum/diff fusion -> single 1x1 + BN'))
    (staging/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    bundle = staging/'runtime_bundle.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        for path, name in payload():
            add(tar, path, f'runtime/{name}')
        add(tar, source, 'source_code.tar.gz')
        add(tar, staging/'source_manifest.json', 'source_manifest.json')
    with _conn() as client:
        run(client, f'test ! -f {REMOTE_ROOT}/readiness.json')
        with client.open_sftp() as sftp:
            for folder in (str(REMOTE_ROOT), str(RUNTIME), str(CHECKPOINTS)):
                try:
                    sftp.stat(folder)
                except IOError:
                    sftp.mkdir(folder)
            sftp.put(str(bundle), str(REMOTE_ROOT/'runtime_bundle.tar.gz'))
        run(client, f'tar -xzf {REMOTE_ROOT}/runtime_bundle.tar.gz -C {REMOTE_ROOT} '
                    f'&& rm {REMOTE_ROOT}/runtime_bundle.tar.gz')
        run(client, f'cd {RUNTIME}/train_scripts/CASA-Pyramid/Run13 && '
                    f'bash -lc "source environment.sh && python environment_probe.py"')
        run(client, f'cd {RUNTIME}/train_scripts/CASA-Pyramid/Run13 && '
                    f'bash -lc "source environment.sh && python -c \'import config;'
                    f'print(config.experiments())\'"')
    print(f'[STAGED] {RUNTIME} from {commit}; bundle {bundle.stat().st_size} bytes')


if __name__ == '__main__':
    main()
