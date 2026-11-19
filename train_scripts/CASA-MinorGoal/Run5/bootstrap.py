"""Stage the Run5 runtime (current models tree) and record its fingerprint.

The server copy from Run1/Run2 is a frozen tree; this script is the only way to
produce a new one, and it writes a per-file SHA-256 manifest so a later audit can
tell which tree a run actually trained on.
"""
import hashlib
import json
import subprocess
import sys
import io
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / '.codex'))
from ssh import _conn

from config import ARMS, CHECKPOINTS, DATASETS, GPUS, JOBS, PROTOCOL, ROOT as REMOTE_ROOT, RUNTIME

SCRIPTS = HERE
SUFFIXES = {'.py', '.md', '.txt', '.sh'}
MODEL_SUFFIXES = {'.py', '.md', '.txt'}


def payload():
    tree = REPO / 'models'
    for path in sorted(tree.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in MODEL_SUFFIXES:
            yield path, f'models/{path.relative_to(tree).as_posix()}'
    for path in sorted(SCRIPTS.iterdir()):
        if path.is_file() and path.suffix in SUFFIXES:
            yield path, f'train_scripts/CASA-MinorGoal/{SCRIPTS.name}/{path.name}'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(tar, source, name):
    info = tar.gettarinfo(str(source), arcname=name)
    data = source.read_bytes()
    if source.suffix == '.sh':
        # the local tree is CRLF; bash on the server chokes on it (conda.sh\r)
        data = data.replace(b'\r\n', b'\n')
    if len(data) != info.size:
        info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def run(client, command):
    stdin, stdout, stderr = client.exec_command(command)
    out, err = stdout.read().decode(), stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(out.strip())
    if code:
        raise RuntimeError(f'remote command failed ({code}): {command}\n{err}')


def main():
    staging = REPO / 'outputs/CASA-MinorGoal/Run5'
    staging.mkdir(parents=True, exist_ok=True)
    source = staging / 'source_code.tar.gz'
    entries = list(payload())
    with tarfile.open(source, 'w:gz') as tar:
        for path, name in entries:
            add(tar, path, name)
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip()
    manifest = dict(protocol=PROTOCOL, base_commit=commit, source_archive='source_code.tar.gz',
                    runtime=str(RUNTIME), authorized_physical_gpus=list(GPUS),
                    datasets=list(DATASETS), arms=list(ARMS), jobs=['/'.join(job) for job in JOBS],
                    calibration_split='val', test_primary_threshold=.5, selection_split='val',
                    initialized_from='fresh MobileNetV2 pretrained weights',
                    runtime_manifest='runtime_manifest.json')
    manifest['experiments'] = len(JOBS)
    (staging / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    fingerprint = {name: digest(path) for path, name in entries}
    (staging / 'runtime_manifest.json').write_text(json.dumps(
        dict(protocol=PROTOCOL, base_commit=commit, files=fingerprint,
             count=len(fingerprint)), indent=2))
    bundle = staging / 'runtime_bundle.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        for path, name in entries:
            add(tar, path, f'runtime/{name}')
        add(tar, source, 'source_code.tar.gz')
        add(tar, staging / 'source_manifest.json', 'source_manifest.json')
        add(tar, staging / 'runtime_manifest.json', 'runtime_manifest.json')
    with _conn() as client:
        run(client, f'test ! -f {REMOTE_ROOT}/readiness.json')
        run(client, f'mkdir -p {REMOTE_ROOT} {RUNTIME} {CHECKPOINTS}')
        with client.open_sftp() as sftp:
            sftp.put(str(bundle), str(REMOTE_ROOT / 'runtime_bundle.tar.gz'))
        run(client, f'tar -xzf {REMOTE_ROOT}/runtime_bundle.tar.gz -C {REMOTE_ROOT} '
                    f'&& rm {REMOTE_ROOT}/runtime_bundle.tar.gz')
        run(client, f'cd {RUNTIME}/train_scripts/CASA-MinorGoal/Run5 && '
                    f'bash -lc "source environment.sh && python environment_probe.py"')
        run(client, f'cd {RUNTIME}/train_scripts/CASA-MinorGoal/Run5 && '
                    f'bash -lc "source environment.sh && python -c \'import config;'
                    f'print(len(config.experiments()), config.GPUS, config.ARM_ORDER)\'"')
    print(f'[STAGED] {RUNTIME} from {commit}; bundle {bundle.stat().st_size} bytes; '
          f'{len(fingerprint)} files fingerprinted')


if __name__ == '__main__':
    main()
