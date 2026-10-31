"""Freeze the exploration source before validation, probes and formal training."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from config import PROTOCOL, REPO, ROOT, experiments


def main():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().replace(b'\0', b' ')
        except FileNotFoundError:
            continue
        if b'/CASA-MoE/Run3/' in command and any(name in command for name in
                (b'pipeline.py', b'health_probe.py', b'train_CASA.py')):
            raise RuntimeError('an exploration process is active; retain its runtime source')
    if any((ROOT / name / 'train_log.txt').exists() for name in experiments()):
        raise RuntimeError('formal Run3 logs already exist; retain the recorded runtime')
    label = datetime.now().strftime('%Y%m%d_%H%M%S')
    destination = ROOT / 'runtime' / label
    destination.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(REPO / 'models_v1', destination / 'models_v1', ignore=ignore)
    scripts = Path('train_scripts/CASA-MoE/Run3')
    shutil.copytree(REPO / scripts, destination / scripts, ignore=ignore)
    with tarfile.open(ROOT / 'source_code.tar.gz', 'w:gz') as archive:
        archive.add(destination / 'models_v1', arcname='models_v1')
        archive.add(destination / scripts, arcname=scripts.as_posix())
    git = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True,
                         capture_output=True)
    import torch
    record = dict(protocol=PROTOCOL, directory=str(destination), created=label,
                  source_checkout_commit=git.stdout.strip() if git.returncode == 0 else None,
                  source_archive='source_code.tar.gz', physical_gpu=0, max_active=1,
                  python=subprocess.check_output(['python', '--version'], text=True).strip(),
                  pytorch=torch.__version__, cuda=torch.version.cuda,
                  experiments=experiments(), formal_initialization='fresh MobileNetV2 pretrained')
    (ROOT / 'runtime.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
