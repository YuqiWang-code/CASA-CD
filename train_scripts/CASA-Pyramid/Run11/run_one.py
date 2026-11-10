"""Launch or resume a declared Run11 job."""
import argparse
import json
import os
import sys
from pathlib import Path
from config import ROOT, RUNTIME, PROTOCOL, DATASETS, VARIANTS, PRETRAINED, training_options


def main():
    p = argparse.ArgumentParser()
    p.add_argument('variant', choices=VARIANTS)
    p.add_argument('dataset', choices=DATASETS)
    p.add_argument('gpu', type=int, choices=[1])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    ready = json.loads((ROOT / 'readiness.json').read_text())
    if ready['protocol'] != PROTOCOL or not ready['passed']:
        raise RuntimeError('Run11 needs completed model checks and healthy paired probes')
    log = ROOT / args.variant / args.dataset
    ckpt = Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run11') / args.variant / args.dataset
    log.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    logfile = log / 'train_log.txt'
    last = ckpt / 'last.pth'
    if args.resume:
        if not last.is_file() or not (ckpt / 'best_model.pth').is_file() or not logfile.is_file():
            raise RuntimeError('Run11 resume requires last/best checkpoints and the existing log')
        if '=== END TEST RESULTS ===' in logfile.read_text(errors='replace'):
            raise RuntimeError('Run11 job already completed')
    elif last.exists() or logfile.exists():
        raise RuntimeError('Run11 job already has training artifacts; inspect before resuming')
    command = [sys.executable, str(RUNTIME / 'models/train_CASA.py'),
               '--dataset-root', '/share_datasets/CD/' + args.dataset,
               '--output-dir', str(log), '--checkpoint-dir', str(ckpt), '--pretrained', PRETRAINED,
               *training_options(args.variant), '--encoder-lr-mult', '1.0',
               '--batch-size', '32', '--max-steps', '40000', '--num-workers', '1']
    if args.resume:
        command.append('--resume')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with logfile.open('a' if args.resume else 'x', buffering=1) as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
