"""Launch one declared formal experiment from pretrained initialization."""
import argparse
import os
from pathlib import Path
import sys
from config import ROOT, VARIANTS, DATASETS, experiments, training_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=list(VARIANTS))
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('gpu', type=int, choices=[1])
    args = parser.parse_args()
    name = f'{args.variant}/{args.dataset}'
    if name not in experiments():
        parser.error('experiment is outside the Run9 matrix')
    repo = Path('/home/yqwang/projects/CASA-CD')
    logdir = ROOT / name
    checkpoint = Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run9') / name
    logdir.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    logfile = logdir / 'train_log.txt'
    text = logfile.read_text(errors='replace') if logfile.exists() else ''
    if '=== END TEST RESULTS ===' in text:
        print(f'{name} already has complete TEST RESULTS')
        return
    extra = []
    if (checkpoint / 'last.pth').exists():
        extra = ['--resume']
    elif text:
        raise RuntimeError(f'{name} has a prior log but no resumable checkpoint')
    command = [sys.executable, str(repo / 'models/train_CASA.py'),
               '--dataset-root', str(Path('/share_datasets/CD') / args.dataset),
               '--output-dir', str(logdir), '--checkpoint-dir', str(checkpoint),
               '--pretrained', str(repo / 'pretrained_weight/mobilenet_v2-b0353104.pth'),
               *training_options(args.variant), '--encoder-lr-mult', '1.0',
               '--batch-size', '32', '--max-steps', '40000', '--num-workers', '1', *extra]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.chdir(repo / 'models')
    with logfile.open('a', buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
