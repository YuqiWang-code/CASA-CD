"""Launch or resume one Run13 ablation arm on one physical GPU."""
import argparse
import os
import sys
import time

from config import CHECKPOINTS, DATA, GPUS, MATRIX, PRETRAINED, ROOT, RUNTIME, VARIANTS, model_options, seed_for
from pipeline import complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=VARIANTS)
    parser.add_argument('dataset', choices=MATRIX)
    parser.add_argument('--gpu', type=int, choices=GPUS, default=GPUS[0])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.variant not in MATRIX[args.dataset]:
        parser.error('experiment is outside the Run13 matrix')
    name = f'{args.variant}/{args.dataset}'
    if complete(name):
        return
    output, checkpoint = ROOT / name, CHECKPOINTS / name
    output.mkdir(parents=True, exist_ok=True)
    logfile = output/'train_log.txt'
    last = checkpoint/'last.pth'
    if not args.resume and (last.exists() or logfile.exists()):
        raise RuntimeError(f'{name} has existing artifacts; use pipeline.py --resume')
    if last.exists() and not logfile.is_file():
        raise RuntimeError(f'{name} has a checkpoint but no original training log')
    if logfile.exists() and not last.exists():
        logfile.rename(logfile.with_name(f'train_log_before_restart_{time.time_ns()}.txt'))
    seed = str(seed_for(args.variant))
    command = [sys.executable, str(RUNTIME / 'models/train_CASA.py'),
               '--dataset-root', str(DATA / args.dataset), '--output-dir', str(output),
               '--checkpoint-dir', str(checkpoint), '--pretrained', PRETRAINED,
               '--module', VARIANTS[args.variant][0], *model_options(args.variant),
               '--seed', seed, '--context-seed', seed, '--batch-size', '32',
               '--max-steps', '40000', '--num-workers', '1']
    if last.exists():
        command.append('--resume')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    with logfile.open('a' if last.exists() else 'x', buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
