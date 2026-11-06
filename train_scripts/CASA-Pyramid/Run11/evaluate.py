"""Evaluate a saved Run11 best checkpoint using its frozen runtime."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from pathlib import Path
import sys
import torch
from config import ROOT, RUNTIME, VARIANTS, DATASETS

sys.path.insert(0, str(RUNTIME / 'models'))
from model.casa_net import CASANet
from train_CASA import make_loader, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('variant', choices=VARIANTS)
    p.add_argument('dataset', choices=DATASETS)
    p.add_argument('--split', choices=['val','test'], default='val')
    a = p.parse_args()
    model = CASANet(module=VARIANTS[a.variant]['module'])
    path = Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run11') / a.variant / a.dataset / 'best_model.pth'
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.cuda().eval()
    torch.set_num_threads(4)
    print(evaluate(model, make_loader(Path('/share_datasets/CD')/a.dataset, a.split),
                   torch.device('cuda'), return_diagnostics=True))


if __name__ == '__main__':
    main()
