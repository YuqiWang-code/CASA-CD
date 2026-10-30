"""Retest one selected Run8 checkpoint on physical GPU1."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'models'))
from model.casa_net import CASANet
from train_CASA import PROTOCOL, evaluate, make_loader, verify_deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['H_CLASS', 'H_NONE'], required=True)
    parser.add_argument('--dataset', choices=['SYSU-CD-256', 'WHU-CD-256'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    module = 'hyperbolic'
    checkpoint = Path('/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run8') / args.variant / args.dataset / 'best_model.pth'
    destination = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8/retests') / args.variant / args.dataset
    loader = make_loader(Path('/share_datasets/CD') / args.dataset, 'test', 32, 1)
    model = CASANet(None, module).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cuda:0', weights_only=True))
    image, _ = next(iter(loader))
    deploy, deployment = verify_deployment(model, image[:1].cuda().float())
    loss, metrics, spatial = evaluate(deploy, loader, torch.device('cuda:0'), return_diagnostics=True)
    result = dict(protocol=PROTOCOL, checkpoint=str(checkpoint), physical_gpu=1,
                  selection_split='test', independent_test=False, test_loss=loss,
                  metrics=metrics, spatial_diagnostics=spatial, deployment=deployment)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
