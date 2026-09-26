"""Smoke test for the ChangeViT baseline (no real data needed).

Builds ChangeViT-Tiny, loads the pretrained DeiT-Tiny weights, runs a forward +
backward pass on random 256x256 bi-temporal input, and reports params / FLOPs.

Usage:
    python smoke_test.py --model_type tiny --pretrained_weight_path <deit_tiny.pth> [--gpu_id 0]
"""
import os
import sys
import argparse

import torch

_MODELS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from model.trainer import Trainer
from model.utils import BCEDiceLoss


def measure_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_effective_params(model):
    """Params excluding unused ResNet head (layer4 + fc + avgpool never run in forward)."""
    total = measure_params(model)
    dead = 0
    for name, mod in model.named_modules():
        if name in ("encoder.resnet.layer4", "encoder.resnet.fc", "encoder.resnet.avgpool"):
            dead += sum(p.numel() for p in mod.parameters())
    return total - dead


def measure_flops(model, size=256):
    from fvcore.nn import flop_count

    model.eval()
    pre = torch.randn(1, 3, size, size).cuda()
    post = torch.randn(1, 3, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post))
    return sum(counts.values()), len(unsupported)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='tiny')
    parser.add_argument('--pretrained_weight_path', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    print("[SMOKE] building ChangeViT-" + args.model_type)
    model = Trainer(args.model_type, pretrained_path=args.pretrained_weight_path).float()
    if torch.cuda.is_available():
        model = model.cuda()

    n_params = measure_params(model)
    n_eff = measure_effective_params(model)
    print(f"[SMOKE] total params = {n_params / 1e6:.3f} M, effective = {n_eff / 1e6:.3f} M")

    pre = torch.randn(2, 3, 256, 256)
    post = torch.randn(2, 3, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    if torch.cuda.is_available():
        pre, post, target = pre.cuda(), post.cuda(), target.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 2e-4)
    model.train()
    for step in range(3):
        out = model(pre, post)
        assert out.shape == (2, 1, 256, 256), f"bad output shape {out.shape}"
        loss = BCEDiceLoss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[SMOKE] step {step + 1}/3 ok, loss = {loss.item():.4f}")

    flops, n_unsup = measure_flops(model, size=256)
    print(f"[SMOKE] FLOPs = {flops:.4f} G (unsupported_ops={n_unsup})")
    print("[SMOKE] ALL OK")


if __name__ == "__main__":
    main()
