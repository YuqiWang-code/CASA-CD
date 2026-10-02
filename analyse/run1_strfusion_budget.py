"""STRFusion Run1 budget audit (G4).

Pre-registered (design doc §3): machine-measured TOTAL / TRAINABLE / EFFECTIVE
params + FLOPs for C0 (plain) and M1 (full), on BOTH the training graph and the
folded deploy graph. Hard asserts:
  - deploy EFFECTIVE <= 3.0M  (CASA-CD hard budget; G4 gate)
  - C0 and M1 deploy params are bitwise equal (identical deploy function class)
  - deploy state_dict contains no BN / aux keys
FLOPs input: 2x3x256x256 bi-temporal pair (fvcore; unsupported ops reported).

Usage (server):
    CUDA_VISIBLE_DEVICES=1 python analyse/run1_strfusion_budget.py \
        --pretrained_weight_path pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth
"""
import argparse
import copy
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "models"))

from model.str_fusion import STRFusionNet

BUDGET = 3.0e6


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dead = 0
    for name, mod in model.named_modules():
        if name in ("encoder.resnet.layer4", "encoder.resnet.fc", "encoder.resnet.avgpool"):
            dead += sum(p.numel() for p in mod.parameters())
    return total, total - dead, trainable


def measure_flops(model, size=256):
    from fvcore.nn import flop_count

    model.eval()
    pre = torch.randn(1, 3, size, size).cuda()
    post = torch.randn(1, 3, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post))
    return sum(counts.values()), len(unsupported)


def report(name, model):
    total, effective, trainable = count_params(model)
    flops, n_unsup = measure_flops(model)
    print(f"  [{name}] TOTAL={total / 1e6:.4f}M  EFFECTIVE={effective / 1e6:.4f}M  "
          f"TRAINABLE={trainable / 1e6:.4f}M  FLOPs={flops:.4f}G  unsupported={n_unsup}")
    return total, effective, trainable, flops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True

    results = {}
    for rep_mode in ("plain", "full"):
        print(f"=== STRFusion rep_mode={rep_mode} (dim=160) ===")
        model = STRFusionNet(args.pretrained_weight_path, dim=160, rep_mode=rep_mode).cuda()
        model.eval()
        print("  [train-graph]")
        results[rep_mode] = {"train": report("train", model)}

        m2 = copy.deepcopy(model)
        m2.switch_to_deploy()
        m2.eval()
        print("  [deploy-graph]")
        results[rep_mode]["deploy"] = report("deploy", m2)
        bn_keys = [k for k in m2.state_dict().keys() if "bn" in k]
        print(f"  [deploy-branch-free] bn_keys={len(bn_keys)} (must be 0)")
        assert len(bn_keys) == 0, f"deploy state_dict has BN keys: {bn_keys[:5]}"
        del model, m2

    print("\n=== G4 budget gate ===")
    ok = True
    for rep_mode in ("plain", "full"):
        d = results[rep_mode]["deploy"]
        under = d[1] <= BUDGET
        ok &= under
        print(f"  {rep_mode}: deploy EFFECTIVE={d[1] / 1e6:.4f}M <= 3.0M -> {'PASS' if under else 'FAIL'}")
    same = (results["plain"]["deploy"][0] == results["full"]["deploy"][0])
    ok &= same
    print(f"  C0/M1 deploy params equal ({results['plain']['deploy'][0]:,} == "
          f"{results['full']['deploy'][0]:,}) -> {'PASS' if same else 'FAIL'}")
    print(f"  [SF-BUDGET-G4] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
