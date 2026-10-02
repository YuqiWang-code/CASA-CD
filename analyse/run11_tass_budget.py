"""Run11 TASS budget audit（方案 §6/§13.9，机器实测优先）。

输出 C0（token）/ M1（tass）的 train-graph 与 deploy-graph 参数 + FLOPs、
TASS-only 参数与 headroom_to_5M；硬断言 deploy EFFECTIVE <= 5.0M。
FLOPs 输入 2x3x256x256（fvcore，unsupported ops 上报）。

Usage (server):
    CUDA_VISIBLE_DEVICES=1 python analyse/run11_tass_budget.py \
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

from model.str_tass_fusion import STRTASSNet

BUDGET = 5.0e6


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
    for mode in ("token", "tass"):
        print(f"=== STRTASS spatial_mode={mode} (dim=160) ===")
        model = STRTASSNet(args.pretrained_weight_path, dim=160, spatial_mode=mode).cuda()
        model.eval()
        tass_params = model.tass.param_count() if model.tass is not None else 0
        print(f"  [TASS-only params] {tass_params:,} ({tass_params / 1e6:.4f}M)")
        print("  [train-graph]")
        results[mode] = {"train": report("train", model), "tass": tass_params}

        m2 = copy.deepcopy(model)
        m2.switch_to_deploy()
        m2.eval()
        print("  [deploy-graph]")
        results[mode]["deploy"] = report("deploy", m2)
        # TAR/DCR 的 BN 必须全部折叠；TASS 是单路径静态 stem，其 BN 合法留在部署图
        bn_keys = [k for k in m2.state_dict().keys() if "bn" in k and not k.startswith("tass.")]
        print(f"  [deploy-branch-free] non-TASS bn_keys={len(bn_keys)} (must be 0)")
        assert len(bn_keys) == 0, f"deploy state_dict has non-TASS BN keys: {bn_keys[:5]}"
        del model, m2

    print("\n=== budget gate (deploy EFFECTIVE <= 5.0M) ===")
    ok = True
    for mode in ("token", "tass"):
        d = results[mode]["deploy"]
        under = d[1] <= BUDGET
        ok &= under
        print(f"  {mode}: deploy EFFECTIVE={d[1] / 1e6:.4f}M <= 5.0M -> {'PASS' if under else 'FAIL'}")
        headroom = (BUDGET - d[1]) / 1e6
        print(f"  {mode}: headroom_to_5M = {headroom:.4f}M")
    print(f"  [TASS-BUDGET] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
