"""Run4 U2：DeiT 预训练加载审计（corrected loader 的原位继承核对）。

核对内容（决策文档 §3 / §33 T0）：
  - source checkpoint 的 patch_embed / pos_embed / norm / blocks key 与 shape；
  - pos_embed：source (1, 1+196, C) → 去 cls/dist token → 14×14 → bicubic → 16×16；
  - 对 depth 内的继承张量：model 张量 vs source 张量 max_abs_diff 必须 == 0；
  - depth 之外的 block（如 blocks.4-11）不得残留在模型 state_dict 里。

用法（服务器，CPU 即可）：
    python analyse/vit_pretrain_audit.py \
        --pretrained_weight_path /home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth \
        --vit_depth 4
"""
import os
import sys
import argparse

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "models"))

from model.encoder import Encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--vit_depth", type=int, default=4)
    args = ap.parse_args()

    sd = torch.load(args.pretrained_weight_path, map_location="cpu")["model"]
    print("=== source checkpoint keys/shapes ===")
    for k in ("patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed",
              "norm.weight", "norm.bias", "blocks.0.attn.qkv.weight",
              "blocks.3.attn.qkv.weight"):
        if k in sd:
            print(f"  {k}: {tuple(sd[k].shape)}  dtype={sd[k].dtype}")
        else:
            print(f"  {k}: MISSING")
    print(f"  total source keys: {len(sd)}")

    enc = Encoder("tiny", pretrained_path=args.pretrained_weight_path,
                  resnet_pretrained=False, mode="baseline", vit_depth=args.vit_depth)

    # T0: 原位继承核对（max_abs_diff == 0）
    print(f"\n=== exact-load audit (depth={args.vit_depth}) ===")
    model_sd = enc.vit.state_dict()
    worst = 0.0
    for i in range(args.vit_depth):
        prefix = f"blocks.{i}."
        for k in ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
                  "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
                  "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"):
            src = sd[prefix + k].float()
            dst = model_sd[prefix + k].float()
            d = (src - dst).abs().max().item()
            worst = max(worst, d)
            assert d == 0.0, f"{prefix + k} NOT loaded in place (diff {d})"
    for k in ("patch_embed.proj.weight", "patch_embed.proj.bias", "norm.weight", "norm.bias"):
        src = sd[k].float()
        dst = model_sd[k].float()
        d = (src - dst).abs().max().item()
        worst = max(worst, d)
        assert d == 0.0, f"{k} NOT loaded in place (diff {d})"
    print(f"  all inherited tensors max_abs_diff == 0 (worst={worst:.2e})")

    # pos_embed 插值核对
    print("\n=== pos_embed interpolation ===")
    print(f"  source: {tuple(sd['pos_embed'].shape)}")
    print(f"  model : {tuple(model_sd['pos_embed'].shape)}")
    assert tuple(model_sd["pos_embed"].shape) == (1, 256, 192)
    # depth 之外的 block 不得残留
    resid = [k for k in model_sd if any(k.startswith(f"blocks.{j}.") for j in range(args.vit_depth, 12))]
    print(f"  residual blocks >= {args.vit_depth} in model: {len(resid)}")
    assert len(resid) == 0, f"leftover deeper blocks in model: {resid[:5]}"
    print("  pos_embed interpolated 14x14 -> 16x16 OK; no deeper-block params remain")
    print("[AUDIT] PASS")


if __name__ == "__main__":
    main()
