"""Run4 U1：按组件拆分模型参数预算。

用法（服务器）：
    python analyse/param_breakdown.py \
        --pretrained_weight_path /home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth \
        [--vit_depth 12|4]

口径（决策文档 §36-37）：
  - EFFECTIVE = 实际出现在 inference forward 图中的全部 learnable 参数（冻结参数照计）；
  - ResNet18 的 layer4/avgpool/fc 不参与 forward → 计 dead，不计 effective；
  - 训练期辅助模块若推理删除 → 不计 deployed，需注明。
"""
import os
import sys
import argparse

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "models"))

from model.trainer import Trainer

DEAD_RESNET = ("encoder.resnet.layer4", "encoder.resnet.fc", "encoder.resnet.avgpool")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--vit_depth", type=int, default=12)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu_id)
    model = Trainer("tiny", pretrained_path=args.pretrained_weight_path,
                    resnet_pretrained=False, mode="baseline",
                    vit_depth=args.vit_depth).float().cuda()

    def count(mod):
        return sum(p.numel() for p in mod.parameters())

    print(f"=== param breakdown (vit_depth={args.vit_depth}) ===")
    vit = model.encoder.vit
    print("--- ViT ---")
    print(f"  patch_embed: {count(vit.patch_embed):>10,}")
    print(f"  pos_embed:   {vit.pos_embed.numel():>10,}")
    print(f"  mask_token:  {vit.mask_token.numel():>10,}")
    for i, blk in enumerate(vit.blocks):
        print(f"  block{i}:      {count(blk):>10,}")
    print(f"  norm:        {count(vit.norm):>10,}")
    vit_total = sum(p.numel() for p in vit.parameters())
    print(f"  ViT total:   {vit_total:>10,}")

    print("--- Detail (ResNet18) ---")
    rn = model.encoder.resnet
    parts = [("stem(conv1/bn1)", None), ("layer1", rn.layer1), ("layer2", rn.layer2),
             ("layer3", rn.layer3), ("layer4(dead)", rn.layer4),
             ("avgpool+fc(dead)", None)]
    stem = count(rn.conv1) + count(rn.bn1)
    print(f"  stem(conv1/bn1): {stem:>10,}")
    for name, m in [("layer1", rn.layer1), ("layer2", rn.layer2), ("layer3", rn.layer3),
                    ("layer4(dead)", rn.layer4)]:
        print(f"  {name:16s}: {count(m):>10,}")
    dead = count(rn.layer4) + count(rn.avgpool) + count(rn.fc)
    print(f"  avgpool+fc(dead): {count(rn.avgpool) + count(rn.fc):>10,}")
    resnet_total = sum(p.numel() for p in rn.parameters())
    print(f"  ResNet total: {resnet_total:>10,}  (dead {dead:>10,})")

    print("--- Decoder ---")
    dec = model.decoder
    fi = count(dec.structure_enhance)
    print(f"  FeatureInjector:      {fi:>10,}")
    for name in ("up_c5", "up_c4", "up_c3", "classfier"):
        print(f"  {name:16s}: {count(getattr(dec, name)):>10,}")
    mlp = sum(count(m) for m in dec.mlp)
    print(f"  difference MLP x4:    {mlp:>10,}")
    dec_total = sum(p.numel() for p in dec.parameters())
    print(f"  Decoder total:        {dec_total:>10,}")

    total = sum(p.numel() for p in model.parameters())
    effective = total - dead
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("---")
    print(f"TOTAL:       {total:>10,}")
    print(f"EFFECTIVE:   {effective:>10,}  ({effective / 1e6:.4f} M)")
    print(f"TRAINABLE:   {trainable:>10,}  ({trainable / 1e6:.4f} M)")
    print(f"dead params: {dead:>10,}  ({dead / 1e6:.4f} M)")


if __name__ == "__main__":
    main()
