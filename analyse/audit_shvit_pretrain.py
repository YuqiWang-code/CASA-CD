"""SHViT-S1 checkpoint / pretrain audit（主线重构 P0，用户 6 问）。

回答：
  1. checkpoint 最外层是 model / state_dict / 还是直接权重；
  2. 是否 ImageNet pretrained（分类头 1000 类 + 总参数量 ≈6.3M 证据）；
  3. backbone 去掉分类头后参数；
  4. stem / blocks1 / blocks2 的 key 是否 100% 对齐；
  5. 改造成多尺度输出（截断 trunk）后预训练参数实际加载率；
  6. 加 CASAA 后，原 SHSA 的 q/k/v 哪些可继承、哪些必须重新初始化。

Usage (server):
    CUDA_VISIBLE_DEVICES=1 python analyse/audit_shvit_pretrain.py \
        --pretrained_path /home/yqwang/projects/CASA-CD/pretrained_weight/shvit_s1.pth
"""
import argparse
import hashlib
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "models"))

from model.shvit_s1_trunc import SHViTS1Truncated, build_shvit_s1


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_path", type=str, required=True)
    args = ap.parse_args()

    print(f"[AUDIT-SHVIT] file sha256 = {sha256_of_file(args.pretrained_path)}")

    # ---- Q1: outermost structure ----
    ckpt = torch.load(args.pretrained_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        print(f"[Q1] outermost = dict, keys = {list(ckpt.keys())}")
        if "model" in ckpt:
            sd = ckpt["model"]
            print("[Q1] state_dict source = ckpt['model'] (official format)")
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            print("[Q1] state_dict source = ckpt['state_dict']")
        else:
            sd = ckpt
            print("[Q1] state_dict source = raw dict")
    else:
        sd = ckpt
        print(f"[Q1] outermost = {type(ckpt).__name__} (raw weights)")

    # ---- Q2: ImageNet evidence ----
    head_keys = [k for k in sd if k.startswith("head.")]
    head_shapes = {k: tuple(sd[k].shape) for k in head_keys[:4]}
    n_total = sum(v.numel() for v in sd.values())
    print(f"[Q2] head keys sample = {head_shapes}")
    print(f"[Q2] checkpoint total params = {n_total:,} (official S1 ≈ 6.3M -> "
          f"{'ImageNet-1K classifier head present' if any('head.l' in k for k in head_keys) else 'NO head'})")

    # ---- Q3: backbone params without head ----
    n_backbone = sum(v.numel() for k, v in sd.items() if not k.startswith(("head.", "head_dist.")))
    print(f"[Q3] backbone (no head) params = {n_backbone:,} ({n_backbone / 1e6:.4f}M)")

    # ---- Q4/Q5: truncated trunk key alignment + load rate ----
    full = build_shvit_s1(num_classes=1000, distillation=False)
    full_sd = full.state_dict()
    trunc = SHViTS1Truncated(pretrained_path=args.pretrained_path)
    trunc_sd = trunc.state_dict()
    retained = list(trunc_sd.keys())
    groups = {"patch_embed": [], "blocks1": [], "blocks2": []}
    for k in retained:
        for g in groups:
            if k.startswith(g + "."):
                groups[g].append(k)
    mismatch = [k for k in retained if k in sd and sd[k].shape != trunc_sd[k].shape]
    missing = [k for k in retained if k not in sd]
    loaded = [k for k in retained if k in sd and sd[k].shape == trunc_sd[k].shape]
    bitwise = sum(1 for k in loaded if torch.equal(trunc_sd[k], sd[k]))
    print(f"[Q4] retained groups: patch_embed={len(groups['patch_embed'])} "
          f"blocks1={len(groups['blocks1'])} blocks2={len(groups['blocks2'])}")
    print(f"[Q4] shape mismatch = {len(mismatch)} {mismatch[:3]}")
    print(f"[Q4] missing = {len(missing)} {missing[:3]}")
    print(f"[Q5] load rate = {len(loaded)}/{len(retained)} = {len(loaded) / len(retained):.4f} "
          f"(bitwise-equal {bitwise})")
    n_trunc = trunc.param_count()
    print(f"[Q5] truncated trunk params = {n_trunc:,} ({n_trunc / 1e6:.4f}M)")

    # ---- Q6: SHSA inheritance for CASAA ----
    blocks1_mixers = [k for k in full_sd if k.startswith("blocks1.") and "mixer" in k and "m." in k]
    blocks2_shsa = [k for k in full_sd if k.startswith("blocks2.") and "mixer.m.qkv" in k]
    blocks2_shsa_proj = [k for k in full_sd if k.startswith("blocks2.") and "mixer.m.proj" in k]
    print(f"[Q6] blocks1 mixer params (should be 0, type='i') = {len(blocks1_mixers)}")
    print(f"[Q6] blocks2 SHSA qkv keys = {len(blocks2_shsa)}; sample {blocks2_shsa[:2]}")
    print(f"[Q6] blocks2 SHSA proj keys = {len(blocks2_shsa_proj)}")
    print("[Q6-CONCLUSION]")
    print("  - CASAA@1/16 (blocks1, type='i'): 原位没有 SHSA -> q/k/v 全部需要新初始化，")
    print("    只有 backbone 主干本身继承 ImageNet 预训练（zero-gate 接入，不破坏 trunk）。")
    print("  - 若做 Partial-Channel CASAA@1/32 (blocks2 SHSA): qkv(48->16+16+48) 与 proj(224->224)")
    print("    的权重/形状可直接继承 4 个 blocks2 SHSA 块；但压缩后 K=16、N=64，")
    print("    '高分辨率完整 Query' 故事变弱（本地实施注意事项 §3.2，第一版不采用）。")
    print("[AUDIT-SHVIT] DONE")


if __name__ == "__main__":
    main()
