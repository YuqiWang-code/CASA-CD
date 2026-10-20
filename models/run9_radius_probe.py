"""HSAM 半径诊断 v2：分别看 `outnorm` **之前**（`o`）与**之后**（`outnorm(o)`）的激活。

`VMambaBackbone.forward_features` 是：
    o = layer.blocks(x); x = layer.downsample(o); outs.append(outnorm{i}(o))
所以网络实际用的是 `outnorm(o)`（LayerNorm2d ⇒ ‖v‖² = C 精确成立）。
HSAM 若想拿到**有幅度信息**的特征，应当用 `o`（outnorm 之前）。

本脚本对两者都算一遍，回答：
  1. `outnorm(o)` 的范数是否真的恒定（已实测：cv ≈ 1e-6 ⇒ 是）；
  2. `o` 的范数是否有内容相关的变化（cv 够不够大）；
  3. 换成 `o` 之后，Poincaré 半径的 cv 与 1-‖p‖² 的分布是否真的散开。
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dataset.Transforms as myTransforms  # noqa: E402
import dataset.dataset as myDataLoader  # noqa: E402
from model.casa_vmamba_net_v2 import CASAVMambaSTRNetV2  # noqa: E402
from model.utils import init_seed  # noqa: E402

MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


def stats(tag, a, C):
    v = a.flatten(2).transpose(1, 2).float()      # (B,N,C)
    r = v.norm(dim=-1)
    r_cc = r / (C ** 0.5)
    th = torch.tanh(r_cc)
    p = th.unsqueeze(-1) * v / (r.unsqueeze(-1) + 1e-8)
    rad = p.norm(dim=-1)
    om = (1.0 - rad.pow(2)).clamp_min(0.0)
    return (f"  {tag:22s} ‖v‖/√C={float(r_cc.mean()):8.4f}±{float(r_cc.std()):7.4f} "
            f"(cv={float(r_cc.std() / r_cc.mean()):8.3e}) | "
            f"‖p‖={float(rad.mean()):.6f} (cv={float(rad.std() / rad.mean()):8.3e}) | "
            f"1-‖p‖²={float(om.mean()):8.4f} min={float(om.min()):8.4e} "
            f"(cv={float(om.std() / om.mean()):8.3e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--vmamba-pretrained-weight-path", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=16)
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_seed(args.seed)
    tf = myTransforms.Compose([myTransforms.Normalize(mean=MEAN6, std=STD6),
                               myTransforms.Scale(256, 256), myTransforms.ToTensor()])
    ds = myDataLoader.Dataset(file_root=args.dataset_root, list_path=args.train_list,
                              transform=tf, mode="train")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, drop_last=True)
    torch.manual_seed(args.seed)
    m = CASAVMambaSTRNetV2(vmamba_pretrained_path=args.vmamba_pretrained_weight_path,
                           hqkv=False, str_dim=192, geometry="lattice", hgsl=False,
                           inject_mode="dcr1", rep_mode="full", fs_tar=False,
                           hsam=False).float().to(dev).eval()
    enc = m.encoder

    img, _ = next(iter(dl))
    pre, post = img[:, 0:3].to(dev), img[:, 3:6].to(dev)
    with torch.no_grad():
        x = enc.patch_embed(torch.cat([pre, post], 0))
        for i, layer in enumerate(enc.layers):
            o = layer.blocks(x)
            x = layer.downsample(o)
            if i in enc.out_indices:
                C = o.shape[1]
                print(f"\n=== stage{i}  C={C} ===")
                print(stats("outnorm(o)  [网络用]", getattr(enc, f"outnorm{i}")(o), C))
                print(stats("o           [HSAM 可用]", o, C))
    print("\n判读：")
    print("  * `outnorm(o)` 的 ‖v‖/√C 恒定 ⇒ 半径恒定 ⇒ HD 与 (1-cos) 严格单调等价（**已证实**）")
    print("  * 若 `o` 的 cv 明显 > 0 且 1-‖p‖² 的 min ≪ mean ⇒ 用 outnorm 之前的激活就能恢复变半径")
    return 0


if __name__ == "__main__":
    sys.exit(main())
