"""Epoch-0 恒等性核对（三模式，用于证明 `CTRL_R8` 是本探索线的有效同期对照）。

  python epoch0_check.py --mode dump --pkg-root <models|models_v1> --arch <a> --out <pt>
  python epoch0_check.py --mode cmp  --a <pt> --b <pt>

`dump` 用**固定种子 + 固定合成 batch**（跨进程可复现）构造网络并保存前向输出；
`cmp` 逐位比较两份输出（torch.equal + max|Δ|）。

- `--pkg-root /home/yqwang/projects/CASA-CD/models`   → Run8 代码（CTRL_R8 架构，hqkv=0）
- `--pkg-root /home/yqwang/projects/CASA-CD/models_v1` → 探索线代码（casa_vmamba_stgeo）
两者输出**逐位相同** ⇒ epoch-0 与本模块关/开无关，CTRL_R8 可直接作对照。
"""
import argparse
import os
import sys

import torch


def build_fixed_batch(size=256, batch=4, seed=1234, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    pre = torch.randn(batch, 3, size, size, generator=g)
    post = torch.randn(batch, 3, size, size, generator=g)
    return pre.to(device), post.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dump", "cmp"])
    ap.add_argument("--pkg-root", default=None)
    ap.add_argument("--arch", default="stgeo")
    ap.add_argument("--stgeo", default="none")
    ap.add_argument("--stgeo-pool", dest="stgeo_pool", default="max_shape")
    ap.add_argument("--stgeo-xscale", dest="stgeo_xscale", type=int, default=0)
    ap.add_argument("--vmamba", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--seed", type=int, default=16)
    args = ap.parse_args()

    if args.mode == "cmp":
        ta = torch.load(args.a, map_location="cpu")
        tb = torch.load(args.b, map_location="cpu")
        eq = bool(torch.equal(ta, tb))
        md = float((ta - tb).abs().max())
        print(f"[EPOCH0-CMP] {os.path.basename(args.a)} vs {os.path.basename(args.b)} "
              f"torch.equal={eq} max|d|={md:.3e}")
        return 0 if eq else 1

    root = os.path.abspath(args.pkg_root)
    sys.path.insert(0, root)
    torch.manual_seed(args.seed)
    if args.arch == "run8":
        from model.casa_vmamba_net import CASAVMambaSTRNet
        model = CASAVMambaSTRNet(vmamba_pretrained_path=args.vmamba, hqkv=False,
                                 hgsl=False, geometry="hyperbolic",
                                 inject_mode="none").float().cuda().eval()
    else:
        from model.casa_vmamba_stgeo_net import CASAVMambaSTGeoNet
        model = CASAVMambaSTGeoNet(vmamba_pretrained_path=args.vmamba,
                                   stgeo=args.stgeo, stgeo_pool=args.stgeo_pool,
                                   stgeo_xscale=bool(args.stgeo_xscale)
                                   ).float().cuda().eval()
    st = model.pretrain_load_stats() or {}
    pre, post = build_fixed_batch(device="cuda")
    with torch.no_grad():
        out = model(pre, post)
    torch.save(out.cpu(), args.out)
    print(f"[EPOCH0-DUMP] arch={args.arch} stgeo={args.stgeo} "
          f"pool={args.stgeo_pool} xscale={args.stgeo_xscale} out={args.out} "
          f"shape={tuple(out.shape)} sum={float(out.sum()):.6f} "
          f"pretrain_retained={st.get('retained')}/{st.get('pretrained_keys')} "
          f"worst_diff={st.get('worst_diff')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
