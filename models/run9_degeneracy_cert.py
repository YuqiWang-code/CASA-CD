"""非退化证书：HSAM 的双曲距离矩阵到底是不是"余弦距离的单调变换"？

对同一批真实特征，分别用
  * `post_norm`：`outnorm(o)`（网络实际用的特征，‖v‖²=C 恒定）
  * `pre_norm` ：`o`（outnorm 之前）
  * `pre_norm + 全局 RMS 缩放`（本模块的推荐修法）
计算 HD 矩阵与 (1−cos) 矩阵的 Pearson 相关。
  corr ≈ 1.0000 ⇒ HD 与余弦距离**等价**，双曲性是装饰（退化）
  corr 明显 < 1   ⇒ HD 携带了角度之外的信息（半径），双曲性有实质作用

还报告：半径 cv、1-‖p‖² 的分布、HD 与 (1-cos) 的**秩相关**（Spearman，用排序近似）。
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
from model.layers.hsam import expmap0, poincare_pairwise_distance  # noqa: E402
from model.utils import init_seed  # noqa: E402

MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


def pearson(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def spearman(a, b, n_max=200_000, seed=0):
    """秩相关：**这才是判断"是否只是单调变换"的正确统计量**。

    秩相关 = 1 ⇔ 两个量严格单调等价（= 只差一个重参数化，没有新信息）。
    Pearson 会被"单调但非线性"的变换本身拉低，不能用来判退化。
    """
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = a.numel()
    if n > n_max:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:n_max]
        a, b = a[idx], b[idx]

    def rank(x):
        order = torch.argsort(x)
        r = torch.empty_like(order, dtype=torch.float32)
        r[order] = torch.arange(x.numel(), dtype=torch.float32)
        return r

    return pearson(rank(a), rank(b))


def report(tag, feat, C, global_rms_scale=False):
    v = feat.flatten(2).transpose(1, 2).float()      # (B,N,C)
    if global_rms_scale:
        s = v.pow(2).mean().clamp_min(0.0).sqrt()
        v = v / s.clamp_min(1e-12)                   # 只除以一个**标量**：保序、保持相对幅度
    v = v[:2]                                        # 只取 2 个样本算矩阵，够用且省内存
    p = expmap0(v, dim_scale=(C ** 0.5))
    rad = p.norm(dim=-1)
    hd = poincare_pairwise_distance(p, p)
    # 余弦距离
    vn = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos = vn @ vn.transpose(1, 2)
    one_minus_cos = (1.0 - cos)
    one_minus_p2 = (1.0 - rad.pow(2))
    # 相关只在非对角元上算（对角恒 0，会人为抬高相关）
    B, N, _ = hd.shape
    mask = ~torch.eye(N, dtype=torch.bool, device=hd.device).unsqueeze(0).expand(B, N, N)
    r_pear = pearson(hd[mask], one_minus_cos[mask])
    r_rank = spearman(hd[mask].cpu(), one_minus_cos[mask].cpu())
    print(f"  {tag:32s} C={C:4d} r_cv={float(rad.std() / rad.mean()):8.3e} "
          f"(1-r²)cv={float(one_minus_p2.std() / one_minus_p2.mean()):8.3e} "
          f"min(1-r²)={float(one_minus_p2.min()):8.3e} | "
          f"rank_corr={r_rank:.6f} pearson={r_pear:.4f}")
    return r_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--vmamba-pretrained-weight-path", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=16)
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_seed(args.seed)
    tf = myTransforms.Compose([myTransforms.Normalize(mean=MEAN6, std=STD6),
                               myTransforms.Scale(256, 256), myTransforms.ToTensor()])
    ds = myDataLoader.Dataset(file_root=args.dataset_root, list_path=args.train_list,
                              transform=tf, mode="train")
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False,
                                     num_workers=args.num_workers, drop_last=True)
    torch.manual_seed(args.seed)
    m = CASAVMambaSTRNetV2(vmamba_pretrained_path=args.vmamba_pretrained_weight_path,
                           hqkv=False, str_dim=192, geometry="lattice", hgsl=False,
                           inject_mode="dcr1", rep_mode="full", fs_tar=False,
                           hsam=False).float().to(dev).eval()
    img, _ = next(iter(dl))
    x = torch.cat([img[:, 0:3].to(dev), img[:, 3:6].to(dev)], 0)
    with torch.no_grad():
        post = m.encoder.forward_features(x)
        pre = m.encoder.forward_pre_norm(x)

    print("=== 非退化证书：HD 与 (1-cos) 的相关性 ===")
    print("（corr≈1 说明双曲距离只是余弦距离的单调变换 ⇒ 双曲性无实质作用）\n")
    for i in (2, 3):
        C = post[i].shape[1]
        print(f"--- stage{i} ---")
        report("A post_norm（网络实际用的）", post[i], C)
        report("B pre_norm（outnorm 之前）", pre[i], C)
        report("C pre_norm + 全局RMS缩放", pre[i], C, global_rms_scale=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
