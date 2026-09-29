"""CASAA Router Audit：用已训练 checkpoint 在 test split 上做无训练路由–GT 对齐诊断。

对应 docs/temporary/CASA-CD_CASAA_Run1复盘与下一步实验决策.md §4.2：
用现有 baseline / A2 checkpoint（state dict 与 CASAA 架构零参数兼容），在 LEVIR/SYSU
test 上记录 blocks 8-11 的 cosine change score 与 GT patch occupancy，输出：

  - Spearman(s_cos, g)            逐 block
  - ROC-AUC / PR-AUC（y = g > 0） 逐 block
  - Top32 changed-pixel coverage（TopK 变化像素覆盖）
  - Top32 precision（TopK 中有真实变化的 patch 比例）
  - 每图真实 change-patch 数分布 P10/P50/P90/mean
  - TopK score gap（第 32 与第 33 个 score 的差）
  - change bank / bg prototype 的 token norm、attention mass、cluster size

用法（服务器）：
    python analyse/casaa_router_diagnostic.py \
        --dataset LEVIR-CD-256 \
        --dataset_root /share_datasets/CD/LEVIR-CD-256 \
        --pretrained_weight_path /home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth \
        --ckpt /share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run1/A2_CASAA/LEVIR-CD-256/best_F1=0.9186.pth \
        --ckpt_tag A2_CASAA --gpu_id 0 [--limit 8]

说明：模型按 mode=casaa/router=change 实例化（与 A2 同架构、零参数差异），
因此加载 baseline checkpoint 也能对「未经 router 训练的特征」做同一套诊断。
"""
import os
import sys
import argparse
import math

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "models"))

from model.trainer import Trainer
import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms

CASAA_LAYERS = [8, 9, 10, 11]
MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


def rankdata(a):
    """numpy rank（升序，1-based）；分数为连续浮点，tie 极少，直接用 argsort 双层。"""
    return np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(np.float64) + 1


def spearman(s, g):
    rs, rg = rankdata(s), rankdata(g)
    if rs.std() == 0 or rg.std() == 0:
        return float("nan")
    return float(np.corrcoef(rs, rg)[0, 1])


def roc_auc(s, y):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order].astype(np.float64)
    n_pos = ys.sum()
    n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(ys) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ys) / n_neg])
    return float(np.trapz(tpr, fpr))


def pr_auc(s, y):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order].astype(np.float64)
    n_pos = ys.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / n_pos
    rec = np.concatenate([[0.0], rec])
    prec = np.concatenate([[prec[0]], prec])
    return float(np.trapz(prec, rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="best_F1=*.pth state dict")
    ap.add_argument("--ckpt_tag", type=str, required=True, help="e.g. A2_CASAA / baseline")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 batch（0=全部）")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True

    model = Trainer("tiny", pretrained_path=args.pretrained_weight_path,
                    resnet_pretrained=False, mode="casaa",
                    casaa_layers=CASAA_LAYERS, casaa_keep_ratio=0.25,
                    casaa_change_share=0.5, casaa_router="change").float().cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    print(f"[AUDIT] loaded {args.ckpt} ({args.ckpt_tag})")

    val_transform = myTransforms.Compose([
        myTransforms.Normalize(mean=MEAN, std=STD),
        myTransforms.Scale(256, 256),
        myTransforms.ToTensor(),
    ])
    ds_root = args.dataset_root
    test_data = myDataLoader.Dataset(
        file_root=ds_root, list_path=os.path.join(ds_root, "list", "test.txt"),
        transform=val_transform)
    loader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True)

    # 收集：per-block score / occupancy / bank stats
    s_list = {i: [] for i in CASAA_LAYERS}
    g_list = []
    bank_stats = {i: {"c_norm": [], "bg_norm": [], "m_change": [], "m_bg": [],
                      "clusters": []} for i in CASAA_LAYERS}

    with torch.no_grad():
        for bi, (img, target) in enumerate(loader):
            pre = img[:, 0:3].cuda().float()
            post = img[:, 3:6].cuda().float()
            g = torch.nn.functional.avg_pool2d(target.cuda().float(), 16).view(target.shape[0], -1)
            g_list.append(g.cpu().numpy())
            model(pre, post)
            for i in CASAA_LAYERS:
                r = model.encoder.vit.blocks[i].attn._routing
                s_list[i].append(r["s"].cpu().numpy())
                st = bank_stats[i]
                st["c_norm"].append(r["c_norm_mean"])
                st["bg_norm"].append(r["bg_norm_mean"])
                st["m_change"].append(r["attn_mass_change"])
                st["m_bg"].append(r["attn_mass_bg"])
                st["clusters"].append(r["cluster_size"].cpu().numpy())
            if args.limit and bi + 1 >= args.limit:
                break

    g_all = np.concatenate(g_list, axis=0)                       # (n_img, 256)
    print(f"[AUDIT] images={g_all.shape[0]}  (dataset={args.dataset}, ckpt={args.ckpt_tag})")
    y_all = (g_all > 0)

    # 每图真实 change-patch 数分布
    n_changed = y_all.sum(axis=1)
    print(f"[GT] changed-patch count per image: P10={np.percentile(n_changed, 10):.1f} "
          f"P50={np.percentile(n_changed, 50):.1f} P90={np.percentile(n_changed, 90):.1f} "
          f"mean={n_changed.mean():.2f}  zero-change-img={float((n_changed == 0).mean()) * 100:.1f}%")
    print(f"[GT] mean patch occupancy (changed px ratio) = {g_all.mean() * 100:.2f}%")

    print(f"\n=== per-block score quality ({args.ckpt_tag}) ===")
    print("block | Spearman | ROC-AUC | PR-AUC | Top32 coverage | Top32 precision | TopK gap | c_norm | bg_norm | attn->change | attn->bg | cluster mean/max/min")
    for i in CASAA_LAYERS:
        s = np.concatenate(s_list[i], axis=0).reshape(-1)         # (n_img*256,)
        g = g_all.reshape(-1)
        y = y_all.reshape(-1)
        sp = spearman(s, g)
        roc = roc_auc(s, y)
        pr = pr_auc(s, y)

        # per-image TopK 统计（与训练时 Kc=32 一致）
        S = np.concatenate(s_list[i], axis=0)                     # (n_img, 256)
        G = g_all
        top_idx = np.argsort(-S, axis=1, kind="mergesort")[:, :32]
        cov = np.take_along_axis(G, top_idx, axis=1).sum(axis=1)
        denom = G.sum(axis=1)
        cov_mean = float((cov[denom > 0] / denom[denom > 0]).mean())
        prec_mean = float((np.take_along_axis(G, top_idx, axis=1) > 0).mean())
        gap = float((np.sort(-S, axis=1)[:, 31] - np.sort(-S, axis=1)[:, 32]).mean())

        st = bank_stats[i]
        cl = np.concatenate(st["clusters"], axis=0)               # (n_img, Kb)
        c_norm = float(np.nanmean(st["c_norm"]))
        bg_norm = float(np.nanmean(st["bg_norm"]))
        m_c = float(np.nanmean(st["m_change"]))
        m_b = float(np.nanmean(st["m_bg"]))
        print(f"  {i:2d}   | {sp:+.4f}  | {roc:.4f}  | {pr:.4f}  | {cov_mean:.4f}        | {prec_mean:.4f}         | {gap:.4f}  | {c_norm:.3f} | {bg_norm:.3f} | {m_c:.4f}     | {m_b:.4f} | {cl.mean():.1f}/{cl.max():.0f}/{cl.min():.0f}")
    print("[AUDIT] DONE")


if __name__ == "__main__":
    main()
