#!/usr/bin/env python
"""Run2 零成本诊断（调研文档 §15.1 D1/D2）：用 Run1 已完成的 M1_FULL / A2_STR best
checkpoint 对 4 数据集 test 集做机制证据分析，不消耗任何新训练。

D1（score 置信度）：按 GT 变化率分组（==0 / (0,1%] / (1%,5%] / >5%）统计
    - abs_cos score 的 mean / p95
    - 2×2 cell 权重熵（rank 模式，与 M1 训练口径一致；均匀=ln4）
    - β、|c_ca - c_avg| RMS
    若 LEVIR 零变化组的 abs 分数低但 cell 熵仍显著低于 ln4（rank 强制不均匀），
    即直接支持 CP-CAACP 的机制叙事。
D2（boundary / object size）：M1 与 A2 的 0.5 二值 mask 在
    - boundary ±2px / ±4px 环带内像素级 F1
    - GT 连通域面积分组（small/medium/large）F1（scipy.ndimage.label，可用时）
    若 LEVIR/SYSU 的损失集中在 boundary/small 成分，即支持 FRH 的机制叙事。

用法（服务器，Run2 训练结束后；需 casacd env + LD_LIBRARY_PATH）：
    cd models && python ../analyse/run2_zero_cost_diag.py --gpu_id 0
输出：打印表格 + 写 docs/temporary/run2_zero_cost_diag.json（本地仓库路径写不了时仅打印）。
"""
import os
import sys
import json
import glob
import argparse

_MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

import torch
import torch.nn.functional as F
import numpy as np

import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms
from model.casa_tvim_str_net import CASATViMSTRNet
from model.layers.caacp_ss2d import change_score_cosine_2d, rank_normalize_2d

MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
CKPT_ROOT = "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run1"
DATA_ROOT = "/share_datasets/CD"
PRETRAIN = "/home/yqwang/projects/CASA-CD/pretrained_weight/tinyvim_s_1000e.pth"
LN4 = 1.3862943611198906

RATIO_BINS = {"zero": (0.0, 0.0), "0-1%": (0.0, 0.01), "1-5%": (0.01, 0.05), ">5%": (0.05, 1.01)}


def pick_best(ckpt_dir):
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "best_F1=*.pth")))
    if not cands:
        raise FileNotFoundError(f"no best_F1=*.pth in {ckpt_dir}")
    return cands[-1]


def make_loader(dataset_root, batch_size):
    transform = myTransforms.Compose([
        myTransforms.Normalize(mean=MEAN, std=STD),
        myTransforms.Scale(256, 256),
        myTransforms.ToTensor(),
    ])
    ds = myDataLoader.Dataset(file_root=dataset_root,
                              list_path=os.path.join(dataset_root, "list", "test.txt"),
                              transform=transform)
    return torch.utils.data.DataLoader(ds, shuffle=False, batch_size=batch_size,
                                       num_workers=4, pin_memory=True)


def cell_weight_entropy(score, eps=1e-6):
    """rank 模式 cell 权重熵（与 M1 训练口径 change_weighted_pool2x2 一致），per-image 标量。"""
    B, H, W = score.shape
    s = score.view(B, 1, H // 2, 2, W // 2, 2)
    w = s + eps
    w = w / w.sum(dim=(3, 5), keepdim=True)
    ent = -(w * (w + 1e-12).log()).sum(dim=(3, 5)).mean(dim=(1, 2, 3))  # per-image 均值
    return ent


def boundary_band(label, px):
    """GT 变化区域 ±px 环带 mask（单张 256²，0/1）。label: (1,1,256,256) float。"""
    lab = label.float().view(1, 1, 256, 256)
    k = 2 * px + 1
    w = torch.ones(1, 1, k, k, device=lab.device)
    dil = (F.conv2d(lab, w, padding=px) > 0).float()          # 膨胀
    ero = (F.conv2d(lab, w, padding=px) >= (k * k)).float()   # 腐蚀
    return (dil - ero).view(256, 256).bool()


def band_metrics(pred_b, gt_b, band):
    """band 内像素级 P/R/F1。"""
    b = band
    tp = (pred_b & gt_b & b).sum().item()
    fp = (pred_b & (~gt_b) & b).sum().item()
    fn = ((~pred_b) & gt_b & b).sum().item()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return p, r, f1


def component_f1(pred_np, gt_np):
    """按 GT 连通域面积分组（small<256px / medium<1024 / large）的组件级 F1。"""
    try:
        from scipy import ndimage
    except ImportError:
        return None
    lab, n = ndimage.label(gt_np > 0)
    out = {}
    for name, lo, hi in (("small", 0, 256), ("medium", 256, 1024), ("large", 1024, 10 ** 9)):
        sizes = ndimage.sum(gt_np > 0, lab, range(1, n + 1))
        sel = set(np.where((sizes >= lo) & (sizes < hi))[0] + 1)
        m = np.isin(lab, list(sel)) if sel else np.zeros_like(lab, dtype=bool)
        if m.sum() == 0:
            out[name] = None
            continue
        tp = ((pred_np > 0) & (gt_np > 0) & m).sum()
        fp = ((pred_np > 0) & (gt_np == 0) & m).sum()
        fn = ((pred_np == 0) & (gt_np > 0) & m).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        out[name] = 2 * p * r / max(p + r, 1e-12)
    return out


def run_d1(model, loader, device):
    """D1：score 置信度分组统计。"""
    stats = {b: {"n": 0, "abs_mean": [], "abs_p95": [], "cell_ent": []} for b in RATIO_BINS}
    beta = float(model.encoder.caacp_op.beta.detach().cpu().item())
    delta_rms = 0.0
    n_delta = 0
    with torch.no_grad():
        for img, label in loader:
            pre = img[:, 0:3].to(device).float()
            post = img[:, 3:6].to(device).float()
            gt = (label.to(device) > 0.5).float()   # Normalize 已把 label 二值化为 0/1
            x2b = torch.cat([pre, post], dim=0)
            _, _, x = model.encoder.forward_stem_s2_prefix(x2b)
            xa, xb = x.chunk(2, dim=0)
            s_abs = change_score_cosine_2d(xa, xb)
            s_rank = rank_normalize_2d(s_abs)
            ent = cell_weight_entropy(s_rank).cpu()          # per-image
            am = s_abs.flatten(1).mean(dim=1).cpu()
            ap95 = torch.quantile(s_abs.flatten(1), 0.95, dim=1).cpu()
            ratio = gt.flatten(1).mean(dim=1).cpu()
            # 完整 forward 一次，让 CAACP 实际执行以记录 context delta
            _ = model(pre, post)
            op = model.encoder.caacp_op
            delta_rms += op._score_delta ** 2
            n_delta += 1
            for i in range(len(ratio)):
                for b, (lo, hi) in RATIO_BINS.items():
                    if lo < ratio[i].item() <= hi or (b == "zero" and ratio[i].item() == 0.0):
                        stats[b]["n"] += 1
                        stats[b]["abs_mean"].append(am[i].item())
                        stats[b]["abs_p95"].append(ap95[i].item())
                        stats[b]["cell_ent"].append(ent[i].item())
                        break
    out = {"beta": beta, "context_delta_rms": (delta_rms / max(n_delta, 1)) ** 0.5}
    for b in RATIO_BINS:
        s = stats[b]
        out[b] = {
            "n": s["n"],
            "abs_mean": float(np.mean(s["abs_mean"])) if s["n"] else None,
            "abs_p95": float(np.mean(s["abs_p95"])) if s["n"] else None,
            "cell_entropy": float(np.mean(s["cell_ent"])) if s["n"] else None,
            "cell_entropy_gap_to_ln4": (float(np.mean(s["cell_ent"])) - LN4) if s["n"] else None,
        }
    return out


def run_d2(model, loader, device):
    """D2：boundary 环带与组件面积分组 F1（逐样本处理，band/连通域均为单张口径）。"""
    band_px = {2: {"tp": 0, "fp": 0, "fn": 0}, 4: {"tp": 0, "fp": 0, "fn": 0}}
    comp = {c: [] for c in ("small", "medium", "large")}
    with torch.no_grad():
        for img, label in loader:
            pre = img[:, 0:3].to(device).float()
            post = img[:, 3:6].to(device).float()
            gt = (label.to(device) > 0.5)   # Normalize 已把 label 二值化为 0/1
            pred = model(pre, post) > 0.5
            for i in range(len(gt)):
                gt_i = gt[i:i + 1]
                pred_i = pred[i:i + 1]
                for px in band_px:
                    band = boundary_band(gt_i, px)
                    tp = (pred_i & gt_i & band).sum().item()
                    fp = (pred_i & (~gt_i) & band).sum().item()
                    fn = ((~pred_i) & gt_i & band).sum().item()
                    band_px[px]["tp"] += tp
                    band_px[px]["fp"] += fp
                    band_px[px]["fn"] += fn
                cf = component_f1(pred_i.squeeze().cpu().numpy(), gt_i.squeeze().cpu().numpy())
                if cf:
                    for c in comp:
                        if cf[c] is not None:
                            comp[c].append(cf[c])
    out = {}
    for px, m in band_px.items():
        p = m["tp"] / max(m["tp"] + m["fp"], 1)
        r = m["tp"] / max(m["tp"] + m["fn"], 1)
        out[f"band{px}"] = {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-12)}
    for c in comp:
        out[f"comp_{c}"] = float(np.mean(comp[c])) if comp[c] else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}"
    torch.backends.cudnn.benchmark = True

    results = {}
    for ds in DATASETS:
        root = os.path.join(DATA_ROOT, ds)
        loader = make_loader(root, args.batch_size)
        for variant, cfg in (("M1_FULL", dict(caacp=True)), ("A2_STR", dict(caacp=False))):
            ckpt = pick_best(os.path.join(CKPT_ROOT, variant, ds))
            model = CASATViMSTRNet(PRETRAIN, caacp=cfg["caacp"], rep_mode="full").to(device).eval()
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
            tag = f"{ds}/{variant}"
            d1 = run_d1(model, loader, device) if cfg["caacp"] else None
            d2 = run_d2(model, loader, device)
            results[tag] = {"d1": d1, "d2": d2}
            print(f"\n===== {tag} =====", flush=True)
            if d1:
                print(f"  beta={d1['beta']:.4e} context_delta_rms={d1['context_delta_rms']:.4e}")
                for b in RATIO_BINS:
                    s = d1[b]
                    print(f"  [{b:>5}] n={s['n']:>4} abs_mean={s['abs_mean']} abs_p95={s['abs_p95']} "
                          f"cell_ent={s['cell_entropy']} gap_to_ln4={s['cell_entropy_gap_to_ln4']}")
            print(f"  D2: band2 F1={d2['band2']['F1']:.4f} band4 F1={d2['band4']['F1']:.4f} "
                  f"comp_small={d2['comp_small']} comp_medium={d2['comp_medium']} comp_large={d2['comp_large']}")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "docs", "temporary", "run2_zero_cost_diag.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {out_path}")
    except OSError as e:
        print(f"\n[json write skipped: {e}]")


if __name__ == "__main__":
    main()
