"""CASAA Router Audit：用已训练 checkpoint 在 test split 上做无训练路由–GT 对齐诊断。

对应决策文档（CASA-CD_CASAA-v2_A4_Run3审查与主线二启动方案.md §5）：
在 Run2 A1_SAA_FROZEN checkpoint 上同时测三路 score：

  - V：ViT late-stage cosine（blocks 8-11，来自 _routing["s"]）
  - D：detail-only（已有 detail branch 1/8 = resnet.layer3 输出 32×32
       → AvgPool2d(2) → 16×16 → 1−cos(d̄1,d̄2) → per-image rank 归一化）
  - F：fused = 0.5*rank(V) + 0.5*D（D 已 rank 归一化）

输出指标（对 V/F 逐 block，对 D 一次）：
  Spearman / ROC-AUC / PR-AUC（y = GT occupancy > 0）/ Top32 changed-pixel coverage /
  Top32 precision / K32-K33 gap；Top32 Jaccard(V,D / V,F / D,F)；detail 零向量率；
  score tie 率；零变化图像的 score 统计；按真实变化 patch 数分桶的 coverage。

用法（服务器）：
    python analyse/casaa_router_diagnostic.py \
        --dataset SYSU-CD-256 --dataset_root /share_datasets/CD/SYSU-CD-256 \
        --pretrained_weight_path .../deit_tiny_patch16_224-a1311bcf.pth \
        --ckpt /share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run2/A1_SAA_FROZEN/SYSU-CD-256/best_F1=0.8204.pth \
        --ckpt_tag A1_SAA_FROZEN --gpu_id 0 [--limit 8] [--out /path/to/router_audit.txt]
"""
import os
import sys
import argparse

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


def rank_normalize_np(s):
    """per-image ordinal rank / (N-1)（numpy，与 casaa.rank_normalize_per_image 等价）。"""
    N = s.shape[1]
    order = np.argsort(s, axis=1, kind="mergesort")          # 升序
    base = np.arange(N, dtype=np.float64)[None, :]
    out = np.empty_like(s, dtype=np.float64)
    np.put_along_axis(out, order, base, axis=1)
    return out / max(N - 1, 1)


def rankdata(a):
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
    return float(np.trapezoid(tpr, fpr))


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
    return float(np.trapezoid(prec, rec))


def topk_jaccard(idx_a, idx_b):
    """Top32 集合的逐图 Jaccard 均值。idx_*: (n_img, K)"""
    a = set()
    inter, union = 0.0, 0.0
    n = 0
    for i in range(idx_a.shape[0]):
        sa = set(idx_a[i].tolist())
        sb = set(idx_b[i].tolist())
        inter += len(sa & sb)
        union += len(sa | sb)
        n += 1
    return inter / union if union else float("nan")


def score_stats(S, G, y):
    """对 (n_img, N) score 矩阵输出各项 ranking 指标 dict。"""
    s = S.reshape(-1)
    g = G.reshape(-1)
    yy = y.reshape(-1)
    top_idx = np.argsort(-S, axis=1, kind="mergesort")[:, :32]
    topG = np.take_along_axis(G, top_idx, axis=1)
    denom = G.sum(axis=1)
    cov = topG.sum(axis=1)
    cov_mean = float((cov[denom > 0] / denom[denom > 0]).mean()) if (denom > 0).any() else float("nan")
    prec_mean = float((topG > 0).mean())
    gap = float((np.sort(-S, axis=1, kind="mergesort")[:, 31] - np.sort(-S, axis=1, kind="mergesort")[:, 32]).mean())
    tie = float(len(np.unique(np.round(s, 6))) / len(s))
    return {
        "spearman": spearman(s, g),
        "roc": roc_auc(s, yy),
        "pr": pr_auc(s, yy),
        "coverage": cov_mean,
        "precision": prec_mean,
        "gap": gap,
        "tie_ratio": tie,
        "top_idx": top_idx,
    }


def stratified_coverage(S, G, n_changed):
    buckets = {"0": n_changed == 0, "1-32": (n_changed >= 1) & (n_changed <= 32),
               "33-128": (n_changed >= 33) & (n_changed <= 128), "129+": n_changed > 128}
    out = {}
    for name, mask in buckets.items():
        if not mask.any():
            out[name] = float("nan")
            continue
        Gb = G[mask]
        Sb = S[mask]
        top_idx = np.argsort(-Sb, axis=1, kind="mergesort")[:, :32]
        topG = np.take_along_axis(Gb, top_idx, axis=1)
        denom = Gb.sum(axis=1)
        cov = topG.sum(axis=1)
        out[name] = float((cov[denom > 0] / denom[denom > 0]).mean()) if (denom > 0).any() else float("nan")
    return out


def zero_change_stats(S, n_changed):
    mask = n_changed == 0
    if not mask.any():
        return {"max": float("nan"), "top32_mean": float("nan"), "gap": float("nan")}
    Ss = S[mask]
    top_idx = np.argsort(-Ss, axis=1, kind="mergesort")[:, :32]
    top32_mean = float(np.take_along_axis(Ss, top_idx, axis=1).mean())
    gap = float((np.sort(-Ss, axis=1, kind="mergesort")[:, 31] - np.sort(-Ss, axis=1, kind="mergesort")[:, 32]).mean())
    return {"max": float(Ss.max()), "top32_mean": top32_mean, "gap": gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="best_F1=*.pth state dict")
    ap.add_argument("--ckpt_tag", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 batch（0=全部）")
    ap.add_argument("--out", type=str, default="", help="额外写入的报告文件路径")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True

    # 模型按 router='change' 实例化（V 分数从 _routing 收集）；detail 分数单独计算。
    # 冻结 ViT 下任一 frozen checkpoint 的 ViT 权重相同，A1 为推荐的 primary。
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

    v_list = {i: [] for i in CASAA_LAYERS}
    d_list = []
    g_list = []
    d_norm_list = []   # detail pooled token norm（查零向量）

    with torch.no_grad():
        for bi, (img, target) in enumerate(loader):
            pre = img[:, 0:3].cuda().float()
            post = img[:, 3:6].cuda().float()
            g = torch.nn.functional.avg_pool2d(target.cuda().float(), 16).view(target.shape[0], -1)
            g_list.append(g.cpu().numpy())
            model(pre, post)
            for i in CASAA_LAYERS:
                v_list[i].append(model.encoder.vit.blocks[i].attn._routing["s"].cpu().numpy())
            # detail-only score（复用模型自身的 detail branch，eval 无 dropout）
            c1 = model.encoder.detail_capture(pre)
            c2 = model.encoder.detail_capture(post)
            d_pool1 = torch.nn.functional.avg_pool2d(c1[2], 2)
            d_pool2 = torch.nn.functional.avg_pool2d(c2[2], 2)
            d_norm_list.append(d_pool1.norm(dim=1).cpu().numpy().reshape(d_pool1.shape[0], -1))
            sd_raw = 1.0 - torch.nn.functional.cosine_similarity(
                d_pool1.flatten(2).transpose(1, 2), d_pool2.flatten(2).transpose(1, 2),
                dim=-1, eps=1e-8)
            d_list.append(rank_normalize_np(sd_raw.cpu().numpy()))
            if args.limit and bi + 1 >= args.limit:
                break

    G = np.concatenate(g_list, axis=0)                       # (n_img, 256)
    D = np.concatenate(d_list, axis=0)                       # (n_img, 256) rank-normalized
    n_changed = (G > 0).sum(axis=1)
    y = (G > 0)
    d_norm = np.concatenate(d_norm_list, axis=0).reshape(-1)
    n_img = G.shape[0]
    print(f"[AUDIT] images={n_img}  (dataset={args.dataset}, ckpt={args.ckpt_tag})")

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"[GT] changed-patch count per image: P10={np.percentile(n_changed, 10):.1f} "
         f"P50={np.percentile(n_changed, 50):.1f} P90={np.percentile(n_changed, 90):.1f} "
         f"mean={n_changed.mean():.2f}  zero-change-img={(n_changed == 0).mean() * 100:.1f}%")
    emit(f"[GT] mean patch occupancy (changed px ratio) = {G.mean() * 100:.2f}%")
    emit(f"[DETAIL] pooled-token zero-norm rate (<1e-6) = {(d_norm < 1e-6).mean() * 100:.3f}%")

    d_stats = score_stats(D, G, y)
    emit(f"\n=== D detail-only ===  Spearman={d_stats['spearman']:+.4f} ROC={d_stats['roc']:.4f} "
         f"PR={d_stats['pr']:.4f} coverage={d_stats['coverage']:.4f} precision={d_stats['precision']:.4f} "
         f"gap={d_stats['gap']:.4f} tie={d_stats['tie_ratio']:.4f}")

    emit("\n=== per-block V (ViT cosine) / F (0.5R(V)+0.5D) ===")
    emit("block | V Spearman | V ROC | V PR | V cov | V prec | V gap | F Spearman | F ROC | F PR | F cov | F prec | F gap | Jacc(V,D) | Jacc(V,F) | Jacc(D,F)")
    v_stats = {}
    f_stats = {}
    for i in CASAA_LAYERS:
        V = np.concatenate(v_list[i], axis=0)                # (n_img, 256) raw cosine
        Rv = rank_normalize_np(V)
        F = 0.5 * Rv + 0.5 * D
        v_stats[i] = score_stats(V, G, y)
        f_stats[i] = score_stats(F, G, y)
        j_vd = topk_jaccard(v_stats[i]["top_idx"], d_stats["top_idx"])
        j_vf = topk_jaccard(v_stats[i]["top_idx"], f_stats[i]["top_idx"])
        j_df = topk_jaccard(d_stats["top_idx"], f_stats[i]["top_idx"])
        emit(f"  {i:2d}   | {v_stats[i]['spearman']:+.4f}   | {v_stats[i]['roc']:.4f} | {v_stats[i]['pr']:.4f} | {v_stats[i]['coverage']:.4f} | {v_stats[i]['precision']:.4f} | {v_stats[i]['gap']:.4f} | "
             f"{f_stats[i]['spearman']:+.4f}   | {f_stats[i]['roc']:.4f} | {f_stats[i]['pr']:.4f} | {f_stats[i]['coverage']:.4f} | {f_stats[i]['precision']:.4f} | {f_stats[i]['gap']:.4f} | {j_vd:.3f}    | {j_vf:.3f}    | {j_df:.3f}")

    # 分桶 coverage + 零变化图像统计（D / V@8 / F@8）
    emit("\n=== stratified Top32 coverage by true changed-patch count ===")
    emit("bucket | D | V@8 | F@8")
    sc_d = stratified_coverage(D, G, n_changed)
    V8 = np.concatenate(v_list[8], axis=0)
    F8 = 0.5 * rank_normalize_np(V8) + 0.5 * D
    sc_v = stratified_coverage(V8, G, n_changed)
    sc_f = stratified_coverage(F8, G, n_changed)
    for b in ("0", "1-32", "33-128", "129+"):
        emit(f"  {b:6s} | {sc_d[b]:.4f} | {sc_v[b]:.4f} | {sc_f[b]:.4f}")

    emit("\n=== zero-change images (score stats) ===")
    z_d = zero_change_stats(D, n_changed)
    z_v = zero_change_stats(V8, n_changed)
    z_f = zero_change_stats(F8, n_changed)
    emit(f"  D : max={z_d['max']:.4f} top32_mean={z_d['top32_mean']:.4f} gap={z_d['gap']:.4f}")
    emit(f"  V8: max={z_v['max']:.4f} top32_mean={z_v['top32_mean']:.4f} gap={z_v['gap']:.4f}")
    emit(f"  F8: max={z_f['max']:.4f} top32_mean={z_f['top32_mean']:.4f} gap={z_f['gap']:.4f}")

    emit("[AUDIT] DONE")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[AUDIT] report saved to {args.out}")


if __name__ == "__main__":
    main()
