"""Run4 R4-D0：Detail Interface Audit（零训练成本，只在 GPU1 前向）。

对应决策文档（CASA-CD_Run4_R4-2失败后_下一步决策与PSD-Detail方案.md §2-4）：
用已训练完的 R4-1 / R4-2 / R4-2b best checkpoint，在完整 SYSU test（4000 对）上：

  - R4-1（resnet）：ResNet C2-C4 direct feature；
  - R4-2（light32）：Light raw（adapter 前）+ Light adapted（1×1 adapter 后）；
  - R4-2b（light48）：同上。

每个尺度（1/2、1/4、1/8）先 AdaptiveAvgPool2d(16×16) 对齐到 ViT 的 256 个位置，
对两个时相 feature 计算 s_i = 1 - cos(f1_i, f2_i)，GT = label → AvgPool16×16 →
occupancy。统计：

  - Primary：PR-AUC(score, GT_patch_changed) / Spearman(score, GT_occupancy) /
    Top32 precision / Top32 changed-pixel coverage；
  - Feature-stat：mean / std / L2 norm / zero fraction / negative fraction /
    T1/T2 cosine mean / mean|F1-F2| / mean|F|。

预注册决策规则（§4）：
  A（adapter mismatch）：≥2/3 尺度 PR(Light raw) >= PR(ResNet)-0.02 且
    PR(Light adapted) <= PR(Light raw)-0.03 且 Spearman 同方向下降 → R4-2c ADAPTER_ALIGN；
  B（representation/pretraining/topology）：≥2/3 尺度 PR(Light raw) <= PR(ResNet)-0.04 且
    Light48 raw 对 Light32 raw 改善 < 0.02 → 直接 R4-2d PSD_DETAIL；
  C（模糊）：不满足 A/B → 仍优先 R4-2d PSD_DETAIL。

用法（服务器）：
    python analyse/run4_detail_interface_audit.py \
        --dataset_root /share_datasets/CD/SYSU-CD-256 \
        --pretrained_weight_path .../deit_tiny_patch16_224-a1311bcf.pth \
        --ckpt_r41_dir .../R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
        --ckpt_r42_dir .../R4_2_VIT4_LIGHTDETAIL/SYSU-CD-256 \
        --ckpt_r42b_dir .../R4_2b_VIT4_LIGHTDETAIL48/SYSU-CD-256 \
        --gpu_id 0 [--out /path/to/audit_report.txt]
"""
import os
import sys
import argparse
import glob

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "models"))

from model.trainer import Trainer
import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms

MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
SCALES = ["1/2", "1/4", "1/8"]


# ------------------------------------------------------------------ ranking 指标（与 casaa_router_diagnostic.py 同口径）
def rankdata(a):
    return np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(np.float64) + 1


def spearman(s, g):
    rs, rg = rankdata(s), rankdata(g)
    if rs.std() == 0 or rg.std() == 0:
        return float("nan")
    return float(np.corrcoef(rs, rg)[0, 1])


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


def top32_stats(S, G):
    top_idx = np.argsort(-S, axis=1, kind="mergesort")[:, :32]
    topG = np.take_along_axis(G, top_idx, axis=1)
    denom = G.sum(axis=1)
    cov = topG.sum(axis=1)
    cov_mean = float((cov[denom > 0] / denom[denom > 0]).mean()) if (denom > 0).any() else float("nan")
    prec_mean = float((topG > 0).mean())
    return cov_mean, prec_mean


class FeatStatAccum:
    """按 batch 累计 feature-stat：mean/std/L2 norm/zero/negative/cosine/absdiff 比值。"""

    def __init__(self):
        self.n = 0            # 元素总数（两个时相）
        self.n_pos = 0        # 位置数（单时相）
        self.sum = 0.0
        self.sumsq = 0.0
        self.l2_sum = 0.0     # 位置级 L2 范数之和（两个时相）
        self.zero = 0
        self.neg = 0
        self.cos_sum = 0.0
        self.absdiff_sum = 0.0
        self.abssum = 0.0

    def add(self, f1, f2):
        """f1/f2: (B, C, 16, 16) 已对齐到 16×16 的 pooled feature。"""
        a = f1.detach().float().cpu().numpy()
        b = f2.detach().float().cpu().numpy()
        n_el = a.size
        n_pos = a.shape[0] * a.shape[2] * a.shape[3]
        self.n += 2 * n_el
        self.n_pos += n_pos
        self.sum += float(a.sum() + b.sum())
        self.sumsq += float((a ** 2).sum() + (b ** 2).sum())
        n1 = np.linalg.norm(a, axis=1)   # (B,16,16) 位置级 L2
        n2 = np.linalg.norm(b, axis=1)
        self.l2_sum += float(n1.sum() + n2.sum())
        self.zero += int((n1 < 1e-6).sum() + (n2 < 1e-6).sum())
        self.neg += int((a < 0).sum() + (b < 0).sum())
        cos = np.sum(a * b, axis=1) / (n1 * n2 + 1e-12)
        self.cos_sum += float(cos.sum())
        self.absdiff_sum += float(np.abs(a - b).sum())
        self.abssum += float(np.abs(a).sum() + np.abs(b).sum())

    def report(self):
        mean = self.sum / self.n
        var = self.sumsq / self.n - mean ** 2
        return {
            "mean": mean,
            "std": float(var ** 0.5) if var > 0 else 0.0,
            "l2_norm": self.l2_sum / (2 * self.n_pos),   # 位置级 L2 范数均值
            "zero_frac": self.zero / (2 * self.n_pos),
            "neg_frac": self.neg / self.n,
            "t1t2_cos": self.cos_sum / self.n_pos,
            "absdiff_ratio": self.absdiff_sum / max(self.abssum, 1e-12),
        }


def pick_best(ckpt_dir):
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "best_F1=*.pth")))
    if not cands:
        raise FileNotFoundError(f"no best_F1=*.pth in {ckpt_dir}")
    return cands[-1]


def run_one(ckpt_dir, detail_mode, pretrained_path, loader, gpu_id, tag):
    """对单个 checkpoint 跑完整 test，返回 {source: {scale: (S, G, FeatStatAccum)}}。"""
    torch.cuda.set_device(gpu_id)
    model = Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=4, detail_mode=detail_mode).float().cuda().eval()
    sd = torch.load(pick_best(ckpt_dir), map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    enc = model.encoder
    print(f"[AUDIT] loaded {pick_best(ckpt_dir)} ({tag}, detail_mode={detail_mode})", flush=True)

    # source 名 → 特征提取函数
    if detail_mode == "resnet":
        sources = {"resnet_direct": lambda img: enc.detail_capture(img)}
    else:
        sources = {"light_raw": lambda img: list(enc.detail(img)),
                   "light_adapted": lambda img: enc.detail_capture(img)}

    acc = {s: {k: {"S": [], "G": [], "feat": FeatStatAccum()} for k in range(3)} for s in sources}
    pool = torch.nn.AdaptiveAvgPool2d((16, 16))

    with torch.no_grad():
        for bi, (img, target) in enumerate(loader):
            pre = img[:, 0:3].cuda().float()
            post = img[:, 3:6].cuda().float()
            g = torch.nn.functional.avg_pool2d(target.cuda().float(), 16).view(target.shape[0], -1)
            for sname, fn in sources.items():
                f1 = fn(pre)
                f2 = fn(post)
                for k in range(3):
                    p1, p2 = pool(f1[k]), pool(f2[k])
                    sd_score = 1.0 - torch.nn.functional.cosine_similarity(
                        p1.flatten(2).transpose(1, 2), p2.flatten(2).transpose(1, 2), dim=-1, eps=1e-8)
                    acc[sname][k]["S"].append(sd_score.cpu().numpy())
                    acc[sname][k]["G"].append(g.cpu().numpy())
                    acc[sname][k]["feat"].add(p1, p2)
            if bi % 50 == 0:
                print(f"  [{tag}] batch {bi + 1}/{len(loader)}", flush=True)

    out = {}
    for sname in sources:
        out[sname] = {}
        for k in range(3):
            S = np.concatenate(acc[sname][k]["S"], axis=0)
            G = np.concatenate(acc[sname][k]["G"], axis=0)
            out[sname][k] = {"S": S, "G": G, "feat": acc[sname][k]["feat"].report()}
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--ckpt_r41_dir", type=str, required=True)
    ap.add_argument("--ckpt_r42_dir", type=str, required=True)
    ap.add_argument("--ckpt_r42b_dir", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="", help="额外写入的报告文件路径")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True

    val_transform = myTransforms.Compose([
        myTransforms.Normalize(mean=MEAN, std=STD),
        myTransforms.Scale(256, 256),
        myTransforms.ToTensor(),
    ])
    test_data = myDataLoader.Dataset(
        file_root=args.dataset_root, list_path=os.path.join(args.dataset_root, "list", "test.txt"),
        transform=val_transform)
    loader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True)
    print(f"[AUDIT] SYSU test batches = {len(loader)}", flush=True)

    r41 = run_one(args.ckpt_r41_dir, "resnet", args.pretrained_weight_path, loader, args.gpu_id, "R4-1")
    r42 = run_one(args.ckpt_r42_dir, "light", args.pretrained_weight_path, loader, args.gpu_id, "R4-2")
    r42b = run_one(args.ckpt_r42b_dir, "light48", args.pretrained_weight_path, loader, args.gpu_id, "R4-2b")

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 100)
    emit("R4-D0 Detail Interface Audit — SYSU test (full)")
    emit("score s_i = 1 - cos(f1_i, f2_i) after AdaptiveAvgPool2d(16x16); GT occupancy = AvgPool16(label)")
    emit("=" * 100)

    # Primary 指标
    emit("\n=== Primary：per-scale ranking 指标 ===")
    emit("source              | scale | PR-AUC | Spearman | Top32 prec | Top32 coverage")
    primary = {}   # (source, scale) -> {pr, sp, prec, cov}
    for src_name, data in (("R4-1 resnet_direct", r41["resnet_direct"]),
                           ("R4-2 light_raw", r42["light_raw"]),
                           ("R4-2 light_adapted", r42["light_adapted"]),
                           ("R4-2b light48_raw", r42b["light_raw"]),
                           ("R4-2b light48_adapted", r42b["light_adapted"])):
        for k in range(3):
            S, G = data[k]["S"], data[k]["G"]
            s_flat, g_flat = S.reshape(-1), G.reshape(-1)
            y_flat = (g_flat > 0)
            pr = pr_auc(s_flat, y_flat)
            sp = spearman(s_flat, g_flat)
            cov, prec = top32_stats(S, G)
            primary[(src_name, k)] = {"pr": pr, "sp": sp, "prec": prec, "cov": cov}
            emit(f"  {src_name:20s} | {SCALES[k]:>4s} | {pr:.4f}  |  {sp:+.4f}  |  {prec:.4f}  |  {cov:.4f}")

    # Feature-stat 指标
    emit("\n=== Feature-stat：per-scale（pooled 16x16） ===")
    emit("source              | scale | mean   | std    | L2norm | zero% | neg%  | cos(T1,T2) | |dF|/|F|")
    for src_name, data in (("R4-1 resnet_direct", r41["resnet_direct"]),
                           ("R4-2 light_raw", r42["light_raw"]),
                           ("R4-2 light_adapted", r42["light_adapted"]),
                           ("R4-2b light48_raw", r42b["light_raw"]),
                           ("R4-2b light48_adapted", r42b["light_adapted"])):
        for k in range(3):
            f = data[k]["feat"]
            emit(f"  {src_name:20s} | {SCALES[k]:>4s} | {f['mean']:+.3f} | {f['std']:.3f} | {f['l2_norm']:.3f} "
                 f"| {f['zero_frac'] * 100:5.2f} | {f['neg_frac'] * 100:5.2f} | {f['t1t2_cos']:.4f} | {f['absdiff_ratio']:.4f}")

    # 预注册决策（§4）
    emit("\n=== R4-D0 预注册决策 ===")
    a_support = 0
    b_support = 0
    for k in range(3):
        pr_res = primary[("R4-1 resnet_direct", k)]["pr"]
        pr_raw = primary[("R4-2 light_raw", k)]["pr"]
        pr_ad = primary[("R4-2 light_adapted", k)]["pr"]
        pr_raw48 = primary[("R4-2b light48_raw", k)]["pr"]
        sp_raw = primary[("R4-2 light_raw", k)]["sp"]
        sp_ad = primary[("R4-2 light_adapted", k)]["sp"]
        cond_a = (pr_raw >= pr_res - 0.02) and (pr_ad <= pr_raw - 0.03) and (sp_ad < sp_raw)
        cond_b = (pr_raw <= pr_res - 0.04) and ((pr_raw48 - pr_raw) < 0.02)
        a_support += int(cond_a)
        b_support += int(cond_b)
        emit(f"  scale {SCALES[k]}: A=[PR_raw>={pr_res - 0.02:.4f} ({pr_raw:.4f}) & "
             f"PR_ad<={pr_raw - 0.03:.4f} ({pr_ad:.4f}) & Spearman down] -> {cond_a} | "
             f"B=[PR_raw<={pr_res - 0.04:.4f} ({pr_raw:.4f}) & PR48-PR32={pr_raw48 - pr_raw:+.4f}<0.02] -> {cond_b}")
    emit(f"  A 支持尺度数 = {a_support}/3 ; B 支持尺度数 = {b_support}/3")

    if a_support >= 2:
        decision = "A：支持 adapter mismatch → R4-2c ADAPTER_ALIGN（唯一一次 80K）"
    elif b_support >= 2:
        decision = "B：支持 representation/pretraining/topology 问题 → 跳过 adapter，直接 R4-2d PSD_DETAIL"
    else:
        decision = "C：audit 模糊 → 仍优先 R4-2d PSD_DETAIL（已有 2 个 width run 失败 + pretrained-vs-random confound）"
    emit(f"[DECISION] {decision}")
    emit("[AUDIT] DONE")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[AUDIT] report saved to {args.out}")


if __name__ == "__main__":
    main()
