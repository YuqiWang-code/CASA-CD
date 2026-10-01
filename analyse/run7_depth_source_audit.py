"""Run7-D0：Change-Sensitive Depth-Pyramid 四数据集零训练审计（预注册）。

背景（Run7 方案）：R6-D0 已在 SYSU 显示 B2≈B4>B12（token change PR-AUC 0.6008 /
0.6127 / 0.5143）。本审计把同一测量扩展到 CDD/LEVIR/SYSU/WHU 四个数据集，
检验「change-sensitive depth saturation」是否跨数据集成立：

预注册 gate（三个条件各自需在 >=3/4 数据集成立，全部满足才 PASS）：
  C1: PR-AUC(B2) >= PR-AUC(B12) + 0.02
  C2: PR-AUC(B2) >= PR-AUC(B4)  - 0.03
  C3: Top32 precision(B2) >= Top32 precision(B4) - 0.05

任何一项 <3/4 → [R7-D0-GATE] FAIL → CSDP-CD 路线停止（不改阈值、不改 B2→B3 救）。

方法：只用一个按 corrected DeiT loader 构建的 frozen depth-12 ViT（权重=ImageNet
预训练；R6-D0 已证明其与 R4-1/R4-0 冻结 checkpoint 的 ViT 逐位一致），对每个
数据集完整 test 集提取 P0（PatchEmbed 后、pos 前）与 B1..B12（各 block 输出过
final LN）的 1-cos 变化 score。SYSU 附带 R4-1 ResNet 1/8 对照自检（0.6535/0.5948
±0.005），超出容差 → [AUDIT-INVALID]。

返回码：PASS=0 / FAIL=2 / AUDIT_INVALID=3。

用法（服务器）：
    python analyse/run7_depth_source_audit.py \
        --data_root /share_datasets/CD \
        --datasets CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256 \
        --pretrained_weight_path .../deit_tiny_patch16_224-a1311bcf.pth \
        --ckpt_r41_dir .../R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
        --gpu_id 0
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "models"))
if os.path.join(_ROOT, "analyse") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "analyse"))

from model.trainer import Trainer
from run4_detail_interface_audit import pr_auc, spearman, top32_stats
from run6_semantic_token_audit import token_sources, make_loader, pick_best

SOURCES = ("P0", "B1", "B2", "B3", "B4", "B12")


def checksum_vit(model, pretrained_path, depth=12):
    """frozen ViT 与 corrected DeiT 初始化逐位一致（patch_embed/blocks0..depth-1/norm/pos）。"""
    sd = torch.load(pretrained_path, map_location="cpu")["model"]
    vit = model.encoder.vit
    keys = ["patch_embed.proj.weight", "patch_embed.proj.bias", "norm.weight", "norm.bias"]
    for i in range(depth):
        keys += [f"blocks.{i}.norm1.weight", f"blocks.{i}.attn.qkv.weight", f"blocks.{i}.attn.qkv.bias"]
    for k in keys:
        if not torch.equal(vit.state_dict()[k].cpu(), sd[k]):
            print(f"[AUDIT-INVALID] checksum mismatch at {k}")
            return False
    pe = sd["pos_embed"]
    n_extra = pe.shape[1] - 196
    pe_map = pe[:, n_extra:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
    pe_new = F.interpolate(pe_map.float(), size=(16, 16), mode="bicubic", align_corners=False)
    pos_ref = pe_new.permute(0, 2, 3, 1).reshape(1, 256, -1).to(pe.dtype)
    if not torch.equal(vit.state_dict()["pos_embed"].cpu(), pos_ref):
        print("[AUDIT-INVALID] pos_embed mismatch")
        return False
    return True


def run_dataset(model, loader):
    acc = {s: {"S": [], "G": []} for s in SOURCES}
    with torch.no_grad():
        for img, target in loader:
            pre = img[:, 0:3].cuda().float()
            post = img[:, 3:6].cuda().float()
            g = F.avg_pool2d(target.cuda().float(), 16).view(target.shape[0], -1).cpu().numpy()
            o1 = token_sources(model.encoder.vit, pre)
            o2 = token_sources(model.encoder.vit, post)
            for s in SOURCES:
                score = 1.0 - F.cosine_similarity(o1[s], o2[s], dim=-1, eps=1e-8)
                acc[s]["S"].append(score.cpu().numpy())
                acc[s]["G"].append(g)
    out = {}
    for s in SOURCES:
        S = np.concatenate(acc[s]["S"], axis=0)
        G = np.concatenate(acc[s]["G"], axis=0)
        out[s] = {"S": S, "G": G}
    return out


def report_dataset(ds, res):
    print(f"\n=== {ds} ===")
    m = {}
    for s in SOURCES:
        pr = pr_auc(res[s]["S"].reshape(-1), (res[s]["G"].reshape(-1) > 0))
        sp = spearman(res[s]["S"].reshape(-1), res[s]["G"].reshape(-1))
        cov, prec = top32_stats(res[s]["S"], res[s]["G"])
        m[s] = {"pr": pr, "sp": sp, "prec": prec, "cov": cov}
        print(f"  [{s}] PR-AUC={pr:.4f} Spearman={sp:+.4f} Top32 prec={prec:.4f} coverage={cov:.4f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--datasets", type=str, required=True, help="comma separated, e.g. CDD-CD-256,LEVIR-CD-256")
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--ckpt_r41_dir", type=str, default="", help="SYSU 对照自检用 R4-1 checkpoint 目录（可为空则跳过）")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    model = Trainer("tiny", pretrained_path=args.pretrained_weight_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=12, detail_mode="resnet").float().cuda().eval()
    print("[R7-D0] frozen depth-12 ViT built (corrected DeiT loader)")
    if not checksum_vit(model, args.pretrained_weight_path, depth=12):
        return 3

    # SYSU 对照自检（与 R6-D0 同口径）
    if args.ckpt_r41_dir and "SYSU-CD-256" in datasets:
        r41 = Trainer("tiny", pretrained_path=args.pretrained_weight_path, resnet_pretrained=False,
                      mode="baseline", vit_depth=4, detail_mode="resnet").float().cuda().eval()
        r41.load_state_dict(torch.load(pick_best(args.ckpt_r41_dir), map_location="cpu", weights_only=False))
        loader = make_loader(os.path.join(args.data_root, "SYSU-CD-256"), args.batch_size, args.num_workers)
        pool = torch.nn.AdaptiveAvgPool2d((16, 16))
        S, G = [], []
        with torch.no_grad():
            for img, target in loader:
                pre = img[:, 0:3].cuda().float()
                post = img[:, 3:6].cuda().float()
                g = F.avg_pool2d(target.cuda().float(), 16).view(target.shape[0], -1).cpu().numpy()
                c1 = r41.encoder.detail_capture(pre)[2]
                c2 = r41.encoder.detail_capture(post)[2]
                p1, p2 = pool(c1), pool(c2)
                s = 1.0 - F.cosine_similarity(p1.flatten(2).transpose(1, 2), p2.flatten(2).transpose(1, 2),
                                              dim=-1, eps=1e-8)
                S.append(s.cpu().numpy())
                G.append(g)
        Sc = np.concatenate(S, axis=0)
        Gc = np.concatenate(G, axis=0)
        ctrl_pr = pr_auc(Sc.reshape(-1), (Gc.reshape(-1) > 0))
        _, ctrl_prec = top32_stats(Sc, Gc)
        print(f"\n[CTRL] SYSU R4-1 ResNet 1/8: PR-AUC={ctrl_pr:.4f} (0.6535±0.005) "
              f"Top32 prec={ctrl_prec:.4f} (0.5948±0.005)")
        if not (abs(ctrl_pr - 0.6535) <= 0.005 and abs(ctrl_prec - 0.5948) <= 0.005):
            print("[AUDIT-INVALID] SYSU control mismatch")
            return 3
        del r41
        torch.cuda.empty_cache()

    metrics = {}
    for ds in datasets:
        loader = make_loader(os.path.join(args.data_root, ds), args.batch_size, args.num_workers)
        res = run_dataset(model, loader)
        metrics[ds] = report_dataset(ds, res)

    print("\n=== R7-D0 预注册 gate（三项各需 >=3/4 数据集） ===")
    c1 = c2 = c3 = 0
    for ds in datasets:
        m = metrics[ds]
        a1 = m["B2"]["pr"] >= m["B12"]["pr"] + 0.02
        a2 = m["B2"]["pr"] >= m["B4"]["pr"] - 0.03
        a3 = m["B2"]["prec"] >= m["B4"]["prec"] - 0.05
        c1 += int(a1)
        c2 += int(a2)
        c3 += int(a3)
        print(f"  {ds:14s} | C1 B2>=B12+0.02: {m['B2']['pr']:.4f}>={m['B12']['pr'] + 0.02:.4f} {a1} | "
              f"C2 B2>=B4-0.03: {m['B2']['pr']:.4f}>={m['B4']['pr'] - 0.03:.4f} {a2} | "
              f"C3 prec B2>=B4-0.05: {m['B2']['prec']:.4f}>={m['B4']['prec'] - 0.05:.4f} {a3}")
    n = len(datasets)
    ok = (c1 >= 0.75 * n) and (c2 >= 0.75 * n) and (c3 >= 0.75 * n)
    print(f"  C1={c1}/{n}  C2={c2}/{n}  C3={c3}/{n}")
    print(f"  [R7-D0-GATE] {'PASS' if ok else 'FAIL'}")
    print("[R7-D0] DONE")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
