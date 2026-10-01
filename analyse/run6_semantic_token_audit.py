"""Run6-D0：零训练 Semantic / Patch-Token Audit（预注册 §4）。

用 R4-1（ViT4）+ R4-0（full12）best checkpoint 与 corrected DeiT 预训练，
在完整 SYSU test（4000 对）上无训练前向，回答：
  1. P0（PatchEmbed 后、pos_embed 前的 raw token）是否有足够局部 change ranking；
  2. B4（block3 + final LN）是否保留足够 semantic change ranking；
  3. 参数自由融合 Fuse = 0.5*Rank(S_P0) + 0.5*Rank(S_B4) 是否达到 Mobile-D4 等级。

预注册 gate（§4.6）：
  G1 P0 : PR-AUC>=0.44   Top32 prec>=0.40
  G2 B4 : PR-AUC>=0.40   Top32 prec>=0.36   Spearman>=0.18
  G3 Fuse: PR-AUC>=0.50  Top32 prec>=0.46   Spearman>=0.25
  全部通过 → [R6-D0-GATE] PASS（才允许实现/训练 R6-1）；任一不过 → FAIL，Run6 停止。

audit 自检（§4.5）：先复现 R4-1 ResNet 1/8 的 PR-AUC=0.6535±0.005、
Top32 prec=0.5948±0.005，超出容差 → [AUDIT-INVALID]（exit 3）。
另外断言 R4-1 冻结 ViT 与 corrected DeiT 初始化逐位一致（§4.2）。

返回码：PASS=0 / FAIL=2 / AUDIT_INVALID=3。

用法（服务器）：
    python analyse/run6_semantic_token_audit.py \
        --dataset_root /share_datasets/CD/SYSU-CD-256 \
        --pretrained_weight_path .../deit_tiny_patch16_224-a1311bcf.pth \
        --ckpt_r41_dir .../R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
        --ckpt_r40_dir .../R4_0_A0_FULL12_FROZEN/SYSU-CD-256 \
        --gpu_id 0 [--batch_size 16]
"""
import os
import sys
import glob
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

import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms

MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


def pick_best(ckpt_dir):
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "best_F1=*.pth")))
    if not cands:
        raise FileNotFoundError(f"no best_F1=*.pth in {ckpt_dir}")
    return cands[-1]


def make_loader(dataset_root, batch_size, num_workers):
    transform = myTransforms.Compose([
        myTransforms.Normalize(mean=MEAN, std=STD),
        myTransforms.Scale(256, 256),
        myTransforms.ToTensor(),
    ])
    ds = myDataLoader.Dataset(file_root=dataset_root,
                              list_path=os.path.join(dataset_root, "list", "test.txt"),
                              transform=transform)
    return torch.utils.data.DataLoader(ds, shuffle=False, batch_size=batch_size,
                                       num_workers=num_workers, pin_memory=True)


def rank_normalize_np(s):
    """per-image ordinal rank / (N-1)（与 casaa.rank_normalize_per_image 等价）。"""
    N = s.shape[1]
    order = np.argsort(s, axis=1, kind="mergesort")
    base = np.arange(N, dtype=np.float64)[None, :]
    out = np.empty_like(s, dtype=np.float64)
    np.put_along_axis(out, order, base, axis=1)
    return out / max(N - 1, 1)


def checksum_assert(model, pretrained_path):
    """§4.2：R4-1 冻结 ViT 必须与 corrected DeiT 初始化逐位一致，否则 AUDIT-INVALID。"""
    sd = torch.load(pretrained_path, map_location="cpu")["model"]
    vit = model.encoder.vit
    keys = ["patch_embed.proj.weight", "patch_embed.proj.bias", "norm.weight", "norm.bias"]
    for i in range(4):
        keys += [f"blocks.{i}.norm1.weight", f"blocks.{i}.attn.qkv.weight", f"blocks.{i}.attn.qkv.bias"]
    for k in keys:
        if not torch.equal(vit.state_dict()[k].cpu(), sd[k]):
            print(f"[AUDIT-INVALID] checksum mismatch at {k}")
            return False
    # pos_embed：与 corrected loader 的 14×14→16×16 bicubic 一致
    pe = sd["pos_embed"]
    n_extra = pe.shape[1] - 196
    pe_map = pe[:, n_extra:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
    pe_new = F.interpolate(pe_map.float(), size=(16, 16), mode="bicubic", align_corners=False)
    pos_ref = pe_new.permute(0, 2, 3, 1).reshape(1, 256, -1).to(pe.dtype)
    if not torch.equal(vit.state_dict()["pos_embed"].cpu(), pos_ref):
        print("[AUDIT-INVALID] pos_embed mismatch vs corrected DeiT interpolation")
        return False
    print("  [CHECKSUM] R4-1 frozen ViT == corrected DeiT init (patch_embed/blocks0-3/norm/pos_embed)")
    return True


def token_sources(vit, img):
    """对单时相提取 P0 与 B1..B{len(blocks)}（每个 block 输出过 final LN）。"""
    with torch.no_grad():
        p0 = vit.patch_embed(img)                       # (B,256,192)，pos 之前
        x = p0 + vit.interpolate_pos_encoding(p0, img.shape[-1], img.shape[-2])
        outs = {"P0": p0}
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            outs[f"B{i + 1}"] = vit.norm(x)
    return outs


def score_from_tokens(f1, f2):
    """s = 1 - cos(f1_i, f2_i) per token（输入 (B,256,C)）。"""
    return 1.0 - F.cosine_similarity(f1, f2, dim=-1, eps=1e-8)


def report_source(name, S, G):
    y = (G.reshape(-1) > 0)
    pr = pr_auc(S.reshape(-1), y)
    sp = spearman(S.reshape(-1), G.reshape(-1))
    cov, prec = top32_stats(S, G)
    print(f"  [{name}] PR-AUC={pr:.4f} Spearman={sp:+.4f} Top32 prec={prec:.4f} coverage={cov:.4f}", flush=True)
    return {"pr": pr, "sp": sp, "prec": prec, "cov": cov}


def bucketed(S, G, n_changed):
    """按 changed-patch 数分桶：每组 Top8 hit-rate（top8 含≥1 changed）与 Top32 coverage。"""
    out = {}
    for bname, mask in (("1-16", (n_changed >= 1) & (n_changed <= 16)),
                        ("17-64", (n_changed >= 17) & (n_changed <= 64)),
                        (">64", n_changed > 64)):
        if not mask.any():
            out[bname] = (float("nan"), float("nan"))
            continue
        Sb, Gb = S[mask], G[mask]
        top8 = np.take_along_axis(Gb, np.argsort(-Sb, axis=1, kind="mergesort")[:, :8], axis=1)
        hit = float((top8.sum(axis=1) > 0).mean())
        top32 = np.take_along_axis(Gb, np.argsort(-Sb, axis=1, kind="mergesort")[:, :32], axis=1)
        denom = Gb.sum(axis=1)
        cov = top32.sum(axis=1)
        out[bname] = (hit, float((cov[denom > 0] / denom[denom > 0]).mean()) if (denom > 0).any() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--ckpt_r41_dir", type=str, required=True)
    ap.add_argument("--ckpt_r40_dir", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    loader = make_loader(args.dataset_root, args.batch_size, args.num_workers)
    print(f"[R6-D0] SYSU test batches = {len(loader)}", flush=True)

    # 两个模型：R4-1（ViT4，主）；R4-0（full12，诊断）
    r41 = Trainer("tiny", pretrained_path=args.pretrained_weight_path, resnet_pretrained=False,
                  mode="baseline", vit_depth=4, detail_mode="resnet").float().cuda().eval()
    r41.load_state_dict(torch.load(pick_best(args.ckpt_r41_dir), map_location="cpu", weights_only=False))
    r40 = Trainer("tiny", pretrained_path=args.pretrained_weight_path, resnet_pretrained=False,
                  mode="baseline", vit_depth=12, detail_mode="resnet").float().cuda().eval()
    r40.load_state_dict(torch.load(pick_best(args.ckpt_r40_dir), map_location="cpu", weights_only=False))

    if not checksum_assert(r41, args.pretrained_weight_path):
        return 3

    pool = torch.nn.AdaptiveAvgPool2d((16, 16))
    acc = {}
    for name in ("P0", "B1", "B2", "B3", "B4", "B12", "CTRL"):
        acc[name] = {"S": [], "G": []}

    with torch.no_grad():
        for img, target in loader:
            pre = img[:, 0:3].cuda().float()
            post = img[:, 3:6].cuda().float()
            g = F.avg_pool2d(target.cuda().float(), 16).view(target.shape[0], -1).cpu().numpy()

            o1_pre, o1_post = token_sources(r41.encoder.vit, pre), token_sources(r41.encoder.vit, post)
            for name in ("P0", "B1", "B2", "B3", "B4"):
                acc[name]["S"].append(score_from_tokens(o1_pre[name], o1_post[name]).cpu().numpy())
                acc[name]["G"].append(g)
            o0_pre, o0_post = token_sources(r40.encoder.vit, pre), token_sources(r40.encoder.vit, post)
            acc["B12"]["S"].append(score_from_tokens(o0_pre["B12"], o0_post["B12"]).cpu().numpy())
            acc["B12"]["G"].append(g)
            # CTRL：R4-1 ResNet 1/8（layer3 32×32 → pool 16×16，与 R4-D0 同口径）
            c1 = r41.encoder.detail_capture(pre)[2]
            c2 = r41.encoder.detail_capture(post)[2]
            p1, p2 = pool(c1), pool(c2)
            acc["CTRL"]["S"].append(score_from_tokens(p1.flatten(2).transpose(1, 2),
                                                      p2.flatten(2).transpose(1, 2)).cpu().numpy())
            acc["CTRL"]["G"].append(g)

    results = {}
    for name in ("P0", "B1", "B2", "B3", "B4", "B12", "CTRL"):
        S = np.concatenate(acc[name]["S"], axis=0)
        G = np.concatenate(acc[name]["G"], axis=0)
        results[name] = {"S": S, "G": G}

    # Fuse = 0.5*Rank(S_P0) + 0.5*Rank(S_B4)（参数自由，仅 audit）
    Sf = 0.5 * rank_normalize_np(results["P0"]["S"]) + 0.5 * rank_normalize_np(results["B4"]["S"])
    results["FUSE-P0-B4"] = {"S": Sf, "G": results["B4"]["G"]}

    # §4.5 自检：CTRL 复现 R4-D0 的 ResNet 1/8 数字
    ctrl_pr = pr_auc(results["CTRL"]["S"].reshape(-1), (results["CTRL"]["G"].reshape(-1) > 0))
    _, ctrl_prec = top32_stats(results["CTRL"]["S"], results["CTRL"]["G"])
    print(f"\n=== audit self-check (§4.5) ===")
    print(f"  [CTRL] R4-1 ResNet 1/8: PR-AUC={ctrl_pr:.4f} (expect 0.6535±0.005) "
          f"Top32 prec={ctrl_prec:.4f} (expect 0.5948±0.005)", flush=True)
    if not (abs(ctrl_pr - 0.6535) <= 0.005 and abs(ctrl_prec - 0.5948) <= 0.005):
        print("[AUDIT-INVALID] control mismatch — score/data 口径与历史 audit 不一致，停止。")
        return 3

    print("\n=== per-source ranking metrics (complete SYSU test) ===")
    m = {}
    for name in ("P0", "B1", "B2", "B3", "B4", "B12", "FUSE-P0-B4"):
        m[name] = report_source(name, results[name]["S"], results[name]["G"])

    print("\n=== bucketed diagnostics (§4.7, 不参与 gate) ===")
    n_changed = (results["P0"]["G"] > 0).sum(axis=1)
    print("  source | 1-16 (Top8 hit / Top32 cov) | 17-64 | >64")
    for name in ("P0", "B1", "B2", "B3", "B4", "B12", "FUSE-P0-B4"):
        b = bucketed(results[name]["S"], results[name]["G"], n_changed)
        print(f"  {name:11s} | {b['1-16'][0]:.3f} / {b['1-16'][1]:.3f} | "
              f"{b['17-64'][0]:.3f} / {b['17-64'][1]:.3f} | {b['>64'][0]:.3f} / {b['>64'][1]:.3f}", flush=True)

    print("\n=== R6-D0 预注册 gate (§4.6) ===")
    g1 = (m["P0"]["pr"] >= 0.44) and (m["P0"]["prec"] >= 0.40)
    g2 = (m["B4"]["pr"] >= 0.40) and (m["B4"]["prec"] >= 0.36) and (m["B4"]["sp"] >= 0.18)
    g3 = (m["FUSE-P0-B4"]["pr"] >= 0.50) and (m["FUSE-P0-B4"]["prec"] >= 0.46) and (m["FUSE-P0-B4"]["sp"] >= 0.25)
    print(f"  G1 P0   : PR {m['P0']['pr']:.4f}>=0.44 | prec {m['P0']['prec']:.4f}>=0.40 -> {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 B4   : PR {m['B4']['pr']:.4f}>=0.40 | prec {m['B4']['prec']:.4f}>=0.36 | "
          f"Spearman {m['B4']['sp']:+.4f}>=0.18 -> {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 Fuse : PR {m['FUSE-P0-B4']['pr']:.4f}>=0.50 | prec {m['FUSE-P0-B4']['prec']:.4f}>=0.46 | "
          f"Spearman {m['FUSE-P0-B4']['sp']:+.4f}>=0.25 -> {'PASS' if g3 else 'FAIL'}")
    ok = g1 and g2 and g3
    print(f"  [R6-D0-GATE] {'PASS' if ok else 'FAIL'}")
    print("[R6-D0] DONE")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
