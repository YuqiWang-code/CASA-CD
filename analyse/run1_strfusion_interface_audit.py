"""STRFusion Run1 SF-D0: depth-as-scale interface audit (zero-training, four datasets).

Pre-registered (design doc §6.1). Answers, BEFORE any 80K: does the four-depth
token pyramid {B1..B4} of the frozen ViT4 carry strictly more boundary-band
change evidence than the B4-only semantic source (R8-D0's own metric)? This is
the fusion's direct response to R8-D0's "ranking != dense reconstruction" —
evidence is measured at the boundary band, not just token ranking.

Method (R8-D0 methodology, reused): frozen depth-4 corrected DeiT; per image,
per depth k: s_k = 1 - cos(T_A^k, T_B^k) over 256 tokens -> ordinal rank ->
reshape 16x16 -> bilinear to 256x256 -> map_k. Then
    fuse4 = mean(map_1..map_4),  fuse2 = mean(map_1, map_2),  b4 = map_4.
Metrics: pixel PR-AUC (diagnostic) and boundary-band PR-AUC (gate) per source.

Pre-registered gates:
    G0 (audit validity): ViT4 checksum == corrected DeiT init; SYSU R4-1 ResNet
        1/8 control 0.6535 / 0.5948 (+-0.005); B4 boundary PR-AUC reproduces
        R8-D0 (CDD 0.5216 / LEVIR 0.5369 / SYSU 0.6113 / WHU 0.5718, +-0.01).
    G1 (multi-depth non-redundancy): PRbnd(fuse4) >= PRbnd(B4) + 0.02,
        required on >=2/4 datasets AND on SYSU.
    G2 (fine scales don't destroy semantics): PRbnd(fuse2) >= PRbnd(B4) - 0.02,
        required on >=3/4 datasets.
Verdict: G1 or G2 fail -> [SF-D0-GATE] FAIL -> no 80K, fusion mainline stops.

Exit codes: PASS=0 / FAIL=2 / AUDIT_INVALID=3.

Usage (server):
    python analyse/run1_strfusion_interface_audit.py \
        --data_root /share_datasets/CD \
        --datasets CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256 \
        --pretrained_weight_path .../deit_tiny_patch16_224-a1311bcf.pth \
        --ckpt_r41_dir .../UltraLight/Run4/R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
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
from run4_detail_interface_audit import pr_auc, top32_stats
from run6_semantic_token_audit import token_sources, make_loader, pick_best
from run7_depth_source_audit import checksum_vit
from run8_b4_dense_recoverability_audit import boundary_band, pixel_pr_auc, rank_normalize_np

# R8-D0 B4 boundary-band PR-AUC reference values (pre-registered reproduction window)
B4_BND_REF = {"CDD-CD-256": 0.5216, "LEVIR-CD-256": 0.5369,
              "SYSU-CD-256": 0.6113, "WHU-CD-256": 0.5718}
DEPTHS = (1, 2, 3, 4)


class _EncoderHolder(torch.nn.Module):
    """Shim so run7's checksum_vit(model.encoder.vit) reaches the Encoder inside STRFusionNet."""

    def __init__(self, enc):
        super().__init__()
        self.encoder = enc


def run_dataset(vit, loader, device):
    acc = {}
    for name in ("b4", "fuse4", "fuse2"):
        acc[name] = {"map": [], "lab": [], "bnd_s": [], "bnd_l": []}
    with torch.no_grad():
        for img, target in loader:
            pre = img[:, 0:3].to(device).float()
            post = img[:, 3:6].to(device).float()
            lab = target.to(device).float()                      # (B,1,256,256) 0/1

            o1 = token_sources(vit, pre)
            o2 = token_sources(vit, post)
            maps = []
            for d in DEPTHS:
                s = 1.0 - F.cosine_similarity(o1[f"B{d}"], o2[f"B{d}"], dim=-1, eps=1e-8)  # (B,256)
                r = torch.from_numpy(rank_normalize_np(s.cpu().numpy())).float().to(device)
                m = F.interpolate(r.reshape(-1, 1, 16, 16), size=(256, 256),
                                  mode="bilinear", align_corners=False)                    # (B,1,256,256)
                maps.append(m)
            fuse4 = torch.stack(maps, dim=0).mean(dim=0)          # (B,1,256,256)
            fuse2 = (maps[0] + maps[1]) * 0.5
            b4 = maps[3]

            band = boundary_band(lab)                             # (B,1,256,256) bool
            for key, m in (("b4", b4), ("fuse4", fuse4), ("fuse2", fuse2)):
                acc[key]["map"].append(m.squeeze(1).cpu().numpy())
                acc[key]["lab"].append(lab.squeeze(1).cpu().numpy())
                acc[key]["bnd_s"].append(m[band].cpu().numpy())
                acc[key]["bnd_l"].append(lab[band].cpu().numpy())

    out = {}
    for key in acc:
        M = np.concatenate(acc[key]["map"], axis=0)
        L = np.concatenate(acc[key]["lab"], axis=0)
        Bs = np.concatenate(acc[key]["bnd_s"], axis=0)
        Bl = np.concatenate(acc[key]["bnd_l"], axis=0)
        out[key] = {"pix": pixel_pr_auc(M, L), "bnd": pixel_pr_auc(Bs, Bl)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--datasets", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--ckpt_r41_dir", type=str, default="")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Reference frozen ViT4: same corrected DeiT loader the fusion model uses
    holder = Trainer("tiny", pretrained_path=args.pretrained_weight_path, resnet_pretrained=False,
                     mode="baseline", vit_depth=4, detail_mode="none_b4",
                     head_mode="b4_spe").float().cuda().eval()
    print("[SF-D0] frozen depth-4 ViT built (corrected DeiT loader)")
    if not checksum_vit(_EncoderHolder(holder.encoder), args.pretrained_weight_path, depth=4):
        return 3

    # G0c: SYSU R4-1 ResNet 1/8 control self-check
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
                s = 1.0 - F.cosine_similarity(p1.flatten(2).transpose(1, 2),
                                              p2.flatten(2).transpose(1, 2), dim=-1, eps=1e-8)
                S.append(s.cpu().numpy())
                G.append(g)
        Sc = np.concatenate(S, axis=0)
        Gc = np.concatenate(G, axis=0)
        ctrl_pr = pr_auc(Sc.reshape(-1), (Gc.reshape(-1) > 0))
        _, ctrl_prec = top32_stats(Sc, Gc)
        print(f"\n[CTRL] SYSU R4-1 ResNet 1/8: PR-AUC={ctrl_pr:.4f} (0.6535+-0.005) "
              f"Top32 prec={ctrl_prec:.4f} (0.5948+-0.005)")
        if not (abs(ctrl_pr - 0.6535) <= 0.005 and abs(ctrl_prec - 0.5948) <= 0.005):
            print("[AUDIT-INVALID] SYSU control mismatch")
            return 3
        del r41
        torch.cuda.empty_cache()

    results = {}
    for ds in datasets:
        print(f"\n=== {ds} ===", flush=True)
        loader = make_loader(os.path.join(args.data_root, ds), args.batch_size, args.num_workers)
        res = run_dataset(holder.encoder.vit, loader, device)
        results[ds] = res
        for key in ("b4", "fuse4", "fuse2"):
            print(f"  [{key}] pixel PR-AUC={res[key]['pix']:.4f}  boundary PR-AUC={res[key]['bnd']:.4f}")
        ref = B4_BND_REF[ds]
        repro = abs(res["b4"]["bnd"] - ref) <= 0.01
        print(f"  [G0-repro] B4 boundary {res['b4']['bnd']:.4f} vs R8-D0 {ref:.4f} -> "
              f"{'OK' if repro else 'MISMATCH'}")

    print("\n=== SF-D0 预注册 gate ===")
    g0 = all(abs(results[ds]["b4"]["bnd"] - B4_BND_REF[ds]) <= 0.01 for ds in datasets)
    g1_count = 0
    g1_sysu = False
    g2_count = 0
    for ds in datasets:
        r = results[ds]
        g1 = r["fuse4"]["bnd"] >= r["b4"]["bnd"] + 0.02
        g2 = r["fuse2"]["bnd"] >= r["b4"]["bnd"] - 0.02
        g1_count += int(g1)
        g2_count += int(g2)
        if ds == "SYSU-CD-256":
            g1_sysu = g1
        print(f"  {ds:14s} | G1 fuse4>=B4+0.02: {r['fuse4']['bnd']:.4f}>={r['b4']['bnd'] + 0.02:.4f} {g1} | "
              f"G2 fuse2>=B4-0.02: {r['fuse2']['bnd']:.4f}>={r['b4']['bnd'] - 0.02:.4f} {g2}")
    n = len(datasets)
    g1_ok = g1_count >= max(1, int(0.5 * n)) and g1_sysu
    g2_ok = g2_count >= int(0.75 * n)
    print(f"  G0={g0}  G1={g1_count}/{n} (SYSU={g1_sysu})  G2={g2_count}/{n}")
    ok = g0 and g1_ok and g2_ok
    print(f"  [SF-D0-GATE] {'PASS' if ok else 'FAIL'}")
    print("[SF-D0] DONE")
    if not g0:
        return 3
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
