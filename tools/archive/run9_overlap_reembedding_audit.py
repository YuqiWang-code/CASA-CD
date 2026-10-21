"""Run9-D0：O-PRE Complementarity Audit（零训练，四数据集）。

对应方案（CASA-CD_Run9_可执行预注册方案.md §8）：
回答「同一个冻结 DeiT patch kernel 在 stride=8 重叠 lattice 上，相对 canonical
stride=16 P0 是否增加边界信息，且与 B4 semantic 互补」。

三张零训练 score map（每图 ordinal rank → [0,1] → bilinear 256×256）：
  s_P0 = 1−cos(P0_A, P0_B)         （canonical 16×16）
  s_B4 = 1−cos(B4_A, B4_B)         （16×16）
  s_O  = 1−cos(O_A, O_B)           （O-PRE 32×32，同权重 stride=8 reflect pad4）
  proxy = U_B4 · (1 + 0.5·U_O)     （固定 λ=0.5，只验证互补性，非训练 head）

指标：全像素 PR-AUC、边界带 PR-AUC（Run8 同定义）、changed-patch 分桶
（1-16/17-64/>64 的 Top8 hit / Top32 coverage）、O-PRE vs B4 Spearman（诊断）。

预注册 gate（§8.5）：
  G0: PRbnd(OPRE) >= PRbnd(P0) + 0.03   于 >=3/4 数据集
  G1: PRbnd(proxy) >= PRbnd(B4) + 0.02  于 >=3/4 数据集
  G2: PRpix(proxy) >= PRpix(B4) - 0.005 于 4/4 且 >= PRpix(B4)+0.01 于 >=2/4
  G3: SYSU 单独强 gate：ΔPRbnd >= +0.03 且 ΔPRpix >= +0.015
  [R9-D0-GATE] PASS iff G0 & G1 & G2 & G3；FAIL → Run9 停止，0 个 80K。

SYSU 附带 R4-1 ResNet 1/8 对照自检（0.6535/0.5948 ±0.005）→ 越界 AUDIT_INVALID。
返回码：PASS=0 / FAIL=2 / AUDIT_INVALID=3。
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
from run8_b4_dense_recoverability_audit import rank_normalize_np, pixel_pr_auc, boundary_band


def opre_map(vit, img):
    """O-PRE：共享 patch_embed.proj 权重，stride=8、reflect pad4 → B×192×32×32。"""
    w = vit.patch_embed.proj.weight.data
    b = vit.patch_embed.proj.bias.data
    x_pad = F.pad(img, (4, 4, 4, 4), mode="reflect")
    return F.conv2d(x_pad, w, b, stride=8)


def to_pixel_map(score_np, device):
    """(B, N) score → per-image rank → 16×16（或 32×32）→ bilinear 256×256 (B,1,256,256)。"""
    s = torch.from_numpy(rank_normalize_np(score_np)).float().to(device)
    side = int(round(score_np.shape[1] ** 0.5))
    return F.interpolate(s.reshape(-1, 1, side, side), size=(256, 256),
                         mode="bilinear", align_corners=False)


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
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    model = Trainer("tiny", pretrained_path=args.pretrained_weight_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=12, detail_mode="resnet").float().cuda().eval()
    print("[R9-D0] frozen depth-12 ViT built (corrected DeiT loader)")
    if not checksum_vit(model, args.pretrained_weight_path, depth=12):
        return 3

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

    results = {}
    for ds in datasets:
        print(f"\n=== {ds} ===", flush=True)
        loader = make_loader(os.path.join(args.data_root, ds), args.batch_size, args.num_workers)
        acc = {k: {"pix": [], "bnd": [], "lab": []} for k in ("P0", "B4", "OPRE")}
        lab_pix_list = []
        proxy_pix_list, proxy_bnd_list = [], []
        o_tok_list, b_tok_list, p0_tok_list = [], [], []
        g16_list, g8_list = [], []
        with torch.no_grad():
            for bi, (img, target) in enumerate(loader):
                pre = img[:, 0:3].cuda().float()
                post = img[:, 3:6].cuda().float()
                lab = target.cuda().float()                      # (B,1,256,256)
                o1 = token_sources(model.encoder.vit, pre)
                o2 = token_sources(model.encoder.vit, post)
                # canonical P0 / B4 token scores
                s_p0 = 1.0 - F.cosine_similarity(o1["P0"], o2["P0"], dim=-1, eps=1e-8)
                s_b4 = 1.0 - F.cosine_similarity(o1["B4"], o2["B4"], dim=-1, eps=1e-8)
                # O-PRE：共享 patch kernel，stride=8 reflect pad4
                oe1 = opre_map(model.encoder.vit, pre)           # (B,192,32,32)
                oe2 = opre_map(model.encoder.vit, post)
                s_o = 1.0 - F.cosine_similarity(
                    oe1.flatten(2).transpose(1, 2), oe2.flatten(2).transpose(1, 2), dim=-1, eps=1e-8)
                # rank → bilinear 256×256
                u_p0 = to_pixel_map(s_p0.cpu().numpy(), pre.device)
                u_b4 = to_pixel_map(s_b4.cpu().numpy(), pre.device)
                u_o = to_pixel_map(s_o.cpu().numpy(), pre.device)
                proxy = u_b4 * (1.0 + 0.5 * u_o)                 # 固定 λ=0.5，仅审计
                band = boundary_band(lab)
                for k, u in (("P0", u_p0), ("B4", u_b4), ("OPRE", u_o)):
                    acc[k]["pix"].append(u.squeeze(1).cpu().numpy())
                    acc[k]["bnd"].append(u[band].cpu().numpy())
                    acc[k]["lab"].append(lab[band].cpu().numpy())
                proxy_pix_list.append(proxy.squeeze(1).cpu().numpy())
                proxy_bnd_list.append(proxy[band].cpu().numpy())
                lab_pix_list.append(lab.squeeze(1).cpu().numpy())
                o_tok_list.append(s_o.cpu().numpy())
                b_tok_list.append(s_b4.cpu().numpy())
                p0_tok_list.append(s_p0.cpu().numpy())
                g16_list.append(F.avg_pool2d(lab, 16).view(lab.shape[0], -1).cpu().numpy())
                g8_list.append(F.avg_pool2d(lab, 8).view(lab.shape[0], -1).cpu().numpy())
                if bi % 50 == 0:
                    print(f"  batch {bi + 1}/{len(loader)}", flush=True)

        LAB = np.concatenate(lab_pix_list, axis=0)               # (n,256,256)
        m = {}
        for k in ("P0", "B4", "OPRE"):
            pix = np.concatenate(acc[k]["pix"], axis=0)
            bnd_s = np.concatenate(acc[k]["bnd"], axis=0)
            bnd_l = np.concatenate(acc[k]["lab"], axis=0)
            m[k] = {"pr_pix": pixel_pr_auc(pix, LAB), "pr_bnd": pixel_pr_auc(bnd_s, bnd_l)}
        m["proxy"] = {
            "pr_pix": pixel_pr_auc(np.concatenate(proxy_pix_list, axis=0), LAB),
            "pr_bnd": pixel_pr_auc(np.concatenate(proxy_bnd_list, axis=0),
                                   np.concatenate(acc["B4"]["lab"], axis=0)),
        }
        # 诊断：O-PRE（池化到 16×16）与 B4 token score 的 Spearman
        o_tok = np.concatenate(o_tok_list, axis=0)               # (n,1024)
        b_tok = np.concatenate(b_tok_list, axis=0)               # (n,256)
        o_pool = (o_tok.reshape(o_tok.shape[0], 16, 2, 16, 2)
                  .mean(axis=(2, 4)).reshape(o_tok.shape[0], 256))
        from run6_semantic_token_audit import spearman as _spearman
        sp_o_b4 = _spearman(o_pool.reshape(-1), b_tok.reshape(-1))
        print(f"  [P0]     pix={m['P0']['pr_pix']:.4f}  bnd={m['P0']['pr_bnd']:.4f}")
        print(f"  [B4]     pix={m['B4']['pr_pix']:.4f}  bnd={m['B4']['pr_bnd']:.4f}")
        print(f"  [OPRE]   pix={m['OPRE']['pr_pix']:.4f}  bnd={m['OPRE']['pr_bnd']:.4f}")
        print(f"  [proxy]  pix={m['proxy']['pr_pix']:.4f}  bnd={m['proxy']['pr_bnd']:.4f}")
        print(f"  [diag]   Spearman(OPRE,B4)={sp_o_b4:+.4f} (冗余度参考)")
        # 分桶诊断（不参与 gate）：P0/B4 用 16×16 GT，O-PRE 用 32×32 GT
        from run6_semantic_token_audit import bucketed
        G16 = np.concatenate(g16_list, axis=0)
        G8 = np.concatenate(g8_list, axis=0)
        P0t = np.concatenate(p0_tok_list, axis=0)
        n16 = (G16 > 0).sum(axis=1)
        for name, S in (("P0", P0t), ("B4", b_tok)):
            b = bucketed(S, G16, n16)
            print(f"  [bucket-{name}] 1-16 hit/cov={b['1-16'][0]:.3f}/{b['1-16'][1]:.3f} "
                  f"17-64={b['17-64'][0]:.3f}/{b['17-64'][1]:.3f} >64={b['>64'][0]:.3f}/{b['>64'][1]:.3f}")
        n8 = (G8 > 0).sum(axis=1)
        b = bucketed(o_tok, G8, n8)
        print(f"  [bucket-OPRE] 1-16 hit/cov={b['1-16'][0]:.3f}/{b['1-16'][1]:.3f} "
              f"17-64={b['17-64'][0]:.3f}/{b['17-64'][1]:.3f} >64={b['>64'][0]:.3f}/{b['>64'][1]:.3f}")
        results[ds] = m

    print("\n=== R9-D0 预注册 gate（§8.5） ===")
    g0 = g1 = 0
    g2a = g2b = 0
    g3 = False
    for ds in datasets:
        r = results[ds]
        a0 = r["OPRE"]["pr_bnd"] >= r["P0"]["pr_bnd"] + 0.03
        a1 = r["proxy"]["pr_bnd"] >= r["B4"]["pr_bnd"] + 0.02
        a2a = r["proxy"]["pr_pix"] >= r["B4"]["pr_pix"] - 0.005
        a2b = r["proxy"]["pr_pix"] >= r["B4"]["pr_pix"] + 0.01
        g0 += int(a0)
        g1 += int(a1)
        g2a += int(a2a)
        g2b += int(a2b)
        line = (f"  {ds:14s} | G0 {r['OPRE']['pr_bnd']:.4f}>={r['P0']['pr_bnd'] + 0.03:.4f} {a0} | "
                f"G1 {r['proxy']['pr_bnd']:.4f}>={r['B4']['pr_bnd'] + 0.02:.4f} {a1} | "
                f"G2 {r['proxy']['pr_pix']:.4f} in [{r['B4']['pr_pix'] - 0.005:.4f}, ..] {a2a}/{a2b}")
        print(line, flush=True)
        if "SYSU" in ds:
            dbnd = r["proxy"]["pr_bnd"] - r["B4"]["pr_bnd"]
            dpix = r["proxy"]["pr_pix"] - r["B4"]["pr_pix"]
            g3 = (dbnd >= 0.03) and (dpix >= 0.015)
            print(f"    [G3-SYSU] Δbnd={dbnd:+.4f}>=+0.03 {dbnd >= 0.03} | "
                  f"Δpix={dpix:+.4f}>=+0.015 {dpix >= 0.015}")
    n = len(datasets)
    ok = (g0 >= 0.75 * n) and (g1 >= 0.75 * n) and (g2a == n) and (g2b >= 0.5 * n) and g3
    print(f"  G0={g0}/{n}  G1={g1}/{n}  G2a={g2a}/{n}  G2b={g2b}/{n}  G3={g3}")
    print(f"  [R9-D0-GATE] {'PASS' if ok else 'FAIL'}")
    print("[R9-D0] DONE")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
