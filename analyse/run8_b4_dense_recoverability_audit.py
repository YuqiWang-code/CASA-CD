"""Run8-D0：B4 Dense Recoverability Audit（零训练，四数据集）。

对应方案（CASA-CD_Run8_B4-SPE最终路线与预注册.md §4）：
回答「16×16 B4 change evidence 在无训练、无 detail branch 情况下投影到像素域后，
是否仍保留足够的整体定位与边界排序能力？」——不再重复测 B4 token PR-AUC。

方法：同一个 frozen corrected depth-12 DeiT，一次前向取得 B4/B12，
S=1−cos(T_A,T_B)，每图 ordinal rank → [0,1]，reshape 16×16，bilinear → 256×256
得到 U4/U12。指标：
  M1 像素 PR-AUC（全 256×256）；
  M2 边界带 PR-AUC（edge = mask XOR erode3×3；band = dilate9×9；仅 band 内）；
  M3 稀疏变化分桶（1-16/17-64/>64 changed patches 的 Top8 hit / Top32 coverage，诊断）；
  M4 fixed patch-unembedding（|B4_A−B4_B| → conv_transpose2d 用 patch_embed.proj.weight
     反投影 → 通道 L2 像素 score，仅诊断，不参与 gate）。

预注册 gate（§4.4，四数据集）：
  G1_pixel(ds):   PRpix(B4)  >= PRpix(B12)  + 0.03
  G2_boundary(ds): PRbnd(B4) >= PRbnd(B12)  + 0.02
  C1: G1_pixel  至少 3/4 数据集；C2: G2_boundary 至少 3/4；
  C3: 任何数据集都不允许 B4 比 B12 下降 >0.01（pixel 或 boundary）。
  C1&C2&C3 → PASS；否则 FAIL → 永久停止 B4-only/no-detail dense reconstruction 路线。

SYSU 附带 R4-1 ResNet 1/8 对照自检（0.6535/0.5948 ±0.005）→ 越界 AUDIT-INVALID。
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
from run4_detail_interface_audit import pr_auc
from run6_semantic_token_audit import token_sources, make_loader, pick_best
from run7_depth_source_audit import checksum_vit


def rank_normalize_np(s):
    N = s.shape[1]
    order = np.argsort(s, axis=1, kind="mergesort")
    base = np.arange(N, dtype=np.float64)[None, :]
    out = np.empty_like(s, dtype=np.float64)
    np.put_along_axis(out, order, base, axis=1)
    return out / max(N - 1, 1)


def pixel_pr_auc(score_maps, labels):
    """全局像素 PR-AUC：float32 + quicksort 精确计算（同 pr_auc 口径，避免 mergesort 大内存）。"""
    s = np.ascontiguousarray(score_maps.reshape(-1), dtype=np.float32)
    y = (np.ascontiguousarray(labels.reshape(-1)) > 0)
    n_pos = float(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="quicksort")
    ys = y[order].astype(np.float64)
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / n_pos
    rec = np.concatenate([[0.0], rec])
    prec = np.concatenate([[prec[0]], prec])
    return float(np.trapezoid(prec, rec))


def boundary_band(label):
    """edge = mask XOR erode3×3；band = dilate(edge, 9×9)。label: (B,1,256,256) 0/1。"""
    mask = (label > 0.5).float()
    ero = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    edge = (mask != ero).float()
    band = F.max_pool2d(edge, kernel_size=9, stride=1, padding=4)
    return band > 0.5


def unembed_score(diff_map, proj_weight):
    """M4：|B4_A−B4_B| (B,192,16,16) 用 DeiT patch_embed.proj.weight 做固定反投影。"""
    # proj_weight: (192,3,16,16) → conv_transpose2d(in=192, out=3, k=16, stride=16)
    out = F.conv_transpose2d(diff_map, proj_weight, stride=16)   # (B,3,256,256)
    return out.norm(dim=1)                                        # (B,256,256)


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
    print("[R8-D0] frozen depth-12 ViT built (corrected DeiT loader)")
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
        from run4_detail_interface_audit import top32_stats
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
        # 逐图累积：pixel score maps + labels（float32）；band 内 score/GT 列表
        u4_list, u12_list, lab_list = [], [], []
        b4_list, b12_list, bnd_lab_list = [], [], []
        s4_tok, s12_tok, g_tok = [], [], []
        ue4_list, ue12_list, lab4_list = [], [], []
        proj_w = model.encoder.vit.patch_embed.proj.weight.data
        with torch.no_grad():
            for img, target in loader:
                pre = img[:, 0:3].cuda().float()
                post = img[:, 3:6].cuda().float()
                lab = target.cuda().float()                      # (B,1,256,256) 0/1
                o1 = token_sources(model.encoder.vit, pre)
                o2 = token_sources(model.encoder.vit, post)
                s4 = 1.0 - F.cosine_similarity(o1["B4"], o2["B4"], dim=-1, eps=1e-8)   # (B,256)
                s12 = 1.0 - F.cosine_similarity(o1["B12"], o2["B12"], dim=-1, eps=1e-8)
                u4 = F.interpolate(torch.from_numpy(rank_normalize_np(s4.cpu().numpy())).float().cuda()
                                   .reshape(-1, 1, 16, 16), size=(256, 256), mode="bilinear",
                                   align_corners=False)                                  # (B,1,256,256)
                u12 = F.interpolate(torch.from_numpy(rank_normalize_np(s12.cpu().numpy())).float().cuda()
                                    .reshape(-1, 1, 16, 16), size=(256, 256), mode="bilinear",
                                    align_corners=False)
                u4_list.append(u4.squeeze(1).cpu().numpy())
                u12_list.append(u12.squeeze(1).cpu().numpy())
                lab_list.append(lab.squeeze(1).cpu().numpy())

                band = boundary_band(lab)                          # (B,1,256,256)
                b4_list.append(u4[band].cpu().numpy())
                b12_list.append(u12[band].cpu().numpy())
                bnd_lab_list.append(lab[band].cpu().numpy())

                s4_tok.append(s4.cpu().numpy())
                s12_tok.append(s12.cpu().numpy())
                g_tok.append(F.avg_pool2d(lab, 16).view(lab.shape[0], -1).cpu().numpy())

                # M4：固定 patch-unembedding（诊断）
                d4 = torch.abs(o1["B4"] - o2["B4"]).transpose(1, 2).reshape(-1, 192, 16, 16)
                d12 = torch.abs(o1["B12"] - o2["B12"]).transpose(1, 2).reshape(-1, 192, 16, 16)
                ue4_list.append(unembed_score(d4, proj_w).cpu().numpy())
                ue12_list.append(unembed_score(d12, proj_w).cpu().numpy())
                lab4_list.append(lab.squeeze(1).cpu().numpy())

        U4 = np.concatenate(u4_list, axis=0)
        U12 = np.concatenate(u12_list, axis=0)
        LAB = np.concatenate(lab_list, axis=0)
        B4b = np.concatenate(b4_list, axis=0)
        B12b = np.concatenate(b12_list, axis=0)
        BND = np.concatenate(bnd_lab_list, axis=0)
        S4t = np.concatenate(s4_tok, axis=0)
        S12t = np.concatenate(s12_tok, axis=0)
        Gt = np.concatenate(g_tok, axis=0)
        UE4 = np.concatenate(ue4_list, axis=0)
        UE12 = np.concatenate(ue12_list, axis=0)

        pr4 = pixel_pr_auc(U4, LAB)
        pr12 = pixel_pr_auc(U12, LAB)
        b4 = pixel_pr_auc(B4b, BND)
        b12 = pixel_pr_auc(B12b, BND)
        m4_4 = pixel_pr_auc(UE4, LAB)
        m4_12 = pixel_pr_auc(UE12, LAB)
        print(f"  [M1-pixel]     B4={pr4:.4f}  B12={pr12:.4f}  lift={pr4 - pr12:+.4f}")
        print(f"  [M2-boundary]  B4={b4:.4f}  B12={b12:.4f}  lift={b4 - b12:+.4f}")
        print(f"  [M4-unembed]   B4={m4_4:.4f}  B12={m4_12:.4f}  (diagnostic only)")

        # M3 分桶（token 级，诊断）
        from run6_semantic_token_audit import bucketed
        n_changed = (Gt > 0).sum(axis=1)
        for name, S in (("B4", S4t), ("B12", S12t)):
            b = bucketed(S, Gt, n_changed)
            print(f"  [M3-{name}] 1-16 hit/cov={b['1-16'][0]:.3f}/{b['1-16'][1]:.3f} "
                  f"17-64={b['17-64'][0]:.3f}/{b['17-64'][1]:.3f} >64={b['>64'][0]:.3f}/{b['>64'][1]:.3f}")

        results[ds] = {"pix4": pr4, "pix12": pr12, "bnd4": b4, "bnd12": b12}

    print("\n=== R8-D0 预注册 gate（§4.4） ===")
    c1 = c2 = c3 = 0
    for ds in datasets:
        r = results[ds]
        g1 = r["pix4"] >= r["pix12"] + 0.03
        g2 = r["bnd4"] >= r["bnd12"] + 0.02
        ok3 = (r["pix4"] >= r["pix12"] - 0.01) and (r["bnd4"] >= r["bnd12"] - 0.01)
        c1 += int(g1)
        c2 += int(g2)
        c3 += int(ok3)
        print(f"  {ds:14s} | G1_pixel: {r['pix4']:.4f}>={r['pix12'] + 0.03:.4f} {g1} | "
              f"G2_boundary: {r['bnd4']:.4f}>={r['bnd12'] + 0.02:.4f} {g2} | C3(no drop>0.01): {ok3}")
    n = len(datasets)
    ok = (c1 >= 0.75 * n) and (c2 >= 0.75 * n) and (c3 == n)
    print(f"  C1={c1}/{n}  C2={c2}/{n}  C3={c3}/{n}")
    print(f"  [R8-D0-GATE] {'PASS' if ok else 'FAIL'}")
    print("[R8-D0] DONE")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
