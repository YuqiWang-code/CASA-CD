"""Run11 TASS-D0: Task-Adaptive Spatial Stem 的 raw source feasibility gate（零训练，四数据集）。

预注册（CASA-CD_下一步方案_Run11_TASS_设计与预注册.md §7）：
回答「原始像素网格在 1/4–1/16 上是否含有相对 B4 互补的边界带 evidence」——
参数自由 raw-RGB 差分代理：
    R64 = mean_c |AvgPool4(A) - AvgPool4(B)|   (64x64)
    R32 = mean_c |AvgPool8(A) - AvgPool8(B)|   (32x32)
    R16 = mean_c |AvgPool16(A) - AvgPool16(B)| (16x16)
各自 bilinear -> 256x256 -> 逐图像素 rank -> [0,1]；
    Rspatial = mean(rank(R64), rank(R32), rank(R16))
    Rfuse    = 0.5*rank_pixel(B4_up) + 0.5*Rspatial   （0.5/0.5 运行前固定，不 sweep）

Gate（§7.3）：
  G0 审计有效性：ViT4 checksum == corrected DeiT；样本数与 list/test.txt 一致；
     label 二值化 gray>=128；B4-only 边界带 PR-AUC 复现 SF-D0（CDD 0.5216 /
     LEVIR 0.5369 / SYSU 0.6113 / WHU 0.5718，±0.01）。不过 -> AUDIT-INVALID（修审计）。
  G1 fine-scale 边界互补：PRbnd(Rfuse) - PRbnd(B4) >= +0.020，SYSU 必过且 >=2/4。
  G2 不靠全局伪变化换 lift：PRpix(Rfuse) >= PRpix(B4) - 0.010，SYSU 必过且 >=3/4。
裁决：G1/G2 任一 FAIL -> [TASS-D0-GATE] FAIL -> 不实现/不训练 TASS，转方向 (c)。
返回码：PASS=0 / FAIL=2 / AUDIT_INVALID=3。

注（口径）：B4 走与训练完全一致的归一化（BGR 域 Normalize 后 ToTensor 翻转为 RGB；
本审计用 raw [0,1] RGB loader + 手动按 RGB 通道序复现归一化数学，B4 复现自检兜底）。
raw proxy 用通道均值差，对通道顺序不变。
"""
import os
import sys
import json
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
from run8_b4_dense_recoverability_audit import boundary_band, pixel_pr_auc, rank_normalize_np

import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms

# B4-only 边界带 PR-AUC 参考（SF-D0 / R8-D0 记录，±0.01 复现窗）
B4_BND_REF = {"CDD-CD-256": 0.5216, "LEVIR-CD-256": 0.5369,
              "SYSU-CD-256": 0.6113, "WHU-CD-256": 0.5718}


class _EncoderHolder(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.encoder = enc


def make_raw_loader(dataset_root, batch_size, num_workers):
    """Raw [0,1] RGB loader（无 Normalize；label 未二值化，审计内手动 gray>=128）。"""
    transform = myTransforms.Compose([
        myTransforms.Scale(256, 256),
        myTransforms.ToTensor(),
    ])
    ds = myDataLoader.Dataset(file_root=dataset_root,
                              list_path=os.path.join(dataset_root, "list", "test.txt"),
                              transform=transform)
    return torch.utils.data.DataLoader(ds, shuffle=False, batch_size=batch_size,
                                       num_workers=num_workers, pin_memory=True)


def rank_normalize_pixel_map(m):
    """逐图对 256x256 像素 map 做 ordinal rank -> [0,1]（(B,256,256) float64 in/out）。"""
    out = np.empty_like(m, dtype=np.float64)
    for i in range(m.shape[0]):
        s = np.ascontiguousarray(m[i].reshape(-1), dtype=np.float64)
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty_like(s)
        ranks[order] = np.arange(s.shape[0], dtype=np.float64)
        out[i] = (ranks / max(s.shape[0] - 1, 1)).reshape(m.shape[1], m.shape[2])
    return out


def run_dataset(vit, raw_loader, norm_loader, device):
    """ViT 路径用参考管线 loader（Normalize→Scale→ToTensor，与 SF-D0 逐位一致）；
    raw 代理用 raw loader（Scale→ToTensor 的 [0,1] RGB，通道均值差对通道序不变）。"""
    acc = {"b4": {"map": [], "lab": [], "bnd_s": [], "bnd_l": []},
           "rfuse": {"map": [], "lab": [], "bnd_s": [], "bnd_l": []},
           "rspatial": {"map": [], "lab": []}}
    with torch.no_grad():
        for (img_raw, _t_raw), (img_norm, target_norm) in zip(raw_loader, norm_loader):
            raw = img_raw.to(device).float()                   # (B,6,256,256) [0,1] RGB
            lab = target_norm.to(device).float()               # Normalize 已按 gray>=128 二值化
            norm = img_norm.to(device).float()                 # 参考管线输出
            pre = norm[:, 0:3]
            post = norm[:, 3:6]

            # B4 token change score（SF-D0 口径：rank over tokens -> bilinear）
            o1 = token_sources(vit, pre)
            o2 = token_sources(vit, post)
            s4 = 1.0 - F.cosine_similarity(o1["B4"], o2["B4"], dim=-1, eps=1e-8)  # (B,256)
            r4 = torch.from_numpy(rank_normalize_np(s4.cpu().numpy())).float().to(device)
            b4_up = F.interpolate(r4.reshape(-1, 1, 16, 16), size=(256, 256),
                                  mode="bilinear", align_corners=False)            # (B,1,256,256)

            # raw spatial proxy at 1/4 / 1/8 / 1/16
            d = (raw[:, 0:3] - raw[:, 3:6]).abs().mean(dim=1, keepdim=True)        # (B,1,256,256)
            r64 = F.avg_pool2d(d, kernel_size=4, stride=4)
            r32 = F.avg_pool2d(d, kernel_size=8, stride=8)
            r16 = F.avg_pool2d(d, kernel_size=16, stride=16)
            maps = [F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
                    for x in (r64, r32, r16)]                                       # 3x (B,256,256)
            ranked = [rank_normalize_pixel_map(m.astype(np.float64)) for m in maps]
            rspatial = np.mean(np.stack(ranked, axis=0), axis=0)                    # (B,256,256)
            b4_up_np = rank_normalize_pixel_map(b4_up.squeeze(1).cpu().numpy().astype(np.float64))
            rfuse = 0.5 * b4_up_np + 0.5 * rspatial                                # (B,256,256)

            band = boundary_band(lab)                                               # (B,1,256,256) bool
            for key, m in (("b4", b4_up.squeeze(1).cpu().numpy()),
                           ("rfuse", rfuse),
                           ("rspatial", rspatial)):
                acc[key]["map"].append(m)
                acc[key]["lab"].append(lab.squeeze(1).cpu().numpy())
            for key in ("b4", "rfuse"):
                m = acc[key]["map"][-1]
                acc[key]["bnd_s"].append(m[band.squeeze(1).cpu().numpy()])
                acc[key]["bnd_l"].append(lab.squeeze(1).cpu().numpy()[band.squeeze(1).cpu().numpy()])

    out = {}
    for key in acc:
        M = np.concatenate(acc[key]["map"], axis=0)
        L = np.concatenate(acc[key]["lab"], axis=0)
        out[key] = {"pix": pixel_pr_auc(M, L)}
        if key in ("b4", "rfuse"):
            Bs = np.concatenate(acc[key]["bnd_s"], axis=0)
            Bl = np.concatenate(acc[key]["bnd_l"], axis=0)
            out[key]["bnd"] = pixel_pr_auc(Bs, Bl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--datasets", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    holder = Trainer("tiny", pretrained_path=args.pretrained_weight_path, resnet_pretrained=False,
                     mode="baseline", vit_depth=4, detail_mode="none_b4",
                     head_mode="b4_spe").float().cuda().eval()
    print("[TASS-D0] frozen depth-4 ViT built (corrected DeiT loader)")
    if not checksum_vit(_EncoderHolder(holder.encoder), args.pretrained_weight_path, depth=4):
        return 3

    results = {}
    g0_repro = {}
    for ds in datasets:
        print(f"\n=== {ds} ===", flush=True)
        raw_loader = make_raw_loader(os.path.join(args.data_root, ds), args.batch_size, args.num_workers)
        norm_loader = make_loader(os.path.join(args.data_root, ds), args.batch_size, args.num_workers)
        # G0: 样本数一致
        with open(os.path.join(args.data_root, ds, "list", "test.txt"), encoding="utf-8") as f:
            n_list = sum(1 for _ in f if _.strip())
        n_raw = len(raw_loader.dataset)
        n_norm = len(norm_loader.dataset)
        print(f"  [G0-sample] list/test.txt={n_list}  raw={n_raw}  norm={n_norm}")
        if not (n_list == n_raw == n_norm):
            print("[AUDIT-INVALID] sample count mismatch")
            return 3

        res = run_dataset(holder.encoder.vit, raw_loader, norm_loader, device)
        results[ds] = res
        for key in ("b4", "rfuse", "rspatial"):
            bnd = res[key].get("bnd", float("nan"))
            print(f"  [{key}] pixel PR-AUC={res[key]['pix']:.4f}  boundary PR-AUC={bnd:.4f}")
        ref = B4_BND_REF[ds]
        repro = abs(res["b4"]["bnd"] - ref) <= 0.01
        g0_repro[ds] = repro
        print(f"  [G0-repro] B4 boundary {res['b4']['bnd']:.4f} vs SF-D0 {ref:.4f} -> "
              f"{'OK' if repro else 'MISMATCH'}")

    print("\n=== TASS-D0 预注册 gate（§7.3） ===")
    g0 = all(g0_repro.values())
    g1_count = g1_sysu = g2_count = g2_sysu = 0
    for ds in datasets:
        r = results[ds]
        g1 = r["rfuse"]["bnd"] >= r["b4"]["bnd"] + 0.020
        g2 = r["rfuse"]["pix"] >= r["b4"]["pix"] - 0.010
        g1_count += int(g1)
        g2_count += int(g2)
        if ds == "SYSU-CD-256":
            g1_sysu = g1
            g2_sysu = g2
        print(f"  {ds:14s} | G1 bnd lift {r['rfuse']['bnd'] - r['b4']['bnd']:+.4f} >= +0.020: {g1} | "
              f"G2 pix {r['rfuse']['pix']:.4f} >= {r['b4']['pix'] - 0.010:.4f}: {g2}")
    n = len(datasets)
    g1_ok = g1_sysu and g1_count >= max(1, int(0.5 * n))
    g2_ok = g2_sysu and g2_count >= int(0.75 * n)
    print(f"  G0={g0}  G1={g1_count}/{n} (SYSU={g1_sysu})  G2={g2_count}/{n} (SYSU={g2_sysu})")
    ok = g0 and g1_ok and g2_ok
    print(f"  [TASS-D0-GATE] {'PASS' if ok else 'FAIL'}")
    print("[TASS-D0] DONE")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "tass_d0_results.json"), "w", encoding="utf-8") as f:
            json.dump({"results": results, "g0": g0, "g1": {"count": g1_count, "sysu": g1_sysu},
                       "g2": {"count": g2_count, "sysu": g2_sysu}, "verdict": "PASS" if ok else "FAIL"}, f, indent=2)
        print(f"[TASS-D0] results saved to {args.out_dir}/tass_d0_results.json")

    if not g0:
        return 3
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
