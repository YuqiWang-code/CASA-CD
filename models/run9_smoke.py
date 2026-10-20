#!/usr/bin/env python
"""Run9 HSAM 真实数据 smoke：在**真实 VMamba 特征**上回答 4 个开跑前必须回答的问题。

    cd models
    python run9_smoke.py --dataset-root /share_datasets/CD/CDD-CD-256 \
        --train-list /share_datasets/CD/CDD-CD-256/list/train.txt \
        --vmamba-pretrained-weight-path ../pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth \
        --output-dir /tmp/run9/smoke --steps 6

1. **预训练是否逐位加载**（`worst_diff == 0`）——任何 run 的前提；
2. **HSAM 在真实特征上到底做了多少事**（`agg/feat`、注意力熵、`prob_max`、半径 `cv`）：
   CPU 上用标准化随机特征只得到"最坏情形"估计，真实特征里 token 之间有相关性，
   聚合项会更大。若真实读数仍然 ≈0，说明这个模块对本项目**没有作用**，不该开跑；
3. **吞吐代价**：`hsam=0/1` 各跑同样步数，给 step/s；
4. **峰值显存**：决定能不能与同门共卡。

不训练、不写 checkpoint、不改任何文件；只打印读数。
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dataset.Transforms as myTransforms  # noqa: E402
import dataset.dataset as myDataLoader  # noqa: E402
from model.casa_vmamba_net_v2 import CASAVMambaSTRNetV2  # noqa: E402
from model.utils import BCEDiceLoss, init_seed  # noqa: E402

MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


def build_transform():
    return myTransforms.Compose([myTransforms.Normalize(mean=MEAN6, std=STD6),
                                 myTransforms.Scale(256, 256),
                                 myTransforms.ToTensor()])


def make_loader(path, root, list_path, batch_size, workers):
    ds = myDataLoader.Dataset(file_root=root, list_path=list_path,
                              transform=build_transform(), mode="train")
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=workers, pin_memory=True, drop_last=True)


def make_model(args, hsam, device):
    torch.manual_seed(args.seed)
    m = CASAVMambaSTRNetV2(
        vmamba_pretrained_path=args.vmamba_pretrained_weight_path,
        hqkv=False, str_dim=args.str_dim, geometry="lattice",
        hgsl=False, inject_mode="dcr1",
        rep_mode="full", fs_tar=False,
        hsam=hsam,
        hsam_stages=tuple(int(s) for s in args.hsam_stages.split(",") if s != ""),
        hsam_logit_norm=args.hsam_logit_norm,
        hsam_radius=args.hsam_radius,
        hsam_feature_source=args.hsam_feature_source,
    ).float().to(device)
    return m


def run(loader, args, hsam, device, out):
    tag = f"hsam={int(hsam)}"
    print(f"\n=== {tag} ===")
    m = make_model(args, hsam, device)
    st = m.pretrain_load_stats() or {}
    print(f"  [PRETRAIN-LOAD] retained={st.get('retained')}/{st.get('pretrained_keys')} "
          f"worst_diff={st.get('worst_diff')} missing_new={len(st.get('missing_new', []))}")
    pb = m.param_breakdown()
    print(f"  [PARAMS] total={pb['total']:,} backbone={pb['backbone']:,} "
          f"hsam={pb['hsam']:,} non_backbone={pb['non_backbone']:,}")
    out[tag] = {"pretrain": {"retained": st.get("retained"),
                             "keys": st.get("pretrained_keys"),
                             "worst_diff": st.get("worst_diff")},
                "params": pb}

    opt = torch.optim.Adam(m.parameters(), args.lr)
    if hsam:
        m.set_hsam_collect(True)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    it = iter(loader)
    times, diags = [], []
    m.train()
    for i in range(args.steps):
        try:
            img, label = next(it)
        except StopIteration:
            it = iter(loader)
            img, label = next(it)
        pre, post = img[:, 0:3].to(device), img[:, 3:6].to(device)
        label = label.to(device).float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out_map = m(pre, post)
        loss = BCEDiceLoss(out_map, label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        if hsam:
            diags.append(m.hsam_diag())
        if i == 0:
            print(f"  [STEP0] loss={float(loss):.4f} finite={bool(torch.isfinite(loss))}")

    warm = 1 if len(times) > 2 else 0
    sps = (len(times) - warm) / max(1e-9, sum(times[warm:]))
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device.type == "cuda" else 0.0
    print(f"  [THROUGHPUT] steps={len(times)} step/s={sps:.4f} "
          f"mean_step={sum(times) / len(times) * 1000:.1f}ms")
    print(f"  [VRAM] peak_allocated={peak:.0f} MiB")
    out[tag]["throughput"] = {"step_per_s": sps, "steps": len(times)}
    out[tag]["vram_peak_mib"] = peak

    if hsam and diags:
        last = diags[-1]
        print("  [HSAM real-feature readings]")
        for s in last.get("stages", []):
            d = last.get(f"stage{s}") or {}
            print(f"    stage{s}: N={d.get('tokens')} hd_mean={d.get('hd_mean', 0):.4f} "
                  f"hd_cv={d.get('hd_cv', 0):.3e} r_cv={d.get('radius_cv', 0):.3e} "
                  f"prob_max={d.get('attn_prob_max', 0):.4e} "
                  f"ent={d.get('attn_entropy_norm', 0):.4f} "
                  f"ent_paper={d.get('attn_entropy_paper', 0):.4f} "
                  f"rank_corr={d.get('rank_corr_hd_vs_cos', 0):.6f} "
                  f"agg/feat={d.get('agg_over_feat', 0):.4e} "
                  f"out/in={d.get('out_over_in', 0):.4f} "
                  f"clip={d.get('atanh_clip_frac', 0):.2e} "
                  f"finite={int(d.get('finite', 0))}")
        out[tag]["hsam_diag"] = {f"stage{s}": last.get(f"stage{s}") for s in last.get("stages", [])}

    del m, opt
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--vmamba-pretrained-weight-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--str-dim", type=int, default=192)
    ap.add_argument("--hsam-stages", default="2,3")
    ap.add_argument("--hsam-logit-norm", default="std", choices=["std", "paper", "raw"])
    ap.add_argument("--hsam-radius", default="feature_energy",
                    choices=["feature_energy", "local_dispersion"])
    ap.add_argument("--hsam-feature-source", default="pre_norm",
                    choices=["pre_norm", "post_norm"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    init_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    loader = make_loader(args.dataset_root, args.dataset_root, args.train_list,
                         args.batch_size, args.num_workers)
    print(f"=== Run9 HSAM smoke | device={device} | batches={len(loader)} "
          f"| batch={args.batch_size} | stages={args.hsam_stages} "
          f"| logit_norm={args.hsam_logit_norm} ===")
    out = {}
    for hsam in (False, True):
        out = run(loader, args, hsam, device, out)
    p = os.path.join(args.output_dir, "smoke.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[out] {p}")
    a = out.get("hsam=0", {}).get("throughput", {}).get("step_per_s")
    b = out.get("hsam=1", {}).get("throughput", {}).get("step_per_s")
    if a and b:
        print(f"[COST] HSAM 吞吐代价 = {(a / b - 1.0) * 100:+.2f}% "
              f"(hsam=0 {a:.4f} -> hsam=1 {b:.4f} step/s)")
    print(f"[VRAM] hsam=0 {out.get('hsam=0', {}).get('vram_peak_mib', 0):.0f} MiB -> "
          f"hsam=1 {out.get('hsam=1', {}).get('vram_peak_mib', 0):.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
