"""对象级小目标指标复算（Exp1 机制证据；4 连通、native 256²、与 Diag1 同口径）。

同时支持两种架构（用 `--pkg-root` 选择代码树）：
  --arch run8  --pkg-root <repo>/models      → Run8 `CTRL_R8`（VMamba + TAR/DCR）
  --arch stgeo --pkg-root <repo>/models_v1   → Exp1 `STG_INJ`（+ SPD(2)/H² 证据）
两者**同一份指标实现**（`analyse/tvim_object_metrics.py`），因此数字可直接对比。

用法（服务器，GPU1）：
  CUDA_VISIBLE_DEVICES=1 python eval_object_metrics.py --arch stgeo --pkg-root ../models_v1 \
      --ckpt /share/.../best_F1=0.9114.pth --dataset LEVIR-CD-256 \
      --dataset-root /share_datasets/CD/LEVIR-CD-256 --test-list .../list/test.txt --out x.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

MEAN6 = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD6 = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg-root", required=True)
    ap.add_argument("--arch", required=True, choices=["run8", "stgeo"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--test-list", required=True)
    ap.add_argument("--analyse-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.analyse_dir))
    from tvim_object_metrics import ObjectMetricAccumulator, METRICS_PROTOCOL  # noqa: E402

    sys.path.insert(0, os.path.abspath(args.pkg_root))
    torch.manual_seed(16)
    if args.arch == "run8":
        from model.casa_vmamba_net import CASAVMambaSTRNet as Net
        model = Net(vmamba_pretrained_path=None, hqkv=False, hgsl=False,
                    inject_mode="none").float().cuda()
    else:
        from model.casa_vmamba_stgeo_net import CASAVMambaSTGeoNet as Net
        model = Net(vmamba_pretrained_path=None, stgeo="inject").float().cuda()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.switch_to_deploy()
    model.eval()
    print(f"[LOAD] {os.path.basename(args.ckpt)} missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)

    import dataset.dataset as myData  # noqa: E402
    import dataset.Transforms as myT  # noqa: E402
    tf = myT.Compose([myT.Normalize(mean=MEAN6, std=STD6),
                      myT.Scale(256, 256), myT.ToTensor()])
    ds = myData.Dataset(file_root=args.dataset_root, list_path=args.test_list,
                        transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True)
    acc = ObjectMetricAccumulator(connectivity=4, bands=(2, 4))
    with torch.no_grad():
        for bi, (img, target) in enumerate(loader):
            pre, post = img[:, 0:3].cuda(), img[:, 3:6].cuda()
            prob = model(pre.float(), post.float())
            pred = (prob[:, 0] > 0.5).cpu().numpy().astype(np.uint8)
            gt = target[:, 0].cpu().numpy().astype(np.uint8)
            for k in range(pred.shape[0]):
                acc.add(pred[k], gt[k], name=f"{bi}_{k}")
            if (bi + 1) % 50 == 0:
                print(f"  batch {bi + 1}/{len(loader)}", flush=True)
    s = acc.summary(label=args.dataset)
    s["_meta"] = {"arch": args.arch, "ckpt": os.path.basename(args.ckpt),
                  "protocol": METRICS_PROTOCOL["version"]}
    json.dump(s, open(args.out, "w", encoding="utf-8"), indent=2, default=str)

    lg = s["legacy_groups"]
    om = s["object"]
    print(f"\n=== {args.dataset} @ {os.path.basename(args.ckpt)} ===")
    print(f"pixel F1={s['pixel']['F1']:.4f} R={s['pixel']['recall']:.4f} "
          f"P={s['pixel']['precision']:.4f}")
    for g in ("small", "medium", "large"):
        d = lg[g]
        print(f"  {g:6s} n={d['n_objects']:5d} pooledR={d['pixel_recall_micro']:.4f} "
              f"macroR={d['object_macro_pixel_recall']:.4f} Hit@25={d['hit25']:.4f}")
    small_um = sum(om["unmatched_pred_cc_count_by_area"][k]
                   for k in ("tiny_1_15", "tiny_16_63", "small_64_255"))
    print(f"  objects: n_gt={om['n_gt']} n_pred={om['n_pred']} "
          f"ObjP_loose={om['ObjPrecision_loose']:.4f} ObjR_loose={om['ObjRecall_loose']:.4f} "
          f"ObjF1_loose={om['ObjF1_loose']:.4f}")
    print(f"  unmatched_pred_loose={om['unmatched_pred_loose']} unmatched<256px={small_um}")


if __name__ == "__main__":
    main()
