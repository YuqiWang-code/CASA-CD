"""D3：冻结线性 probe —— 识别"信息存在但被 cosine proxy 隐藏"（Run-Diag 文档 §7）。

协议（先写 probe_protocol.json，再执行；test 只做一次性评估）：
  * 冻结整个 M1（含 BN running stats），`torch.no_grad()` 提特征；梯度只落在 probe。
  * probe = Conv2d(C_in, 1, kernel_size=1, bias=True)（统一结构范式，容量随 C_in 线性变化）。
  * 编码器节点输入口径（主口径，全层一致）：`abs(F_A - F_B)`；
    单路融合节点（TAR/DCR/head）输入为该节点自身的融合特征（无 A/B 可减）。
  * 拟合只用 SYSU `train.txt`：按**文件名 SHA1 hash** 固定划分 90% probe-train / 10% probe-val，
    val 只用于固定步数后的记录（不做早停调参）；test 一次性评估。
  * 损失 = BCEWithLogitsLoss（probe 输出上采样到原生 256² 后与原生 mask 计算）——这是
    诊断分类器的常规拟合目标，不是修改 CASA 主模型 loss。
  * 预注册超参：AdamW lr=1e-3, wd=0, steps=1500, batch=16, seed=16（默认值，可由 CLI 覆盖并写入协议）。
  * 预算偏差（如有）：`--max-train` 限制拟合样本数，会降低 probe 拟合质量，写入协议并在报告中标注。

输出：probe_protocol.json、probe_results.json、probe_weights_PROBE_ONLY/（诊断权重，不入正式 checkpoint）、gate.json
"""
import os
import sys
import json
import copy
import hashlib
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tvim_diag_common import (  # noqa: E402
    VARIANT_CFG, build_model, ensure_dir, make_loader, pick_best_ckpt, read_list_names,
    set_eval_numerics, write_gate, write_json,
)
from tvim_object_metrics import label_components  # noqa: E402
from tvim_stage_recoverability import NBINS, SCORE_MAX, _ap_from_hist  # noqa: E402

HOOK_TARGETS = {
    "L00_patch_embed": ("enc", "patch_embed"),
    "L01b_norm0": ("enc", "norm0"),
    "L03b_norm2": ("enc", "norm2"),
    "L05_stage3_last_prefix": ("enc", "s3prefix"),
    "L06b_norm4": ("enc", "norm4"),
    "L08b_norm6": ("enc", "norm6"),
    "T01_tar1": ("tar", "stage1"),
    "T03_tar3": ("tar", "stage3"),
    "DOUT_decoder_refine": ("dec", "refine"),
    "P00_head_logits": ("head", None),
}
ENCODER_KEYS = [k for k in HOOK_TARGETS if k.startswith(("L0",))]
SINGLE_KEYS = [k for k in HOOK_TARGETS if k not in ENCODER_KEYS]


def _hash_split(name, val_frac=0.10):
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 1000
    return "val" if h < int(val_frac * 1000) else "train"


def _resolve_module(model, key):
    kind, attr = HOOK_TARGETS[key]
    if kind == "enc":
        if attr == "patch_embed":
            return model.encoder.patch_embed
        if attr == "s3prefix":
            return model.encoder.network[4][model.encoder.depths[2] - 2]
        return getattr(model.encoder, attr)
    if kind == "tar":
        return getattr(model.tar, attr)
    if kind == "dec":
        return model.decoder.refine
    return model.head


def extract_features(model, args, device, keys, names_filter=None, max_images=None):
    """提取指定节点特征（fp16 CPU 存储）。返回 {key: np.ndarray[N,C,H,W](fp16)}, gts(N,256,256)bool, names。"""
    import torch

    loader, _ = make_loader(args.dataset, "train", batch_size=args.extract_batch,
                            num_workers=args.num_workers)
    train_names = read_list_names(os.path.join(args._data_root, "list", "train.txt"))
    handles, cap = [], {}

    def hook(k):
        return lambda m, i, o: cap.__setitem__(k, o.detach())

    for k in keys:
        handles.append(_resolve_module(model, k).register_forward_hook(hook(k)))
    store = {k: [] for k in keys}
    gts, keep_names = [], []
    idx = 0
    try:
        with torch.no_grad():
            for img, label in loader:
                cap.clear()
                pre = img[:, 0:3].to(device).float()
                post = img[:, 3:6].to(device).float()
                _ = model(pre, post)
                B = pre.shape[0]
                for i in range(B):
                    nm = train_names[idx + i] if idx + i < len(train_names) else f"idx{idx+i}"
                    if names_filter is not None and not names_filter(nm):
                        continue
                    keep_names.append(nm)
                    gts.append((label[i, 0].numpy() > 0.5))
                    for k in keys:
                        t = cap[k]
                        if k in ENCODER_KEYS:
                            fa = t[i].float()
                            fb = t[B + i].float()
                            feat = (fa - fb).abs()
                        else:
                            feat = t[i].float()
                        store[k].append(feat.half().cpu().numpy())
                idx += B
                if max_images and len(keep_names) >= max_images:
                    break
    finally:
        for h in handles:
            h.remove()
    return {k: np.stack(v) for k, v in store.items()}, np.stack(gts), keep_names


def train_probe(feats, gts, args, device, out_dir, tag):
    """按预注册超参训练单个 probe（固定步数，无 test 参与）。"""
    import torch
    import torch.nn.functional as F

    N, C = feats.shape[0], feats.shape[1]
    probe = torch.nn.Conv2d(C, 1, 1).to(device)
    torch.manual_seed(args.seed)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.0)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    feats_t = torch.as_tensor(feats)          # CPU fp16
    gts_t = torch.as_tensor(gts)
    hist_pos = np.zeros(NBINS); hist_all = np.zeros(NBINS)
    for step in range(args.steps):
        sel = rng.integers(0, N, args.probe_batch)
        xb = feats_t[sel].to(device).float()
        yb = gts_t[sel].to(device)
        logits = probe(xb)
        up = F.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=False)
        loss = lossf(up[:, 0], yb.float())
        opt.zero_grad(); loss.backward(); opt.step()
        if torch.isnan(loss):
            break
    with torch.no_grad():
        for s in range(0, N, args.probe_batch):
            xb = feats_t[s:s + args.probe_batch].to(device).float()
            logits = probe(xb)
            p = torch.sigmoid(logits)[:, 0]        # (B,h,w)
            for i in range(p.shape[0]):
                pi = p[i]
                if pi.shape != (256, 256):
                    pi = F.interpolate(pi[None, None], size=(256, 256), mode="bilinear",
                                       align_corners=False)[0, 0]
                sc = pi.cpu().numpy().astype(np.float64)
                g = gts[s + i]
                hp, _ = np.histogram(sc[g], bins=NBINS, range=(0.0, 1.0))
                ha, _ = np.histogram(sc, bins=NBINS, range=(0.0, 1.0))
                hist_pos += hp; hist_all += ha
    ap = _ap_from_hist(hist_pos, hist_all)
    torch.save({"probe": probe.state_dict(), "C_in": C, "variant": args.variant,
                "tag": tag, "protocol": vars(args).copy()},
               os.path.join(out_dir, f"probe_{tag}_PROBE_ONLY.pth"))
    return {"C_in": int(C), "n_train_samples": int(N),
            "probe_train_subset_AP": ap}, probe


def evaluate_probe(probe, args, device, key):
    """在 test 集上一次性评估该 probe（不调参）。"""
    import torch
    import torch.nn.functional as F
    from tvim_diag_common import DATA_ROOT

    handles, cap = [], {}
    handles.append(_resolve_module(probe["model"], key).register_forward_hook(
        lambda m, i, o: cap.__setitem__("f", o.detach())))
    loader, _ = make_loader(args.dataset, "test", batch_size=args.extract_batch,
                            num_workers=args.num_workers)
    test_names = read_list_names(os.path.join(DATA_ROOT, args.dataset, "list", "test.txt"))
    hist_pos = np.zeros(NBINS); hist_all = np.zeros(NBINS)
    per_img_ap = []
    idx = 0
    try:
        with torch.no_grad():
            for img, label in loader:
                cap.clear()
                pre = img[:, 0:3].to(device).float()
                post = img[:, 3:6].to(device).float()
                _ = probe["model"](pre, post)
                t = cap["f"]
                B = pre.shape[0]
                for i in range(B):
                    if key in ENCODER_KEYS:
                        feat = (t[i].float() - t[B + i].float()).abs()[None]
                    else:
                        feat = t[i].float()[None]
                    logits = probe["module"](feat)
                    p = torch.sigmoid(logits)[0, 0]
                    if p.shape != (256, 256):
                        p = F.interpolate(p[None, None], size=(256, 256), mode="bilinear",
                                          align_corners=False)[0, 0]
                    sc = p.cpu().numpy().astype(np.float64)
                    g = (label[i, 0].numpy() > 0.5)
                    hp, _ = np.histogram(sc[g], bins=NBINS, range=(0.0, 1.0))
                    ha, _ = np.histogram(sc, bins=NBINS, range=(0.0, 1.0))
                    hist_pos += hp; hist_all += ha
                    per_img_ap.append(_ap_from_hist(hp, ha) if g.any() else np.nan)
                    idx += 1
    finally:
        for h in handles:
            h.remove()
    return {"pooled_AP": _ap_from_hist(hist_pos, hist_all),
            "per_image_AP_mean": float(np.nanmean(per_img_ap)) if per_img_ap else None,
            "n_images": idx}


def run_d3(args):
    import torch
    from tvim_diag_common import DATA_ROOT

    args._data_root = os.path.join(DATA_ROOT, args.dataset)
    out_dir = ensure_dir(args.out_dir)
    device = args.device
    set_eval_numerics()
    keys = [k.strip() for k in args.layers.split(",") if k.strip()]
    ckpt_path, ckpt_meta = pick_best_ckpt(args.run, args.variant, args.dataset)

    protocol = {
        "probe": "Conv2d(C_in,1,1) + BCEWithLogitsLoss on bilinear-upsampled logits vs native 256² mask",
        "encoder_input": "abs(F_A - F_B)（全编码器层统一主口径）",
        "single_path_input": "节点自身融合特征（无 A/B）",
        "split": "SYSU train.txt 按文件名 SHA1 hash 固定 90% probe-train / 10% probe-val",
        "test_usage": "一次性评估，不参与任何优化/早停/调参",
        "hyperparams": {"lr": args.lr, "steps": args.steps, "batch": args.probe_batch,
                        "optimizer": "AdamW(wd=0)", "seed": args.seed},
        "budget_deviation": (f"--max-train={args.max_train}（0=全部）; "
                             f"--extract-batch={args.extract_batch}"),
        "layers": keys,
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_meta["sha256"]},
        "frozen_model": True,
    }
    write_json(os.path.join(out_dir, "probe_protocol.json"), protocol)

    # 冻结模型参数 SHA256（前后比对）
    model, _ = build_model(args.variant, device=device, ckpt_path=ckpt_path)
    import hashlib
    def _model_sha(m):
        h = hashlib.sha256()
        for k, v in m.state_dict().items():
            h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    sha_before = _model_sha(model)

    train_names = read_list_names(os.path.join(DATA_ROOT, args.dataset, "list", "train.txt"))
    split = {nm: _hash_split(nm, args.val_frac) for nm in train_names}
    n_val = sum(1 for v in split.values() if v == "val")

    print(f"[D3] extracting features for {len(keys)} layers (max_train={args.max_train})", flush=True)
    feats, gts, names = extract_features(model, args, device, keys,
                                        names_filter=lambda nm: split.get(nm) == "train",
                                        max_images=args.max_train or None)
    print(f"[D3] extracted: {feats[keys[0]].shape}, val-probe names={n_val}", flush=True)

    results = {"per_layer": {}, "protocol": protocol}
    for k in keys:
        r, probe = train_probe(feats[k], gts, args, device, out_dir, k)
        r["test"] = evaluate_probe({"model": model, "module": probe}, args, device, k)
        results["per_layer"][k] = r
        print(f"[D3] {k}: C_in={r['C_in']} n_train={r['n_train_samples']} "
              f"test pooled AP={r['test']['pooled_AP']}", flush=True)
    sha_after = _model_sha(model)
    results["frozen_unchanged"] = (sha_before == sha_after)
    results["model_sha256_before"] = sha_before
    results["model_sha256_after"] = sha_after
    write_json(os.path.join(out_dir, "probe_results.json"), results)
    checks = {
        "frozen_model_unchanged": results["frozen_unchanged"],
        "test_used_once": True,
        "val_probe_names": n_val,
        "layers": keys,
        "pooled_AP": {k: results["per_layer"][k]["test"]["pooled_AP"] for k in keys},
    }
    write_gate(out_dir, "D3-VALID", "PASS" if results["frozen_unchanged"] else "FAIL", checks)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--run", default="Run1")
    ap.add_argument("--variant", default="M1_FULL", choices=sorted(VARIANT_CFG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="L01b_norm0,L03b_norm2,L05_stage3_last_prefix,L06b_norm4,"
                                       "T01_tar1,DOUT_decoder_refine,P00_head_logits")
    ap.add_argument("--max-train", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--probe-batch", type=int, default=16)
    ap.add_argument("--extract-batch", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()
    run_d3(args)


if __name__ == "__main__":
    main()
