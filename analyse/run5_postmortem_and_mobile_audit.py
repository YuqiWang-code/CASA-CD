"""Run5-D0：零训练成本的 postmortem（H1/H2/H3）+ MobileDetail-P3 raw gate。

对应方案（CASA-CD_Run5_成熟预训练MicroDetail_SGDP可执行方案.md §5-12）：

  H2（§6）：PSD pretrained stem 是否被重写——R4-1 ResNet stem / R4-2d PSD stem
            vs ImageNet 初始 stem 的权重 rel_L2/cosine + 激活漂移（前 256 对）。
  H1（§7）：TileAdapter inference-only 替换 R4-2 的 learned adapters，
            完整 SYSU test 一次 → ΔF1/ΔIoU vs 82.30/69.92。
  H3（§8）：R4-1/R4-2/R4-2d 的 FI 三尺度接口 hook（post-LN/KV/Q/AttnOut RMS + 注意力熵），
            r_kv / r_a / |Δentropy| 相对 R4-1 判定 interface geometry mismatch。
  D（§10-12）：MobileDetail-P3（官方 ImageNet MobileNetV3-Small features0-3）raw gate：
            1/4、1/8 尺度的 PR-AUC / Spearman / Top32 precision / Top32 coverage，
            预注册门槛 D4 PR≥0.50、D8 PR≥0.52、D4 prec≥0.46、D8 prec≥0.48。

用法（服务器）：
    python analyse/run5_postmortem_and_mobile_audit.py \
        --dataset_root /share_datasets/CD/SYSU-CD-256 \
        --pretrained_weight_path .../deit_tiny_patch16_224-a1311bcf.pth \
        --mobile_pretrained_weight_path .../mobilenet_v3_small-047dcff4.pth \
        --ckpt_r41_dir .../R4_1_VIT4_OLDHEAD/SYSU-CD-256 \
        --ckpt_r42_dir .../R4_2_VIT4_LIGHTDETAIL/SYSU-CD-256 \
        --ckpt_r42d_dir .../R4_2d_PSD_DETAIL/SYSU-CD-256 \
        --gpu_id 0 [--part h2|h1|h3|mobile]
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "models"))
if os.path.join(_ROOT, "analyse") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "analyse"))

from model.trainer import Trainer
from model.mobile_detail import MobileDetail
from model.metric_tool import ConfuseMatrixMeter
from model.utils import BCEDiceLoss
from run4_detail_interface_audit import pr_auc, spearman, top32_stats

import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms

MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
SCALES = ["1/2", "1/4", "1/8"]
FI_BLOCKS = ("c2_c5", "c3_c5", "c4_c5")


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


def build_model(pretrained_path, detail_mode, ckpt_dir, gpu_id, head_mode="legacy"):
    torch.cuda.set_device(gpu_id)
    model = Trainer("tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
                    mode="baseline", vit_depth=4, detail_mode=detail_mode,
                    head_mode=head_mode).float().cuda().eval()
    model.load_state_dict(torch.load(pick_best(ckpt_dir), map_location="cpu", weights_only=False))
    return model


def rel_l2(w_t, w_0):
    return float((w_t - w_0).norm().item() / (w_0.norm().item() + 1e-12))


def cos_sim(w_t, w_0):
    return float(torch.nn.functional.cosine_similarity(
        w_t.reshape(-1).float(), w_0.reshape(-1).float(), dim=0).item())


# ----------------------------------------------------------------- H2 stem drift
def audit_h2(args, loader):
    print("\n=== Run5-D0-A：PSD pretrained stem 漂移（H2） ===", flush=True)
    from model.resnet import resnet18
    ref = resnet18(pretrained=True).cpu().eval()
    r41 = build_model(args.pretrained_weight_path, "resnet", args.ckpt_r41_dir, args.gpu_id)
    r42d = build_model(args.pretrained_weight_path, "psd", args.ckpt_r42d_dir, args.gpu_id)

    r41_conv, r41_bn = r41.encoder.resnet.conv1.weight.data.cpu(), r41.encoder.resnet.bn1
    psd_conv, psd_bn = r42d.encoder.detail.stem_conv.weight.data.cpu(), r42d.encoder.detail.stem_bn
    init_conv, init_bn = ref.conv1.weight.data.cpu(), ref.bn1

    out = {}
    for tag, conv, bn in (("R4-1 ResNet stem", r41_conv, r41_bn), ("R4-2d PSD stem", psd_conv, psd_bn)):
        d = {
            "conv_relL2": rel_l2(conv, init_conv),
            "conv_cos": cos_sim(conv, init_conv),
            "bn_w_relL2": rel_l2(bn.weight.data.cpu(), init_bn.weight.data.cpu()),
            "bn_b_relL2": rel_l2(bn.bias.data.cpu(), init_bn.bias.data.cpu()),
            "bn_rm_relL2": rel_l2(bn.running_mean.cpu(), init_bn.running_mean.cpu()),
            "bn_rv_relL2": rel_l2(bn.running_var.cpu(), init_bn.running_var.cpu()),
        }
        out[tag] = d
        print(f"  {tag}: conv rel_L2={d['conv_relL2']:.4f} cos={d['conv_cos']:.4f} | "
              f"bn w/b={d['bn_w_relL2']:.3f}/{d['bn_b_relL2']:.3f} "
              f"rm/rv={d['bn_rm_relL2']:.3f}/{d['bn_rv_relL2']:.3f}", flush=True)

    # 激活漂移：前 256 对，三套 stem 输出对比
    def stem_fn(name):
        if name == "init":
            m = ref.to(args.gpu_id)
            return lambda x: torch.relu(m.bn1(m.conv1(x)))
        if name == "r41":
            m = r41.encoder.resnet.to(args.gpu_id)
            return lambda x: torch.relu(m.bn1(m.conv1(x)))
        m = r42d.encoder.detail.to(args.gpu_id)
        return lambda x: m.stem_relu(m.stem_bn(m.stem_conv(x)))

    stems = {k: stem_fn(k) for k in ("init", "r41", "psd")}
    cos_a = {k: [] for k in ("r41", "psd")}
    rms_a = {k: [] for k in ("r41", "psd")}
    with torch.no_grad():
        for bi, (img, _) in enumerate(loader):
            if bi * loader.batch_size >= 256:
                break
            pre = img[:, 0:3].cuda().float()
            f_init = stems["init"](pre)
            for k in ("r41", "psd"):
                f = stems[k](pre)
                c = torch.nn.functional.cosine_similarity(
                    f.flatten(1), f_init.flatten(1), dim=-1).mean().item()
                r = (f.pow(2).mean() ** 0.5).item() / max((f_init.pow(2).mean() ** 0.5).item(), 1e-12)
                cos_a[k].append(c)
                rms_a[k].append(r)
    for k in ("r41", "psd"):
        out[f"act_{k}"] = {"cos": float(np.mean(cos_a[k])), "rms_ratio": float(np.mean(rms_a[k]))}
        print(f"  activation drift {k}: mean cos(init,trained)={out[f'act_{k}']['cos']:.4f} "
              f"RMS ratio={out[f'act_{k}']['rms_ratio']:.4f}", flush=True)

    d_psd, d_r41 = out["R4-2d PSD stem"], out["R4-1 ResNet stem"]
    strong = (d_psd["conv_relL2"] >= 0.15 and d_psd["conv_relL2"] >= 1.5 * d_r41["conv_relL2"]) or \
             (out["act_psd"]["cos"] < 0.90 and out["act_r41"]["cos"] >= 0.95)
    weak = d_psd["conv_relL2"] < 0.10 and (d_psd["conv_relL2"] / max(d_r41["conv_relL2"], 1e-12)) <= 1.2 \
        and out["act_psd"]["cos"] >= 0.95
    verdict = "强支持 PSD stem 被重写" if strong else ("基本排除" if weak else "inconclusive")
    print(f"  [H2-VERDICT] {verdict}", flush=True)
    del r41, r42d, ref
    torch.cuda.empty_cache()


# ----------------------------------------------------------------- H1 adapter
class TileAdapter(torch.nn.Module):
    """channel repeat ×2 / sqrt(2)（variance-preserving，零参数，inference-only）。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.repeat_interleave(x, 2, dim=1) / (2 ** 0.5)


def audit_h1(args, loader):
    print("\n=== Run5-D0-B：learned adapter 是否承担关键接口映射（H1） ===", flush=True)
    model = build_model(args.pretrained_weight_path, "light", args.ckpt_r42_dir, args.gpu_id)
    model.encoder.detail_adapters = torch.nn.ModuleList([TileAdapter() for _ in range(3)])
    model = model.cuda().eval()

    meter = ConfuseMatrixMeter(n_class=2)
    with torch.no_grad():
        for img, target in loader:
            pre = img[:, 0:3].cuda().float()
            post = img[:, 3:6].cuda().float()
            tgt = target.cuda().float()
            out = model(pre, post, tgt)
            pred = (out > 0.5).long()
            meter.update_cm(pr=pred.cpu().numpy(), gt=tgt.cpu().numpy())
    scores = meter.get_scores()
    f1, iou = scores["F1"], scores["IoU"]
    d_f1, d_iou = f1 - 0.8230, iou - 0.6992
    print(f"  TileAdapter F1={f1:.4f} IoU={iou:.4f} | ΔF1={d_f1:+.4f} ΔIoU={d_iou:+.4f} "
          f"(ref R4-2 82.30/69.92)", flush=True)
    if d_f1 <= -0.30 and d_iou <= -0.45:
        verdict = "强支持 adapter 功能重要"
    elif abs(d_f1) < 0.15 and abs(d_iou) < 0.25:
        verdict = "adapter 功能较弱（显著削弱 H1）"
    else:
        verdict = "inconclusive"
    print(f"  [H1-VERDICT] {verdict}", flush=True)
    del model
    torch.cuda.empty_cache()


# ----------------------------------------------------------------- H3 interface
class RMSAccum:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def add(self, t):
        self.sum += float(t.pow(2).sum().item())
        self.n += t.numel()

    @property
    def rms(self):
        return (self.sum / max(self.n, 1)) ** 0.5


class EntAccum:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def add(self, w, nk):
        p = w.clamp_min(1e-12)
        h = -(p * p.log()).sum(-1) / np.log(nk)   # (B, H, Nq)
        self.sum += float(h.sum().item())
        self.n += h.numel()

    @property
    def mean(self):
        return self.sum / max(self.n, 1)


def audit_h3(args, loader):
    print("\n=== Run5-D0-C：PSD 与旧 FI 接口 geometry（H3） ===", flush=True)
    results = {}
    for tag, dmode, ckpt in (("R4-1", "resnet", args.ckpt_r41_dir),
                             ("R4-2", "light", args.ckpt_r42_dir),
                             ("R4-2d", "psd", args.ckpt_r42d_dir)):
        model = build_model(args.pretrained_weight_path, dmode, ckpt, args.gpu_id)
        fi = model.decoder.structure_enhance
        acc = {b: {"ln": RMSAccum(), "kv": RMSAccum(), "q": RMSAccum(),
                   "out": RMSAccum(), "ent": EntAccum()} for b in FI_BLOCKS}
        hooks = []
        for bname in FI_BLOCKS:
            blk = getattr(fi, bname)
            hooks.append(blk.norm2.register_forward_hook(
                lambda m, i, o, a=acc[bname]: a["ln"].add(o)))
            hooks.append(blk.attn.kv.register_forward_hook(
                lambda m, i, o, a=acc[bname]: a["kv"].add(o)))
            hooks.append(blk.attn.q.register_forward_hook(
                lambda m, i, o, a=acc[bname]: a["q"].add(o)))
            hooks.append(blk.attn.register_forward_hook(
                lambda m, i, o, a=acc[bname]: a["out"].add(o)))
            hooks.append(blk.attn.attn_drop.register_forward_pre_hook(
                lambda m, i, a=acc[bname]: a["ent"].add(i[0], i[0].shape[-1])))
        with torch.no_grad():
            for img, _ in loader:
                pre = img[:, 0:3].cuda().float()
                post = img[:, 3:6].cuda().float()
                model(pre, post)
        for h in hooks:
            h.remove()
        results[tag] = acc
        print(f"  {tag} hooks done", flush=True)
        del model
        torch.cuda.empty_cache()

    base = results["R4-1"]
    print(f"\n  === raw RMS / entropy per model (R4-1 / R4-2 / R4-2d) ===")
    for bi, bname in enumerate(FI_BLOCKS):
        for tag in ("R4-1", "R4-2", "R4-2d"):
            a = results[tag][bname]
            print(f"  {SCALES[bi]:>3s} {tag:>5s}: LN={a['ln'].rms:.4f} KV={a['kv'].rms:.4f} "
                  f"Q={a['q'].rms:.4f} AttnOut={a['out'].rms:.4f} Hn={a['ent'].mean:.4f}")
    print(f"\n  scale | r_ln | r_kv | r_a(AttnOut/Q) | |ΔH_n|")
    h3_strong = 0
    h3_weak_all = True
    for bi, bname in enumerate(FI_BLOCKS):
        row = {}
        for tag in ("R4-2", "R4-2d"):
            for k in ("ln", "kv"):
                b_rms = base[bname][k].rms
                v_rms = results[tag][bname][k].rms
                row[f"{tag}_{k}"] = (v_rms / b_rms) if b_rms > 1e-8 else float("inf")
            q_b, q_v = base[bname]["q"].rms, results[tag][bname]["q"].rms
            o_b, o_v = base[bname]["out"].rms, results[tag][bname]["out"].rms
            rb = (o_b / q_b) if q_b > 1e-8 else float("inf")
            rv = (o_v / q_v) if q_v > 1e-8 else float("inf")
            row[f"{tag}_a"] = (rv / rb) if rb > 1e-8 else float("inf")
            row[f"{tag}_ent"] = abs(results[tag][bname]["ent"].mean - base[bname]["ent"].mean)
        line = (f"  {SCALES[bi]:>3s} | R4-2 r_kv={row['R4-2_kv']:.2f} r_a={row['R4-2_a']:.2f} "
                f"|ΔH|={row['R4-2_ent']:.3f} || R4-2d r_kv={row['R4-2d_kv']:.2f} "
                f"r_a={row['R4-2d_a']:.2f} |ΔH|={row['R4-2d_ent']:.3f}")
        print(line, flush=True)
        # 对 PSD（R4-2d）判定（§8.2：至少 2/3 scale 三项中至少两项越界 → 强支持）。
        # base 近零（<1e-6）导致比值爆炸的尺度标记为 "base≈0"，不计入强支持判定。
        kv_ok = 0.70 <= row["R4-2d_kv"] <= 1.30
        a_ok = 0.70 <= row["R4-2d_a"] <= 1.30
        kv_base_zero = base[bname]["kv"].rms < 1e-6
        q_base_zero = base[bname]["q"].rms < 1e-6 or (base[bname]["out"].rms / max(base[bname]["q"].rms, 1e-12)) < 1e-6
        conds = [not kv_ok and not kv_base_zero,
                 not a_ok and not q_base_zero,
                 row["R4-2d_ent"] >= 0.08]
        if sum(conds) >= 2:
            h3_strong += 1
        ok = (0.80 <= row["R4-2d_kv"] <= 1.25) and (0.80 <= row["R4-2d_a"] <= 1.25) \
            and row["R4-2d_ent"] < 0.05
        h3_weak_all = h3_weak_all and ok
    if h3_strong >= 2:
        verdict = "强支持 interface geometry mismatch（≥2/3 scale）"
    elif h3_weak_all:
        verdict = "基本削弱 H3（全部 3 scale 在容差内）"
    else:
        verdict = "inconclusive"
    print(f"  [H3-VERDICT] {verdict}", flush=True)


# ----------------------------------------------------------------- D mobile gate
def audit_d_mobile(args, loader):
    print("\n=== Run5-D0-D：MobileDetail-P3 raw gate（预注册 §12） ===", flush=True)
    torch.cuda.set_device(args.gpu_id)
    mob = MobileDetail(pretrained_weight_path=args.mobile_pretrained_weight_path).cuda().eval()
    pool = torch.nn.AdaptiveAvgPool2d((16, 16))
    acc = {k: {"S": [], "G": []} for k in (1, 2)}
    with torch.no_grad():
        for bi, (img, target) in enumerate(loader):
            pre = img[:, 0:3].cuda().float()
            post = img[:, 3:6].cuda().float()
            g = torch.nn.functional.avg_pool2d(target.cuda().float(), 16).view(target.shape[0], -1)
            f1 = mob(pre)
            f2 = mob(post)
            for k in (1, 2):
                p1, p2 = pool(f1[k]), pool(f2[k])
                s = 1.0 - torch.nn.functional.cosine_similarity(
                    p1.flatten(2).transpose(1, 2), p2.flatten(2).transpose(1, 2), dim=-1, eps=1e-8)
                acc[k]["S"].append(s.cpu().numpy())
                acc[k]["G"].append(g.cpu().numpy())
            if bi % 50 == 0:
                print(f"  batch {bi + 1}/{len(loader)}", flush=True)

    gates = {}
    for k in (1, 2):
        S = np.concatenate(acc[k]["S"], axis=0)
        G = np.concatenate(acc[k]["G"], axis=0)
        y = (G.reshape(-1) > 0)
        pr = pr_auc(S.reshape(-1), y)
        sp = spearman(S.reshape(-1), G.reshape(-1))
        cov, prec = top32_stats(S, G)
        gates[k] = {"pr": pr, "sp": sp, "prec": prec, "cov": cov}
        print(f"  scale {SCALES[k]}: PR-AUC={pr:.4f} Spearman={sp:+.4f} "
              f"Top32 prec={prec:.4f} Top32 coverage={cov:.4f}", flush=True)

    d4, d8 = gates[1], gates[2]
    ok = (d4["pr"] >= 0.50) and (d8["pr"] >= 0.52) and (d4["prec"] >= 0.46) and (d8["prec"] >= 0.48)
    print(f"  gate check: D4 PR {d4['pr']:.4f}>=0.50 | D8 PR {d8['pr']:.4f}>=0.52 | "
          f"D4 prec {d4['prec']:.4f}>=0.46 | D8 prec {d8['prec']:.4f}>=0.48", flush=True)
    print(f"  [MOBILE-GATE] {'PASS' if ok else 'FAIL'}", flush=True)
    del mob
    torch.cuda.empty_cache()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--mobile_pretrained_weight_path", type=str, required=True)
    ap.add_argument("--ckpt_r41_dir", type=str, required=True)
    ap.add_argument("--ckpt_r42_dir", type=str, required=True)
    ap.add_argument("--ckpt_r42d_dir", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--skip_postmortem", type=int, default=0, help="1 = 只跑 mobile gate")
    ap.add_argument("--part", type=str, default="all", choices=["all", "h2", "h1", "h3", "mobile"],
                    help="只跑指定部分")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    loader = make_loader(args.dataset_root, args.batch_size, args.num_workers)
    print(f"[AUDIT-R5] SYSU test batches = {len(loader)}", flush=True)

    if args.part == "all":
        if not args.skip_postmortem:
            audit_h2(args, loader)
            audit_h1(args, loader)
            audit_h3(args, loader)
        ok = audit_d_mobile(args, loader)
    else:
        ok = True
        fn = {"h2": audit_h2, "h1": audit_h1, "h3": audit_h3, "mobile": audit_d_mobile}[args.part]
        out = fn(args, loader)
        if args.part == "mobile":
            ok = bool(out)
    print("[AUDIT-R5] DONE", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
