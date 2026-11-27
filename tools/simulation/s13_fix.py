#!/usr/bin/env python
"""S13a-fix: support-prior gradient path, with identical inputs and a real-model check.

Copy of ``s13_support_final.py`` with the two reported defects fixed; the
historical script and its CSV are untouched.

Fixes
-----
1. **identical inputs across arms.**  ``base`` attention logits, the feature map,
   the head initialisation and the loss target are generated once and deep-copied
   into every arm.  The head is *not* re-created per arm from a shared module, so
   the base logits are provably the same tensor (their SHA-256 is recorded).

2. **LOGSHARE definition.**  The prior is ``logsigmoid(mean_cell logit)`` -- the
   log-sigmoid of the *pooled logit* -- instead of the old
   ``logsigmoid(mean_cell sigmoid)`` which squashes the already-squashed share and
   is neither the log of a share nor a log-sigmoid of a logit.

3. **weight *and* bias are updated.**  The synthetic step applies the same SGD
   delta to every parameter of the prior head (the old code updated ``weight`` and
   carried ``bias`` unchanged), and both gradient norms are reported.

4. **per-slot x per-cell layout.**  Each class owns several cells; the share is a
   mean over its cells and the log prior is placed on the individual key slots of
   that class.  With a single cell per class, ``log(mean sigmoid)`` and
   ``logsigmoid(mean logit)`` are indistinguishable at zero init, which is why the
   old one-cell test could not tell them apart.

5. **real-model backward check.**  ``CASANet`` is loaded from a frozen checkpoint
   for a ``mass`` arm (``R_FULL``) and a ``confidence`` arm (``R_COMPACT``).  The
   backward pass is run twice: once with the checkpoint's ``support_head`` and
   once with that head re-zeroed.  At zero init the confidence branch yields
   ``|2*0.5-1| = 0`` -> ``clamp_min(1e-6)`` -> zero gradient, while the ``mass``
   branch differentiates through ``sigmoid`` and must keep a non-zero gradient.

Usage (server, conda env casacd)::

    python s13_fix.py --out .../FinalClosure/S13
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
SIM_DIR = HERE.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, str(SIM_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S9 = _load("s9a_fix", "s9a_fix.py")
S7 = S9.S7

ARMS = ("OLD", "SHARE", "NONZERO", "LOGSHARE", "OFF")
CELLS_PER_CLASS = 8
SLOTS = 2 * CELLS_PER_CLASS


def build_inputs(channels=8):
    """Fixed feature map, head initial state and base logits (one call only)."""
    g = torch.Generator().manual_seed(7)
    feat = torch.randn(SLOTS, channels, generator=g)
    base = torch.randn(16, SLOTS, generator=g)
    target = torch.zeros(16, dtype=torch.long)
    head_zero = torch.nn.Conv2d(channels, 2, 1)
    torch.nn.init.zeros_(head_zero.weight)
    torch.nn.init.zeros_(head_zero.bias)
    head_kaiming = torch.nn.Conv2d(channels, 2, 1)
    torch.nn.init.kaiming_uniform_(head_kaiming.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(head_kaiming.bias)
    head_mid = torch.nn.Conv2d(channels, 2, 1)
    with torch.no_grad():
        head_mid.weight.copy_(0.3 * torch.randn(head_mid.weight.shape, generator=g))
        head_mid.bias.copy_(0.2 * torch.randn(head_mid.bias.shape, generator=g))
    class_ids = torch.tensor([0] * CELLS_PER_CLASS + [1] * CELLS_PER_CLASS)
    return (feat, base, target, class_ids,
            {"zero": head_zero, "kaiming": head_kaiming, "mid": head_mid})


def class_share(logits, class_ids):
    """(2,) share per class: mean sigmoid over the cells owned by the class."""
    sig = logits.sigmoid().reshape(2, -1)                   # (2, slots)
    share = torch.stack([sig[c][class_ids == c].mean() for c in (0, 1)])
    return share


def class_pooled_logit(logits, class_ids):
    logit = logits.reshape(2, -1)
    return torch.stack([logit[c][class_ids == c].mean() for c in (0, 1)])


def prior_for(mode, logits, class_ids):
    """Return (log_prior per slot, share per class) for one mode."""
    share = class_share(logits, class_ids)
    if mode == "OFF":
        return torch.zeros(SLOTS), share
    if mode == "LOGSHARE":
        per_class = F.logsigmoid(class_pooled_logit(logits, class_ids))
    elif mode == "SHARE":
        per_class = share.clamp_min(1e-6).log()
    else:                                                    # OLD / NONZERO
        per_class = (2 * share - 1).abs().clamp_min(1e-6).log()
    return per_class[class_ids], share


def synthetic_row(mode, feat, base, target, head_state, class_ids, label,
                  init_name, lr=0.1):
    head = torch.nn.Conv2d(head_state.in_channels, 2, 1)
    head.load_state_dict(head_state.state_dict())
    feat_map = feat[None].permute(0, 2, 1).reshape(1, feat.shape[1], 1, SLOTS)

    def forward():
        return head(feat_map)

    logits = forward()
    log_prior, share = prior_for(mode, logits, class_ids)
    attn = base + log_prior[None, :]
    loss = F.cross_entropy(attn, target)

    if mode == "OFF":
        wg = bg = None
        wnorm = bnorm = 0.0
        wnz = bnz = 0
    else:
        loss.backward(retain_graph=False)
        wg, bg = head.weight.grad, head.bias.grad
        wnorm = float(wg.norm()) if wg is not None else 0.0
        bnorm = float(bg.norm()) if bg is not None else 0.0
        wnz = int((wg.abs() > 0).sum()) if wg is not None else 0
        bnz = int((bg.abs() > 0).sum()) if bg is not None else 0

    with torch.no_grad():
        before = attn.detach().softmax(-1)
        for p in head.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-lr)
        log_prior2, share2 = prior_for(mode, forward(), class_ids)
        after = (base + log_prior2[None, :]).softmax(-1)
        l1 = float((before - after).abs().sum(-1).mean())
        delta_prior = float((log_prior2 - log_prior).abs().sum())
        delta_param = float(sum((p.detach() - q.detach()).abs().sum()
                               for p, q in zip(head.parameters(), head_state.parameters())))

    return dict(kind="synthetic", arm=label, slots=SLOTS, init=init_name,
                cells_per_class=CELLS_PER_CLASS,
                base_logit_sha256=hashlib.sha256(base.numpy().tobytes()).hexdigest()[:16],
                log_prior_min=round(float(log_prior.detach().min()), 6),
                log_prior_max=round(float(log_prior.detach().max()), 6),
                log_prior_std=round(float(log_prior.detach().std(unbiased=False)), 6),
                share_min=round(float(share.detach().min()), 6),
                share_max=round(float(share.detach().max()), 6),
                share_change=round(float((share2 - share).detach().abs().max()), 8),
                loss=round(float(loss.detach()), 6),
                weight_grad_norm=round(wnorm, 8), bias_grad_norm=round(bnorm, 8),
                weight_grad_nonzero=wnz, bias_grad_nonzero=bnz,
                prior_l1_after_step=round(delta_prior, 8),
                param_l1_after_step=round(delta_param, 8),
                attn_l1_after_one_step=round(l1, 8),
                share_definition=("log(mean_cell sigmoid)" if mode == "SHARE"
                                  else "logsigmoid(mean_cell logit)" if mode == "LOGSHARE"
                                  else "|2*share-1|" if mode in ("OLD", "NONZERO")
                                  else "none"))


def sample_manifest(arm, dataset):
    """The S7c sample manifests were captured per R_FULL checkpoint; reuse them."""
    for candidate in (f"{arm}_{dataset}", f"R_FULL-2334_{dataset}", f"R_FULL-2335_{dataset}"):
        path = S7.OUT_ROOT / candidate / "samples.json"
        if path.exists():
            return json.loads(path.read_text()), candidate
    raise FileNotFoundError(f"no S7c samples.json for {arm}/{dataset} under {S7.OUT_ROOT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "s13_results_fix"))
    ap.add_argument("--arm", default="R_COMPACT-2334")
    ap.add_argument("--dataset", default="SYSU-CD-256")
    ap.add_argument("--no-real", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("S7B_THREADS", "6")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feat, base, target, class_ids, states = build_inputs()
    rows = []
    plan = [("OLD", "zero", "OLD"), ("SHARE", "zero", "SHARE"),
            ("NONZERO", "kaiming", "NONZERO"), ("LOGSHARE", "zero", "LOGSHARE"),
            ("OFF", "zero", "OFF"),
            ("SHARE", "mid", "SHARE-mid"), ("LOGSHARE", "mid", "LOGSHARE-mid")]
    for mode, init_name, label in plan:
        rows.append(synthetic_row(mode, feat, base, target, states[init_name],
                                  class_ids, label, init_name))
        r = rows[-1]
        print(f"[s13] {label:11s} init={init_name:7s} "
              f"prior=[{r['log_prior_min']:.3e},{r['log_prior_max']:.3e}] "
              f"w_grad={r['weight_grad_norm']:.3e} b_grad={r['bias_grad_norm']:.3e} "
              f"attn_l1={r['attn_l1_after_one_step']:.3e}",
              flush=True)

    if not args.no_real:
        C, CASANet, Dataset, transforms, T = S7.bootstrap_paths()
        m, manifest_dir = sample_manifest(args.arm, args.dataset)
        print(f"[s13] real samples from {manifest_dir}", flush=True)
        names = [r["filename"] for r in m["val"]][:1]
        listing = out / "_s13_list.txt"
        listing.write_text("\n".join(names) + "\n")
        tf = transforms.Compose([transforms.Scale(256, 256),
                                 transforms.Normalize(T.MEAN, T.STD),
                                 transforms.ToTensor()])
        ds = Dataset(str(S7.DATA_ROOT / args.dataset), str(listing), transform=tf)
        image, tgt = next(iter(torch.utils.data.DataLoader(ds, batch_size=1)))

        for arm in ("R_FULL-2334", "R_COMPACT-2334"):
            for zero_init in (False, True):
                model = CASANet(None, **C.context_kwargs(arm)).eval()
                model.load_state_dict(torch.load(
                    str(S7.CKPT_ROOT / arm / args.dataset / "best_model.pth"),
                    map_location="cpu", weights_only=True), strict=True)
                head = model.context.support_head
                if zero_init:
                    torch.nn.init.zeros_(head.weight)
                    torch.nn.init.zeros_(head.bias)
                model.zero_grad(set_to_none=True)
                outputs, _aux = model(image[:, :3], image[:, 3:], return_aux=True)
                loss = F.binary_cross_entropy_with_logits(outputs[0], tgt.float())
                loss.backward()
                wg, bg = head.weight.grad, head.bias.grad
                record = dict(
                    kind="real", arm=arm, dataset=args.dataset, image=names[0],
                    zero_init=zero_init, mode="confidence" if "COMPACT" in arm else "mass",
                    loss=round(float(loss.detach()), 6),
                    support_head_grad_norm=round(float(wg.norm()) if wg is not None else 0.0, 8),
                    support_head_bias_grad_norm=round(float(bg.norm()) if bg is not None else 0.0, 8),
                    support_head_grad_nonzero=int((wg.abs() > 0).sum()) if wg is not None else 0)
                try:
                    with torch.no_grad():
                        from model.layers.pyramid_context import channel_norm
                        fa, fb = model.encoder(image[:, :3]), model.encoder(image[:, 3:])
                        res = model.tar(fa[1:], fb[1:])
                        tar, diff = (res[0], res[1]) if len(res) >= 2 else (res, None)
                        normed = [channel_norm(model.context.norm, x) for x in tar]
                        sources = model.context.prototype_features(normed, diff)
                        shares = []
                        for x, grid in zip(sources, model.context.grid_sizes):
                            cell = F.adaptive_avg_pool2d(head(x).sigmoid(), grid)
                            shares.append(cell.flatten(2).transpose(1, 2).reshape(x.shape[0], -1))
                        share = torch.cat(shares, dim=1)
                        support = share if "COMPACT" not in arm else (2 * share - 1).abs()
                        record["share_min"] = round(float(share.min()), 8)
                        record["share_max"] = round(float(share.max()), 8)
                        record["support_min"] = round(float(support.clamp_min(1e-6).min()), 8)
                        record["support_max"] = round(float(support.max()), 8)
                except Exception as exc:                       # pragma: no cover
                    record["share_error"] = repr(exc)
                rows.append(record)
                print(f"[s13-real] {arm} zero_init={zero_init} "
                      f"grad={record['support_head_grad_norm']:.3e} "
                      f"bias_grad={record['support_head_bias_grad_norm']:.3e} "
                      f"share=[{record.get('share_min')},{record.get('share_max')}]", flush=True)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in
                                                                 ("kind", "arm"), k))
    with open(out / "s13_fix.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "s13_fix.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out / 's13_fix.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
