"""S13a-final: reproduce the ``confidence`` prior gradient cut with the *real*
log-prior path, and isolate what each prior term contributes.

The frozen ``ManifoldContext.support_term`` computes, per cell,
``share = mean sigmoid(support_head(x))`` and then ``|2 share - 1|`` clamped at
1e-6 when ``support_prior == "confidence"``; the attention logits add
``log(support)``.  With the zero-initialised head every share is exactly 0.5, so
``|2*0.5-1| = 0`` and ``clamp_min(1e-6)`` makes ``log(support)`` a constant while
its gradient is zero through both the ``abs`` kink and the clamp.

Arm list (same inputs, loss and seed; a single linear classifier on top of a
random feature map keeps the test independent of the real checkpoint):

* ``OLD``      confidence branch, zero init  -> expected zero gradient
* ``SHARE``    ``share.clamp_min(eps)``, zero init -> gradient must flow
* ``NONZERO``  confidence branch, Kaiming init -> gradient can flow
* ``LOGSHARE`` ``logsigmoid(pooled logits)``, zero init -> neutral, non-zero grad
* ``OFF``      no prior at all (constant) -> reference logits

The script also reports the log-prior statistics and the attention change caused
by a single SGD step of the prior head, so we can tell "gradient exists" from
"gradient is useful".
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("S7B_THREADS", "6")))
torch.manual_seed(20261126)


def build_arms(channels=8, cells=6, keys=24):
    torch.manual_seed(7)
    feat = torch.randn(keys, channels)
    head_zero = torch.nn.Conv2d(channels, 2, 1)
    torch.nn.init.zeros_(head_zero.weight)
    torch.nn.init.zeros_(head_zero.bias)
    head_kaiming = torch.nn.Conv2d(channels, 2, 1)
    torch.nn.init.kaiming_uniform_(head_kaiming.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(head_kaiming.bias)
    query = torch.randn(1, 1, 4, 4, channels)
    return feat, head_zero, head_kaiming, query


def prior_from_head(head, feat, mode):
    """Return (log_prior (keys,), pooled logits, share) for one cell batch."""
    feats = feat[None].permute(0, 2, 1)          # (1, C, keys) -> use as 1x1 "map"
    feats = feats.reshape(1, feats.shape[1], 1, feats.shape[2])
    logits = head(feats).reshape(1, 2, -1)        # (1, 2, keys)
    share = logits.sigmoid().mean(-1)             # (1, 2)
    if mode == "OLD":
        support = (2 * share - 1).abs().clamp_min(1e-6)
        return support.log().squeeze(0), logits, share
    if mode == "NONZERO":
        support = (2 * share - 1).abs().clamp_min(1e-6)
        return support.log().squeeze(0), logits, share
    if mode == "SHARE":
        return share.clamp_min(1e-6).log().squeeze(0), logits, share
    if mode == "LOGSHARE":
        return F.logsigmoid(share).squeeze(0), logits, share
    if mode == "OFF":
        return torch.zeros(2), logits, share
    raise ValueError(mode)


def run_arm(mode, feat, head, query, steps=1):
    head = type(head)(head.in_channels, head.out_channels, 1)
    state = json.loads(json.dumps({"w": None}))
    if mode in ("OLD", "SHARE", "LOGSHARE", "OFF"):
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    else:
        torch.nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(head.bias)

    log_prior, logits, share = prior_from_head(head, feat, mode)
    rows = 16
    base = torch.randn(rows, log_prior.shape[0])
    # attention over keys for every query row: logits = base + log_prior
    attn_logits = base + log_prior[None, :]
    target = torch.zeros(rows, dtype=torch.long)
    loss = F.cross_entropy(attn_logits, target)
    if mode == "OFF":
        grad, gnorm, nz = None, 0.0, 0
    else:
        loss.backward()
        grad = head.weight.grad
        gnorm = float(grad.norm()) if grad is not None else 0.0
        nz = int((grad.abs() > 0).sum()) if grad is not None else 0
    info = dict(arm=mode, log_prior_min=float(log_prior.min()),
                log_prior_max=float(log_prior.max()),
                log_prior_std=float(log_prior.std(unbiased=False)),
                loss=float(loss), grad_norm=gnorm, grad_nonzero=nz,
                share_min=float(share.min()), share_max=float(share.max()))
    with torch.no_grad():
        before = attn_logits.softmax(-1)
        step = 0.1
        updated = head.weight - step * (grad if grad is not None else torch.zeros_like(head.weight))
        head2 = type(head)(head.in_channels, head.out_channels, 1)
        head2.weight.copy_(updated)
        head2.bias.copy_(head.bias)
        log2, _, _ = prior_from_head(head2, feat, mode)
        after = (base + log2[None, None, :]).softmax(-1)
        info["attn_l1_after_one_step"] = float((before - after).abs().sum(-1).mean())
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "s13_results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feat, head_zero, head_kaiming, query = build_arms()
    rows = []
    for mode in ("OLD", "SHARE", "NONZERO", "LOGSHARE", "OFF"):
        head = head_kaiming if mode == "NONZERO" else head_zero
        info = run_arm(mode, feat, head, query)
        rows.append(info)
        print(f"[s13] {mode:9s} log_prior=[{info['log_prior_min']:.3e},"
              f"{info['log_prior_max']:.3e}] grad_norm={info['grad_norm']:.3e} "
              f"nonzero={info['grad_nonzero']} attn_l1={info['attn_l1_after_one_step']:.3e}",
              flush=True)
    with open(out / "s13_grad.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "s13_grad.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
