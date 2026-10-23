"""Euclidean compressed QKV with signed and magnitude change values."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.context_compression import pool_tokens, relative_prototype_weights, token_map


class ChangeAwareCompressedQKV(nn.Module):
    """Dense queries retrieve signed/magnitude values from local contrast prototypes."""

    def __init__(self, channels=64, dim=32):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Conv2d(channels, dim, 1, bias=False)
        self.k = nn.Conv2d(channels, dim, 1, bias=False)
        self.v = nn.Conv2d(channels, dim, 1, bias=False)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.out = nn.Conv2d(3 * dim, channels, 1, bias=False)
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(0.15))))
        self.gamma = nn.Parameter(torch.zeros(()))
        self.collect_diag = False
        self.last_diag = {}

    def normalize(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def attention(self, q, k):
        temperature = F.softplus(self.raw_temperature) + 0.05
        logits = F.normalize(q, dim=-1, eps=1e-6) @ F.normalize(k, dim=-1, eps=1e-6).transpose(-1, -2)
        return (logits / temperature).softmax(-1)

    def forward(self, a, b):
        _, _, height, width = a.shape
        na, nb = self.normalize(a), self.normalize(b)
        difference = (na - nb).abs().mean(1, keepdim=True)
        weights = relative_prototype_weights(difference)
        qa, qb = (self.q(x).flatten(2).transpose(1, 2) for x in (na, nb))
        ka, kb = (pool_tokens(self.k(x), weights) for x in (na, nb))
        va, vb = self.v(a), self.v(b)
        difference_values = va - vb
        change_values = pool_tokens(torch.cat((difference_values, difference_values.abs()), dim=1), weights)
        pab, pba = self.attention(qa, kb), self.attention(qb, ka)
        global_a = token_map(pab @ change_values, height, width)
        signed, magnitude = change_values.chunk(2, dim=-1)
        global_b = token_map(pba @ torch.cat((-signed, magnitude), dim=-1), height, width)
        local_a = F.silu(self.local(difference_values))
        local_b = F.silu(self.local(-difference_values))
        update_a = self.out(torch.cat((global_a, local_a), dim=1))
        update_b = self.out(torch.cat((global_b, local_b), dim=1))
        if self.collect_diag:
            with torch.no_grad():
                entropy = sum(-(p * p.clamp_min(1e-9).log()).sum(-1).mean()
                              for p in (pab, pba)) / 2
                self.last_diag = dict(
                    queries=height * width, contexts=ka.shape[1],
                    entropy=float(entropy / math.log(ka.shape[1])),
                    temperature=float(F.softplus(self.raw_temperature) + 0.05),
                    gamma=float(self.gamma),
                    assignment_tv=float((weights[:, :, 0] - weights[:, :, 1]).abs().sum(-1).mean() / 2),
                    global_value_rms=float(global_a.square().mean().sqrt()),
                    local_value_rms=float(local_a.square().mean().sqrt()),
                    signed_value_rms=float(signed.square().mean().sqrt()),
                    magnitude_value_rms=float(magnitude.square().mean().sqrt()),
                    finite=bool(torch.isfinite(update_a).all() & torch.isfinite(update_b).all()))
        return a + self.gamma * update_a, b + self.gamma * update_b
