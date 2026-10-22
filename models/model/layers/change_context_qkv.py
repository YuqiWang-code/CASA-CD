"""Euclidean change-aware compressed QKV with aligned difference values."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.context_compression import pool_tokens, prototype_weights, token_map
from model.layers.change_assignment import ChangeAssignmentHead


class ChangeAwareCompressedQKV(nn.Module):
    """Full queries attend to paired change/stable key and difference tokens."""

    def __init__(self, channels=64, dim=32):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Conv2d(channels, dim, 1, bias=False)
        self.k = nn.Conv2d(channels, dim, 1, bias=False)
        self.v = nn.Conv2d(channels, dim, 1, bias=False)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.out = nn.Conv2d(2 * dim, channels, 1, bias=False)
        self.assignment = ChangeAssignmentHead(2 * channels, dim)
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

    def forward(self, a, b, return_aux=False):
        _, _, height, width = a.shape
        na, nb = self.normalize(a), self.normalize(b)
        assignment_features = torch.cat(((na + nb) * 0.5, (na - nb).abs()), dim=1)
        logits = self.assignment(assignment_features)
        probability = logits.sigmoid()
        weights = prototype_weights(probability)
        qa, qb = (self.q(x).flatten(2).transpose(1, 2) for x in (na, nb))
        ka, kb = (pool_tokens(self.k(x), weights) for x in (na, nb))
        va, vb = self.v(a), self.v(b)
        change_values = pool_tokens(va - vb, weights)
        pab, pba = self.attention(qa, kb), self.attention(qb, ka)
        global_a = token_map(pab @ change_values, height, width)
        global_b = token_map(pba @ -change_values, height, width)
        local_a = F.silu(self.local(va - vb))
        local_b = F.silu(self.local(vb - va))
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
                    assignment_mean=float(probability.mean()),
                    assignment_std=float(probability.std(unbiased=False)),
                    finite=bool(torch.isfinite(update_a).all() & torch.isfinite(update_b).all()))
        auxiliary_logits = self.assignment(assignment_features.detach()) if return_aux else logits
        return a + self.gamma * update_a, b + self.gamma * update_b, auxiliary_logits
