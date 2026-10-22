"""Manifold Q/K/V compression, displacement values and hyperbolic attention."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.context_compression import cell_tokens, prototype_weights, token_map
from model.layers.poincare_ops import (
    aligned_distance, displacement, expmap0, from_lorentz_sum, logmap0,
    pairwise_distance, to_lorentz, weighted_centroid,
)


class HyperbolicCompressedQKV(nn.Module):
    """Full manifold queries retrieve compressed manifold change displacements."""

    def __init__(self, channels=64, dim=32):
        super().__init__()
        self.dim = dim
        self.embed = nn.Conv2d(channels, dim, 1, bias=False)
        self.metric = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.metric.weight)
        nn.init.eye_(self.value.weight)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.out = nn.Conv2d(2 * dim, channels, 1, bias=False)
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(0.95))))
        self.gamma = nn.Parameter(torch.zeros(()))
        self.collect_diag = False
        self.last_diag = {}

    def encode(self, x):
        tangent = self.embed(x).flatten(2).transpose(1, 2) / math.sqrt(self.dim)
        return expmap0(tangent)

    @staticmethod
    def mobius_linear(x, layer):
        return expmap0(layer(logmap0(x)))

    @staticmethod
    def compress(x, weights, height, width):
        cells = cell_tokens(token_map(to_lorentz(x), height, width))
        sums = torch.einsum("bltp,blpc->bltc", weights, cells).flatten(1, 2)
        return from_lorentz_sum(sums)

    def attention(self, q, k):
        temperature = F.softplus(self.raw_temperature) + 0.05
        return (-pairwise_distance(q, k) / temperature).softmax(-1)

    def forward(self, a, b):
        _, _, height, width = a.shape
        xa, xb = self.encode(a), self.encode(b)
        qa, qb = (self.mobius_linear(x, self.metric) for x in (xa, xb))
        distance = aligned_distance(qa, qb)
        change = (distance / (1 + distance)).reshape(a.shape[0], 1, height, width)
        weights = prototype_weights(change)
        ka, kb = (self.compress(x, weights, height, width) for x in (qa, qb))
        va, vb = (self.mobius_linear(x, self.value) for x in (xa, xb))
        cva, cvb = (self.compress(x, weights, height, width) for x in (va, vb))
        delta_a, delta_b = displacement(cva, cvb), displacement(cvb, cva)
        pab, pba = self.attention(qa, kb), self.attention(qb, ka)
        global_a = token_map(logmap0(weighted_centroid(pab, delta_a)), height, width)
        global_b = token_map(logmap0(weighted_centroid(pba, delta_b)), height, width)
        local_a = token_map(logmap0(displacement(va, vb)), height, width)
        local_b = token_map(logmap0(displacement(vb, va)), height, width)
        update_a = self.out(torch.cat((global_a, F.silu(self.local(local_a))), dim=1))
        update_b = self.out(torch.cat((global_b, F.silu(self.local(local_b))), dim=1))
        if self.collect_diag:
            with torch.no_grad():
                radii = torch.linalg.vector_norm(torch.cat((qa, qb), dim=1), dim=-1)
                entropy = sum(-(p * p.clamp_min(1e-9).log()).sum(-1).mean()
                              for p in (pab, pba)) / 2
                self.last_diag = dict(
                    queries=height * width, contexts=ka.shape[1],
                    entropy=float(entropy / math.log(ka.shape[1])),
                    temperature=float(F.softplus(self.raw_temperature) + 0.05),
                    gamma=float(self.gamma), radius_mean=float(radii.mean()),
                    radius_max=float(radii.max()),
                    radius_cv=float(radii.std(unbiased=False) / radii.mean().clamp_min(1e-6)),
                    change_mean=float(distance.mean()),
                    finite=bool(torch.isfinite(update_a).all() & torch.isfinite(update_b).all()))
        return a + self.gamma * update_a, b + self.gamma * update_b
