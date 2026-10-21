"""Four-scale change-weighted context compression with cross-temporal QKV."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def compress_pair(a, b, cell=4):
    """One shared, change-weighted context token per regular spatial cell."""
    batch, channels, height, width = a.shape
    score = (a - b).abs().mean(1, keepdim=True)
    score = score / score.mean((2, 3), keepdim=True).clamp_min(1e-6) + 1e-3
    weights = F.unfold(score, cell, stride=cell).softmax(dim=1)

    def pool(x):
        patches = F.unfold(x, cell, stride=cell)
        patches = patches.reshape(batch, channels, cell * cell, -1)
        return (patches * weights[:, None]).sum(2).reshape(
            batch, channels, height // cell, width // cell)

    return pool(a), pool(b)


def ball_map(x, rms):
    tangent = x / rms
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    radius = norm.clamp_max(2.3)
    return torch.tanh(radius) * tangent / norm


def hyperbolic_distance(p, q):
    p2 = p.square().sum(-1).clamp_max(1 - 1e-5)
    q2 = q.square().sum(-1).clamp_max(1 - 1e-5)
    squared = (p2[..., None] + q2[:, None, :] - 2 * (p @ q.transpose(-1, -2))).clamp_min(0)
    denominator = ((1 - p2)[..., None] * (1 - q2)[:, None, :]).clamp_min(1e-8)
    return 2 * torch.asinh(torch.sqrt((squared / denominator).clamp_min(1e-8)))


class ChangeContextQKV(nn.Module):
    """Full-resolution queries attend to compressed opposite-time context.

    hyperbolic: Poincare distance replaces dot-product logits, with positive
    distance emphasizing cross-time discrepancy as in HSAM. qkv: dot-product
    attention. Both modes share compression, projections and residual routing.
    """

    def __init__(self, channels=64, dim=32, geometry="qkv"):
        super().__init__()
        if geometry not in ("qkv", "hyperbolic"):
            raise ValueError(geometry)
        self.geometry = geometry
        self.dim = dim
        self.q = nn.Conv2d(channels, dim, 1, bias=False)
        self.k = nn.Conv2d(channels, dim, 1, bias=False)
        self.v = nn.Conv2d(channels, dim, 1, bias=False)
        self.out = nn.Conv2d(dim, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.collect_diag = False
        self.last_diag = {}

    def _attention(self, query, context):
        batch, _, height, width = query.shape
        q = self.q(query).flatten(2).transpose(1, 2)
        k = self.k(context).flatten(2).transpose(1, 2)
        v = self.v(context).flatten(2).transpose(1, 2)
        if self.geometry == "hyperbolic":
            rms = ((q.square().mean((1, 2), keepdim=True) +
                    k.square().mean((1, 2), keepdim=True)) / 2).clamp_min(1e-12).sqrt()
            rms = rms * math.sqrt(self.dim)
            p, r = ball_map(q, rms), ball_map(k, rms)
            logits = hyperbolic_distance(p, r) / math.sqrt(self.dim)
        else:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dim)
        logits = (logits - logits.mean(-1, keepdim=True)) / logits.std(
            -1, keepdim=True, unbiased=False).clamp_min(1e-4)
        probability = logits.softmax(-1)
        result = probability @ v
        if self.collect_diag:
            with torch.no_grad():
                entropy = -(probability * probability.clamp_min(1e-9).log()).sum(-1).mean()
                self.last_diag = {
                    "queries": height * width, "contexts": context.shape[-2] * context.shape[-1],
                    "entropy": float(entropy / math.log(max(2, probability.shape[-1]))),
                    "gamma": float(self.gamma), "finite": bool(torch.isfinite(result).all()),
                }
                if self.geometry == "hyperbolic":
                    radii = p.norm(dim=-1)
                    self.last_diag["radius_cv"] = float(radii.std() / radii.mean().clamp_min(1e-6))
        result = result.transpose(1, 2).reshape(batch, self.dim, height, width)
        return self.out(result)

    def forward(self, a, b):
        context_a, context_b = compress_pair(a, b)
        enhanced_a = a + self.gamma * self._attention(a, context_b)
        enhanced_b = b + self.gamma * self._attention(b, context_a)
        return enhanced_a, enhanced_b
