"""Change-guided multi-head retrieval from a shared four-scale prototype pool."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.poincare_ops import (
    RADIUS, expmap0, from_lorentz_sum, logmap0, pairwise_distance, to_lorentz,
)


class PyramidContext(nn.Module):
    """Keep dense queries and pool two supervised prototypes per grid region."""

    def __init__(self, channels=64, attention_dim=64, heads=4, grid_size=4,
                 geometry="euclidean"):
        super().__init__()
        if geometry not in ("none", "euclidean", "hyperbolic"):
            raise ValueError(geometry)
        self.geometry = geometry
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(channels)
        self.guide = nn.Conv2d(channels, 1, 1)
        if geometry != "none":
            self.q = nn.Conv2d(channels, attention_dim, 1, bias=False)
            self.k = nn.Conv2d(channels, attention_dim, 1, bias=False)
            self.v = nn.Conv2d(channels, attention_dim, 1, bias=False)
            self.out = nn.Conv2d(attention_dim, channels, 1, bias=False)
            self.scale_bias = nn.Parameter(torch.zeros(heads, 4, 4))
            self.raw_temperature = nn.Parameter(torch.full((heads,), math.log(math.expm1(0.2))))
            self.gamma = nn.Parameter(torch.zeros(4))
        self.collect_diag = False
        self.last_diag = {}

    def regions(self, x):
        """B,H,N,D tokens -> B,H,16,P,D spatially aligned regions."""
        batch, heads, height, width, dim = x.shape
        grid = self.grid_size
        return x.reshape(batch, heads, grid, height // grid, grid, width // grid, dim).permute(
            0, 1, 2, 4, 3, 5, 6).reshape(batch, heads, grid * grid, -1, dim)

    def assignments(self, probability):
        batch, _, height, width = probability.shape
        cells = self.regions(probability.permute(0, 2, 3, 1).reshape(
            batch, 1, height, width, 1)).squeeze(1).squeeze(-1)
        membership = torch.stack((cells, 1 - cells), dim=2) + 1e-6
        return membership / membership.sum(-1, keepdim=True)

    def project_heads(self, layer, x):
        batch, _, height, width = x.shape
        tokens = layer(x).reshape(batch, self.heads, self.head_dim, height, width).permute(
            0, 1, 3, 4, 2) / math.sqrt(self.head_dim)
        return expmap0(tokens) if self.geometry == "hyperbolic" else tokens

    def compress(self, tokens, weights):
        values = to_lorentz(tokens) if self.geometry == "hyperbolic" else tokens
        pooled = torch.einsum("brtp,bhrpd->bhrtd", weights, self.regions(values)).flatten(2, 3)
        return from_lorentz_sum(pooled) if self.geometry == "hyperbolic" else pooled

    def distance(self, query, key):
        if self.geometry == "hyperbolic":
            return pairwise_distance(query, key) / 2
        squared = (query.square().sum(-1, keepdim=True)
                   + key.square().sum(-1).unsqueeze(-2)
                   - 2 * query @ key.transpose(-1, -2)).clamp_min(0)
        return (squared + 1e-8).sqrt() - 1e-4

    def forward(self, features):
        normalized = [self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for x in features]
        logits = tuple(self.guide(x) for x in normalized)
        if self.geometry == "none":
            return list(features), logits
        weights = [self.assignments(z.sigmoid()) for z in logits]
        queries = [self.project_heads(self.q, x) for x in normalized]
        keys = [self.compress(self.project_heads(self.k, x), w) for x, w in zip(normalized, weights)]
        values = [self.compress(self.project_heads(self.v, x), w) for x, w in zip(normalized, weights)]
        key, value = torch.cat(keys, dim=2), torch.cat(values, dim=2)
        value_sum = to_lorentz(value) if self.geometry == "hyperbolic" else value
        temperature = F.softplus(self.raw_temperature).view(1, self.heads, 1, 1)
        enhanced, diagnostics = [], {}
        tokens_per_scale = 2 * self.grid_size ** 2
        for index, (feature, query, assignment) in enumerate(zip(features, queries, weights)):
            batch, _, height, width = feature.shape
            query = query.flatten(2, 3)
            bias = self.scale_bias[:, index].repeat_interleave(tokens_per_scale, dim=-1)[None, :, None]
            probabilities = (-self.distance(query, key) / temperature + bias).softmax(-1)
            context = probabilities @ value_sum
            if self.geometry == "hyperbolic":
                context = logmap0(from_lorentz_sum(context))
            context = context.transpose(-1, -2).reshape(batch, -1, height, width)
            update = self.out(context)
            enhanced.append(feature + self.gamma[index] * update)
            if self.collect_diag:
                with torch.no_grad():
                    diag = dict(queries=height * width, contexts=key.shape[2],
                        gamma=float(self.gamma[index]),
                        temperature=temperature.flatten().tolist(),
                        entropy=float(-(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()
                                      / math.log(key.shape[2])),
                        scale_mass=probabilities.reshape(batch, self.heads, -1, 4, tokens_per_scale).sum(-1).mean((0, 2)).tolist(),
                        assignment_tv=float((assignment[:, :, 0] - assignment[:, :, 1]).abs().sum(-1).mean() / 2),
                        guide_mean=float(logits[index].sigmoid().mean()),
                        branch_rms=float(update.square().mean().sqrt()),
                        finite=bool(torch.isfinite(update).all()))
                    if self.geometry == "hyperbolic":
                        radii = query.norm(dim=-1)
                        diag.update(radius_mean=float(radii.mean()), radius_max=float(radii.max()),
                            radius_cv=float(radii.std(unbiased=False) / radii.mean().clamp_min(1e-6)),
                            query_clip_fraction=float((radii >= RADIUS - 1e-5).float().mean()))
                    diagnostics[f"stage{index + 1}"] = diag
        if self.collect_diag:
            self.last_diag = diagnostics
        return enhanced, logits
