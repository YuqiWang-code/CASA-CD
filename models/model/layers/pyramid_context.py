"""Change-guided multi-head retrieval from a shared four-scale prototype pool."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.poincare_ops import RADIUS, expmap0, from_lorentz_sum, pairwise_distance, to_lorentz


class PyramidContext(nn.Module):
    """Keep dense queries and pool two supervised prototypes per grid region."""

    def __init__(self, channels=64, attention_dim=64, heads=4, grid_sizes=(8, 4, 4, 4),
                 geometry="euclidean", value_geometry="euclidean"):
        super().__init__()
        if geometry not in ("none", "euclidean", "hyperbolic"):
            raise ValueError(geometry)
        if value_geometry != "euclidean":
            raise ValueError(value_geometry)
        self.geometry = geometry
        self.value_geometry = value_geometry
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.grid_sizes = tuple(grid_sizes)
        self.tokens_per_scale = tuple(2 * grid ** 2 for grid in self.grid_sizes)
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

    def regions(self, x, grid):
        """B,H,height,width,D -> B,H,grid*grid,pixels,D aligned regions."""
        batch, heads, height, width, dim = x.shape
        return x.reshape(batch, heads, grid, height // grid, grid, width // grid, dim).permute(
            0, 1, 2, 4, 3, 5, 6).reshape(batch, heads, grid * grid, -1, dim)

    def assignments(self, probability, grid):
        batch, _, height, width = probability.shape
        cells = self.regions(probability.permute(0, 2, 3, 1).reshape(
            batch, 1, height, width, 1), grid).squeeze(1).squeeze(-1)
        membership = torch.stack((cells, 1 - cells), dim=2) + 1e-6
        return membership / membership.sum(-1, keepdim=True)

    def project_heads(self, layer, x):
        batch, _, height, width = x.shape
        tokens = layer(x).reshape(batch, self.heads, self.head_dim, height, width).permute(
            0, 1, 3, 4, 2) / math.sqrt(self.head_dim)
        return tokens

    def pool_features(self, x, weights, grid):
        """Pool normalized features before the bias-free Euclidean K/V maps."""
        cells = self.regions(x.permute(0, 2, 3, 1).unsqueeze(1), grid).squeeze(1)
        return torch.einsum("brtp,brpc->brtc", weights, cells).flatten(1, 2)

    def project_prototypes(self, layer, prototypes):
        projected = F.linear(prototypes, layer.weight[:, :, 0, 0]) / math.sqrt(self.head_dim)
        return projected.reshape(projected.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)

    def compress_hyperbolic_keys(self, keys, weights, grid):
        cells = self.regions(to_lorentz(keys), grid)
        pooled = torch.einsum("brtp,bhrpd->bhrtd", weights, cells).flatten(2, 3)
        return from_lorentz_sum(pooled)

    def attention(self, query, key, query_scale):
        counts = torch.tensor(self.tokens_per_scale, device=query.device)
        # Divide each source scale's prior mass equally among its tokens.
        scale_logits = self.scale_bias[:, query_scale] - counts.to(query.dtype).log()
        bias = scale_logits.repeat_interleave(counts, dim=-1)[None, :, None]
        temperature = F.softplus(self.raw_temperature).view(1, self.heads, 1, 1)
        return (-self.distance(query, key) / temperature + bias).softmax(-1)

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
        weights = [self.assignments(z.sigmoid(), grid) for z, grid in zip(logits, self.grid_sizes)]
        prototypes = [self.pool_features(x, w, grid)
                      for x, w, grid in zip(normalized, weights, self.grid_sizes)]
        queries = [self.project_heads(self.q, x) for x in normalized]
        if self.geometry == "hyperbolic":
            queries = [expmap0(query) for query in queries]
            keys = [self.compress_hyperbolic_keys(expmap0(self.project_heads(self.k, x)), w, grid)
                    for x, w, grid in zip(normalized, weights, self.grid_sizes)]
        else:
            keys = [self.project_prototypes(self.k, p) for p in prototypes]
        values = [self.project_prototypes(self.v, p) for p in prototypes]
        key, value = torch.cat(keys, dim=2), torch.cat(values, dim=2)
        enhanced, diagnostics = [], {}
        for index, (feature, query, assignment) in enumerate(zip(features, queries, weights)):
            batch, _, height, width = feature.shape
            query = query.flatten(2, 3)
            probabilities = self.attention(query, key, index)
            context = probabilities @ value
            context = context.transpose(-1, -2).reshape(batch, -1, height, width)
            update = self.out(context)
            enhanced.append(feature + self.gamma[index] * update)
            if self.collect_diag:
                with torch.no_grad():
                    feature_rms = feature.square().mean().sqrt()
                    residual_rms = (self.gamma[index] * update).square().mean().sqrt()
                    query_tv = (probabilities - probabilities.mean(-2, keepdim=True)).abs().sum(-1) / 2
                    diag = dict(queries=height * width, contexts=key.shape[2],
                        gamma=float(self.gamma[index]),
                        temperature=F.softplus(self.raw_temperature).tolist(),
                        entropy=float(-(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()
                                      / math.log(key.shape[2])),
                        scale_mass=torch.stack([p.sum(-1) for p in probabilities.split(self.tokens_per_scale, -1)],
                                               dim=-1).mean((0, 2)).tolist(),
                        assignment_tv=float((assignment[:, :, 0] - assignment[:, :, 1]).abs().sum(-1).mean() / 2),
                        guide_mean=float(logits[index].sigmoid().mean()),
                        branch_rms=float(update.square().mean().sqrt()),
                        feature_rms=float(feature_rms), residual_rms=float(residual_rms),
                        residual_to_feature_rms=float(residual_rms / feature_rms.clamp_min(1e-8)),
                        query_attention_tv=float(query_tv.mean()),
                        query_attention_tv_per_head=query_tv.mean((0, 2)).tolist(),
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
