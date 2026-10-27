"""Lorentz prototype retrieval relative to each query's own geometric point."""
import math

import torch

from model.layers.lorentz_ops import lift, normalize_sum, pairwise_squared_distance
from model.layers.pyramid_context import ResidualPyramidContext, regions, split_heads


class HyperbolicContext(ResidualPyramidContext):
    geometry = "hyperbolic"

    def project_queries(self, feature):
        space = split_heads(self.qk(feature), self.heads) / math.sqrt(feature.shape[1] // self.heads)
        return lift(space)

    def pool_keys(self, query, prototype, weights, grid):
        mean = torch.einsum("brtp,bhrpd->bhrtd", weights, regions(query, grid))
        return normalize_sum(mean).flatten(2, 3)

    squared_distance = staticmethod(pairwise_squared_distance)

    @staticmethod
    def geometry_diagnostics(query, key):
        radius = torch.asinh(query[..., 1:].norm(dim=-1))
        key_radius = torch.asinh(key[..., 1:].norm(dim=-1))
        error = (query[..., :1].square() - query[..., 1:].square().sum(-1, keepdim=True) - 1).abs()
        return dict(radius_mean=float(radius.mean()), radius_max=float(radius.max()),
                    radius_cv=float(radius.std(unbiased=False) / radius.mean().clamp_min(1e-8)),
                    key_radius_mean=float(key_radius.mean()), manifold_max_error=float(error.max()))
