"""Euclidean retrieval of reliable change/background prototypes."""
import math

from model.layers.pyramid_context import ResidualPyramidContext, project_tokens, split_heads


class EuclideanContext(ResidualPyramidContext):
    geometry = "euclidean"

    def project_queries(self, feature):
        return split_heads(self.qk(feature), self.heads) / math.sqrt(feature.shape[1] // self.heads)

    def pool_keys(self, query, prototype, weights, grid):
        # LN precedes compression; the shared linear map commutes with weighted pooling.
        return project_tokens(self.qk, prototype, self.heads) / math.sqrt(prototype.shape[-1] // self.heads)

    @staticmethod
    def squared_distance(query, key):
        return (query.square().sum(-1, keepdim=True) + key.square().sum(-1).unsqueeze(-2)
                - 2 * (query @ key.transpose(-1, -2))).clamp_min(0)

    @staticmethod
    def geometry_diagnostics(query, key):
        return dict(query_norm_mean=float(query.norm(dim=-1).mean()),
                    query_norm_max=float(query.norm(dim=-1).max()))
