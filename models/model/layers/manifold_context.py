"""Manifold-valued retrieval: aggregate values on the hyperboloid.

The plain context only uses the Lorentz metric to shape attention weights; the
value average stays Euclidean. Here the aggregation itself is a Klein/Hyperboloid
barycenter, the prior comes from an independent change-evidence head, and the
write-back carries a level path in addition to the zero-mean contrast path.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.edge_supervision import EdgeHead
from model.layers.hyperbolic_context import FLAT_GRIDS, HyperbolicContext
from model.layers.lorentz_ops import lift
from model.layers.pyramid_context import channel_norm, merge_heads


class ManifoldContext(HyperbolicContext):
    """Change-evidence prototypes retrieved and averaged on the Lorentz model."""

    geometry = "hyperbolic_manifold"
    key_mode = "pooled_change_evidence_prototypes"
    value_geometry = "lorentz_barycenter"
    readout_mode = "lorentz_barycenter_with_level_and_position"

    def __init__(self, channels=64, heads=1, head_dim=32, grid_sizes=FLAT_GRIDS,
                 support_mode="mass", kv_hierarchy="flat", freeze_tau=False,
                 edge_supervision=False, edge_detach="retrieval", aggregate_mode="lorentz",
                 prototype_pooling="euclidean"):
        if aggregate_mode not in ("lorentz", "euclidean"):
            raise ValueError(f"unknown aggregate_mode {aggregate_mode!r}")
        super().__init__(channels, heads, head_dim, grid_sizes,
                         support_mode, kv_hierarchy, freeze_tau, edge_supervision,
                         edge_detach=edge_detach, prototype_pooling=prototype_pooling)
        self.aggregate_mode = aggregate_mode
        width = heads * head_dim
        self.scale_log_temperature = nn.Parameter(torch.zeros(len(self.grid_sizes)))
        self.support_head = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.support_head.weight)
        nn.init.zeros_(self.support_head.bias)
        self.level = nn.Sequential(nn.Linear(width, channels), nn.SiLU(),
                                   nn.Linear(channels, channels))
        self.level_scale = nn.Parameter(torch.zeros(1))
        self.gate = nn.Conv2d(2 * width + 5, heads, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        # the change-evidence stream is normalised on its own so the prototypes are
        # built from D_s, exactly like the difference-conditioned context.
        self.norm_diff = nn.LayerNorm(channels)

    def prototype_features(self, normalized, difference=None):
        """Keys and values are pooled from the change evidence, not the content."""
        if difference is None:
            raise ValueError("ManifoldContext requires the temporal difference stream")
        if len(difference) != len(normalized):
            raise ValueError("the difference stream must match the number of scales")
        return [channel_norm(self.norm_diff, d) for d in difference]

    def scale_temperature(self, query_scale):
        base = F.softplus(self.raw_temperature).view(1, self.heads, 1, 1)
        return base * self.scale_log_temperature[query_scale].exp()

    def support_term(self, sources, support):
        pooled = []
        for x, grid in zip(sources, self.grid_sizes):
            cell = F.adaptive_avg_pool2d(self.support_head(x).sigmoid(), grid)
            pooled.append(cell.flatten(2).transpose(1, 2).reshape(cell.shape[0], -1))
        return torch.cat(pooled, dim=1).clamp_min(1e-6)

    def aggregate(self, probability, values):
        """Klein barycenter of the lifted values, mapped back to tangent space."""
        if self.aggregate_mode == "euclidean":
            return super().aggregate(probability, values)
        centroid = probability @ lift(values)
        norm2 = centroid[..., :1].square() - centroid[..., 1:].square().sum(-1, keepdim=True)
        return centroid[..., 1:] / norm2.clamp_min(1).sqrt()

    @staticmethod
    def position_channels(height, width, reference, scale_index):
        ys = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        grid_y = ys.view(1, 1, height, 1).expand(reference.shape[0], 1, height, width)
        grid_x = xs.view(1, 1, 1, width).expand(reference.shape[0], 1, height, width)
        scale = torch.full_like(grid_x, (scale_index + 1) / 4.0)
        return torch.cat((grid_x, grid_y, scale), dim=1)

    def readout(self, feature, probability, values, occupancy, prototype_occupancy,
                scale_index=None):
        height, width = feature.shape[-2:]
        retrieved = merge_heads(self.aggregate(probability, values), height, width)
        own_value = self.v(feature)
        discrepancy = retrieved - retrieved.mean((-2, -1), keepdim=True)
        retrieved_occupancy = (probability @ prototype_occupancy[:, None]).mean(1)
        retrieved_occupancy = retrieved_occupancy.transpose(1, 2).reshape(-1, 1, height, width)
        index = 0 if scale_index is None else scale_index
        position = self.position_channels(height, width, feature, index)
        gate = self.gate(torch.cat((own_value, discrepancy, occupancy,
                                    retrieved_occupancy, position), 1)).sigmoid()
        gate_map = gate[:, :, None].expand(-1, -1, self.head_dim, -1, -1).flatten(1, 2)
        gated = gate_map * discrepancy
        level = self.level(retrieved.mean((-2, -1)))
        update = self.out(gated) + self.level_scale.tanh() * level[:, :, None, None]
        return update, gate, retrieved, own_value, discrepancy, gated
