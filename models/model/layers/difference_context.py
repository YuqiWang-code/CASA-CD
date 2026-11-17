"""Difference-conditioned cross-scale retrieval.

Keys and values are pooled from the bi-temporal absolute difference ``|P - Q|``
rather than from the fused feature, so the retrieval answers *"is there changed
content of this kind"* instead of *"what does this location look like"*. Queries
stay at full resolution on the fused feature, so localisation is unchanged and
only the knowledge side of the attention becomes change-specific.

The geometry, support prior, scale bias and centred gated readout are inherited
unchanged from :class:`HyperbolicContext`; this class only swaps the content the
prototypes are pooled from.
"""
import torch
from torch import nn

from model.layers.hyperbolic_context import HyperbolicContext
from model.layers.pyramid_context import channel_norm


class DifferenceContext(HyperbolicContext):
    """Hyperbolic retrieval whose prototype content is the temporal difference."""

    geometry = "hyperbolic_difference"
    key_mode = "project_pooled_temporal_difference_prototypes"
    support_mode = "class_mass_of_temporal_difference"

    def __init__(self, channels=64, heads=1, head_dim=32, grid_sizes=(8, 4, 4, 4)):
        super().__init__(channels, heads, head_dim, grid_sizes)
        self.norm_diff = nn.LayerNorm(channels)

    def prototype_features(self, normalized, difference=None):
        if difference is None:
            raise ValueError("DifferenceContext requires the temporal difference stream")
        if len(difference) != len(normalized):
            raise ValueError("the difference stream must match the number of scales")
        return [channel_norm(self.norm_diff, d) for d in difference]
