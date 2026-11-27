"""HRA: the Run10 retrieval with three independently selectable geometric axes.

One data flow, one parameter set, three switches:

* ``member_pool``     how the per-pixel projections of the change evidence are
                      averaged into the class prototypes (the K and V tokens);
* ``distance``        the metric that compares a query with a prototype;
* ``value_aggregate`` how the retrieved prototype values are combined.

The default (``lorentz`` on every axis) reproduces ``ManifoldContext`` exactly,
so the Run10 ``R_FULL`` arm (224 prototypes, all-hyperbolic) remains a valid
reference for the low-budget arm: the two implementations share the module tree
and only differ in which branch of these three switches runs.  ``euclidean`` on
every axis is the matched E-HRA: identical inputs, guide, prior, readout and
parameter count, no manifold anywhere.  The single-axis variants (E-HV etc.)
change exactly one switch, which is what makes an attribution claim possible.

The query side is always ``T`` (the fused TAR output); the key/value side is
always ``D`` (the propagated signed change evidence), exactly as in Run10.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.hyperbolic_context import FLAT_GRIDS
from model.layers.lorentz_ops import lift
from model.layers.manifold_context import ManifoldContext
from model.layers.pyramid_context import channel_norm, highpass, split_heads


def euclidean_squared_distance(query, key):
    """Squared Euclidean distance between ambient query/key points."""
    return (query.square().sum(-1, keepdim=True)
            + key.square().sum(-1).unsqueeze(-2)
            - 2 * (query @ key.transpose(-1, -2))).clamp_min(0)


class HRAContext(ManifoldContext):
    """Change-evidence prototypes with configurable pooling, metric and averaging.

    ``prototype_pooling`` is pinned to ``"lorentz"`` internally: in the parent
    forward pass that flag only selects *which* K/V constructor runs, and HRA
    replaces that constructor with :meth:`manifold_keys_and_values`, which reads
    the real ``member_pool`` axis.  ``aggregate_mode`` mirrors ``value_aggregate``.
    ``support_mode`` is pinned to ``"mass"`` (the learned share prior); the old
    ``confidence`` prior is not part of this round because its ``|2p-1|`` form
    has zero gradient when the head sits at ``p = 0.5``.
    """

    geometry = "hra"
    key_mode = "pooled_change_evidence_prototypes"

    axes = ("member_pool", "distance", "value_aggregate")
    modes = ("euclidean", "lorentz")

    def __init__(self, channels=64, heads=1, head_dim=32, grid_sizes=FLAT_GRIDS,
                 support_mode="mass", kv_hierarchy="flat", freeze_tau=False,
                 member_pool="lorentz", distance="lorentz", value_aggregate="lorentz",
                 retrieval_scales=(0, 1, 2, 3)):
        for name, value in (("member_pool", member_pool), ("distance", distance),
                            ("value_aggregate", value_aggregate)):
            if value not in self.modes:
                raise ValueError(f"unknown {name} {value!r}; expected one of {self.modes}")
        if support_mode != "mass":
            raise ValueError("HRA uses the learned share prior (support_mode='mass')")
        self.member_pool = member_pool
        self.distance_mode = distance
        self.value_mode = value_aggregate
        super().__init__(channels, heads, head_dim, grid_sizes,
                         support_mode="mass", kv_hierarchy=kv_hierarchy,
                         freeze_tau=freeze_tau,
                         edge_supervision=False, edge_detach="retrieval",
                         aggregate_mode="lorentz" if value_aggregate == "lorentz" else "euclidean",
                         prototype_pooling="lorentz",
                         retrieval_scales=retrieval_scales)
        # both branches of the parent forward are keyed on ``prototype_pooling``;
        # the real axis is kept above and consumed by the overrides below.
        self.prototype_pooling = "lorentz"
        self.value_geometry = ("lorentz_barycenter" if value_aggregate == "lorentz"
                               else "euclidean_weighted_mean")
        self.key_mode = f"pooled_change_evidence_prototypes_{member_pool}"
        if distance == "euclidean":
            self.squared_distance = euclidean_squared_distance

    def project_queries(self, feature):
        """Queries live on the manifold only when the metric needs it."""
        projected = split_heads(self.qk(feature), self.heads)
        if self.distance_mode == "lorentz":
            return self.geometric_point(projected)
        return projected / math.sqrt(projected.shape[-1])

    @staticmethod
    def pool_members(points, weights, manifold):
        """Weighted class prototype of ``(B, heads, R, P, D)`` per-pixel points.

        ``weights`` is ``(B, R, 2, P)`` and the class axis broadcasts exactly as
        in :func:`pool_features`, so the result is ``(B, heads, 2R, D)`` for the
        ambient mean and ``(B, heads, 2R, D+1)`` on the hyperboloid.
        """
        if manifold:
            return ManifoldContext.manifold_pool(points, weights)
        return torch.einsum("brtp,bhrpd->bhrtd", weights, points).flatten(2, 3)

    def manifold_keys_and_values(self, sources, compressed):
        """Build K/V from the change evidence with the configured member pooling.

        ``member_pool="lorentz"`` is bit-for-bit the parent implementation: the
        key projections are scaled by ``1/sqrt(head_dim)``, lifted and normalised
        as a Minkowski barycentre; the value projections are pooled on the same
        manifold and returned in the ambient chart (the parent readout lifts them
        again, so ``value_aggregate`` owns that switch).  ``"euclidean"`` averages
        the ambient projections directly and never lifts anything.

        The *chart* of the keys follows the distance axis, not the member axis:
        a Lorentz metric needs lifted keys (lift the ambient barycentre when the
        members were averaged in the chart), and a Euclidean metric needs ambient
        keys (keep the space part of a hyperboloid barycentre).  This is what
        lets the three axes vary independently without mixing dimensions.
        """
        manifold = self.member_pool == "lorentz"
        keys, values = [], []
        for source, (weights, _), grid in zip(sources, compressed, self.grid_sizes):
            key_points = self.head_regions(split_heads(self.qk(source), self.heads), grid)
            value_points = self.head_regions(split_heads(self.v(source), self.heads), grid)
            keys.append(self.pool_members(key_points / math.sqrt(self.head_dim), weights,
                                          manifold))
            pooled = self.pool_members(value_points, weights, manifold)
            values.append(pooled[..., 1:] if manifold else pooled)
        keys = torch.cat(keys, dim=2)
        if self.distance_mode == "lorentz" and not manifold:
            keys = lift(keys)
        elif self.distance_mode == "euclidean" and manifold:
            keys = keys[..., 1:]
        return keys, torch.cat(values, dim=2)

    def axis_report(self):
        """The exact switch combination, for the run manifest."""
        return dict(member_pool=self.member_pool, distance=self.distance_mode,
                    value_aggregate=self.value_mode, grids=list(self.grid_sizes),
                    kv_tokens=sum(self.tokens_per_scale),
                    retrieval_scales=list(self.retrieval_scales))


class LocalContext(nn.Module):
    """LOCAL arm: the HRA guide and local detail branch, without any retrieval.

    TAR, the change stream, the guide supervision and the depthwise local
    residual are exactly the ones the retrieval arms use; only the global
    retrieve-and-write-back path is removed (so ``qk``/``v``/``out``/gate/level/
    support heads do not exist here and cannot be counted as HRA overhead).
    The arm isolates what the local branch alone contributes, and it is the
    matched control for the retrieval gain because its guide is computed on the
    same temporal-difference stream.
    """

    geometry = "local_only"
    value_geometry = "euclidean"
    grid_sizes = (8, 4, 4, 4)
    tokens_per_scale = (0, 0, 0, 0)
    compression_power = 0
    support_power = 0
    self_reference = False
    key_mode = "none"

    def __init__(self, channels=64):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.norm_diff = nn.LayerNorm(channels)
        self.guide = nn.Conv2d(channels, 1, 1)
        self.local_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                  padding_mode="replicate", groups=channels, bias=False)
        self.local_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma_detail = nn.Parameter(torch.zeros(2, channels))
        self.collect_diag = False
        self.last_diag = {}

    def forward(self, tar, difference=None, return_aux=False):
        if difference is None:
            raise ValueError("LocalContext requires the temporal difference stream")
        normalized = [channel_norm(self.norm, x) for x in tar]
        sources = [channel_norm(self.norm_diff, d) for d in difference]
        guide = [self.guide(x) for x in sources]
        enhanced = []
        for index, original in enumerate(tar):
            result = original
            if index < 2:
                local = self.local_out(self.local_dw(highpass(normalized[index])))
                result = result + self.gamma_detail[index][None, :, None, None] * local
            enhanced.append(result)
        return enhanced, dict(guide_logits=tuple(guide))

    def axis_report(self):
        return dict(member_pool="none", distance="none", value_aggregate="none",
                    grids=[], kv_tokens=0, retrieval_scales=[])
