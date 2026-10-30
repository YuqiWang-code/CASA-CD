"""Shared change prototypes and dense routing of temporal, geometric and frequency evidence."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.lorentz_ops import lift, pairwise_squared_distance
from model.layers.pyramid_context import (
    attention_diagnostics, channel_norm, feature_regions, merge_heads, pool_features,
    project_tokens, rms, source_bias, split_heads,
)


MODEL_VARIANTS = (
    "none", "hyperbolic", "qkv", "frequency", "equal", "moe",
    "no_hyperbolic", "no_qkv", "no_frequency",
)
EXPERTS = ("hyperbolic", "qkv", "frequency")


def subtype_assignments(probability, appearance, grid):
    """Two appearance representatives for each change/background class in a region."""
    positive = feature_regions(probability, grid).squeeze(-1)
    subtype = feature_regions(appearance, grid).squeeze(-1)
    classes = torch.stack((positive, 1 - positive), dim=2)
    modes = torch.stack((subtype, 1 - subtype), dim=2)
    weights = (classes.unsqueeze(3) + 1e-6).square() * (modes.unsqueeze(2) + 1e-6)
    weights = weights / weights.sum(-1, keepdim=True)
    support = (weights * classes.unsqueeze(3)).sum(-1)
    return weights.flatten(2, 3), support.flatten(1).clamp_min(1e-6)


def haar_split(x):
    """Orthonormal Haar analysis of even spatial sizes; retain signed LH/HL/HH."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("Haar analysis requires even height and width")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    low = (a + b + c + d) / 2
    high = torch.stack(((a - b + c - d) / 2, (a + b - c - d) / 2,
                        (a - b - c + d) / 2), dim=2)
    return low, high


def haar_merge(low, high):
    """Exact inverse of haar_split before learned subband transformations."""
    lh, hl, hh = high.unbind(2)
    top = torch.stack(((low + lh + hl + hh) / 2,
                       (low - lh + hl - hh) / 2), -1).flatten(-2)
    bottom = torch.stack(((low + lh - hl - hh) / 2,
                          (low - lh - hl + hh) / 2), -1).flatten(-2)
    return torch.stack((top, bottom), -2).flatten(-3, -2)


class RetrievalExpert(nn.Module):
    """Full-resolution queries read a shared compressed content pool."""
    def __init__(self, geometry, channels=64, width=32):
        super().__init__()
        self.geometry = geometry
        self.heads = 1
        self.qk = nn.Conv2d(channels, width, 1, bias=False)
        self.v = nn.Conv2d(channels, width, 1, bias=False)
        self.out = nn.Conv2d(width, channels, 1, bias=False)
        self.scale_bias = nn.Parameter(torch.zeros(1, 4, 4))
        self.raw_temperature = nn.Parameter(torch.full((1,), math.log(math.expm1(.5))))

    def point(self, projection, occupancy):
        direction = F.normalize(projection, dim=-1, eps=1e-6)
        if self.geometry == "hyperbolic":
            return lift((1 + occupancy) * direction)
        return direction

    def prepare(self, prototype, prototype_occupancy):
        key = self.point(project_tokens(self.qk, prototype, 1), prototype_occupancy[:, None])
        return key, project_tokens(self.v, prototype, 1)

    def forward(self, feature, occupancy, key, values, support, scale, counts):
        height, width = feature.shape[-2:]
        query = self.point(split_heads(self.qk(feature), 1),
                           occupancy.permute(0, 2, 3, 1).unsqueeze(1)).flatten(2, 3)
        if self.geometry == "hyperbolic":
            distance = pairwise_squared_distance(query, key)
        else:
            distance = (query.square().sum(-1, keepdim=True)
                        + key.square().sum(-1).unsqueeze(-2)
                        - 2 * query @ key.transpose(-1, -2)).clamp_min(0)
        temperature = F.softplus(self.raw_temperature).view(1, 1, 1, 1)
        logits = -distance / (2 * temperature.square())
        logits = logits + support.log()[:, None, None] + source_bias(self.scale_bias, scale, counts)
        attention = logits.softmax(-1)
        retrieved = merge_heads(attention @ values, height, width)
        correction = self.out(retrieved - self.v(feature))
        return correction, attention, query


class FrequencyExpert(nn.Module):
    """Low-frequency structure conditions signed directional high-frequency correction."""
    def __init__(self, channels=64, width=16):
        super().__init__()
        self.width = width
        self.input = nn.Conv2d(channels, width, 1, bias=False)
        self.low_dw = nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate",
                                groups=width, bias=False)
        self.low_out = nn.Conv2d(width, width, 1, bias=False)
        self.high_dw = nn.Conv2d(3 * width, 3 * width, 3, padding=1,
                                 padding_mode="replicate", groups=3 * width, bias=False)
        self.high_out = nn.Conv2d(3 * width, 3 * width, 1, groups=3, bias=False)
        self.high_gate = nn.Conv2d(width, 3 * width, 1)
        self.out = nn.Conv2d(width, channels, 1, bias=False)

    def forward(self, feature):
        low, high = haar_split(self.input(feature))
        # Group each directional band across channels; keep its signed coefficients.
        directional = high.transpose(1, 2).flatten(1, 2)
        low_update = self.low_out(F.silu(self.low_dw(low)))
        high_update = self.high_out(self.high_dw(directional)) * self.high_gate(low).sigmoid()
        high_update = high_update.reshape(feature.shape[0], 3, self.width,
                                          *low.shape[-2:]).transpose(1, 2)
        return self.out(haar_merge(low_update, high_update))


class MoEContext(nn.Module):
    """TAR identity and three correction experts with one residual amplitude."""
    geometry = "heterogeneous"
    value_geometry = "euclidean"
    compression_power = 2
    support_power = 1
    self_reference = False
    radius_mode = "hyperbolic: asinh(1 + supervised_change_occupancy)"
    key_mode = "shared_pooled_content_projected_by_each_expert"
    readout_mode = "single_gamma_times_router_weighted_relative_corrections"

    def __init__(self, variant="moe", channels=64, grid_sizes=(8, 4, 4, 4)):
        super().__init__()
        if variant not in MODEL_VARIANTS[1:]:
            raise ValueError(f"Unknown MoE correction variant: {variant}")
        self.variant = variant
        self.channels = channels
        self.grid_sizes = tuple(grid_sizes)
        self.heads = 1
        self.head_dim = 32
        self.norm = nn.LayerNorm(channels)
        self.guide = nn.Conv2d(channels, 1, 1)
        self.enabled_experts = ((variant,) if variant in EXPERTS else
                                tuple(name for name in EXPERTS if variant != f"no_{name}"))
        self.routing_mode = "single" if variant in EXPERTS else "equal" if variant == "equal" else "learned"
        self.route_names = ("tar",) + self.enabled_experts
        self.uses_context = any(name in self.enabled_experts for name in ("hyperbolic", "qkv"))
        self.tokens_per_scale = tuple(4 * grid ** 2 if self.uses_context else 0 for grid in grid_sizes)
        self.experts = nn.ModuleDict()
        # Composition-independent seeds match expert initialization across ablations.
        for name, seed in zip(EXPERTS, (3401, 3402, 3403)):
            if name in self.enabled_experts:
                with torch.random.fork_rng():
                    torch.manual_seed(seed)
                    self.experts[name] = (FrequencyExpert(channels) if name == "frequency"
                                          else RetrievalExpert(name, channels))
        if self.uses_context:
            with torch.random.fork_rng():
                torch.manual_seed(3404)
                self.appearance = nn.Conv2d(channels, 1, 1)
        if self.routing_mode == "learned":
            with torch.random.fork_rng():
                torch.manual_seed(3405)
                self.router = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.SiLU(),
                                             nn.Conv2d(16, 4, 1))
                nn.init.normal_(self.router[2].weight, std=.01)
                nn.init.zeros_(self.router[2].bias)
        self.gamma_context = nn.Parameter(torch.zeros(4, channels))
        self.collect_diag = False
        self.last_diag = {}

    def route_weights(self, feature):
        shape = (feature.shape[0], len(self.route_names), *feature.shape[-2:])
        if self.routing_mode == "learned":
            indices = [0] + [1 + EXPERTS.index(name) for name in self.enabled_experts]
            return self.router(feature)[:, indices].softmax(1)
        weights = feature.new_full(shape, 1 / len(self.route_names))
        if self.routing_mode == "single":
            weights[:, 0] = 0
            weights[:, 1] = 1
        return weights

    def metadata(self):
        return dict(
            variant=self.variant, active_experts=list(self.route_names), routing_mode=self.routing_mode,
            execution="dense_all_enabled_experts", context_grid_sizes=list(self.grid_sizes),
            context_prototypes_per_scale=list(self.tokens_per_scale), joint_contexts=sum(self.tokens_per_scale),
            qk_geometry={name: "lorentz_occupancy_radius" if name == "hyperbolic" else "euclidean_unit_direction"
                         for name in self.enabled_experts if name != "frequency"},
            value_geometry="euclidean", heads=1 if self.uses_context else 0,
            query_key_head_dim=32 if self.uses_context else 0, value_head_dim=32 if self.uses_context else 0,
            qkv_dim=32 if self.uses_context else 0, compression_power=2 if self.uses_context else 0,
            support_power=1 if self.uses_context else 0, self_reference=False,
            hyperbolic_radius=self.radius_mode if "hyperbolic" in self.enabled_experts else None,
            key_mode=self.key_mode, readout_mode=self.readout_mode, context_gamma_shape=[4, self.channels],
            local_gamma_shape=[], frequency_width=16 if "frequency" in self.enabled_experts else 0,
            frequency_transform="single_level_signed_haar" if "frequency" in self.enabled_experts else None,
            router_hidden_channels=16 if self.routing_mode == "learned" else 0,
            initialization_seeds={"hyperbolic": 3401, "qkv": 3402, "frequency": 3403,
                                  "appearance": 3404, "router": 3405},
        )

    def forward(self, tar):
        normalized = [channel_norm(self.norm, x) for x in tar]
        guide = [self.guide(x) for x in normalized]
        occupancies = [g.sigmoid() for g in guide]
        prepared, support = {}, None
        if self.uses_context:
            compression = [subtype_assignments(p, self.appearance(x).sigmoid(), grid)
                           for x, p, grid in zip(normalized, occupancies, self.grid_sizes)]
            prototype = torch.cat([pool_features(x, weights, grid)
                                   for x, (weights, _), grid in zip(normalized, compression, self.grid_sizes)], 1)
            prototype_occupancy = F.linear(prototype, self.guide.weight[:, :, 0, 0], self.guide.bias).sigmoid()
            support = torch.cat([item[1] for item in compression], 1)
            for name in self.enabled_experts:
                if name != "frequency":
                    prepared[name] = self.experts[name].prepare(prototype, prototype_occupancy)
        enhanced, diagnostics = [], {}
        for scale, (original, feature, occupancy) in enumerate(zip(tar, normalized, occupancies)):
            routes = self.route_weights(feature)
            updates, attention_diags = {}, {}
            for name in self.enabled_experts:
                expert = self.experts[name]
                if name == "frequency":
                    updates[name] = expert(feature)
                else:
                    key, values = prepared[name]
                    update, attention, query = expert(feature, occupancy, key, values, support,
                                                       scale, self.tokens_per_scale)
                    updates[name] = update
                    if self.collect_diag:
                        with torch.no_grad():
                            stats = attention_diagnostics(attention, self.tokens_per_scale)
                            stats["temperature"] = F.softplus(expert.raw_temperature).tolist()
                            if name == "hyperbolic":
                                radius = torch.asinh(query[..., 1:].norm(dim=-1))
                                key_radius = torch.asinh(key[..., 1:].norm(dim=-1))
                                stats.update(radius_mean=float(radius.mean()),
                                             radius_cv=float(radius.std(unbiased=False) / radius.mean().clamp_min(1e-8)),
                                             key_radius_mean=float(key_radius.mean()))
                            attention_diags[name] = stats
            weighted = {name: routes[:, index:index + 1] * updates[name]
                        for index, name in enumerate(self.route_names) if name != "tar"}
            gamma = self.gamma_context[scale][None, :, None, None]
            residual = gamma * sum(weighted.values())
            result = original + residual
            enhanced.append(result)
            if self.collect_diag:
                with torch.no_grad():
                    base = rms(original).clamp_min(1e-8)
                    route_mean = routes.mean((0, 2, 3))
                    foreground = (routes * occupancy).sum((0, 2, 3)) / occupancy.sum().clamp_min(1e-8)
                    background = (routes * (1 - occupancy)).sum((0, 2, 3)) / (1 - occupancy).sum().clamp_min(1e-8)
                    cosines = {f"{left}/{right}": float(F.cosine_similarity(updates[left].flatten(1),
                                                                            updates[right].flatten(1)).mean())
                               for i, left in enumerate(self.enabled_experts)
                               for right in self.enabled_experts[i + 1:]}
                    diagnostics[f"stage{scale + 1}"] = dict(
                        queries=feature.shape[-2] * feature.shape[-1], contexts=sum(self.tokens_per_scale),
                        gamma_context=float(self.gamma_context[scale].mean()),
                        gamma_context_rms=float(rms(self.gamma_context[scale])), feature_rms=float(base),
                        residual_to_feature_rms=float(rms(residual) / base),
                        branch_rms=float(rms(sum(weighted.values()))),
                        route_mass=dict(zip(self.route_names, route_mean.tolist())),
                        route_change_mass=dict(zip(self.route_names, foreground.tolist())),
                        route_background_mass=dict(zip(self.route_names, background.tolist())),
                        route_grouping="predicted_change_occupancy_weighted",
                        route_entropy=float(-(routes * routes.clamp_min(1e-9).log()).sum(1).mean()
                                            / math.log(len(self.route_names))),
                        expert_rms={name: float(rms(update)) for name, update in updates.items()},
                        expert_residual_to_feature_rms={name: float(rms(gamma * value) / base)
                                                       for name, value in weighted.items()},
                        expert_cosine=cosines, expert_attention=attention_diags,
                        guide_mean=float(occupancy.mean()), finite=bool(torch.isfinite(result).all()))
        if self.collect_diag:
            self.last_diag = diagnostics
        return enhanced, dict(guide_logits=tuple(guide))
