"""Task-calibrated Lorentz retrieval with aligned content and learned correction."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.lorentz_ops import lift, pairwise_squared_distance
from model.layers.pyramid_context import (
    attention_diagnostics, channel_norm, feature_regions, highpass, merge_heads,
    pool_features, project_tokens, regions, rms, source_bias, split_heads,
)


def class_assignments(probability, grid):
    """One change and one background representative per spatial region."""
    positive = feature_regions(probability, grid).squeeze(-1)
    membership = torch.stack((positive, 1 - positive), dim=2)
    weights = (membership + 1e-6).square()
    weights = weights / weights.sum(-1, keepdim=True)
    support = (weights * membership).sum(-1)
    return weights, support.flatten(1).clamp_min(1e-6)


class HyperbolicContext(nn.Module):
    """Direct projected geometry, pooled K/V content and independently gated readout."""
    geometry = "hyperbolic"
    value_geometry = "euclidean"
    compression_power = 2
    support_power = 1
    self_reference = False
    radius_mode = "asinh(norm(projected_features) / sqrt(head_dim))"
    key_mode = "project_same_pooled_content_as_values_then_lift"
    readout_mode = "learned_gate_times_conditional_context_minus_query_value"
    squared_distance = staticmethod(pairwise_squared_distance)

    def __init__(self, channels=64, heads=1, head_dim=32, grid_sizes=(8, 4, 4, 4)):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.grid_sizes = tuple(grid_sizes)
        self.tokens_per_scale = tuple(2 * grid ** 2 for grid in grid_sizes)
        width = heads * head_dim
        self.norm = nn.LayerNorm(channels)
        self.guide = nn.Conv2d(channels, 1, 1)
        self.qk = nn.Conv2d(channels, width, 1, bias=False)
        self.v = nn.Conv2d(channels, width, 1, bias=False)
        self.out = nn.Conv2d(width, channels, 1, bias=False)
        self.gate = nn.Conv2d(2 * width + 2, heads, 1)
        nn.init.zeros_(self.gate.bias)
        self.local_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                  padding_mode="replicate", groups=channels, bias=False)
        self.local_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale_bias = nn.Parameter(torch.zeros(heads, 4, 4))
        self.raw_temperature = nn.Parameter(torch.full((heads,), math.log(math.expm1(.5))))
        self.gamma_context = nn.Parameter(torch.zeros(4, channels))
        self.gamma_detail = nn.Parameter(torch.zeros(2, channels))
        self.collect_diag = False
        self.last_diag = {}

    @staticmethod
    def geometric_point(projected):
        """Lift unnormalized projections; their norm determines intrinsic radius."""
        return lift(projected / math.sqrt(projected.shape[-1]))

    def project_queries(self, feature):
        projected = split_heads(self.qk(feature), self.heads)
        return self.geometric_point(projected)

    def project_keys(self, prototype):
        projected = project_tokens(self.qk, prototype, self.heads)
        occupancy = F.linear(prototype, self.guide.weight[:, :, 0, 0], self.guide.bias).sigmoid()
        return self.geometric_point(projected), occupancy

    def attention_logits(self, query, key, support, query_scale):
        temperature = F.softplus(self.raw_temperature).view(1, self.heads, 1, 1)
        logits = -self.squared_distance(query, key) / (2 * temperature.square())
        logits = logits + support.log()[:, None, None] + source_bias(
            self.scale_bias, query_scale, self.tokens_per_scale)
        return logits

    def attention(self, query, key, support, query_scale):
        return self.attention_logits(query, key, support, query_scale).softmax(-1)

    @staticmethod
    def geometric_change_logit(logits, height, width):
        """Log odds of change versus background evidence across all prototypes."""
        # Each region contributes its change prototype, then its background prototype.
        positive, background = logits.reshape(*logits.shape[:-1], -1, 2).unbind(-1)
        evidence = positive.logsumexp(-1) - background.logsumexp(-1)
        return evidence.mean(1).reshape(logits.shape[0], 1, height, width)

    def readout(self, feature, probability, values, occupancy, prototype_occupancy):
        height, width = feature.shape[-2:]
        retrieved = merge_heads(probability @ values, height, width)
        own_value = self.v(feature)
        discrepancy = retrieved - own_value
        retrieved_occupancy = (probability @ prototype_occupancy[:, None]).mean(1)
        retrieved_occupancy = retrieved_occupancy.transpose(1, 2).reshape(-1, 1, height, width)
        gate = self.gate(torch.cat((own_value, discrepancy, occupancy, retrieved_occupancy), 1)).sigmoid()
        gate_map = gate[:, :, None].expand(-1, -1, self.head_dim, -1, -1).flatten(1, 2)
        gated = gate_map * discrepancy
        return self.out(gated), gate, retrieved, own_value, discrepancy, gated

    def forward(self, tar, return_aux=False):
        normalized = [channel_norm(self.norm, x) for x in tar]
        guide = [self.guide(x) for x in normalized]
        occupancy = [g.sigmoid() for g in guide]
        compression = [class_assignments(p, grid) for p, grid in zip(occupancy, self.grid_sizes)]
        prototypes = [pool_features(x, weights, grid)
                      for x, (weights, _), grid in zip(normalized, compression, self.grid_sizes)]
        prototype = torch.cat(prototypes, dim=1)
        key, prototype_occupancy = self.project_keys(prototype)
        values = project_tokens(self.v, prototype, self.heads)
        support = torch.cat([item[1] for item in compression], dim=1)
        enhanced, diagnostics, geometric_logits = [], {}, []
        for index, (original, x, p) in enumerate(zip(tar, normalized, occupancy)):
            query = self.project_queries(x)
            logits = self.attention_logits(query.flatten(2, 3), key, support, index)
            probability = logits.softmax(-1)
            if return_aux:
                geometric_logits.append(self.geometric_change_logit(logits, *x.shape[-2:]))
            update, gate, retrieved, own_value, discrepancy, gated = self.readout(
                x, probability, values, p, prototype_occupancy)
            context_residual = self.gamma_context[index][None, :, None, None] * update
            local_residual = torch.zeros_like(original)
            local = None
            if index < 2:
                local = self.local_out(self.local_dw(highpass(x)))
                local_residual = self.gamma_detail[index][None, :, None, None] * local
            result = original + context_residual + local_residual
            enhanced.append(result)
            if self.collect_diag:
                with torch.no_grad():
                    base = rms(original).clamp_min(1e-8)
                    radius = torch.asinh(query[..., 1:].norm(dim=-1))
                    key_radius = torch.asinh(key[..., 1:].norm(dim=-1))
                    q_flat = query.flatten(2, 3)
                    direction_q = F.normalize(q_flat[..., 1:], dim=-1, eps=1e-6)
                    direction_k = F.normalize(key[..., 1:], dim=-1, eps=1e-6)
                    angular = (1 - direction_q @ direction_k.transpose(-1, -2)).clamp_min(0)
                    radial = (radius.flatten(2)[..., None] - key_radius[:, :, None]).abs()
                    weights, class_support = compression[index]
                    local_direction = regions(direction_q.reshape(*query.shape[:-1], -1), self.grid_sizes[index])
                    mean_direction = torch.einsum("brtp,bhrpd->bhrtd", weights, local_direction)
                    variance = (1 - mean_direction.square().sum(-1)).clamp_min(0)
                    diag = dict(
                        queries=x.shape[-2] * x.shape[-1], contexts=key.shape[2],
                        gamma_context=float(self.gamma_context[index].mean()),
                        gamma_context_rms=float(rms(self.gamma_context[index])),
                        gamma_detail=float(self.gamma_detail[index].mean()) if index < 2 else 0.,
                        gamma_detail_rms=float(rms(self.gamma_detail[index])) if index < 2 else 0.,
                        feature_rms=float(base), context_rms=float(rms(update)), branch_rms=float(rms(update)),
                        local_rms=float(rms(local)) if local is not None else 0.,
                        relative_update_rms=float(rms(discrepancy)),
                        gated_difference_rms=float(rms(gated)),
                        ungated_projection_rms=float(rms(F.conv2d(discrepancy, self.out.weight))),
                        residual_rms=float(rms(context_residual + local_residual)),
                        residual_to_feature_rms=float(rms(context_residual + local_residual) / base),
                        context_residual_to_feature_rms=float(rms(context_residual) / base),
                        local_residual_to_feature_rms=float(rms(local_residual) / base),
                        guide_mean=float(p.mean()), class_support_mean=float(class_support.mean()),
                        class_support_min=float(class_support.min()),
                        retrieved_rms=float(rms(retrieved)), self_value_rms=float(rms(own_value)),
                        retrieved_spatial_rms=float(rms(retrieved - retrieved.mean((-2, -1), keepdim=True))),
                        retrieved_self_cosine=float(F.cosine_similarity(retrieved.flatten(1), own_value.flatten(1)).mean()),
                        qk_weight_rms=float(rms(self.qk.weight)), v_weight_rms=float(rms(self.v.weight)),
                        out_weight_rms=float(rms(self.out.weight)),
                        temperature=F.softplus(self.raw_temperature).tolist(),
                        radius_mean=float(radius.mean()), radius_max=float(radius.max()),
                        radius_cv=float(radius.std(unbiased=False) / radius.mean().clamp_min(1e-8)),
                        key_radius_mean=float(key_radius.mean()),
                        radial_gap_attention_mean=float((probability * radial).sum(-1).mean()),
                        angular_gap_attention_mean=float((probability * angular).sum(-1).mean()),
                        prototype_direction_variance_mean=float(variance.mean()),
                        finite=bool(torch.isfinite(result).all()))
                    diag.update(attention_diagnostics(probability, self.tokens_per_scale))
                    diag.pop("self_mass_mean")
                    diag.update(context_mass_mean=float(gate.mean()), context_mass_min=float(gate.min()),
                                context_mass_max=float(gate.max()), context_mass_std=float(gate.std(unbiased=False)),
                                adoption_source="learned_value_correction_gate")
                    diagnostics[f"stage{index+1}"] = diag
        if self.collect_diag:
            self.last_diag = diagnostics
        auxiliary = dict(guide_logits=tuple(guide))
        if return_aux:
            auxiliary["geometric_logits"] = tuple(geometric_logits)
        return enhanced, auxiliary
