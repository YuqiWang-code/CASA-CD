"""Selective change prototypes and self-referenced pyramid retrieval."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def channel_norm(layer, x):
    return layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def regions(x, grid):
    """B,heads,height,width,D -> B,heads,regions,pixels,D."""
    batch, heads, height, width, dim = x.shape
    return x.reshape(batch, heads, grid, height // grid, grid, width // grid, dim).permute(
        0, 1, 2, 4, 3, 5, 6).reshape(batch, heads, grid * grid, -1, dim)


def feature_regions(x, grid):
    return regions(x.permute(0, 2, 3, 1).unsqueeze(1), grid).squeeze(1)


def split_heads(x, heads):
    batch, channels, height, width = x.shape
    return x.reshape(batch, heads, channels // heads, height, width).permute(0, 1, 3, 4, 2)


def merge_heads(x, height, width):
    return x.transpose(-1, -2).reshape(x.shape[0], -1, height, width)


def project_tokens(layer, tokens, heads):
    projected = F.linear(tokens, layer.weight[:, :, 0, 0])
    return projected.reshape(projected.shape[0], -1, heads, projected.shape[-1] // heads).transpose(1, 2)


def assignments(probability, grid):
    """Sharpen class membership within each region and measure class support."""
    positive = feature_regions(probability, grid).squeeze(-1)
    memberships = torch.stack((positive, 1 - positive), dim=2)
    weights = (memberships + 1e-6).square()
    weights = weights / weights.sum(-1, keepdim=True)
    confidence = (weights * memberships.square()).sum(-1).clamp_min(1e-6)
    return weights, confidence.flatten(1, 2)


def pool_features(feature, weights, grid):
    return torch.einsum("brtp,brpc->brtc", weights, feature_regions(feature, grid)).flatten(1, 2)


def source_bias(bias, query_scale, counts):
    return torch.cat([(bias[:, query_scale, scale] - math.log(count))[:, None].expand(-1, count)
                      for scale, count in enumerate(counts)], dim=-1)[None, :, None]


def highpass(x):
    return x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)


def rms(x):
    return x.square().mean().sqrt()


def attention_diagnostics(probability, counts):
    mass = probability.sum(-1, keepdim=True)
    conditional = probability / mass.clamp_min(1e-8)
    tv = (conditional - conditional.mean(-2, keepdim=True)).abs().sum(-1) / 2
    return dict(
        entropy=float(-(conditional * conditional.clamp_min(1e-9).log()).sum(-1).mean()
                      / math.log(probability.shape[-1])),
        scale_mass=torch.stack([p.sum(-1) for p in conditional.split(counts, -1)], -1).mean((0, 2)).tolist(),
        context_mass_mean=float(mass.mean()), context_mass_min=float(mass.min()),
        context_mass_max=float(mass.max()), context_mass_std=float(mass.std(unbiased=False)),
        self_mass_mean=float((1-mass).mean()),
        query_attention_tv=float(tv.mean()), query_attention_tv_per_head=tv.mean((0, 2)).tolist())


class ResidualPyramidContext(nn.Module):
    """Shared compression/readout; subclasses define Euclidean or Lorentz keys."""
    value_geometry = "euclidean"
    compression_power = 2
    support_power = 2
    self_reference = True

    def __init__(self, channels=64, heads=4, grid_sizes=(8, 4, 4, 4)):
        super().__init__()
        self.heads = heads
        self.grid_sizes = tuple(grid_sizes)
        self.tokens_per_scale = tuple(2 * grid ** 2 for grid in grid_sizes)
        self.norm = nn.LayerNorm(channels)
        self.guide = nn.Conv2d(channels, 1, 1)
        self.qk = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, bias=False)
        self.local_dw = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate",
                                  groups=channels, bias=False)
        self.local_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale_bias = nn.Parameter(torch.zeros(heads, 4, 4))
        self.raw_temperature = nn.Parameter(torch.full((heads,), math.log(math.expm1(.5))))
        self.gamma_context = nn.Parameter(torch.zeros(4, channels))
        self.gamma_detail = nn.Parameter(torch.zeros(2, channels))
        self.collect_diag = False
        self.last_diag = {}

    def attention(self, query, key, confidence, query_scale):
        temperature = F.softplus(self.raw_temperature).view(1, self.heads, 1, 1)
        logits = -self.squared_distance(query, key) / (2 * temperature.square())
        logits = logits + confidence.log()[:, None, None] + source_bias(
            self.scale_bias, query_scale, self.tokens_per_scale)
        # The query's own key has distance zero and unit support. Its value is
        # retained locally; a shared softmax makes context adoption query-specific.
        return torch.cat((logits, torch.zeros_like(logits[..., :1])), -1).softmax(-1)[..., :-1]

    def forward(self, tar):
        normalized = [channel_norm(self.norm, x) for x in tar]
        guide = [self.guide(x) for x in normalized]
        compression = [assignments(g.sigmoid(), grid) for g, grid in zip(guide, self.grid_sizes)]
        prototypes = [pool_features(x, w, grid) for x, (w, _), grid in
                      zip(normalized, compression, self.grid_sizes)]
        query = [self.project_queries(x) for x in normalized]
        keys = [self.pool_keys(q, p, w, grid) for q, p, (w, _), grid in
                zip(query, prototypes, compression, self.grid_sizes)]
        key = torch.cat(keys, dim=2)
        confidence = torch.cat([r for _, r in compression], dim=1)
        values = project_tokens(self.v, torch.cat(prototypes, dim=1), self.heads)
        enhanced, diagnostics = [], {}
        for index, (original, x, q) in enumerate(zip(tar, normalized, query)):
            height, width = x.shape[-2:]
            probability = self.attention(q.flatten(2, 3), key, confidence, index)
            retrieved = merge_heads(probability @ values, height, width)
            own_value = self.v(x)
            mass = probability.sum(-1, keepdim=True)
            mass_map = merge_heads(mass.expand(-1, -1, -1, x.shape[1] // self.heads), height, width)
            difference = retrieved - mass_map * own_value
            update = self.out(difference)
            context_residual = self.gamma_context[index][None, :, None, None] * update
            result = original + context_residual
            local = None
            local_residual = torch.zeros_like(original)
            if index < 2:
                local = self.local_out(self.local_dw(highpass(x)))
                local_residual = self.gamma_detail[index][None, :, None, None] * local
                result = result + local_residual
            enhanced.append(result)
            if self.collect_diag:
                with torch.no_grad():
                    base = rms(original).clamp_min(1e-8)
                    weights, reliability = compression[index]
                    diag = dict(
                        queries=height * width, contexts=key.shape[2],
                        gamma_context=float(self.gamma_context[index].mean()),
                        gamma_context_rms=float(rms(self.gamma_context[index])),
                        gamma_detail=float(self.gamma_detail[index].mean()) if index < 2 else 0.,
                        gamma_detail_rms=float(rms(self.gamma_detail[index])) if index < 2 else 0.,
                        context_rms=float(rms(update)), branch_rms=float(rms(update)),
                        local_rms=float(rms(local)) if local is not None else 0.,
                        feature_rms=float(base), relative_update_rms=float(rms(difference)),
                        residual_rms=float(rms(context_residual + local_residual)),
                        residual_to_feature_rms=float(rms(context_residual + local_residual) / base),
                        context_residual_to_feature_rms=float(rms(context_residual) / base),
                        local_residual_to_feature_rms=float(rms(local_residual) / base),
                        guide_mean=float(guide[index].sigmoid().mean()),
                        reliability_mean=float(reliability.mean()), reliability_min=float(reliability.min()),
                        retrieved_rms=float(rms(retrieved)), self_value_rms=float(rms(own_value)),
                        self_subtraction_rms=float(rms(mass_map * own_value)),
                        retrieved_spatial_rms=float(rms(retrieved-retrieved.mean((-2,-1),keepdim=True))),
                        retrieved_self_cosine=float(F.cosine_similarity(retrieved.flatten(1),own_value.flatten(1)).mean()),
                        assignment_tv=float((weights[:, :, 0] - weights[:, :, 1]).abs().sum(-1).mean() / 2),
                        qk_weight_rms=float(rms(self.qk.weight)), v_weight_rms=float(rms(self.v.weight)),
                        out_weight_rms=float(rms(self.out.weight)),
                        temperature=F.softplus(self.raw_temperature).tolist(),
                        finite=bool(torch.isfinite(result).all()))
                    diag.update(attention_diagnostics(probability, self.tokens_per_scale))
                    diag.update(self.geometry_diagnostics(q, keys[index]))
                    diagnostics[f"stage{index + 1}"] = diag
        if self.collect_diag:
            self.last_diag = diagnostics
        return enhanced, dict(guide_logits=tuple(guide))


class AuxiliaryContext(nn.Module):
    """CTRL retains the same change-guide supervision on unmodified TAR features."""
    geometry = "none"
    value_geometry = "euclidean"
    grid_sizes = (8, 4, 4, 4)
    tokens_per_scale = (0, 0, 0, 0)
    compression_power = 0
    support_power = 0
    self_reference = False

    def __init__(self, channels=64):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.guide = nn.Conv2d(channels, 1, 1)
        self.collect_diag = False
        self.last_diag = {}

    def forward(self, tar):
        return list(tar), dict(guide_logits=tuple(self.guide(channel_norm(self.norm, x)) for x in tar))
