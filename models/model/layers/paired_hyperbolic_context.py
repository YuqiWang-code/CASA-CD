"""Paired hyperbolic midpoint and common-tangent difference retrieval."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.layers.lorentz_pair_ops import (
    euclidean_pairwise_distance, lorentz_normalize, pair_midpoint_difference, pairwise_lorentz_distance,
)
from model.layers.poincare_ops import RADIUS, expmap0, to_lorentz
from model.layers.pyramid_context import (
    attention_diagnostics, binomial_highpass, channel_norm, feature_regions, merge_heads,
    project_tokens, regions, residual_diagnostics, source_bias, split_heads,
)


class PairedHyperbolicContext(nn.Module):
    geometry = "hyperbolic"
    value_geometry = "euclidean"

    def __init__(self, in_channels=(24, 32, 96, 320), channels=64,
                 heads=4, grid_sizes=(8, 4, 4, 4)):
        super().__init__()
        self.heads = heads
        self.grid_sizes = tuple(grid_sizes)
        self.tokens_per_scale = tuple(2 * grid ** 2 for grid in self.grid_sizes)
        self.norm = nn.LayerNorm(channels)
        self.guide = nn.Conv2d(channels, 1, 1)
        self.input_norm = nn.ModuleList(nn.LayerNorm(c) for c in in_channels)
        self.pair_proj = nn.ModuleList(nn.Conv2d(c, 32, 1, bias=False) for c in in_channels)
        self.diff_proj = nn.Conv2d(32, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, 64, 1, bias=False)
        self.out = nn.Conv2d(64, channels, 1, bias=False)
        self.scale_bias = nn.Parameter(torch.zeros(heads, 4, 4))
        self.raw_tau_mid = nn.Parameter(torch.full((heads,), math.log(math.expm1(.2))))
        self.raw_tau_diff = nn.Parameter(torch.full((heads,), math.log(math.expm1(.2))))
        self.gamma_context = nn.Parameter(torch.zeros(4))
        self.gamma_detail = nn.Parameter(torch.zeros(2))
        self.boundary = nn.Conv2d(channels + 64, 1, 1)
        self.collect_diag = False
        self.last_diag = {}

    def paired_features(self, features_a, features_b, tar):
        midpoints, signed, magnitude, normalized, logits, clip_fractions = [], [], [], [], [], []
        for a, b, t, norm, projection in zip(features_a, features_b, tar, self.input_norm, self.pair_proj):
            tangent_a = split_heads(projection(channel_norm(norm, a)), self.heads) / math.sqrt(8)
            tangent_b = split_heads(projection(channel_norm(norm, b)), self.heads) / math.sqrt(8)
            midpoint, difference = pair_midpoint_difference(to_lorentz(expmap0(tangent_a)),
                                                           to_lorentz(expmap0(tangent_b)))
            unsigned = difference.abs()
            height, width = t.shape[-2:]
            feature = channel_norm(self.norm, t + self.diff_proj(merge_heads(unsigned.flatten(2, 3), height, width)))
            midpoints.append(midpoint)
            signed.append(difference)
            magnitude.append(unsigned)
            normalized.append(feature)
            logits.append(self.guide(feature))
            if self.collect_diag:
                clip_fractions.append(((tangent_a.detach().norm(dim=-1) >= 2.3).float().mean()
                                       + (tangent_b.detach().norm(dim=-1) >= 2.3).float().mean()) / 2)
        return midpoints, signed, magnitude, normalized, logits, clip_fractions

    @staticmethod
    def assignments(probability, grid):
        cells = feature_regions(probability, grid).squeeze(-1)
        changed = (cells + 1e-6) / (cells.sum(-1, keepdim=True) + cells.shape[-1] * 1e-6)
        return torch.stack((torch.full_like(changed, 1 / cells.shape[-1]), changed), dim=2)

    def compress(self, midpoint, unsigned, feature, weights, grid):
        mid_cells, unsigned_cells = regions(midpoint, grid), regions(unsigned, grid)
        key_mid = lorentz_normalize(torch.einsum("brtp,bhrpd->bhrtd", weights, mid_cells)).flatten(2, 3)
        key_unsigned = torch.einsum("brtp,bhrpd->bhrtd", weights, unsigned_cells).flatten(2, 3)
        content = torch.einsum("brtp,brpc->brtc", weights, feature_regions(feature, grid)).flatten(1, 2)
        return key_mid, key_unsigned, project_tokens(self.v, content, self.heads)

    def attention(self, midpoint, unsigned, key_mid, key_unsigned, query_scale):
        midpoint_term = -pairwise_lorentz_distance(midpoint, key_mid) / (
            2 * F.softplus(self.raw_tau_mid).view(1, self.heads, 1, 1))
        difference_term = -euclidean_pairwise_distance(unsigned, key_unsigned) / F.softplus(
            self.raw_tau_diff).view(1, self.heads, 1, 1)
        probability = (midpoint_term + difference_term + source_bias(
            self.scale_bias, query_scale, self.tokens_per_scale)).softmax(-1)
        return probability, midpoint_term, difference_term

    def forward(self, features_a, features_b, tar):
        midpoint, signed, magnitude, normalized, guide, clip_fraction = self.paired_features(features_a, features_b, tar)
        weights = [self.assignments(g.sigmoid(), grid) for g, grid in zip(guide, self.grid_sizes)]
        prototypes = [self.compress(m, u, x, w, grid) for m, u, x, w, grid in
                      zip(midpoint, magnitude, normalized, weights, self.grid_sizes)]
        key_mid, key_unsigned, values = (torch.cat(bank, dim=2) for bank in zip(*prototypes))
        enhanced, boundaries, diagnostics = [], [], {}
        for index, (original, feature, mid, signed_diff, unsigned) in enumerate(
                zip(tar, normalized, midpoint, signed, magnitude)):
            height, width = original.shape[-2:]
            probability, mid_term, diff_term = self.attention(mid.flatten(2, 3), unsigned.flatten(2, 3),
                                                            key_mid, key_unsigned, index)
            update = self.out(merge_heads(probability @ values, height, width))
            result = original + self.gamma_context[index] * update
            local, detail_gamma = None, 0.
            unsigned_map = merge_heads(unsigned.flatten(2, 3), height, width)
            if index < 2:
                high = binomial_highpass(merge_heads(signed_diff.flatten(2, 3), height, width)).abs()
                boundary = self.boundary(torch.cat((feature, unsigned_map, high), dim=1))
                local = self.diff_proj(unsigned_map + boundary.sigmoid() * high)
                detail_gamma = self.gamma_detail[index]
                result = result + detail_gamma * local
                boundaries.append(boundary)
            enhanced.append(result)
            if self.collect_diag:
                with torch.no_grad():
                    radii = (mid[..., 1:] / (mid[..., :1] + 1)).norm(dim=-1)
                    diag = residual_diagnostics(original, update, local, self.gamma_context[index], detail_gamma)
                    diag.update(attention_diagnostics(probability, self.tokens_per_scale))
                    diag.update(queries=height * width, contexts=key_mid.shape[2],
                        guide_mean=float(guide[index].sigmoid().mean()),
                        temperature_mid=F.softplus(self.raw_tau_mid).tolist(),
                        temperature_diff=F.softplus(self.raw_tau_diff).tolist(),
                        midpoint_logit_mean=mid_term.mean((0, 2, 3)).tolist(),
                        midpoint_logit_std=mid_term.std((0, 2, 3), unbiased=False).tolist(),
                        difference_logit_mean=diff_term.mean((0, 2, 3)).tolist(),
                        difference_logit_std=diff_term.std((0, 2, 3), unbiased=False).tolist(),
                        midpoint_logit_key_std=mid_term.std(-1, unbiased=False).mean((0, 2)).tolist(),
                        difference_logit_key_std=diff_term.std(-1, unbiased=False).mean((0, 2)).tolist(),
                        radius_mean=float(radii.mean()), radius_max=float(radii.max()),
                        radius_cv=float(radii.std(unbiased=False) / radii.mean().clamp_min(1e-6)),
                        query_clip_fraction=float((radii >= RADIUS - 1e-5).float().mean()),
                        pair_clip_fraction=float(clip_fraction[index]),
                        signed_difference_rms=float(signed_diff.square().mean().sqrt()),
                        unsigned_difference_rms=float(unsigned.square().mean().sqrt()),
                        compressed_difference_rms=float(prototypes[index][1].square().mean().sqrt()),
                        bank_attention_mass=probability.reshape(probability.shape[0], self.heads,
                            height * width, -1, 2).sum(-2).mean((0, 2)).tolist())
                    if index < 2:
                        diag.update(boundary_mean=float(boundary.sigmoid().mean()),
                                    detail_highpass_rms=float(high.square().mean().sqrt()))
                    diagnostics[f"stage{index + 1}"] = diag
        if self.collect_diag:
            self.last_diag = diagnostics
        return enhanced, dict(guide_logits=tuple(guide), boundary_logits=tuple(boundaries))
