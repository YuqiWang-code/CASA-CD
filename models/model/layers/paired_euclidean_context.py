"""Dense paired differences and separate regional mean/contrast retrieval."""
import math

import torch
from torch import nn

from model.layers.pyramid_context import (
    attention_diagnostics, binomial_highpass, channel_norm, feature_regions,
    merge_heads, project_tokens, residual_diagnostics, source_bias, split_heads,
)


class PairedEuclideanContext(nn.Module):
    geometry = "euclidean"
    value_geometry = "euclidean"

    def __init__(self, in_channels=(24, 32, 96, 320), channels=64,
                 heads=4, grid_sizes=(8, 4, 4, 4)):
        super().__init__()
        self.heads = heads
        self.grid_sizes = tuple(grid_sizes)
        self.regions_per_scale = tuple(grid ** 2 for grid in self.grid_sizes)
        self.tokens_per_scale = tuple(2 * count for count in self.regions_per_scale)
        self.norm = nn.LayerNorm(channels)
        self.guide = nn.Conv2d(channels, 1, 1)
        self.input_norm = nn.ModuleList(nn.LayerNorm(c) for c in in_channels)
        self.pair_proj = nn.ModuleList(nn.Conv2d(c, 32, 1, bias=False) for c in in_channels)
        self.diff_proj = nn.Conv2d(32, channels, 1, bias=False)
        self.q = nn.Conv2d(channels, 64, 1, bias=False)
        self.k_mean = nn.Conv2d(channels, 64, 1, bias=False)
        self.k_contrast = nn.Conv2d(channels, 64, 1, bias=False)
        self.v_mean = nn.Conv2d(channels, 32, 1, bias=False)
        self.v_contrast = nn.Conv2d(channels, 32, 1, bias=False)
        self.out = nn.Conv2d(64, channels, 1, bias=False)
        self.scale_bias = nn.Parameter(torch.zeros(2, heads, 4, 4))
        self.gamma_context = nn.Parameter(torch.zeros(4))
        self.gamma_detail = nn.Parameter(torch.zeros(2))
        self.boundary = nn.Conv2d(channels + 64, 1, 1)
        self.collect_diag = False
        self.last_diag = {}

    def paired_features(self, features_a, features_b, tar):
        differences, normalized, logits = [], [], []
        for a, b, t, norm, projection in zip(features_a, features_b, tar, self.input_norm, self.pair_proj):
            difference = (projection(channel_norm(norm, b)) - projection(channel_norm(norm, a))).abs()
            feature = channel_norm(self.norm, t + self.diff_proj(difference))
            differences.append(difference)
            normalized.append(feature)
            logits.append(self.guide(feature))
        return differences, normalized, logits

    @staticmethod
    def prototypes(feature, probability, grid):
        cells = feature_regions(feature, grid)
        membership = feature_regions(probability, grid)
        mean = cells.mean(-2)
        changed = ((membership * cells).sum(-2) + mean) / (membership.sum(-2) + 1.)
        unchanged = (((1 - membership) * cells).sum(-2) + mean) / ((1 - membership).sum(-2) + 1.)
        return mean, changed - unchanged

    def attention(self, query, key, query_scale, bank):
        logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
        return (logits + source_bias(self.scale_bias[bank], query_scale, self.regions_per_scale)).softmax(-1)

    def forward(self, features_a, features_b, tar):
        differences, normalized, guide = self.paired_features(features_a, features_b, tar)
        prototypes = [self.prototypes(x, g.sigmoid(), grid)
                      for x, g, grid in zip(normalized, guide, self.grid_sizes)]
        mean, contrast = (torch.cat(bank, dim=1) for bank in zip(*prototypes))
        keys = (project_tokens(self.k_mean, mean, self.heads),
                project_tokens(self.k_contrast, contrast, self.heads))
        values = (project_tokens(self.v_mean, mean, self.heads),
                  project_tokens(self.v_contrast, contrast, self.heads))
        enhanced, boundaries, diagnostics = [], [], {}
        for index, (original, feature, difference) in enumerate(zip(tar, normalized, differences)):
            height, width = feature.shape[-2:]
            query = split_heads(self.q(feature), self.heads).flatten(2, 3)
            probabilities = [self.attention(query, key, index, bank) for bank, key in enumerate(keys)]
            content = [merge_heads(probability @ value, height, width)
                       for probability, value in zip(probabilities, values)]
            update = self.out(torch.cat(content, dim=1))
            result = original + self.gamma_context[index] * update
            local, detail_gamma = None, 0.
            if index < 2:
                high = binomial_highpass(difference)
                boundary = self.boundary(torch.cat((feature, difference, high), dim=1))
                local = self.diff_proj(difference + boundary.sigmoid() * high)
                detail_gamma = self.gamma_detail[index]
                result = result + detail_gamma * local
                boundaries.append(boundary)
            enhanced.append(result)
            if self.collect_diag:
                with torch.no_grad():
                    diag = residual_diagnostics(original, update, local, self.gamma_context[index], detail_gamma)
                    diag.update(queries=height * width, contexts=mean.shape[1] + contrast.shape[1],
                                contexts_per_bank=mean.shape[1], guide_mean=float(guide[index].sigmoid().mean()),
                                dense_difference_rms=float(difference.square().mean().sqrt()),
                                mean_prototype_rms=float(prototypes[index][0].square().mean().sqrt()),
                                contrast_prototype_rms=float(prototypes[index][1].square().mean().sqrt()))
                    for bank, probability in zip(("mean", "contrast"), probabilities):
                        diag.update({f"{bank}_{key}": value for key, value in
                                     attention_diagnostics(probability, self.regions_per_scale).items()})
                    if index < 2:
                        diag.update(boundary_mean=float(boundary.sigmoid().mean()),
                                    detail_highpass_rms=float(high.square().mean().sqrt()))
                    diagnostics[f"stage{index + 1}"] = diag
        if self.collect_diag:
            self.last_diag = diagnostics
        return enhanced, dict(guide_logits=tuple(guide), boundary_logits=tuple(boundaries))
