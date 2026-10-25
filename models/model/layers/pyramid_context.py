"""Tensor operations and auxiliary supervision for paired pyramid contexts."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def channel_norm(layer, x):
    return layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def regions(x, grid):
    """B,H,height,width,D -> B,H,regions,pixels,D."""
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


def source_bias(bias, query_scale, counts):
    """Give each source scale equal prior mass before its learned bias."""
    return torch.cat([(bias[:, query_scale, scale] - math.log(count))[:, None].expand(-1, count)
                      for scale, count in enumerate(counts)], dim=-1)[None, :, None]


def binomial_highpass(x):
    kernel = x.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    blurred = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"),
                       kernel.expand(x.shape[1], 1, 3, 3), groups=x.shape[1])
    return x - blurred


def attention_diagnostics(probability, counts):
    tv = (probability - probability.mean(-2, keepdim=True)).abs().sum(-1) / 2
    return dict(
        entropy=float(-(probability * probability.clamp_min(1e-9).log()).sum(-1).mean()
                      / math.log(probability.shape[-1])),
        scale_mass=torch.stack([p.sum(-1) for p in probability.split(counts, -1)], -1).mean((0, 2)).tolist(),
        query_attention_tv=float(tv.mean()),
        query_attention_tv_per_head=tv.mean((0, 2)).tolist())


def residual_diagnostics(feature, context, local, gamma_context, gamma_detail):
    def rms(x):
        return x.square().mean().sqrt()
    base = rms(feature).clamp_min(1e-8)
    context_residual = gamma_context * context
    local_residual = gamma_detail * local if local is not None else torch.zeros_like(feature)
    return dict(
        gamma_context=float(gamma_context), gamma_detail=float(gamma_detail),
        branch_rms=float(rms(context)), context_rms=float(rms(context)),
        local_rms=float(rms(local)) if local is not None else 0.,
        feature_rms=float(base), residual_rms=float(rms(context_residual + local_residual)),
        residual_to_feature_rms=float(rms(context_residual + local_residual) / base),
        context_residual_to_feature_rms=float(rms(context_residual) / base),
        local_residual_to_feature_rms=float(rms(local_residual) / base),
        finite=bool(torch.isfinite(context).all() and
                    (local is None or torch.isfinite(local).all())))


class AuxiliaryContext(nn.Module):
    """CTRL retains guide and boundary supervision on the TAR representation."""

    geometry = "none"
    value_geometry = "euclidean"
    grid_sizes = (8, 4, 4, 4)
    tokens_per_scale = (0, 0, 0, 0)

    def __init__(self, channels=64):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.guide = nn.Conv2d(channels, 1, 1)
        self.boundary = nn.Conv2d(channels, 1, 1)
        self.collect_diag = False
        self.last_diag = {}

    def forward(self, features_a, features_b, tar):
        normalized = [channel_norm(self.norm, x) for x in tar]
        return list(tar), dict(guide_logits=tuple(self.guide(x) for x in normalized),
                               boundary_logits=tuple(self.boundary(x) for x in normalized[:2]))
