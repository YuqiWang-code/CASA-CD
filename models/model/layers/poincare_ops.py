"""Curvature-one Poincare operations with tangent clipping at radius 2.3."""
import math

import torch

RADIUS = math.tanh(2.3)


def project(x):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    return x * (RADIUS / norm).clamp_max(1)


def expmap0(v):
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
    return v * (norm.clamp_max(2.3).tanh() / norm)


def logmap0(x):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp(1e-8, RADIUS)
    return x * (torch.atanh(norm) / norm)


def displacement(x, y):
    """Clipped Mobius displacement (-y) (+) x, exactly zero for x == y."""
    delta = x - y
    delta2 = delta.square().sum(-1, keepdim=True)
    y2 = y.square().sum(-1, keepdim=True)
    yd = (y * delta).sum(-1, keepdim=True)
    denominator = (1 - y2).square() - 2 * (1 - y2) * yd + y2 * delta2
    return project(((1 - y2) * delta - delta2 * y) / denominator.clamp_min(1e-8))


def aligned_distance(x, y):
    numerator = torch.linalg.vector_norm(x - y, dim=-1)
    denominator = ((1 - x.square().sum(-1)) * (1 - y.square().sum(-1))).clamp_min(1e-8).sqrt()
    return 2 * torch.asinh(numerator / denominator)


def pairwise_distance(x, y):
    x2, y2 = x.square().sum(-1), y.square().sum(-1)
    squared = (x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2 * (x @ y.transpose(-1, -2))).clamp_min(0)
    denominator = ((1 - x2).unsqueeze(-1) * (1 - y2).unsqueeze(-2)).clamp_min(1e-8)
    return 2 * (torch.asinh((squared / denominator + 1e-8).sqrt()) - math.asinh(1e-4))


def to_lorentz(x):
    squared = x.square().sum(-1, keepdim=True)
    denominator = (1 - squared).clamp_min(1e-8)
    return torch.cat(((1 + squared) / denominator, 2 * x / denominator), dim=-1)


def from_lorentz_sum(x):
    """Normalize a positive Lorentz sum and return its Poincare centroid."""
    time, space = x[..., :1], x[..., 1:]
    norm = (time.square() - space.square().sum(-1, keepdim=True)).clamp_min(1e-8).sqrt()
    return project(space / (time + norm).clamp_min(1e-8))


def weighted_centroid(weights, values):
    """Einstein/Lorentz centroid of manifold values, not a Euclidean mean."""
    return from_lorentz_sum(weights @ to_lorentz(values))
