"""Differentiable first-order change/background pooling."""
import torch
from model.layers.pyramid_context import feature_regions


def class_assignments(probability, grid):
    positive = feature_regions(probability, grid).squeeze(-1)
    membership = torch.stack((positive, 1 - positive), 2)
    mass = membership.sum(-1, keepdim=True)
    denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
    weights = membership / denominator
    support = (weights * membership).sum(-1).flatten(1).clamp_min(1e-6)
    return weights, support
