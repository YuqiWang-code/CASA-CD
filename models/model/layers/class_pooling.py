"""Differentiable first-order change/background pooling."""
import torch
from model.layers.pyramid_context import feature_regions


def class_assignments(probability, grid, support_mode="mass"):
    """Cell-level class weights, plus the prior that scales their attention.

    ``mass`` (historical) is a participation ratio: it shrinks when the two
    classes are mixed inside the cell, so it rewards big homogeneous regions.
    ``confidence`` is ``|2 * share - 1|``: how decisive the cell is, in [0, 1],
    which is the quantity we actually want in ``log(support)``.
    """
    positive = feature_regions(probability, grid).squeeze(-1)
    membership = torch.stack((positive, 1 - positive), 2)
    mass = membership.sum(-1, keepdim=True)
    denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
    weights = membership / denominator
    if support_mode == "confidence":
        support = (2 * weights[..., 0] - 1).abs().flatten(1)
    else:
        support = (weights * membership).sum(-1).flatten(1)
    return weights, support.clamp_min(1e-6)
