"""Aligned change/stable prototypes with full-resolution queries."""
import torch
from torch.nn import functional as F


def cell_tokens(x, cell=4):
    """Return cells as [batch, cells, pixels, channels]."""
    batch, channels, _, _ = x.shape
    return F.unfold(x, cell, stride=cell).reshape(
        batch, channels, cell * cell, -1).permute(0, 3, 2, 1)


def prototype_weights(change):
    """Two shared assignments per cell: change and stable evidence."""
    scores = cell_tokens(change).squeeze(-1)
    membership = torch.stack((scores, 1 - scores), dim=2) + 1e-6
    return membership / membership.sum(-1, keepdim=True)


def relative_prototype_weights(change):
    """Pool high/low local change contrast with opposite softmax assignments."""
    scores = cell_tokens(change).squeeze(-1)
    centered = scores - scores.mean(-1, keepdim=True)
    scale = (centered.square().mean(-1, keepdim=True) + 1e-4).sqrt()
    contrast = centered / scale
    return torch.stack((contrast, -contrast), dim=2).softmax(-1)


def pool_tokens(x, weights):
    return torch.einsum("bltp,blpc->bltc", weights, cell_tokens(x)).flatten(1, 2)


def token_map(tokens, height, width):
    return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
