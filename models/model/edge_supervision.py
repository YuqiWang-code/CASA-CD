"""Sobel edge targets and a cheap 1x1 edge head for the bridge and the retrieval.

The change mask is a step function, so a Sobel magnitude on it is 1 on the
boundary ring and 0 inside flat regions.  Widening that ring by one pixel keeps
the target tolerant to a one-pixel localisation error, which is what the
boundary metric already allows.  Both consumers (TAR-CG evidence and the
retrieval write-back) get at most 65 parameters, so the loss cannot buy accuracy
by growing the model.
"""
import torch
import torch.nn.functional as F
from torch import nn

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]]) / 4.0


def sobel_edge_map(mask, dilate=1):
    """Return an edge strength map in [0, 1] for a binary mask of shape (B, 1, H, W)."""
    if mask.dim() != 4:
        raise ValueError("the edge target expects (B, 1, H, W)")
    kx = _SOBEL_X.to(dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(mask, kx, padding=1)
    gy = F.conv2d(mask, ky, padding=1)
    magnitude = (gx.square() + gy.square()).sqrt().clamp(0.0, 1.0)
    if dilate > 0:
        magnitude = F.max_pool2d(magnitude, 2 * dilate + 1, 1, dilate)
    return magnitude.clamp(0.0, 1.0)


class EdgeHead(nn.Module):
    """1x1 projection from a feature stream to a single edge logit map."""

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, 1, 1)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feature):
        return self.proj(feature)


def edge_loss(logits, target, max_positive_weight=64.0):
    """Balanced BCE plus soft Dice, both on the edge ring."""
    positive = target.mean().clamp_min(1e-6)
    weight = ((1.0 - positive) / positive).clamp(1.0, max_positive_weight).detach()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    probability = logits.sigmoid()
    overlap = (2.0 * (probability * target).sum() + 1.0) / (probability.sum() + target.sum() + 1.0)
    return bce + (1.0 - overlap)
