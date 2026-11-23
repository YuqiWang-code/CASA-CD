"""Cross-scale prototype tree for the hyperbolic retrieval module.

The previous hyperbolic retrieval arms lifted every prototype independently and
computed a Lorentz distance; the geometry therefore only re-weighted the softmax
logits and an Euclidean twin matched it to within noise.  This module gives the
manifold a job the Euclidean chart cannot do: it encodes the *cross-scale
hierarchy* of the prototypes.

Every scale keeps its own grid of class-conditioned prototypes, and a coarse
prototype is the parent of the four (or one) finer prototypes inside its cell:

    scale 3 (8x8,  8 b grid)  <- root, smallest radius
    scale 2 (16x16, 4x4 grid)
    scale 1 (32x32, 4x4 grid)
    scale 0 (64x64, 8x8 grid) <- leaves, largest radius

Two properties are imposed:

* **radial order** -- radii increase monotonically from coarse to fine, built
  by a positive increment parameterisation so the constraint cannot be violated
  by training;
* **entailment** -- a child must sit inside its parent's cone, i.e. its spatial
  direction must stay within ``aperture(depth)`` of the parent's.  The hinge
  loss below pushes violators back in.

The parent links are deterministic (a child belongs to the coarse cell that
contains it), so they can be *shuffled* at test time to check whether the model
actually relies on them.
"""
import math

import torch
from torch import nn


def parent_index(grid_sizes):
    """Map every prototype of every scale to its parent prototype.

    ``grid_sizes`` is ordered fine to coarse, as in the context module.  The
    class axis is preserved: a change prototype can only parent a change
    prototype.  The coarsest scale is its own parent so the tensor stays dense.
    """
    offsets = sum(2 * grid * grid for grid in grid_sizes)
    index = torch.empty(offsets, dtype=torch.long)
    cursor = 0
    for position, grid in enumerate(grid_sizes):
        coarse = grid_sizes[position + 1] if position + 1 < len(grid_sizes) else grid
        rows = torch.arange(grid).view(grid, 1).expand(grid, grid)
        cols = torch.arange(grid).view(1, grid).expand(grid, grid)
        parent_rows = (rows * coarse // grid).reshape(-1)
        parent_cols = (cols * coarse // grid).reshape(-1)
        cells = parent_rows * coarse + parent_cols
        for label in range(2):
            base = cursor + label * grid * grid
            if position + 1 < len(grid_sizes):
                parent_base = cursor + 2 * grid * grid + label * coarse * coarse
            else:
                parent_base = cursor + label * grid * grid   # coarsest scale parents itself
            index[base:base + grid * grid] = parent_base + cells
        cursor += 2 * grid * grid
    return index


def retrieval_mask(grid_sizes, radius=1):
    """Boolean ``(2R, 2R)`` mask of the prototypes a query may retrieve.

    A query keeps, per scale, the ``(2*radius+1)^2`` spatial neighbourhood of the
    same class, plus its own ancestor chain and the descendants of its cell.  The
    point of the tree is that the coarse scales stop competing with every fine
    prototype: the candidate count for a fine query is ``O(radius^2 + depth)``
    instead of ``224``, and shuffling the parent links later tests whether the
    ancestor part is actually used.
    """
    parents = parent_index(grid_sizes)
    sizes = [2 * grid * grid for grid in grid_sizes]
    total = sum(sizes)
    keep = torch.eye(total, dtype=torch.bool)
    offsets, cells = [], 0
    for grid in grid_sizes:
        offsets.append((cells, grid))
        cells += 2 * grid * grid
    for position, (offset, grid) in enumerate(offsets):
        rows = torch.arange(grid).view(grid, 1).expand(grid, grid).reshape(-1)
        cols = torch.arange(grid).view(1, grid).expand(grid, grid).reshape(-1)
        near = ((rows[:, None] - rows[None, :]).abs() <= radius) & \
               ((cols[:, None] - cols[None, :]).abs() <= radius)
        for label in range(2):
            base = offset + label * grid * grid
            keep[base:base + grid * grid, base:base + grid * grid] |= near
    # ancestor / descendant closure over the parent chain
    ancestor = torch.zeros(total, dtype=torch.long) - 1
    cursor = 0
    for index, size in enumerate(sizes):
        for slot in range(size):
            node = cursor + slot
            ancestor[node] = parents[node] if index + 1 < len(sizes) else -1
        cursor += size
    for node in range(total):
        parent = ancestor[node].item()
        while parent >= 0:
            keep[node, parent] = True
            keep[parent, node] = True
            parent = ancestor[parent].item()
    return keep, ancestor


class ScaleHierarchy(nn.Module):
    """Monotone per-scale radii plus the entailment hinge.

    ``radii`` are the Lorentz radii used to lift each scale's prototypes; they
    are produced from positive increments so ``r[0] > r[1] > ...`` holds for any
    parameter value (``scale 0`` is the finest, deepest level).
    """

    def __init__(self, grid_sizes, base_radius=0.5, step=0.35, aperture=math.pi / 6):
        super().__init__()
        self.grid_sizes = tuple(grid_sizes)
        self.scales = len(self.grid_sizes)
        self.base_radius = float(base_radius)
        self.aperture = float(aperture)
        self.increments = nn.Parameter(torch.full((self.scales - 1,), float(step)))
        self.parents = nn.Parameter(parent_index(self.grid_sizes), requires_grad=False)
        self.leaf_scale = self.scales - 1
        mask, ancestor = retrieval_mask(self.grid_sizes)
        self.register_buffer("keep_mask", mask, persistent=False)
        self.register_buffer("ancestor", ancestor, persistent=False)

    def radii(self):
        """Radii from coarse (smallest) to fine (largest), one per scale."""
        steps = nn.functional.softplus(self.increments) + 1e-3
        cumulative = torch.cat([torch.zeros(1, device=steps.device), steps.cumsum(0)])
        radius = self.base_radius * torch.exp(cumulative)
        return radius.flip(0)

    def scale_of(self, count):
        """Index tensor mapping each prototype slot to its scale."""
        labels = torch.cat([torch.full((2 * grid * grid,), position, dtype=torch.long)
                            for position, grid in enumerate(self.grid_sizes)])
        return labels[:count]

    def aperture_of(self, scale_index):
        """Half-angle of the cone at a given scale: coarse cones open wider."""
        depth = (self.scales - 1 - scale_index.float()) / max(self.scales - 1, 1)
        return self.aperture * (1.0 - 0.5 * depth)

    @staticmethod
    def spatial(points):
        """Spatial part of a Lorentz point ``(..., D+1)`` as a unit direction."""
        space = points[..., 1:]
        return space / space.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def entailment(self, points, scale_of, sample=256):
        """Mean hinge violation of child-inside-parent cones, in radians.

        ``points`` is ``(heads, P, D+1)`` or ``(P, D+1)``; ``scale_of`` maps a
        prototype slot to its scale.  Only child/parent pairs are scored, and at
        most ``sample`` of them per call keeps the term cheap.
        """
        if points.dim() == 3:
            points = points.mean(0)
        direction = self.spatial(points)
        parents = self.parents
        child = torch.arange(points.shape[0], device=points.device)
        valid = scale_of < self.scales - 1
        child = child[valid]
        if child.numel() == 0:
            return points.new_zeros(())
        if sample and child.numel() > sample:
            pick = torch.randperm(child.numel(), device=points.device)[:sample]
            child = child[pick]
        parent = parents[child]
        cosine = (direction[child] * direction[parent]).sum(-1).clamp(-1.0, 1.0)
        angle = cosine.acos()
        limit = self.aperture_of(scale_of[child])
        violation = (angle - limit).clamp_min(0.0)
        return violation.mean()

    def subtree_counts(self):
        """How many candidates each query keeps once the tree prunes them."""
        return self.keep_mask.sum(-1)


def hierarchy_loss(hierarchy, points, scales, weight):
    """Weighted entailment term, zero when every child already sits in its cone."""
    return weight * hierarchy.entailment(points, scales)
