"""Checks for the cross-scale prototype tree (run: python models/test_hierarchy.py)."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.layers.hierarchical_prototypes import (ScaleHierarchy, hierarchy_loss,
                                                  parent_index, retrieval_mask)

GRIDS = (8, 4, 4, 4)


def test_parent_links():
    parents = parent_index(GRIDS)
    assert parents.numel() == 224, parents.numel()
    # scale 0 is 8x8x2 = 128 slots starting at 0; its parents live at 128
    assert parents[0].item() == 128
    assert parents[4].item() == 128 + 2           # (0,4) of an 8x8 grid -> (0,2) of 4x4
    assert parents[127].item() == 144 + 15        # class 1 starts at 144, last cell -> (3,3)
    # the coarsest scale parents itself, so no link escapes the tensor
    assert parents.max().item() < 224
    assert parents[-1].item() == 223


def test_mask_prunes():
    keep, ancestor = retrieval_mask(GRIDS)
    assert keep.shape == (224, 224)
    assert bool(keep.diagonal().all())
    counts = keep.sum(-1)
    # a corner cell keeps its 2x2 window plus the three ancestors on its chain
    leaf = keep[0]
    assert 4 <= int(leaf.sum()) <= 12, int(leaf.sum())
    # a root keeps its whole subtree: 1 root + 4 children + 16 grandchildren ...
    root = keep[128]
    assert int(root.sum()) >= 1 + 4, int(root.sum())
    assert int(counts.max()) < 224, int(counts.max())
    assert ancestor[0].item() == 128


def test_radii_are_monotone():
    hierarchy = ScaleHierarchy(GRIDS)
    radii = hierarchy.radii()
    assert radii.shape == (4,)
    assert bool((torch.diff(radii) < 0).all()), radii           # index 0 is the finest, deepest
    with torch.no_grad():
        hierarchy.increments.fill_(-8.0)                       # increments -> 0
    flat = hierarchy.radii()
    assert float(flat.max() - flat.min()) < 1e-2, flat         # degenerates, never inverts
    with torch.no_grad():
        hierarchy.increments.fill_(4.0)
    steep = hierarchy.radii()
    assert bool((torch.diff(steep) < 0).all()), steep


def test_entailment_reacts():
    hierarchy = ScaleHierarchy(GRIDS, aperture=math.pi / 6)
    scales = hierarchy.scale_of(224)
    # a nested tree: every point points the same way, so nothing violates
    nested = torch.zeros(224, 33)
    nested[:, 0] = 1.0
    nested[:, 1] = torch.linspace(0.1, 3.0, 224)
    assert float(hierarchy.entailment(nested, scales, sample=None)) < 1e-5
    # flip the children of the finest scale: they now sit in a different cone
    spread = nested.clone()
    spread[:128, 1] = -2.0
    spread[:128, 2] = 2.0
    violation = float(hierarchy.entailment(spread, scales, sample=None))
    assert violation > 0.5, violation
    loss = hierarchy_loss(hierarchy, spread, scales, 0.5)
    assert float(loss) > 0.25, float(loss)
    zero = hierarchy_loss(hierarchy, nested, scales, 0.5)
    assert float(zero) < 1e-5, float(zero)


def test_pruning_ratio():
    hierarchy = ScaleHierarchy(GRIDS)
    counts = hierarchy.subtree_counts()
    mean = float(counts.float().mean())
    assert mean < 224 / 2, mean
    print(f"  mean candidates per query {mean:.1f} (flat 224), leaf {int(counts[0])}")


def main():
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"[OK] {name}")
    print("cross-scale prototype tree: parents, pruning, radii and entailment PASS")


if __name__ == "__main__":
    main()
