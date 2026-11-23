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


def check_context_wiring():
    """The tree must prune logits, keep radii ordered and carry real gradients."""
    import torch
    from model.layers.hyperbolic_context import HyperbolicContext

    torch.manual_seed(0)
    grids = (8, 4, 4, 4)
    sizes = (64, 32, 16, 8)
    plain = HyperbolicContext(channels=8, heads=1, head_dim=4, grid_sizes=grids)
    tree = HyperbolicContext(channels=8, heads=1, head_dim=4, grid_sizes=grids,
                             prototype_tree="tree", entailment_weight=0.05)
    tree.load_state_dict(plain.state_dict(), strict=False)
    tar = [torch.randn(2, 8, g, g) for g in sizes]
    # gamma_context starts at zero, so the write-back is an exact no-op and every
    # comparison below would look identical; open it first (Run14 lesson).
    for module in (plain, tree):
        with torch.no_grad():
            module.gamma_context.fill_(0.5)

    plain.eval()
    tree.eval()
    out_plain, aux_plain = plain(tar, return_aux=True)
    out_tree, aux_tree = tree(tar, return_aux=True)
    assert "tree_violation" in aux_tree and "tree_loss" in aux_tree, aux_tree.keys()
    assert "tree_violation" not in aux_plain
    assert all(torch.isfinite(x).all() for x in out_tree)
    delta = max(float((a - b).abs().max()) for a, b in zip(out_plain, out_tree))
    assert delta > 1e-4, delta
    print(f"  tree vs flat output max diff {delta:.3e} (pruning + radii both act)")

    radius = tree.hierarchy.radii()
    assert bool((radius[:-1] > radius[1:]).all()), radius
    kept = tree.hierarchy.keep_mask.sum(-1).float().mean()
    print(f"  candidates per query {kept:.1f} of {tree.hierarchy.keep_mask.shape[-1]}")

    tree.train()
    _, aux_train = tree(tar, return_aux=True)
    aux_train["tree_loss"].backward()
    qk_grad = tree.qk.weight.grad
    assert qk_grad is not None and float(qk_grad.abs().sum()) > 0, qk_grad
    print(f"  entailment hinge moves the projections: |grad qk| = {float(qk_grad.abs().sum()):.3e}")
    radius_grad = tree.hierarchy.increments.grad
    print(f"  radii gradient from the hinge {radius_grad.tolist()} "
          f"(radii also enter every distance, where they do carry gradient)")

    shuffled = HyperbolicContext(channels=8, heads=1, head_dim=4, grid_sizes=grids,
                                 prototype_tree="tree", entailment_weight=0.05)
    shuffled.load_state_dict(tree.state_dict())
    generator = torch.Generator().manual_seed(7)
    perm = torch.randperm(shuffled.hierarchy.parents.numel(), generator=generator)
    with torch.no_grad():
        shuffled.hierarchy.parents.copy_(shuffled.hierarchy.parents[perm])
    _, aux_shuffled = shuffled(tar, return_aux=True)
    print(f"  violation ordered {float(aux_tree['tree_violation']):.4f} "
          f"vs shuffled {float(aux_shuffled['tree_violation']):.4f}")


if __name__ == "__main__":
    main()
    print("context wiring")
    check_context_wiring()
