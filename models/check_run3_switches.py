"""Fail-fast checks for the Run3 step-A/B switches (kv-hierarchy / support / freeze-tau).

CPU only: no checkpoint, no dataset, no GPU memory. Checks:
  1. nested parent/child index structure per scale pair;
  2. forward + backward through all four scales (no NaN, q/k gradients alive);
  3. deploy folding maxdiff (flat and nested) and binary agreement;
  4. trainable/deploy parameter counts for the four hierarchy x support combos;
  5. KV token counts (flat 224, nested 170);
  6. nested parent-to-child pooling actually changes the update;
  7. --support pixel really changes the support tensor and logits under identical weights;
  8. --freeze-tau removes the temperature gradient;
  9. the difference context accepts (difference + spp) and (difference + diff, nested).
"""
import itertools
import math
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, ".")

from model.casa_net import CASANet  # noqa: E402
from model.layers.hyperbolic_context import (  # noqa: E402
    FLAT_GRIDS, FROZEN_TAU, NESTED_GRIDS, HyperbolicContext)

EXPECTED = {  # director's Run3 table: hyperbolic + spp, flat == nested
    "trainable": 2129577, "deploy": 2006998,
}


def build(hierarchy, support, seed=2334):
    torch.manual_seed(seed)
    return CASANet(None, module="hyperbolic", context_seed=seed, tar_variant="spp",
                   support=support, kv_hierarchy=hierarchy).eval()


def counts(model):
    return sum(p.numel() for p in model.parameters())


def fold_params(model):
    """Deploy parameter count without mutating the caller's model."""
    import copy
    clone = copy.deepcopy(model)
    with torch.no_grad():
        clone.switch_to_deploy()
    return sum(p.numel() for p in clone.parameters())


def check_hierarchy():
    grids = NESTED_GRIDS
    for parent_grid, child_grid in zip(grids[:-1], grids[1:]):
        assert child_grid * 2 == parent_grid, (parent_grid, child_grid)
        for row in range(child_grid):
            for col in range(child_grid):
                block = [(2 * row + dr) * parent_grid + (2 * col + dc)
                         for dr in (0, 1) for dc in (0, 1)]
                assert len(block) == 4 and len(set(block)) == 4, (row, col, block)
    print(f"[HIERARCHY] ok grids={grids} parent blocks are exact 2x2 neighbourhoods")


def check_forward_backward(model, tag):
    torch.manual_seed(7)
    a, b = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
    model.train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    # Step 1: gamma_context is zero-initialised, so at initialisation only the gate
    # receives gradient; that is by design and must be reported, not asserted away.
    loss = sum(o.float().mean() for o in model(a, b))
    loss.backward()
    assert torch.isfinite(loss).item(), f"{tag}: non-finite loss"
    q_at_init = model.context.qk.weight.grad.norm().item()
    gate = model.context.gamma_context.grad.norm().item()
    assert gate > 0, f"{tag}: gate gradient is zero"
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.context.gamma_context.add_(0.1)
    # Step 2: with the gate off zero the q/k projection must be trainable.
    loss = sum(o.float().mean() for o in model(a, b))
    loss.backward()
    q_norm = model.context.qk.weight.grad.norm().item()
    v_grad = model.context.v.weight.grad
    v_norm = 0.0 if v_grad is None else v_grad.norm().item()
    assert q_norm > 0, f"{tag}: q/k projection gradient is zero after one gate step"
    assert v_norm > 0, f"{tag}: value projection gradient is zero"
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads), f"{tag}: non-finite gradients"
    print(f"[GRAD] {tag} loss={loss.item():.6f} qk_grad@{tag}={q_norm:.3e} "
          f"v_grad={v_norm:.3e} qk_grad_at_init={q_at_init:.3e} gate_grad={gate:.3e}")
    return {"loss": loss.item(), "qk_grad": q_norm, "v_grad": v_norm,
            "qk_grad_at_init": q_at_init, "gate_grad": gate}


def check_fold(hierarchy):
    model = build(hierarchy, "cell")
    torch.manual_seed(11)
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        before = model.predict(a, b).clone()
        model.switch_to_deploy()
        after = model.predict(a, b)
    maxdiff = (before - after).abs().max().item()
    binary = int((before > .5).ne(after > .5).sum().item())
    assert maxdiff <= 2e-4, f"{hierarchy}: fold maxdiff {maxdiff:.3e}"
    assert binary == 0, f"{hierarchy}: {binary} flipped pixels after fold"
    print(f"[FOLD] {hierarchy} maxdiff={maxdiff:.3e} binary_diff={binary} pixels={before.numel()}")
    return {"maxdiff": maxdiff, "binary_diff": binary}


def check_support_effect():
    """Same weights, different support prior: the support tensor and logits must differ."""
    mass = build("flat", "cell")
    pixel = build("flat", "pixel")
    pixel.load_state_dict(mass.state_dict())
    torch.manual_seed(13)
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    captured = {}

    def capture(context, tag):
        original = context.attention_logits

        def wrapped(query, key, support, support_index, *args, **kwargs):
            recorded = captured.setdefault(tag, [])
            recorded.append(support.detach().clone())
            out = original(query, key, support, support_index, *args, **kwargs)
            captured[f"{tag}_logits"] = captured.get(f"{tag}_logits", [])
            captured[f"{tag}_logits"].append(out.detach().clone())
            return out
        context.attention_logits = wrapped

    capture(mass.context, "mass")
    capture(pixel.context, "pixel")
    with torch.no_grad():
        mass.predict(a, b)
        pixel.predict(a, b)
    support_diff = max((x - y).abs().max().item()
                       for x, y in zip(captured["mass"], captured["pixel"]))
    logit_diff = max((x - y).abs().max().item()
                     for x, y in zip(captured["mass_logits"], captured["pixel_logits"]))
    assert support_diff > 1e-4, "support pixel is a silent no-op (support unchanged)"
    assert logit_diff > 1e-4, "support pixel does not change the attention logits"
    print(f"[SUPPORT] support maxdiff={support_diff:.3e} logits maxdiff={logit_diff:.3e}")
    return {"support_maxdiff": support_diff, "logits_maxdiff": logit_diff}


def check_freeze_tau():
    frozen = build("flat", "cell")
    frozen_fold = CASANet(None, module="hyperbolic", context_seed=2334, tar_variant="spp",
                          support="cell", freeze_tau=True).eval()
    tau_frozen = F.softplus(frozen_fold.context.raw_temperature.detach()).mean().item()
    assert not frozen_fold.context.raw_temperature.requires_grad
    assert abs(tau_frozen - FROZEN_TAU) < 1e-4, tau_frozen
    delta = (sum(p.numel() for p in frozen.parameters() if p.requires_grad)
             - sum(p.numel() for p in frozen_fold.parameters() if p.requires_grad))
    assert delta == 1, delta
    assert sum(p.numel() for p in frozen_fold.parameters()) == sum(
        p.numel() for p in frozen.parameters()), "freeze_tau must not change the parameter count"
    print(f"[TAU] frozen softplus(tau)={tau_frozen:.6f} (target {FROZEN_TAU}) params delta={delta}")
    return {"tau": tau_frozen, "params_delta": delta}


def check_nested_effect():
    """Same weights: nested must change the context output, and pooling must be the 2x2 mean."""
    flat = build("flat", "cell")
    nested = build("nested", "cell")
    nested.load_state_dict(flat.state_dict())
    context = nested.context
    batch, dim = 1, 3
    prototypes = []
    for grid in NESTED_GRIDS:
        cells = 2 * grid * grid
        base = torch.arange(batch * cells * dim, dtype=torch.float32).reshape(batch, cells, dim)
        prototypes.append(base)
    aggregated = context.aggregate_nested(prototypes)
    parent = prototypes[0].reshape(batch, 8, 8, 2, dim)
    pooled = parent.reshape(batch, 4, 2, 4, 2, 2, dim).mean(dim=(2, 4)).reshape(batch, -1, dim)
    expected = 0.5 * (prototypes[1] + pooled)
    maxdiff = (aggregated[1] - expected).abs().max().item()
    assert maxdiff < 1e-6, f"nested pooling is not the 2x2 parent mean: {maxdiff}"
    torch.manual_seed(17)
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    updates = {}

    def capture(context, tag):
        original = context.readout

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            updates.setdefault(tag, []).append(out[0].detach().clone())
            return out
        context.readout = wrapped

    capture(flat.context, "flat")
    capture(nested.context, "nested")
    with torch.no_grad():
        flat.context.gamma_context.fill_(1.0)
        nested.context.gamma_context.fill_(1.0)
        out_flat = flat.predict(a, b)
        out_nested = nested.predict(a, b)
    update_diff = max((x - y).abs().max().item()
                      for x, y in zip(updates["flat"], updates["nested"]))
    assert update_diff > 1e-4, f"nested hierarchy leaves the readout update unchanged: {update_diff}"
    diff = (out_flat - out_nested).abs().max().item()
    print(f"[NESTED] pooling maxdiff={maxdiff:.3e} update maxdiff={update_diff:.3e} "
          f"output maxdiff={diff:.3e} (gamma=1)")
    return {"pooling_maxdiff": maxdiff, "update_maxdiff": update_diff, "output_maxdiff": diff}


def check_difference_passthrough():
    torch.manual_seed(2334)
    model = CASANet(None, module="difference", context_seed=2334, tar_variant="spp",
                    kv_hierarchy="nested", support="pixel").eval()
    trainable = sum(p.numel() for p in model.parameters())
    deploy = fold_params(model)
    assert model.context.grid_sizes == NESTED_GRIDS, model.context.grid_sizes
    assert (trainable, deploy) == (2129705, 2007126), (trainable, deploy)
    print(f"[DIFF-NESTED] grids={model.context.grid_sizes} trainable={trainable} deploy={deploy}")


def main():
    report = {"hierarchy": {}, "params": {}, "fold": {}, "kv": {}}
    check_hierarchy()
    for hierarchy in ("flat", "nested"):
        model = build(hierarchy, "cell")
        grids = model.context.grid_sizes
        kv = sum(2 * g * g for g in grids)
        report["kv"][hierarchy] = kv
        print(f"[KV] {hierarchy} grids={grids} kv={kv} context_grids={model.context.grid_sizes}")
        report["hierarchy"][hierarchy] = check_forward_backward(model, hierarchy)
        report["fold"][hierarchy] = check_fold(hierarchy)
    for hierarchy, support in itertools.product(("flat", "nested"), ("cell", "pixel")):
        model = build(hierarchy, support)
        trainable = sum(p.numel() for p in model.parameters())
        deploy = fold_params(model)
        report["params"][f"{hierarchy}/{support}"] = {"trainable": trainable, "deploy": deploy}
        print(f"[PARAMS] {hierarchy}/{support} trainable={trainable} deploy={deploy}")
        assert trainable == EXPECTED["trainable"], (hierarchy, support, trainable)
        assert deploy == EXPECTED["deploy"], (hierarchy, support, deploy)
    report["support"] = check_support_effect()
    report["tau"] = check_freeze_tau()
    report["nested"] = check_nested_effect()
    check_difference_passthrough()
    assert report["kv"] == {"flat": 224, "nested": 170}, report["kv"]
    print("[OK] all Run3 switch checks passed")
    return report


if __name__ == "__main__":
    main()
