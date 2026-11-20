"""Run7: prototypes pooled on the manifold, and no learnable scale in the bridge."""
import math

import torch

from model.casa_net import CASANet
from model.layers.hyperbolic_context import HyperbolicContext
from model.str_tar import ChangeGateTARStage


def test_manifold_pool_stays_on_the_hyperboloid():
    points = torch.randn(2, 1, 4, 9, 8)
    weights = torch.rand(2, 4, 2, 9)
    weights = weights / weights.sum(-1, keepdim=True)
    pooled = HyperbolicContext.manifold_pool(points, weights)
    assert pooled.shape == (2, 1, 8, 9), pooled.shape
    time, space = pooled[..., 0], pooled[..., 1:]
    norm = time.square() - space.square().sum(-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), norm.min()
    assert pooled[..., 0].min() > 0


def test_manifold_pool_differs_from_the_euclidean_mean():
    torch.manual_seed(0)
    points = torch.randn(1, 1, 3, 16, 8) * 2.0        # (B, H, R, P, D)
    weights = torch.rand(1, 3, 2, 16)
    weights = weights / weights.sum(-1, keepdim=True)
    manifold = HyperbolicContext.manifold_pool(points, weights)[..., 1:]   # (B,H,2R,D)
    euclidean = torch.einsum("brtp,bhrpd->bhrtd", weights, points)         # (B,H,R,2,D)
    euclidean = euclidean.reshape(1, 1, -1, points.shape[-1])
    assert manifold.shape == euclidean.shape
    assert (manifold - euclidean).abs().max() > 1e-3


def test_manifold_pool_is_idempotent_on_a_single_point():
    points = torch.randn(1, 1, 2, 1, 8)
    weights = torch.ones(1, 2, 2, 1)
    pooled = HyperbolicContext.manifold_pool(points, weights)[..., 1:]
    expected = points[:, :, :, 0].repeat_interleave(2, dim=2)
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_context_switch_changes_the_kv_without_adding_parameters():
    torch.manual_seed(0)
    sources = [torch.randn(1, 64, 16, 16)]
    weights = torch.rand(1, 64, 2, 4)          # 8x8 cells, 2 classes, 4 pixels each
    weights = weights / weights.sum(-1, keepdim=True)
    compressed = [(weights, torch.ones(1, 128))]

    torch.manual_seed(1)
    euclidean = HyperbolicContext(heads=1, prototype_pooling="euclidean")
    torch.manual_seed(1)
    manifold = HyperbolicContext(heads=1, prototype_pooling="lorentz")
    assert sum(p.numel() for p in euclidean.parameters()) == \
        sum(p.numel() for p in manifold.parameters())

    key_e, value_e = euclidean.manifold_keys_and_values(sources, compressed) \
        if euclidean.prototype_pooling == "lorentz" else (None, None)
    assert key_e is None and value_e is None  # the flag really is consulted

    torch.manual_seed(2)
    prototype = torch.randn(1, 128, 64)
    key_e, _ = euclidean.project_keys(prototype)
    value_e = torch.nn.functional.linear(prototype, euclidean.v.weight[:, :, 0, 0])
    key_m, value_m = manifold.manifold_keys_and_values(sources, compressed)
    assert key_m.shape[-1] == key_e.shape[-1]
    assert value_m.shape[-1] == value_e.shape[-1]
    assert key_m.shape[2] == key_e.shape[2]


def test_unknown_pooling_is_rejected():
    try:
        HyperbolicContext(prototype_pooling="poincare")
    except ValueError as error:
        assert "prototype_pooling" in str(error)
    else:
        raise AssertionError("unknown pooling should raise")
    try:
        HyperbolicContext(kv_hierarchy="nested", prototype_pooling="lorentz")
    except ValueError as error:
        assert "flat hierarchy" in str(error)
    else:
        raise AssertionError("nested lorentz pooling should raise")


def test_bridge_has_no_learnable_scale_between_streams():
    stage = ChangeGateTARStage(64, 64)
    names = [name for name, _ in stage.named_parameters()]
    assert not any(name.startswith("beta") for name in names), names

    p = torch.randn(2, 64, 8, 8)
    q = torch.randn(2, 64, 8, 8)
    fused, delta = stage(p, q)
    content = torch.nn.functional.silu(stage.bn_content(stage.content(torch.cat((p, q), 1))))
    cp, cq = stage.contrast(p), stage.contrast(q)
    positive, negative = (cp - cq).relu(), (cq - cp).relu()
    expected_delta = stage.bn_difference(stage.difference(torch.cat((positive, negative), 1)))
    gate = stage.gate(torch.cat((expected_delta, positive + negative), 1)).sigmoid()
    expected = stage.block(content * (1 + gate) + expected_delta)
    assert torch.allclose(delta, expected_delta, atol=1e-6)
    assert torch.allclose(fused, expected, atol=1e-6)


def test_end_to_end_folds_and_keeps_shapes():
    for pooling in ("euclidean", "lorentz"):
        model = CASANet(module="manifold", tar_variant="cg", prototype_pooling=pooling)
        deploy = CASANet(module="manifold", tar_variant="cg", prototype_pooling=pooling)
        deploy.load_state_dict(model.state_dict())
        deploy.switch_to_deploy()
        model.eval()
        deploy.eval()
        a, b = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            reference = model(a, b)[0]
            folded = deploy(a, b)[0]
        assert reference.shape == (2, 1, 256, 256)
        assert torch.isfinite(reference).all()
        assert (reference - folded).abs().max() <= 2e-4


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[OK] {len(tests)} Run7 checks passed")


if __name__ == "__main__":
    main()
