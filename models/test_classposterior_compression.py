"""Regional class correction: symmetry, direction, common content and gradients."""
import torch

from model.layers.hyperbolic_context import HyperbolicContext, MatchedEuclideanContext
from model.layers.posterior_compression import class_assignments, compress_posterior, posterior_assignments
from model.layers.pyramid_context import feature_regions, pool_features


def inputs(op, feature, guide, grid):
    weights, _ = class_assignments(guide.sigmoid(), grid)
    initial = pool_features(feature, weights, grid)
    return op.project_queries(feature), op.project_keys(initial)[0]


def check(op_type):
    op = op_type(channels=4, head_dim=3, grid_sizes=(2, 2, 2, 2)).double()
    grid = 2
    guide = torch.randn(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    feature = torch.randn(1, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    query, key = inputs(op, feature, guide, grid)
    temperature = torch.nn.functional.softplus(op.raw_compression_temperature).reshape(1)
    prototype, support, stats = compress_posterior(feature, guide, query, key, grid, temperature, op.squared_distance)
    blocks = feature_regions(feature, grid)
    expected = []
    for region in range(4):
        for category in range(2):
            membership = stats["posterior"][0, region]
            if category:
                membership = 1 - membership
            expected.append((blocks[0, region] * membership[:, None]).sum(0) / membership.sum())
    expected = torch.stack(expected)[None]
    torch.testing.assert_close(prototype, expected)
    original = torch.stack((stats["prior"], 1 - stats["prior"]), 2)
    torch.testing.assert_close(support, (stats["weights"] * original).sum(-1).flatten(1))
    weights, _, swapped = posterior_assignments(-guide, query, key.reshape(1, 1, 4, 2, 4).flip(3).flatten(2, 3),
                                                 grid, temperature, op.squared_distance)
    torch.testing.assert_close(swapped["posterior"], 1 - stats["posterior"])
    torch.testing.assert_close(weights, stats["weights"].flip(2))
    torch.testing.assert_close(swapped["delta"], -stats["delta"])
    loss = (prototype * torch.randn_like(prototype)).sum() + (support * torch.randn_like(support)).sum()
    loss.backward()
    for name, gradient in (("feature", feature.grad), ("guide", guide.grad),
                           ("qk", op.qk.weight.grad), ("temperature", op.raw_compression_temperature.grad)):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name

    for zero in (False, True):
        constant = torch.zeros(1, 4, 1, 1, dtype=torch.float64) if zero else torch.randn(1, 4, 1, 1, dtype=torch.float64)
        field = constant.expand(1, 4, 4, 4)
        query, key = inputs(op, field, guide.detach(), grid)
        pooled, _, constant_stats = compress_posterior(field, guide.detach(), query, key, grid, temperature, op.squared_distance)
        torch.testing.assert_close(constant_stats["posterior"], constant_stats["prior"], rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(pooled, constant.flatten(2).transpose(1, 2).expand(1, 8, 4))

    # A query at its positive center increases p; exchanging centers decreases p.
    origin = op.geometric_point(torch.zeros(1, 1, 1, 3, dtype=torch.float64))
    distant = op.geometric_point(torch.tensor([[[[4., 0., 0.]]]], dtype=torch.float64))
    dense = origin.reshape(1, 1, 1, 1, 4).expand(1, 1, 2, 2, 4)
    two_keys = torch.cat((origin, distant), 2)
    _, _, positive = posterior_assignments(torch.zeros(1, 1, 2, 2, dtype=torch.float64), dense, two_keys,
                                            1, temperature, op.squared_distance)
    _, _, negative = posterior_assignments(torch.zeros(1, 1, 2, 2, dtype=torch.float64), dense, two_keys.flip(2),
                                            1, temperature, op.squared_distance)
    assert (positive["posterior"] > .99).all() and (negative["posterior"] < .01).all()
    print(op.geometry + ": constant/zero, class swap, signed correction, shared K/V content and gradients PASS")


def main():
    torch.manual_seed(2333)
    torch.set_num_threads(4)
    for kind in (HyperbolicContext, MatchedEuclideanContext):
        check(kind)


if __name__ == "__main__":
    main()
