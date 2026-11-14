"""Check grid geometry, cross-scale connectivity and manifold gradients."""
import torch

from model.layers.pyramid_context import PyramidContext
from model.layers.poincare_ops import expmap0, logmap0, pairwise_distance, weighted_centroid


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    sizes = (64, 32, 16, 8)
    features = [torch.randn(1, 64, n, n, requires_grad=True) for n in sizes]
    for geometry in ("euclidean", "hyperbolic"):
        op = PyramidContext(geometry=geometry)
        weights = op.assignments(torch.full((1, 1, 64, 64), 0.5))
        assert weights.shape == (1, 16, 2, 256)
        assert torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]))
        # Region averages must correspond to the same 4x4 image partition.
        coordinate = torch.arange(64 * 64).reshape(1, 1, 64, 64, 1).float()
        regions = op.regions(coordinate)
        assert regions[0, 0, 0, 0, 0] == 0 and regions[0, 0, 1, 0, 0] == 16
        assert regions[0, 0, 4, 0, 0] == 16 * 64
        outputs, logits = op(features)
        assert all(torch.equal(x, y) for x, y in zip(features, outputs))
        assert [z.shape[-2:] for z in logits] == [(n, n) for n in sizes]
        op.gamma.data.fill_(0.1)
        op.collect_diag = True
        outputs, _ = op(features)
        assert all(x.shape == y.shape for x, y in zip(features, outputs))
        assert all(d["contexts"] == 128 and d["finite"] for d in op.last_diag.values())
        gradients = torch.autograd.grad(outputs[0].square().mean(), features, retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), "cross-scale path absent"
        op.zero_grad()
        sum(x.square().mean() for x in outputs).backward()
        for name, parameter in op.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        print(geometry, "aligned grids, 128 prototypes, residual identity and cross-scale task gradients PASS")
    # Exercise the reused geometry on four independent head dimensions.
    tangent = (torch.randn(1, 4, 3, 16, dtype=torch.double) * 0.03).requires_grad_()
    assert torch.allclose(logmap0(expmap0(tangent)), tangent, atol=1e-10)
    probability = torch.softmax(torch.randn(1, 4, 2, 3, dtype=torch.double), -1)
    assert torch.autograd.gradcheck(lambda x: weighted_centroid(probability, expmap0(x)), (tangent,))
    distance = pairwise_distance(expmap0(tangent), expmap0(tangent))
    assert torch.isfinite(distance).all() and torch.allclose(distance, distance.transpose(-1, -2))
    print("head-wise Exp/Log, distance and centroid gradients PASS")


if __name__ == "__main__":
    main()
