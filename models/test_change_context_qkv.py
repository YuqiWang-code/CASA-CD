"""Geometry, aligned change evidence and dense-query interface tests."""
import math

import torch

from model.layers.change_context_qkv import ChangeAwareCompressedQKV
from model.layers.context_compression import pool_tokens, prototype_weights, relative_prototype_weights
from model.layers.hyperbolic_context_qkv import HyperbolicCompressedQKV
from model.layers.poincare_ops import (
    RADIUS, aligned_distance, displacement, expmap0, logmap0,
    pairwise_distance, weighted_centroid,
)


def geometry():
    tangent = torch.randn(2, 5, 4, dtype=torch.float64) * 0.1
    assert torch.allclose(logmap0(expmap0(tangent)), tangent, atol=1e-12)
    point = torch.tensor([[[0.6, 0.0]]], dtype=torch.float64)
    origin = torch.zeros_like(point)
    assert torch.allclose(aligned_distance(point, origin), torch.tensor([[2 * math.atanh(0.6)]], dtype=torch.float64))
    midpoint = weighted_centroid(torch.tensor([[[0.5, 0.5]]], dtype=torch.float64), torch.cat((origin, point), 1))
    assert torch.allclose(midpoint, torch.tensor([[[1/3, 0.0]]], dtype=torch.float64), atol=1e-12)
    assert torch.equal(displacement(point, point), origin)
    x, y = expmap0(tangent), expmap0(tangent.flip(1))
    x2, y2, xy = x.square().sum(-1, keepdim=True), y.square().sum(-1, keepdim=True), (x*y).sum(-1, keepdim=True)
    exact = (-(1-2*xy+x2)*y + (1-y2)*x) / (1-2*xy+x2*y2)
    assert torch.allclose(displacement(x, y), exact, atol=1e-12)
    assert torch.linalg.vector_norm(expmap0(tangent * 1000), dim=-1).max() <= RADIUS + 1e-12
    inputs = tuple((torch.randn(1, n, 3, dtype=torch.float64) * 0.03).requires_grad_() for n in (2, 3))
    assert torch.autograd.gradcheck(pairwise_distance, inputs)
    values = (torch.randn(1, 3, 3, dtype=torch.float64) * 0.03).requires_grad_()
    weights = torch.tensor([[[0.2, 0.3, 0.5]]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda z: weighted_centroid(weights, z), (values,))
    print("Poincare distances, Mobius displacement, geodesic midpoint and geometry gradients PASS")


def modules():
    for cls in (ChangeAwareCompressedQKV, HyperbolicCompressedQKV):
        module = cls().eval()
        module.gamma.data.fill_(0.5)
        module.collect_diag = True
        for size in (8, 16, 32, 64):
            a, b = torch.randn(1, 64, size, size), torch.randn(1, 64, size, size)
            with torch.no_grad():
                ya, yb = module(a, b)
                reverse_b, reverse_a = module(b, a)
                same_a, same_b = module(a, a)
            assert ya.shape == a.shape and yb.shape == b.shape
            assert torch.allclose(ya, reverse_a, atol=1e-6)
            assert torch.allclose(yb, reverse_b, atol=1e-6)
            assert torch.equal(same_a, a) and torch.equal(same_b, a)
            assert module.last_diag['queries'] == size * size
            assert module.last_diag['contexts'] == size * size // 8
            assert module.last_diag['finite']
        for scale in (0.0, 1.0, 100.0):
            a = (torch.randn(2, 64, 8, 8) * scale).requires_grad_()
            b = (torch.randn_like(a) * scale).requires_grad_()
            module.zero_grad()
            ya, yb = module(a, b)
            (ya.square().mean() + yb.square().mean()).backward()
            assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
            if scale == 1:
                assert all(p.grad.abs().sum() > 0 for p in module.parameters())
        if cls is HyperbolicCompressedQKV:
            q = torch.tensor([[[0.1, 0.0]]])
            probability = module.attention(q, torch.tensor([[[0.1, 0.0], [-0.2, 0.0]]]))
            assert probability[0, 0, 0] > probability[0, 0, 1]
            a = torch.randn(1, 64, 8, 8) * 0.1
            small, large = module.encode(a), module.encode(2*a)
            assert torch.linalg.vector_norm(large, dim=-1).mean() > torch.linalg.vector_norm(small, dim=-1).mean()
        print(cls.__name__, "four-scale N-to-N queries, compressed KV, exchange symmetry, identical-pair identity and finite task gradients PASS")


def change_values():
    ordinary = ChangeAwareCompressedQKV(channels=2, dim=2).eval()
    with torch.no_grad():
        ordinary.gamma.fill_(1)
        ordinary.q.weight.zero_()
        ordinary.k.weight.zero_()
        ordinary.v.weight.zero_()
        ordinary.v.weight[0, 0, 0, 0] = 1
        ordinary.out.weight.zero_()
        ordinary.out.weight[0, 2, 0, 0] = 1
    a = torch.zeros(1, 2, 8, 8)
    a[:, 0, :, ::2] = 1
    a[:, 0, :, 1::2] = -1
    b = torch.zeros_like(a)
    with torch.no_grad():
        ya, yb = ordinary(a, b)
    assert torch.allclose(ya[:, 0] - a[:, 0], torch.ones_like(a[:, 0]), atol=1e-6)
    assert torch.allclose(yb[:, 0], torch.ones_like(a[:, 0]), atol=1e-6)
    print("Opposite signed changes retain magnitude evidence in both temporal branches PASS")

    hyperbolic = HyperbolicCompressedQKV(channels=2, dim=2).eval()
    with torch.no_grad():
        hyperbolic.gamma.fill_(1)
        hyperbolic.embed.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1) * math.sqrt(2))
        hyperbolic.out.weight.zero_()
        hyperbolic.out.weight[:, 2:, 0, 0].copy_(torch.eye(2))
        hyperbolic.local.weight.zero_()
        hyperbolic.local.weight[:, 0, 1, 1] = 1
    a = torch.zeros(1, 2, 8, 8)
    a[:, 0, :, ::2] = 0.5
    a[:, 1, :, 1::2] = 0.5
    b = a.flip(-1)
    with torch.no_grad():
        ya, yb = hyperbolic(a, b)
        weights = prototype_weights(torch.full((1, 1, 8, 8), 0.5))
        ca, cb = (hyperbolic.compress(hyperbolic.encode(x), weights, 8, 8) for x in (a, b))
    assert torch.allclose(ca, cb, atol=1e-6)
    assert float((ya - a).abs().max()) > 1e-3
    assert float((yb - b).abs().max()) > 1e-3
    print("Spatially rearranged pairs retain local manifold displacement despite identical regional centroids PASS")


def relative_contexts():
    uniform = relative_prototype_weights(torch.ones(1, 1, 8, 8))
    assert torch.allclose(uniform, torch.full_like(uniform, 1 / 16))
    evidence = torch.zeros(1, 1, 8, 8)
    evidence[:, :, :, ::2] = 1
    assignments = relative_prototype_weights(evidence)
    separation = (assignments[:, :, 0] - assignments[:, :, 1]).abs().sum(-1).mean() / 2
    assert float(separation) > .7
    assert torch.allclose(assignments.sum(-1), torch.ones_like(assignments.sum(-1)))
    values = expmap0(torch.randn(1, 8, 4) * .1)
    probability = torch.full((1, 5, 8), 1 / 8)
    result = HyperbolicCompressedQKV.context_residual(probability, values)
    assert float(result.abs().max()) < 1e-6
    fixed = values[:, :1].expand_as(values)
    selective = torch.randn_like(probability).softmax(-1)
    assert float(HyperbolicCompressedQKV.context_residual(selective, fixed).abs().max()) < 1e-6
    inputs = (torch.randn(1, 8, 4, dtype=torch.float64) * .03).requires_grad_()
    assert torch.autograd.gradcheck(lambda z: HyperbolicCompressedQKV.context_residual(selective.double(), z), (inputs,))
    print("Opposite contrast assignments, uniform-cell pooling and common-manifold context rejection PASS")


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2333)
    x = torch.randn(1, 64, 8, 8)
    weights = prototype_weights(torch.zeros(1, 1, 8, 8))
    pooled = pool_tokens(x, weights).reshape(1, 4, 2, 64)
    average = torch.nn.functional.avg_pool2d(x, 4).flatten(2).transpose(1, 2)
    assert torch.allclose(pooled[:, :, 0], average, atol=1e-6)
    assert torch.equal(pooled[:, :, 0], pooled[:, :, 1])
    relative_contexts()
    geometry()
    modules()
    change_values()


if __name__ == '__main__':
    main()
