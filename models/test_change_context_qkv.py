"""Geometry, aligned change evidence and dense-query interface tests."""
import math

import torch

from model.layers.change_context_qkv import ChangeAwareCompressedQKV
from model.layers.context_compression import pool_tokens, prototype_weights
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
                ya, yb, logits = module(a, b)
                reverse_b, reverse_a, reverse_logits = module(b, a)
                same_a, same_b, _ = module(a, a)
            assert ya.shape == a.shape and yb.shape == b.shape
            assert torch.allclose(ya, reverse_a, atol=1e-6)
            assert torch.allclose(yb, reverse_b, atol=1e-6)
            assert logits.shape == (1, 1, size, size)
            assert torch.allclose(logits, reverse_logits, atol=1e-6)
            assert torch.equal(same_a, a) and torch.equal(same_b, a)
            assert module.last_diag['queries'] == size * size
            assert module.last_diag['contexts'] == size * size // 8
            assert module.last_diag['finite']
        for scale in (0.0, 1.0, 100.0):
            a = (torch.randn(2, 64, 8, 8) * scale).requires_grad_()
            b = (torch.randn_like(a) * scale).requires_grad_()
            module.zero_grad()
            ya, yb, _ = module(a, b)
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


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2333)
    x = torch.randn(1, 64, 8, 8)
    weights = prototype_weights(torch.zeros(1, 1, 8, 8))
    pooled = pool_tokens(x, weights).reshape(1, 4, 2, 64)
    average = torch.nn.functional.avg_pool2d(x, 4).flatten(2).transpose(1, 2)
    assert torch.allclose(pooled[:, :, 0], average, atol=1e-6)
    assert torch.equal(pooled[:, :, 0], pooled[:, :, 1])
    geometry()
    modules()


if __name__ == '__main__':
    main()
