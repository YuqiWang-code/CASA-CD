"""Paired Lorentz geometry: identities, reconstruction and FP32 coincidence gradients."""
import torch

from model.layers.lorentz_pair_ops import (
    euclidean_pairwise_distance, lorentz_inner, lorentz_logmap, lorentz_midpoint, lorentz_normalize,
    pair_midpoint_difference, pairwise_lorentz_distance, parallel_transport_to_origin,
)
from model.layers.poincare_ops import aligned_distance, expmap0, to_lorentz


def tangent_exp(base, tangent):
    squared = lorentz_inner(tangent, tangent, keepdim=True).clamp_min(0)
    norm = squared.clamp_min(1e-12).sqrt()
    return norm.cosh() * base + (norm.sinh() / norm) * tangent


def geometric_identities():
    torch.manual_seed(31)
    a = torch.randn(2, 4, 13, 8, dtype=torch.float64)
    b = torch.randn_like(a)
    a = a / a.norm(dim=-1, keepdim=True) * torch.linspace(0.01, 2.3, 13).double()[None, None, :, None]
    b = b / b.norm(dim=-1, keepdim=True) * 1.2
    x, y = to_lorentz(expmap0(a)), to_lorentz(expmap0(b))
    midpoint, difference = pair_midpoint_difference(x, y)
    reversed_midpoint, reversed_difference = pair_midpoint_difference(y, x)
    direct_log = lorentz_logmap(midpoint, y)
    direct_difference = parallel_transport_to_origin(midpoint, direct_log)
    assert torch.allclose(lorentz_inner(midpoint, midpoint), -torch.ones_like(midpoint[..., 0]), atol=2e-10)
    assert torch.allclose(midpoint, lorentz_normalize(x + y), atol=2e-10, rtol=2e-10)
    assert torch.equal(midpoint, reversed_midpoint)
    assert torch.equal(difference, -reversed_difference)
    assert torch.allclose(difference, direct_difference, atol=2e-10, rtol=2e-10)
    assert torch.allclose(lorentz_inner(midpoint, direct_log), torch.zeros_like(midpoint[..., 0]), atol=2e-10)
    assert torch.allclose(difference.norm(dim=-1), aligned_distance(expmap0(a), expmap0(b)) * 0.5,
                          atol=2e-10, rtol=2e-10)
    assert torch.allclose(tangent_exp(midpoint, direct_log), y, atol=2e-9, rtol=2e-10)
    assert torch.allclose(tangent_exp(midpoint, -direct_log), x, atol=2e-9, rtol=2e-10)
    expected = aligned_distance(expmap0(a).unsqueeze(-2), expmap0(b).unsqueeze(-3))
    assert torch.allclose(pairwise_lorentz_distance(x, y), expected, atol=2e-10, rtol=2e-10)
    weights = torch.rand(2, 4, 7, 13, dtype=torch.float64)
    weights = weights / weights.sum(-1, keepdim=True)
    centroid = lorentz_normalize(weights @ midpoint)
    assert torch.allclose(lorentz_inner(centroid, centroid), -torch.ones_like(centroid[..., 0]), atol=2e-10)
    print("midpoint, Log/PT, half-distance, reconstruction and centroid identities PASS")


def finite_float32_gradients():
    torch.manual_seed(39)
    for radius in (0.0, 0.5, 2.3):
        for perturbation in (0.0, 1e-6, 0.1):
            a = torch.randn(1, 2, 11, 8)
            a = (a / a.norm(dim=-1, keepdim=True) * radius).requires_grad_()
            b = (a.detach() + torch.randn_like(a) * perturbation).requires_grad_()
            x, y = to_lorentz(expmap0(a)), to_lorentz(expmap0(b))
            midpoint, difference = pair_midpoint_difference(x, y)
            general_log = lorentz_logmap(x, y)
            distance = pairwise_lorentz_distance(x, y)
            assert all(torch.isfinite(t).all() for t in (midpoint, difference, general_log, distance))
            if perturbation == 0:
                assert torch.equal(midpoint, x)
                assert torch.count_nonzero(difference) == 0
                assert torch.count_nonzero(general_log) == 0
                assert torch.count_nonzero(distance.diagonal(dim1=-2, dim2=-1)) == 0
            reference = aligned_distance(expmap0(a), expmap0(b)) * 0.5
            assert torch.allclose(difference.norm(dim=-1), reference, atol=1e-4, rtol=2e-3)
            loss = difference.square().mean() + general_log.square().mean() + midpoint.mean() + distance.mean()
            loss.backward()
            assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    print("FP32 origin, identical pairs, near-identical pairs and clipping-radius gradients PASS")


def pair_derivative_check():
    torch.manual_seed(43)
    a = (torch.randn(1, 3, 3, dtype=torch.float64) * 0.2).requires_grad_()
    b = (torch.randn(1, 3, 3, dtype=torch.float64) * 0.2).requires_grad_()

    def operation(left, right):
        x, y = to_lorentz(expmap0(left)), to_lorentz(expmap0(right))
        return pair_midpoint_difference(x, y)

    assert torch.autograd.gradcheck(operation, (a, b), eps=1e-6, atol=2e-6, rtol=2e-4)
    print("paired geometry derivatives PASS")


def distance_derivative_check():
    torch.manual_seed(47)
    x = torch.randn(2, 3, 5, 8, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 3, 7, 8, dtype=torch.float64, requires_grad=True)
    actual = euclidean_pairwise_distance(x, y)
    expected = torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-12)
    direction = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * direction).sum(), (x, y))
    expected_grad = torch.autograd.grad((expected * direction).sum(), (x, y))
    assert all(torch.allclose(a, e, atol=2e-12, rtol=2e-12)
               for a, e in zip(actual_grad, expected_grad))
    assert torch.autograd.gradcheck(euclidean_pairwise_distance, (x, y),
                                    eps=1e-6, atol=2e-6, rtol=2e-4)
    same = torch.randn(2, 3, 1, 8).expand(-1, -1, 5, -1).clone().requires_grad_()
    matching = same.detach().clone().requires_grad_()
    coincident = euclidean_pairwise_distance(same, matching)
    coincident.sum().backward()
    assert torch.count_nonzero(coincident) == 0
    assert torch.count_nonzero(same.grad) == 0 and torch.count_nonzero(matching.grad) == 0
    near = (same.detach() + 1e-6 * torch.randn_like(same)).requires_grad_()
    euclidean_pairwise_distance(same, near).sum().backward()
    assert torch.isfinite(same.grad).all() and torch.isfinite(near.grad).all()
    print("direct distance forward/backward equivalence, gradcheck and coincidence gradients PASS")


if __name__ == "__main__":
    geometric_identities()
    finite_float32_gradients()
    pair_derivative_check()
    distance_derivative_check()
