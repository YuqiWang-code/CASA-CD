"""Lorentz geometry checked against Poincare and Klein coordinate identities."""
import torch

from model.layers.lorentz_ops import (
    acosh1p_squared, lift, normalize_sum, pairwise_squared_distance,
)


def poincare_distance_reference(space_q, space_k):
    """FP64 reference using direct chords in the Poincare ball."""
    q, k = space_q.double(), space_k.double()
    time_q = torch.sqrt(1 + (q * q).sum(-1, keepdim=True))
    time_k = torch.sqrt(1 + (k * k).sum(-1, keepdim=True))
    ball_q, ball_k = q / (time_q + 1), k / (time_k + 1)
    chord = torch.cdist(ball_q, ball_k, compute_mode="donot_use_mm_for_euclid_dist")
    denominator = torch.sqrt(
        (1 - ball_q.square().sum(-1)).unsqueeze(-1)
        * (1 - ball_k.square().sum(-1)).unsqueeze(-2))
    return (2 * torch.asinh(chord / denominator)).square()


def klein_centroid_reference(points, weights):
    """Einstein barycenter in Klein coordinates, converted through the ball."""
    time, space = points[..., :1], points[..., 1:]
    klein = space / time
    weighted_mass = weights.unsqueeze(-1) * time.unsqueeze(-3)
    center = (weighted_mass * klein.unsqueeze(-3)).sum(-2) / weighted_mass.sum(-2)
    ball = center / (1 + torch.sqrt(1 - center.square().sum(-1, keepdim=True)))
    denominator = 1 - ball.square().sum(-1, keepdim=True)
    return torch.cat(((1 + ball.square().sum(-1, keepdim=True)) / denominator,
                      2 * ball / denominator), dim=-1)


def lift_radius_and_gradient():
    directions = torch.randn(7, 16, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    lengths = torch.tensor([0., .01, .25, 1., 3., 10., 30.],
                           dtype=torch.float64, requires_grad=True)
    point = lift(lengths[:, None] * directions)
    unit = point[:, 0].square() - point[:, 1:].square().sum(-1)
    torch.testing.assert_close(unit, torch.ones_like(unit), atol=5e-13, rtol=0)
    radius = torch.asinh(point[:, 1:].norm(dim=-1))
    torch.testing.assert_close(radius, torch.asinh(lengths), atol=2e-15, rtol=2e-15)
    gradient = torch.autograd.grad(radius[1:].sum(), lengths)[0][1:]
    expected = 1 / torch.sqrt(1 + lengths.detach()[1:].square())
    torch.testing.assert_close(gradient, expected, atol=1e-14, rtol=1e-13)
    assert (gradient > 0).all(), "large-radius spatial inputs lost radial gradients"
    assert radius.max() - radius.min() > 4
    print("Lorentz unit norm, variable radius and non-clipped radial gradients PASS")


def centroid_identities():
    space = torch.randn(2, 4, 7, 16, dtype=torch.float64) * .7
    points = lift(space)
    weights = torch.rand(2, 4, 3, 7, dtype=torch.float64).softmax(-1)
    actual = normalize_sum(weights @ points)
    expected = klein_centroid_reference(points, weights)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)

    constant = points[..., :1, :].expand_as(points)
    actual = normalize_sum(weights @ constant)
    expected = points[..., :1, :].expand(-1, -1, 3, -1)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)

    selected = torch.nn.functional.one_hot(torch.tensor([1, 4, 6]), 7).double()
    actual = normalize_sum(selected @ points)
    torch.testing.assert_close(actual, points[..., [1, 4, 6], :], atol=2e-13, rtol=2e-13)

    opposite = torch.stack((space[..., 0, :], -space[..., 0, :]), dim=-2)
    center = normalize_sum(lift(opposite).mean(-2))
    origin = torch.zeros_like(center)
    origin[..., 0] = 1
    torch.testing.assert_close(center, origin, atol=1e-14, rtol=0)
    print("Lorentz centroids agree with Klein barycenters, constants and one-hot weights PASS")


def distance_references():
    q = torch.randn(2, 4, 11, 16, dtype=torch.float64) * 1.2
    k = torch.randn(2, 4, 7, 16, dtype=torch.float64) * .8
    actual = pairwise_squared_distance(lift(q), lift(k))
    expected = poincare_distance_reference(q, k)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

    radial_q = torch.tensor([0., .1, .8, 2., 4.], dtype=torch.float64)
    radial_k = torch.tensor([.03, .4, 1.3, 3.], dtype=torch.float64)
    a = torch.zeros(5, 16, dtype=torch.float64)
    b = torch.zeros(4, 16, dtype=torch.float64)
    a[:, 0], b[:, 0] = radial_q.sinh(), radial_k.sinh()
    expected = (radial_q[:, None] - radial_k[None, :]).square()
    actual = pairwise_squared_distance(lift(a), lift(b))
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    print("FP64 squared distances agree with Poincare chords and radial geodesics PASS")


def float32_distance_range():
    lengths = torch.tensor([0., 1e-4, .05, .5, 1., 3., 10., 30.])
    q = torch.randn(2, 4, len(lengths), 16)
    k = torch.randn_like(q)
    q = q / q.norm(dim=-1, keepdim=True) * lengths[None, None, :, None]
    k = k / k.norm(dim=-1, keepdim=True) * lengths[None, None, :, None]
    cases = [(q, k), (q, q), (q, q + torch.randn_like(q) * 1e-3)]
    maximum_error = 0.
    for left, right in cases:
        actual = pairwise_squared_distance(lift(left), lift(right))
        expected = poincare_distance_reference(left, right)
        maximum_error = max(maximum_error, float((actual.double() - expected).abs().max()))
        assert torch.isfinite(actual).all() and (actual >= 0).all()
        torch.testing.assert_close(actual.double(), expected, atol=2e-3, rtol=3e-5)
    print(f"FP32 spatial norm 0..30 squared-distance max absolute error={maximum_error:.3e} PASS")


def small_distance_limit():
    z = torch.cat((torch.tensor([0.], dtype=torch.float64),
                   torch.logspace(-14, -2, 25, dtype=torch.float64),
                   torch.tensor([1e-3 - 1e-9, 1e-3, 1e-3 + 1e-9], dtype=torch.float64)))
    z.requires_grad_()
    actual = acosh1p_squared(z)
    expected = 4 * torch.asinh(torch.sqrt(z.detach() / 2)).square()
    torch.testing.assert_close(actual, expected, atol=3e-15, rtol=2e-13)
    derivative = torch.autograd.grad(actual.sum(), z)[0]
    assert derivative[0] == 2 and torch.isfinite(derivative).all()
    positive_z = z.detach()[1:]
    expected_derivative = (4 * torch.asinh(torch.sqrt(positive_z / 2))
                           / torch.sqrt(positive_z * (positive_z + 2)))
    torch.testing.assert_close(derivative[1:], expected_derivative, atol=2e-12, rtol=2e-12)
    print("Squared acosh analytic value, derivative and branch-boundary limits PASS")


def distance_and_centroid_gradients():
    q = (torch.randn(1, 2, 3, dtype=torch.float64) * .6).requires_grad_()
    k = (torch.randn(1, 3, 3, dtype=torch.float64) * .6).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: pairwise_squared_distance(lift(a), lift(b)),
                                    (q, k), eps=1e-6, atol=2e-6, rtol=2e-4)
    logits = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda points, scores: normalize_sum(scores.softmax(-1) @ lift(points)),
                                    (k, logits), eps=1e-6, atol=2e-6, rtol=2e-4)

    for radius in (0., .5, 3., 30.):
        source = torch.randn(1, 2, 5, 16)
        source = source / source.norm(dim=-1, keepdim=True) * radius
        for perturbation in (0., 1e-6, .01):
            a = source.clone().requires_grad_()
            b = (source + perturbation * torch.randn_like(source)).requires_grad_()
            distance = pairwise_squared_distance(lift(a), lift(b))
            distance.sum().backward()
            assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
            if radius == 0 and perturbation == 0:
                assert distance.count_nonzero() == 0
                assert a.grad.count_nonzero() == 0 and b.grad.count_nonzero() == 0
    print("Noncoincident distance/centroid gradcheck and FP32 coincident gradients PASS")


def euclidean_origin_limit():
    q = torch.randn(2, 3, 5, 16, dtype=torch.float64)
    k = torch.randn(2, 3, 7, 16, dtype=torch.float64)
    euclidean = torch.cdist(q, k, compute_mode="donot_use_mm_for_euclid_dist").square()
    scale = 1e-4
    intrinsic = pairwise_squared_distance(lift(scale * q), lift(scale * k)) / scale ** 2
    torch.testing.assert_close(intrinsic, euclidean, atol=2e-5, rtol=2e-6)
    print("Lorentz spatial lift has the Euclidean metric at the origin PASS")


def main():
    torch.set_num_threads(4)
    torch.manual_seed(4701)
    lift_radius_and_gradient()
    centroid_identities()
    distance_references()
    float32_distance_range()
    small_distance_limit()
    distance_and_centroid_gradients()
    euclidean_origin_limit()


if __name__ == "__main__":
    main()
