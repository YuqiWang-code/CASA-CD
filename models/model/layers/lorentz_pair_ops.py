"""Curvature -1 paired geometry in Lorentz coordinates (time coordinate first)."""
import torch


class _DirectPairwiseDistance(torch.autograd.Function):
    """Direct Euclidean norms with a matrix-based gradient requiring no N×M×D buffer."""

    @staticmethod
    def forward(ctx, x, y):
        squared = (x[..., 0, None] - y[..., None, :, 0]).square()
        for dim in range(1, x.shape[-1]):
            squared.add_((x[..., dim, None] - y[..., None, :, dim]).square())
        distance = squared.sqrt_()
        ctx.save_for_backward(x, y, distance)
        return distance

    @staticmethod
    def backward(ctx, grad_output):
        x, y, distance = ctx.saved_tensors
        weights = grad_output / distance.clamp_min(torch.finfo(distance.dtype).tiny)
        weights = torch.where(distance > 0, weights, 0.)
        grad_x = x * weights.sum(-1, keepdim=True) - weights @ y
        grad_y = y * weights.sum(-2).unsqueeze(-1) - weights.transpose(-1, -2) @ x
        return grad_x.sum_to_size(x.shape), grad_y.sum_to_size(y.shape)


def euclidean_pairwise_distance(x, y):
    """Exact pairwise Euclidean norms; use the zero subgradient at coincident points."""
    return _DirectPairwiseDistance.apply(x, y)


def lorentz_inner(x, y, keepdim=False):
    """Lorentz inner product with arbitrary broadcastable leading dimensions."""
    value = (x[..., 1:] * y[..., 1:]).sum(-1) - x[..., 0] * y[..., 0]
    return value.unsqueeze(-1) if keepdim else value


def lorentz_normalize(x):
    """Normalize a positive weighted sum of upper-sheet hyperboloid points."""
    norm = (-lorentz_inner(x, x, keepdim=True)).clamp_min(1e-8).sqrt()
    return x / norm


def lorentz_midpoint(x, y):
    """Exact equal-weight two-point Frechet mean, without an iterative solver."""
    delta = y - x
    # -<x+y,x+y> = 4 + <y-x,y-x> for unit hyperboloid points.
    denominator = (4 + lorentz_inner(delta, delta, keepdim=True).clamp_min(0)).sqrt()
    return (x + y) / denominator


def lorentz_logmap(base, target):
    """Log_base(target), with the analytic coincident-point coefficient limit."""
    delta = target - base
    # z = -<base,target>-1; this form avoids subtracting nearly equal products.
    z = (0.5 * lorentz_inner(delta, delta, keepdim=True)).clamp_min(0)
    regular_z = z.clamp_min(1e-3)
    regular = (2 * torch.asinh((regular_z / 2).sqrt()) /
               (regular_z * (regular_z + 2)).sqrt())
    series = 1 - z / 3 + 2 * z.square() / 15 - 2 * z.pow(3) / 35
    coefficient = torch.where(z < 1e-3, series, regular)
    return coefficient * (delta - z * base)


def parallel_transport_to_origin(base, tangent):
    """Transport an ambient Lorentz tangent to the origin; return spatial coordinates."""
    return tangent[..., 1:] - (tangent[..., :1] / (base[..., :1] + 1)) * base[..., 1:]


def pair_midpoint_difference(x, y):
    """Return midpoint m and spatial PT_{m->o}(Log_m(y)); swap negates the latter."""
    midpoint = lorentz_midpoint(x, y)
    half_difference = (y - x) * 0.5
    squared = lorentz_inner(half_difference, half_difference, keepdim=True).clamp_min(0)
    regular_squared = squared.clamp_min(1e-3)
    root = regular_squared.sqrt()
    regular = torch.asinh(root) / root
    series = 1 - squared / 6 + 3 * squared.square() / 40 - 5 * squared.pow(3) / 112
    coefficient = torch.where(squared < 1e-3, series, regular)
    tangent = coefficient * half_difference
    return midpoint, parallel_transport_to_origin(midpoint, tangent)


def pairwise_lorentz_distance(x, y):
    """All-pairs geodesic distances, with finite gradients for coincident points.

    The equivalent Poincare asinh identity avoids cancellation in acosh(-<x,y>)
    for nearby large Lorentz coordinates. Direct differences keep exact zero distances
    without materializing a query-by-key-by-channel tensor.
    """
    x_time, y_time = x[..., :1] + 1, y[..., :1] + 1
    x_ball, y_ball = x[..., 1:] / x_time, y[..., 1:] / y_time
    chord = euclidean_pairwise_distance(x_ball, y_ball)
    factor = (x_time * y_time.transpose(-1, -2)).sqrt() * 0.5
    return 2 * torch.asinh(chord * factor)
