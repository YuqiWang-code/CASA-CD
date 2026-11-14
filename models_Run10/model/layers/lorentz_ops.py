"""Curvature -1 geometry using unconstrained spatial Lorentz coordinates."""
import torch


def lift(space):
    """The intrinsic radius is asinh(||space||); there is no radial clipping."""
    return torch.cat(((1 + space.square().sum(-1, keepdim=True)).sqrt(), space), dim=-1)


def normalize_sum(point):
    """Lorentz centroid of a positive weighted mean of unit hyperboloid points."""
    norm2 = point[..., :1].square() - point[..., 1:].square().sum(-1, keepdim=True)
    # Convex weights imply norm2 >= 1. Rebuild time to preserve the unit manifold.
    return lift(point[..., 1:] / norm2.clamp_min(1).sqrt())


def acosh1p_squared(z):
    """Squared geodesic distance with its finite analytic derivative at z=0."""
    regular_z = z.clamp_min(1e-3)
    regular = torch.log1p(regular_z + (regular_z * (regular_z + 2)).sqrt()).square()
    series = 2 * z - z.square() / 3 + 4 * z.pow(3) / 45 - z.pow(4) / 35
    return torch.where(z <= 1e-3, series, regular)


def pairwise_squared_distance(query, key):
    space_q, space_k = query[..., 1:], key[..., 1:]
    squared_chord = (space_q.square().sum(-1, keepdim=True)
                     + space_k.square().sum(-1).unsqueeze(-2)
                     - 2 * (space_q @ space_k.transpose(-1, -2)))
    time_difference = query[..., :1] - key[..., :1].transpose(-1, -2)
    z = (0.5 * (squared_chord - time_difference.square())).clamp_min(0)
    return acosh1p_squared(z)
