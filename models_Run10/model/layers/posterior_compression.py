"""One geometric correction of regional change/background membership."""
import torch

from model.layers.pyramid_context import feature_regions, pool_features, regions


def normalized_class_weights(membership):
    """Normalize B,R,2,P first-order class memberships over regional pixels."""
    mass = membership.sum(-1)
    denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
    return membership / denominator[..., None], mass


def class_assignments(probability, grid):
    """Confidence-weighted initial class anchors; final posterior pooling is linear."""
    positive = feature_regions(probability, grid).squeeze(-1)
    membership = torch.stack((positive, 1 - positive), 2)
    weights, _ = normalized_class_weights((membership + 1e-6).square())
    support = (weights * membership).sum(-1).flatten(1).clamp_min(1e-6)
    return weights, support


def posterior_assignments(guide_logits, query, initial_key, grid, temperature, distance):
    """Correct image-derived guide logits using the two local geometric centers."""
    grouped_query = regions(query, grid)
    batch, heads, count, _, dim = grouped_query.shape
    key = initial_key.reshape(batch, heads, count, 2, dim)
    distances = distance(grouped_query, key)
    delta = ((distances[..., 1] - distances[..., 0]) /
             (2 * temperature[None, :, None, None].square())).mean(1)
    logits = feature_regions(guide_logits, grid).squeeze(-1)
    prior = logits.sigmoid()
    posterior_logits = logits + delta
    posterior = posterior_logits.sigmoid()
    weights, mass = normalized_class_weights(torch.stack((posterior, 1 - posterior), 2))
    support = (weights * torch.stack((prior, 1 - prior), 2)).sum(-1).flatten(1).clamp_min(1e-6)
    square_sum = weights.square().sum(-1)
    effective = torch.where(square_sum > 0, square_sum.clamp_min(torch.finfo(square_sum.dtype).tiny).reciprocal(),
                            torch.zeros_like(square_sum))
    return weights, support, dict(prior=prior, posterior=posterior, posterior_logits=posterior_logits, delta=delta,
                                  class_mass=mass, effective_pixels=effective)


def compress_posterior(feature, guide_logits, query, initial_key, grid, temperature, distance):
    weights, support, diagnostics = posterior_assignments(
        guide_logits, query, initial_key, grid, temperature, distance)
    prototype = pool_features(feature, weights, grid)
    diagnostics["weights"] = weights
    return prototype, support, diagnostics


def regional_logits_to_image(logits, grid, height, width):
    """Invert regional row/column then within-region row/column ordering."""
    return logits.reshape(logits.shape[0], grid, grid, height // grid, width // grid).permute(
        0, 1, 3, 2, 4).reshape(logits.shape[0], 1, height, width)
