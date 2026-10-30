"""Occupancy radii, pooled-content keys, subtypes and independent adoption."""
import torch

from model.layers.hyperbolic_context import HyperbolicContext, subtype_assignments
from model.layers.pyramid_context import channel_norm, pool_features, project_tokens
from model.layers.lorentz_ops import pairwise_squared_distance


def compression_and_geometry():
    op = HyperbolicContext().double()
    p = torch.full((1, 1, 8, 8), .5, dtype=torch.double, requires_grad=True)
    appearance = torch.linspace(.05, .95, 64, dtype=torch.double).reshape_as(p).requires_grad_()
    x = torch.randn(1, 64, 8, 8, dtype=torch.double, requires_grad=True)
    weights, support = subtype_assignments(p, appearance, 1)
    assert weights.shape == (1, 1, 4, 64)
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]))
    assert torch.allclose(support, torch.full_like(support, .5))
    prototype = pool_features(x, weights, 1)
    assert not torch.allclose(prototype[:, 0], prototype[:, 1])
    loss = prototype[:, 0].square().mean()
    gradients = torch.autograd.grad(loss, (x, p, appearance))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    sparse = torch.full_like(p, .05)
    sparse[..., 0, 0] = .6
    sparse_weights, _ = subtype_assignments(sparse, torch.full_like(p, .5), 1)
    assert .69 < sparse_weights[0, 0, 0, 0] < .70
    q = torch.randn(1, op.heads, 12, op.head_dim, dtype=torch.double)
    occupancy = torch.linspace(0, 1, 12, dtype=torch.double).reshape(1, 1, 12, 1)
    points = op.geometric_point(q, occupancy)
    radius = torch.asinh(points[..., 1:].norm(dim=-1))
    assert torch.allclose(radius, torch.asinh(1 + occupancy[..., 0]).expand_as(radius), atol=1e-12)
    assert torch.allclose(points, op.geometric_point(3 * q, occupancy), atol=1e-12)
    assert torch.allclose(points[..., 0].square() - points[..., 1:].square().sum(-1), torch.ones_like(radius), atol=2e-14)
    # K and V depend on the same P, without a separate region-scatter statistic.
    mean = torch.randn(1, 1, 64, dtype=torch.double)
    delta = torch.randn_like(mean)
    two = torch.cat((mean + delta, mean - delta), 1).mean(1, keepdim=True)
    large = torch.cat((mean + 5 * delta, mean - 5 * delta), 1).mean(1, keepdim=True)
    assert torch.allclose(op.project_keys(two)[0], op.project_keys(large)[0], atol=1e-12)
    assert torch.allclose(project_tokens(op.v, two, op.heads), project_tokens(op.v, large, op.heads), atol=1e-12)
    print('Occupancy radius, scatter invariance, sparse targets, two subtypes and gradients PASS')


def independent_adoption():
    op = HyperbolicContext().double()
    feature = torch.randn(1, 64, 2, 2, dtype=torch.double)
    p = op.guide(feature).sigmoid()
    query = op.project_queries(feature, p).flatten(2, 3)
    prototypes = torch.randn(1, 448, 64, dtype=torch.double)
    key, p_key = op.project_keys(prototypes)
    value = project_tokens(op.v, prototypes, op.heads)
    support = torch.ones(1, 448, dtype=torch.double)
    attention = op.attention(query, key, support, 0)
    update, gate, *_ = op.readout(feature, attention, value, p, p_key)
    with torch.no_grad():
        op.scale_bias.add_(7.)
    shifted = op.attention(query, key, support, 0)
    shifted_update, shifted_gate, *_ = op.readout(feature, shifted, value, p, p_key)
    assert torch.allclose(shifted, attention, atol=1e-14)
    assert torch.allclose(shifted_gate, gate, atol=1e-14)
    assert torch.allclose(update, shifted_update, atol=1e-14)
    # A single available fine-scale prototype is not divided by 256 at readout.
    rare = torch.zeros_like(attention)
    rare[..., 0] = 1
    with torch.no_grad():
        op.gate.weight.zero_()
        op.gate.bias.zero_()
    _, rare_gate, *_ = op.readout(feature, rare, value, p, p_key)
    assert torch.equal(rare_gate, torch.full_like(rare_gate, .5))
    radial_flat = HyperbolicContext.geometric_point(query[..., 1:], torch.full_like(query[..., :1], .5))
    assert (pairwise_squared_distance(query, key) - pairwise_squared_distance(radial_flat, key)).abs().max() > 1e-4
    print('Common logit shift invariance, rare-match adoption and radial sensitivity PASS')


def module():
    op = HyperbolicContext()
    sizes = (32, 16, 8, 4)
    tar = [torch.randn(1, 64, n, n, requires_grad=True) for n in sizes]
    output, _ = op(tar)
    assert all(torch.equal(a, b) for a, b in zip(tar, output))
    with torch.no_grad():
        op.gamma_context.fill_(.1)
        op.gamma_detail.fill_(.1)
    constant = torch.randn(1, 64, 1, 1)
    fields = [constant.expand(1, 64, n, n).clone() for n in sizes]
    output, _ = op(fields)
    assert all(torch.allclose(a, b, atol=3e-6) for a, b in zip(fields, output))
    zero = [torch.zeros_like(x) for x in tar]
    output, _ = op(zero)
    assert all(torch.equal(a, b) for a, b in zip(zero, output))
    op.collect_diag = True
    output, _ = op(tar)
    grads = torch.autograd.grad(output[0].square().mean(), tar, retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    sum(o.square().mean() for o in output).backward()
    for name, parameter in op.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for index, stage in enumerate(op.last_diag.values()):
        assert stage['contexts'] == 448 and stage['finite']
        assert 0 < stage['context_mass_mean'] < 1
        if sizes[index] > op.grid_sizes[index]:
            assert stage['subtype_separation_rms'] > 0
        assert stage['adoption_source'] == 'learned_value_correction_gate'
    print('Identity, constant/zero fields, all parameter and cross-scale gradients PASS')


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    compression_and_geometry()
    independent_adoption()
    module()
