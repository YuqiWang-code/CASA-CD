"""Direct projected geometry, class evidence and inference checks."""
import math

import torch

from model.casa_net import CASANet
from model.layers.hyperbolic_context import HyperbolicContext
from model.layers.pyramid_context import project_tokens
from train_CASA import guide_loss, training_losses, loss_function


def manifold_and_radius():
    op = HyperbolicContext().double()
    projected = torch.randn(2, 1, 7, 32, dtype=torch.double)
    point = op.geometric_point(projected)
    space = projected / math.sqrt(32)
    assert torch.equal(point[..., 1:], space)
    assert torch.allclose(point[..., 0].square() - point[..., 1:].square().sum(-1),
                          torch.ones_like(point[..., 0]), atol=1e-12)
    radius = torch.asinh(point[..., 1:].norm(dim=-1))
    assert torch.allclose(radius, torch.asinh(projected.norm(dim=-1) / math.sqrt(32)), atol=1e-12)
    larger = torch.asinh(op.geometric_point(4 * projected)[..., 1:].norm(dim=-1))
    assert (larger > radius).all()
    zeros = torch.zeros_like(projected)
    origin = op.geometric_point(zeros)
    assert torch.equal(origin[..., 0], torch.ones_like(origin[..., 0]))
    assert torch.equal(origin[..., 1:], zeros)
    for initial in (projected, zeros):
        query = initial.clone().requires_grad_()
        key = initial.clone().requires_grad_()
        diagonal = op.squared_distance(op.geometric_point(query), op.geometric_point(key)).diagonal(dim1=-2, dim2=-1)
        assert diagonal.abs().max() < 1e-12
        gradients = torch.autograd.grad(diagonal.sum(), (query, key))
        assert all(torch.isfinite(g).all() and g.abs().max() < 1e-12 for g in gradients)
    print("Lorentz constraint, projected-radius sensitivity and finite zero-distance gradients PASS")


def content_and_guide_independence():
    op = HyperbolicContext().double()
    mean, delta = torch.randn(1, 3, 64, dtype=torch.double), torch.randn(1, 3, 64, dtype=torch.double)
    pooled_a = torch.stack((mean + delta, mean - delta), -2).mean(-2)
    pooled_b = torch.stack((mean + 5 * delta, mean - 5 * delta), -2).mean(-2)
    key_a, occupancy_a = op.project_keys(pooled_a)
    key_b, _ = op.project_keys(pooled_b)
    assert torch.allclose(key_a, key_b, atol=1e-12)
    assert torch.allclose(project_tokens(op.v, pooled_a, op.heads), project_tokens(op.v, pooled_b, op.heads), atol=1e-12)
    field = torch.randn(1, 64, 4, 4, dtype=torch.double)
    query_a = op.project_queries(field)
    with torch.no_grad():
        op.guide.weight.zero_()
        op.guide.bias.fill_(4.)
    query_b = op.project_queries(field)
    key_c, occupancy_c = op.project_keys(pooled_a)
    assert torch.equal(query_a, query_b) and torch.equal(key_a, key_c)
    assert not torch.equal(occupancy_a, occupancy_c)
    print("Shared pooled K/V content and Q/K coordinates independent of guide parameters PASS")


def evidence_and_fields():
    op = HyperbolicContext()
    assert op.tokens_per_scale == (128, 32, 32, 32)
    assert sum(p.numel() for p in op.parameters()) == 11477
    sizes = (16, 8, 4, 4)
    tar = [torch.randn(1, 64, n, n, requires_grad=True) for n in sizes]
    output, auxiliary = op(tar, return_aux=True)
    assert all(torch.equal(a, b) for a, b in zip(output, tar))
    target = torch.zeros(1, 1, 64, 64)
    target[..., 16:48, 16:48] = 1
    geometry_loss = guide_loss(auxiliary["geometric_logits"], target)
    gradients = torch.autograd.grad(geometry_loss, (op.qk.weight, *tar), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    logit = torch.randn(1, 1, 6, 224)
    evidence = op.geometric_change_logit(logit, 2, 3)
    expected = logit[..., ::2].logsumexp(-1) - logit[..., 1::2].logsumexp(-1)
    assert torch.equal(evidence, expected.reshape(1, 1, 2, 3))
    predictions = tuple(torch.randn_like(target) for _ in range(4))
    loss, _, auxiliary_loss = training_losses(predictions, auxiliary, target)
    expected_auxiliary = .5 * (guide_loss(auxiliary["guide_logits"], target) + geometry_loss)
    assert torch.equal(auxiliary_loss, expected_auxiliary)
    assert torch.equal(loss, loss_function(predictions, target) + .2 * expected_auxiliary)
    with torch.no_grad():
        op.gamma_context.fill_(.1)
        op.gamma_detail.fill_(.1)
        constant = torch.randn(1, 64, 1, 1)
        fields = [constant.expand(1, 64, n, n).clone() for n in sizes]
        enhanced, _ = op(fields)
        assert all(torch.allclose(a, b, atol=3e-6) for a, b in zip(fields, enhanced))
        zeros = [torch.zeros_like(x) for x in fields]
        enhanced, _ = op(zeros)
        assert all(torch.equal(a, b) for a, b in zip(zeros, enhanced))
    print("224 prototypes, geometric-evidence QK/TAR gradients, 0.1/0.1 loss, constant/zero fields PASS")


def inference_path():
    model = CASANet(module="hyperbolic").eval()
    def prohibited(*args):
        raise AssertionError("predict must not compute geometric auxiliary logits")
    model.context.geometric_change_logit = prohibited
    with torch.no_grad():
        result = model.predict(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
        _, auxiliary = model(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128), return_aux=True)
        assert "geometric_logits" not in auxiliary
    assert result.shape == (1, 1, 128, 128) and torch.isfinite(result).all()
    print("Prediction and evaluation exclude geometric auxiliary logsumexp PASS")


if __name__ == "__main__":
    torch.manual_seed(2333)
    torch.set_num_threads(4)
    manifold_and_radius()
    content_and_guide_independence()
    evidence_and_fields()
    inference_path()
