"""Centered context identity and differentiable content routing."""
import torch

from model.layers.hyperbolic_context import HyperbolicContext, MatchedEuclideanContext
from model.layers.pyramid_context import project_tokens


def check(kind):
    op = kind().double()
    feature = torch.randn(1, 64, 4, 4, dtype=torch.float64)
    prototype = torch.randn(1, 224, 64, dtype=torch.float64)
    key, occupancy = op.project_keys(prototype)
    query = op.project_queries(feature).flatten(2, 3)
    values = project_tokens(op.v, prototype, op.heads)
    prior = op.guide(feature).sigmoid()
    probability = torch.randn(1, 1, 1, 224, dtype=torch.float64).softmax(-1).expand(1, 1, 16, 224)
    update, _, _, _, discrepancy, _ = op.readout(feature, probability, values, prior, occupancy)
    torch.testing.assert_close(update, torch.zeros_like(update), rtol=0, atol=1e-12)
    torch.testing.assert_close(discrepancy, torch.zeros_like(discrepancy), rtol=0, atol=1e-12)
    value_gradient = torch.autograd.grad((update * torch.randn_like(update)).sum(), values)[0]
    torch.testing.assert_close(value_gradient, torch.zeros_like(value_gradient), rtol=0, atol=1e-12)

    values = project_tokens(op.v, prototype, op.heads)
    probability = op.attention(query, key, torch.ones(1, 224, dtype=torch.float64), 0)
    update, *_ = op.readout(feature, probability, values, prior, occupancy)
    (update * torch.randn_like(update)).sum().backward()
    for name in ("qk", "v", "out", "gate"):
        gradient = getattr(op, name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
    assert torch.count_nonzero(op.gamma_context) == 0
    print(op.geometry + ": constant attention zero update/zero V gradient; varying attention QK/V/out/gate gradients PASS")


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    for kind in (HyperbolicContext, MatchedEuclideanContext):
        check(kind)
