"""Posterior supervision alignment, fixed guide budget and zero-gamma learning."""
import torch
from torch.nn import functional as F

from model.casa_net import CASANet
from model.layers.posterior_compression import regional_logits_to_image
from model.layers.pyramid_context import feature_regions
from train_CASA import GUIDE_OBJECTIVE, guide_loss, training_losses


def alignment_and_budget():
    logits = torch.arange(96, dtype=torch.float64).reshape(2, 1, 6, 8).div(50).sub(1).requires_grad_()
    target = (torch.arange(384).reshape(2, 1, 12, 16) % 3 == 0).double()
    regional = feature_regions(logits, 2).squeeze(-1)
    restored = regional_logits_to_image(regional, 2, 6, 8)
    torch.testing.assert_close(restored, logits, rtol=0, atol=0)
    area = F.adaptive_avg_pool2d(target, logits.shape[-2:])
    gradient_restored = torch.autograd.grad(F.binary_cross_entropy_with_logits(restored, area), logits, retain_graph=True)[0]
    gradient_direct = torch.autograd.grad(F.binary_cross_entropy_with_logits(logits, area), logits)[0]
    torch.testing.assert_close(gradient_restored, gradient_direct)
    prediction = tuple(torch.randn_like(target) for _ in range(4))
    raw = (logits.detach(),)
    expected = guide_loss(raw, target)
    for auxiliary in (dict(guide_logits=raw), dict(guide_logits=raw, posterior_logits=raw)):
        loss, main, guide = training_losses(prediction, auxiliary, target, .1, 0.)
        torch.testing.assert_close(guide, expected)
        torch.testing.assert_close(loss, main + .1 * expected)
    print("Area-label alignment, zero-delta guide budget and CTRL objective equivalence PASS")


def gamma_zero_gradient(variant):
    model = CASANet(module=variant).train()
    a, b = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    target = torch.randint(2, (1, 1, 128, 128)).float()
    outputs, auxiliary = model(a, b, return_aux=True)
    assert torch.count_nonzero(model.context.gamma_context) == 0
    assert all(z.shape[:2] == (1, 1) for z in auxiliary["posterior_logits"])
    posterior = guide_loss(auxiliary["posterior_logits"], target)
    parameters = (model.context.qk.weight, model.context.raw_compression_temperature, model.context.guide.weight)
    posterior_gradients = torch.autograd.grad(posterior, parameters, retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in posterior_gradients)
    total, _, _ = training_losses(outputs, auxiliary, target, .1, 0.)
    actual = torch.autograd.grad(total, model.context.qk.weight)[0]
    if GUIDE_OBJECTIVE == "half_raw_half_posterior_area_bce":
        torch.testing.assert_close(actual, .05 * posterior_gradients[0], rtol=2e-4, atol=2e-7)
        assert actual.abs().sum() > 0
    else:
        assert torch.count_nonzero(actual) == 0
    norms = [float(g.square().mean().sqrt()) for g in posterior_gradients]
    print(variant, GUIDE_OBJECTIVE, "posterior BCE gradient RMS (qk,temp,guide):", norms, "PASS")


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    alignment_and_budget()
    for variant in ("hyperbolic", "matched"):
        gamma_zero_gradient(variant)
