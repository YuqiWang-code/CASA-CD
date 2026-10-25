"""Paired token semantics, dense details, gradients and scale priors."""
import torch

from model.layers.paired_euclidean_context import PairedEuclideanContext
from model.layers.paired_hyperbolic_context import PairedHyperbolicContext
from model.layers.pyramid_context import binomial_highpass, feature_regions, project_tokens, regions
from model.layers.poincare_ops import expmap0, to_lorentz
from model.layers.lorentz_pair_ops import lorentz_normalize


def contrast_semantics():
    op = PairedEuclideanContext().double()
    feature = torch.randn(1, 64, 8, 8, dtype=torch.double, requires_grad=True)
    for level in (0., .01, .5, .99, 1.):
        probability = torch.full((1, 1, 8, 8), level, dtype=torch.double)
        mean, contrast = op.prototypes(feature, probability, 1)
        assert torch.allclose(mean, feature.mean((-2, -1)).unsqueeze(1), atol=1e-12)
        assert contrast.abs().max() < 1e-12, "constant guide must not create contrast content"
        value = project_tokens(op.v_contrast, contrast, 4)
        assert value.abs().max() < 1e-12
    probability = torch.zeros(1, 1, 8, 8, dtype=torch.double)
    probability[..., 0, 0] = 1
    mean, contrast = op.prototypes(feature, probability, 1)
    positive = (feature[..., 0, 0].unsqueeze(1) + mean) / 2
    negative = (feature.flatten(2).sum(-1).unsqueeze(1) - feature[..., 0, 0].unsqueeze(1) + mean) / 64
    assert torch.allclose(contrast, positive - negative, atol=1e-12)
    logits = torch.randn(1, 1, 8, 8, dtype=torch.double, requires_grad=True)
    _, contrast = op.prototypes(feature, logits.sigmoid(), 1)
    grads = torch.autograd.grad(contrast.square().sum(), (feature, logits))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert binomial_highpass(torch.ones(1, 32, 8, 8)).abs().max() == 0
    print("mean/contrast shrinkage, zero constant-guide values and assignment gradients PASS")


def aligned_hyperbolic_tokens():
    op = PairedHyperbolicContext().double()
    midpoint = to_lorentz(expmap0(torch.randn(1, 4, 8, 8, 8, dtype=torch.double) * .1))
    unsigned = torch.rand(1, 4, 8, 8, 8, dtype=torch.double)
    feature = torch.randn(1, 64, 8, 8, dtype=torch.double)
    probability = torch.full((1, 1, 8, 8), .3, dtype=torch.double)
    weights = op.assignments(probability, 2)
    km, ku, value = op.compress(midpoint, unsigned, feature, weights, 2)
    assert torch.allclose(km[:, :, 0::2], km[:, :, 1::2], atol=1e-12)
    assert torch.allclose(ku[:, :, 0::2], ku[:, :, 1::2], atol=1e-12)
    assert torch.allclose(value[:, :, 0::2], value[:, :, 1::2], atol=1e-12)
    probability.zero_()
    probability[..., 0, 0] = 1
    weights = op.assignments(probability, 2)
    km, ku, value = op.compress(midpoint, unsigned, feature, weights, 2)
    wm = weights[:, 0, 1, :, None]
    expected_m = lorentz_normalize((regions(midpoint, 2)[:, :, 0] * wm[:, None]).sum(-2))
    expected_u = (regions(unsigned, 2)[:, :, 0] * wm[:, None]).sum(-2)
    expected_x = (feature_regions(feature, 2)[:, 0] * wm).sum(-2).unsqueeze(1)
    expected_v = project_tokens(op.v, expected_x, 4).squeeze(2)
    assert torch.allclose(km[:, :, 1], expected_m, atol=1e-12)
    assert torch.allclose(ku[:, :, 1], expected_u, atol=1e-12)
    assert torch.allclose(value[:, :, 1], expected_v, atol=1e-12)
    assert weights[0, 0, 1, 0] > .9999, "sparse change weight was diluted"
    print("uniform/change duplicate tokens and aligned midpoint/difference/value pooling PASS")


def scale_priors():
    e = PairedEuclideanContext()
    q, k = torch.zeros(1, 4, 3, 16), torch.zeros(1, 4, 112, 16)
    for bank in range(2):
        attention = e.attention(q, k, 0, bank)
        masses = torch.stack([part.sum(-1) for part in attention.split(e.regions_per_scale, -1)], -1)
        assert torch.allclose(masses, torch.full_like(masses, .25), atol=1e-6)
        assert torch.allclose(attention.sum(-1), torch.ones_like(attention[..., 0]))
    h = PairedHyperbolicContext()
    origin_q, origin_k = torch.zeros(1, 4, 3, 9), torch.zeros(1, 4, 224, 9)
    origin_q[..., 0] = origin_k[..., 0] = 1
    attention, _, _ = h.attention(origin_q, torch.zeros(1, 4, 3, 8), origin_k,
                                   torch.zeros(1, 4, 224, 8), 0)
    masses = torch.stack([part.sum(-1) for part in attention.split(h.tokens_per_scale, -1)], -1)
    assert torch.allclose(masses, torch.full_like(masses, .25), atol=1e-6)
    print("two independent 112-token softmaxes and 224-token source priors PASS")


def contexts():
    sizes, channels = (32, 16, 8, 4), (24, 32, 96, 320)
    for context_type, expected_count in ((PairedEuclideanContext, 39032), (PairedHyperbolicContext, 26688)):
        op = context_type()
        assert sum(p.numel() for p in op.parameters()) == expected_count
        a = [torch.randn(1, c, n, n, requires_grad=True) for c, n in zip(channels, sizes)]
        b = [torch.randn_like(x, requires_grad=True) for x in a]
        tar = [torch.randn(1, 64, n, n, requires_grad=True) for n in sizes]
        output, auxiliary = op(a, b, tar)
        assert all(torch.equal(x, y) for x, y in zip(output, tar)), "zero gates must preserve TAR"
        assert len(auxiliary["guide_logits"]) == 4 and len(auxiliary["boundary_logits"]) == 2
        op.gamma_context.data.fill_(.1)
        op.gamma_detail.data.fill_(.1)
        op.collect_diag = True
        output, _ = op(a, b, tar)
        exchanged, _ = op(b, a, tar)
        assert all(torch.allclose(x, y, atol=2e-6, rtol=1e-5) for x, y in zip(output, exchanged))
        # With TAR held fixed, paired geometry is unordered; TAR itself can remain ordered.
        if isinstance(op, PairedHyperbolicContext):
            pair = op.paired_features(a, b, tar)
            reverse = op.paired_features(b, a, tar)
            assert all(torch.equal(x, y) for x, y in zip(pair[0], reverse[0]))
            assert all(torch.equal(x, -y) for x, y in zip(pair[1], reverse[1]))
            identical = op.paired_features(a, a, tar)
            assert all(x.abs().max() == 0 for x in identical[1])
        else:
            identical = op.paired_features(a, a, tar)
            assert all(x.abs().max() == 0 for x in identical[0])
        gradients = torch.autograd.grad(output[0].square().mean(), tar + a + b, retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
        sum(x.square().mean() for x in output).backward()
        for name, parameter in op.named_parameters():
            if parameter.ndim >= 2:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.abs().sum() > 0, name
        for index, original in enumerate(tar):
            diag = op.last_diag[f"stage{index + 1}"]
            assert diag["finite"] and diag["contexts"] == 224
            assert diag["context_residual_to_feature_rms"] > 0
            if index < 2:
                assert diag["local_residual_to_feature_rms"] > 0
        print(context_type.__name__, expected_count, "identity/swap/dense detail/cross-scale task gradients PASS")


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    contrast_semantics()
    aligned_hyperbolic_tokens()
    scale_priors()
    contexts()


if __name__ == "__main__":
    main()
